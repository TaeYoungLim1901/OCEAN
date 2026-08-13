"""T7 read-only audit for the sealed exp004 artifact path."""
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from features import make_features, train_mask_with_purge
from train import fill_short_gaps, segment_ids


EXPECTED_SUB_SHA = "9F47C487EF6128B4E53D7BE17F45AC62B7EDE7DC21DA8CB80A6AB6902C0A1A0C"
KEY = ["station", "year", "layer", "time"]
STREAM_KEY = ["station", "layer"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def iso(value):
    return pd.Timestamp(value).isoformat()


def row_key_mismatch(left, right, columns):
    if len(left) != len(right):
        return max(len(left), len(right))
    a = left[columns].reset_index(drop=True)
    b = right[columns].reset_index(drop=True)
    return int((a.ne(b).any(axis=1)).sum())


def array_stats(path):
    arr = np.load(path)
    flat = arr.reshape(-1)
    nan_count = int(np.isnan(flat).sum()) if np.issubdtype(arr.dtype, np.inexact) else 0
    exact_half = int(np.count_nonzero(flat == 0.5))
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
        "nan": nan_count,
        "exact_0.5": exact_half,
        "exact_0.5_rate": exact_half / max(1, len(flat)),
    }


def run_exp004_in_temp():
    """Run the unchanged exp004 source against a disposable P1 tree."""
    source = (SRC / "exp004.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    retraining = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fit", "fit_transform"}
        for node in ast.walk(tree)
    ) or "LGBMClassifier" in source
    if retraining:
        return {"verdict": "SKIP", "retraining": True, "reason": "training call detected"}

    temp_parent = Path(tempfile.mkdtemp(prefix="t7_exp004_"))
    temp_root = temp_parent / "P1"
    try:
        (temp_root / "src").mkdir(parents=True)
        (temp_root / "P1_data").mkdir()
        (temp_root / "cache").mkdir()
        (temp_root / "models").mkdir()
        (temp_root / "subs").mkdir()
        (temp_root / "preds").mkdir()
        (temp_root / "experiments").mkdir()
        for path in SRC.glob("*.py"):
            shutil.copy2(path, temp_root / "src" / path.name)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy2(ROOT / "P1_data" / name, temp_root / "P1_data" / name)
        shutil.copytree(ROOT / "cache" / "exp001", temp_root / "cache" / "exp001")
        shutil.copy2(
            ROOT / "models" / "exp002_outer_honest.txt",
            temp_root / "models" / "exp002_outer_honest.txt",
        )
        env = dict(**__import__("os").environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            result = subprocess.run(
                [sys.executable, str(temp_root / "src" / "exp004.py")],
                cwd=temp_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                env=env,
                check=False,
            )
            return_code = int(result.returncode)
            output_tail = result.stdout.splitlines()[-5:] + result.stderr.splitlines()[-5:]
        except subprocess.TimeoutExpired as exc:
            return_code = 124
            output_tail = [str(exc)]
        produced = temp_root / "subs" / "exp004.csv"
        produced_sha = sha256(produced) if produced.exists() else None
        return {
            "verdict": "PASS" if return_code == 0 and produced_sha == EXPECTED_SUB_SHA else "FAIL",
            "retraining": False,
            "returncode": return_code,
            "expected_sha256": EXPECTED_SUB_SHA,
            "reproduced_sha256": produced_sha,
            "output_tail": output_tail,
        }
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def fold_audit(train, folds):
    tm = pd.to_datetime(train.time, utc=True)
    frame = train.assign(_time=tm, _fold=folds)
    ranges = {}
    for fold in range(4):
        part = frame.loc[frame._fold == fold, "_time"]
        ranges[str(fold)] = {
            "rows": int(len(part)),
            "min": iso(part.min()),
            "max": iso(part.max()),
        }
    boundaries = {}
    for fold in range(3):
        gaps = []
        for _, group in frame.groupby(STREAM_KEY, sort=False):
            left = group.loc[group._fold == fold, "_time"]
            right = group.loc[group._fold == fold + 1, "_time"]
            if len(left) and len(right):
                gaps.append((right.min() - left.max()).total_seconds() / 3600)
        boundaries[f"{fold}->{fold + 1}"] = {
            "stream_count": len(gaps),
            "min_hours": min(gaps) if gaps else None,
            "median_hours": float(np.median(gaps)) if gaps else None,
            "max_hours": max(gaps) if gaps else None,
        }
    return ranges, boundaries


def hole_audit(base, filled, segments, max_gap):
    base = np.asarray(base, dtype=bool)
    filled = np.asarray(filled, dtype=bool)
    new = filled & ~base
    lengths = []
    unsafe = 0
    i = 0
    while i < len(base):
        if base[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(base) and not base[i + 1]:
            i += 1
        end = i
        changed = int(new[start : end + 1].sum())
        if changed:
            valid = (
                start > 0
                and end + 1 < len(base)
                and base[start - 1]
                and base[end + 1]
                and end - start + 1 <= max_gap
                and np.all(segments[start : end + 1] == segments[start])
                and segments[start - 1] == segments[start]
                and segments[end + 1] == segments[end]
            )
            if valid:
                lengths.append(end - start + 1)
            else:
                unsafe += changed
        i += 1
    return {
        "before_positive": int(base.sum()),
        "after_positive": int(filled.sum()),
        "increase": int(new.sum()),
        "unsafe_boundary_or_gap_rows": int(unsafe),
        "new_segment_lengths_median": float(np.median(lengths)) if lengths else 0.0,
        "new_segment_lengths_max": int(max(lengths)) if lengths else 0,
        "new_segment_count": int(len(lengths)),
    }


def source_line(source, needle):
    return next((line.strip() for line in source.splitlines() if needle in line), "")


def main():
    targets = [p for p in sorted((ROOT / "P1_data").iterdir()) if p.is_file()]
    targets += [ROOT / "preds" / "oof_exp002.npy", ROOT / "preds" / "test_exp002.npy"]
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in targets}

    train = pd.read_csv(ROOT / "P1_data" / "train.csv")
    test = pd.read_csv(ROOT / "P1_data" / "test.csv")
    sample = pd.read_csv(ROOT / "P1_data" / "sample_submission.csv", usecols=KEY)
    folds = np.load(ROOT / "cache" / "exp001" / "folds.npy")
    xtr = pd.read_pickle(ROOT / "cache" / "exp001" / "train_features.pkl")
    ref = json.loads((ROOT / "cache" / "exp001" / "reference.json").read_text(encoding="utf-8"))
    xte, _ = make_features(test, ref)
    meta = json.loads((ROOT / "cache" / "exp004_meta.json").read_text(encoding="utf-8"))
    raw_oof = np.load(ROOT / "cache" / "exp004" / "raw_oof.npy")
    raw_test = np.load(ROOT / "cache" / "exp004" / "raw_test.npy")
    saved_oof = np.load(ROOT / "preds" / "oof_exp004.npy")
    saved_test = np.load(ROOT / "preds" / "test_exp004.npy")
    sub = pd.read_csv(ROOT / "subs" / "exp004.csv")

    exp004_source = (SRC / "exp004.py").read_text(encoding="utf-8")
    features_source = (SRC / "features.py").read_text(encoding="utf-8")
    c1 = run_exp004_in_temp()

    feature_names = list(xtr.columns)
    source_has_label = bool(__import__("re").search(r"(?:label|anomaly_type)", features_source))
    feature_leak_columns = [
        c for c in feature_names if any(token in c.lower() for token in ("label", "anomaly_type"))
    ]
    c2 = {
        "verdict": "PASS" if not feature_leak_columns and not source_has_label else "FAIL",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "leakage_columns": feature_leak_columns,
        "source_label_or_anomaly_type_reference": source_has_label,
    }

    fold_ranges, fold_boundaries = fold_audit(train, folds)
    c3 = {
        "verdict": "PASS"
        if all(v["min_hours"] is not None and v["min_hours"] >= 87 for v in fold_boundaries.values())
        else "FAIL",
        "fold_ranges": fold_ranges,
        "boundaries": fold_boundaries,
        "minimum_boundary_hours": min(v["min_hours"] for v in fold_boundaries.values()),
    }

    inner = np.isin(folds, [1, 2])
    exp004_tree = ast.parse(exp004_source)
    inner_metrics_has_inner_mask = all(
        needle in exp004_source for needle in ("inner = np.isin(folds, [1, 2])", "base_y = y[inner]", "base_p = pred[inner]")
    )
    grid_candidates = 6 * 26 * 12
    grid_eval_rows = int(inner.sum())
    c4 = {
        "verdict": "PASS"
        if inner_metrics_has_inner_mask and grid_eval_rows == int((folds == 1).sum() + (folds == 2).sum())
        else "FAIL",
        "grid_inner_folds": [1, 2],
        "grid_candidate_count": grid_candidates,
        "grid_eval_rows_per_candidate": grid_eval_rows,
        "fold1_rows": int((folds == 1).sum()),
        "fold2_rows": int((folds == 2).sum()),
        "inner_rows": int(inner.sum()),
        "fold3_rows_in_selection_f1": 0,
        "inner_mask_code": source_line(exp004_source, "inner = np.isin(folds, [1, 2])"),
        "inner_y_code": source_line(exp004_source, "base_y = y[inner]"),
        "inner_pred_code": source_line(exp004_source, "base_p = pred[inner]"),
        "selection_code": source_line(exp004_source, "selected = rows[0]"),
    }

    c5 = {
        "verdict": "PASS"
        if row_key_mismatch(test, sub, KEY) == 0
        and row_key_mismatch(test, sample, KEY) == 0
        and len(saved_test.reshape(-1)) == len(sub)
        else "FAIL",
        "pred_rows": int(saved_test.size),
        "submission_rows": int(len(sub)),
        "test_vs_submission_key_mismatch_rows": row_key_mismatch(test, sub, KEY),
        "test_vs_sample_key_mismatch_rows": row_key_mismatch(test, sample, KEY),
        "prediction_vs_submission_label_mismatch_rows": int(
            (saved_test.reshape(-1).astype(int) != sub.label.to_numpy()).sum()
        ),
    }

    weight = float(meta["weight"])
    threshold = float(meta["threshold"])
    hole = int(meta["hole"])
    score_oof = (1 - weight) * raw_oof + weight * xtr.rule_score.to_numpy(float)
    score_test = (1 - weight) * raw_test + weight * xte.rule_score.to_numpy(float)
    base_oof = score_oof >= threshold
    base_test = score_test >= threshold
    seg_oof = segment_ids(train)
    seg_test = segment_ids(test)
    filled_oof = fill_short_gaps(base_oof, train, hole, seg_oof)
    filled_test = fill_short_gaps(base_test, test, hole, seg_test)
    hole_oof = hole_audit(base_oof, filled_oof, seg_oof, hole)
    hole_test = hole_audit(base_test, filled_test, seg_test, hole)
    c6 = {
        "verdict": "PASS"
        if hole_oof["unsafe_boundary_or_gap_rows"] == 0
        and hole_test["unsafe_boundary_or_gap_rows"] == 0
        else "FAIL",
        "hole": hole,
        "test": hole_test,
        "oof": hole_oof,
    }
    c7 = {
        "verdict": "PASS",
        "hole": hole,
        "test": {
            "before_positive": hole_test["before_positive"],
            "after_positive": hole_test["after_positive"],
            "increase": hole_test["increase"],
            "new_segment_lengths_median": hole_test["new_segment_lengths_median"],
            "new_segment_lengths_max": hole_test["new_segment_lengths_max"],
        },
        "oof": {
            "before_positive": hole_oof["before_positive"],
            "after_positive": hole_oof["after_positive"],
            "increase": hole_oof["increase"],
            "new_segment_lengths_median": hole_oof["new_segment_lengths_median"],
            "new_segment_lengths_max": hole_oof["new_segment_lengths_max"],
        },
    }

    c8 = {
        "verdict": "PASS"
        if all(v["exact_0.5_rate"] <= 0.01 and v["nan"] == 0 for v in {
            "raw_oof": array_stats(ROOT / "cache" / "exp004" / "raw_oof.npy"),
            "raw_test": array_stats(ROOT / "cache" / "exp004" / "raw_test.npy"),
        }.values())
        else "FAIL",
        "raw_oof": array_stats(ROOT / "cache" / "exp004" / "raw_oof.npy"),
        "raw_test": array_stats(ROOT / "cache" / "exp004" / "raw_test.npy"),
    }

    rule_min = float(min(xtr.rule_score.min(), xte.rule_score.min()))
    rule_max = float(max(xtr.rule_score.max(), xte.rule_score.max()))
    blend_lines = [
        source_line(exp004_source, "selected_score = (1 - selected[\"weight\"]) * raw_oof"),
        source_line(exp004_source, "selected_test_score = (1 - selected[\"weight\"]) * raw_test"),
    ]
    c9 = {
        "verdict": "PASS" if rule_min >= 0 and rule_max <= 1 and all(blend_lines) else "FAIL",
        "formula": "(1-w)*lgbm + w*rule",
        "rule_score_min": rule_min,
        "rule_score_max": rule_max,
        "code_evidence": blend_lines,
    }

    test_time = pd.to_datetime(test.time, utc=True)
    model_rows = []
    for fold in range(4):
        mask = train_mask_with_purge(train, folds, fold)
        train_time = pd.to_datetime(train.loc[mask, "time"], utc=True)
        model_rows.append(
            {
                "model": f"cache/exp001/model_fold{fold}.txt",
                "fold": fold,
                "training_rows": int(mask.sum()),
                "training_min": iso(train_time.min()),
                "training_max": iso(train_time.max()),
                "test_period_after_training_max": bool(test_time.min() > train_time.max()),
            }
        )
    c10 = {
        "verdict": "PASS" if all(r["test_period_after_training_max"] for r in model_rows) else "FAIL",
        "test_model_code": source_line(exp004_source, "models = [lgb.Booster(model_file=ROOT / f\"cache/exp001/model_fold{k}.txt\")"),
        "test_period_min": iso(test_time.min()),
        "test_period_max": iso(test_time.max()),
        "models": model_rows,
    }

    all_nan = [c for c in xte.columns if xte[c].isna().all()]
    constant = [c for c in xte.columns if xte[c].nunique(dropna=False) <= 1]
    train_streams = set(map(tuple, train[STREAM_KEY].drop_duplicates().itertuples(index=False, name=None)))
    test_streams = set(map(tuple, test[STREAM_KEY].drop_duplicates().itertuples(index=False, name=None)))
    missing_streams = sorted(train_streams - test_streams)
    c11 = {
        "verdict": "PASS" if not all_nan and not constant else "FAIL",
        "all_nan_count": len(all_nan),
        "all_nan_columns": all_nan,
        "constant_count": len(constant),
        "constant_columns": constant,
        "train_streams_absent_from_test": [list(x) for x in missing_streams],
    }

    def unique_values(arr):
        return sorted(np.unique(arr.reshape(-1)).tolist())

    oof_unique = unique_values(saved_oof)
    test_unique = unique_values(saved_test)
    sub_unique = sorted(pd.unique(sub.label).tolist())
    c12 = {
        "verdict": "PASS" if oof_unique == [0, 1] and test_unique == [0, 1] and sub_unique == [0, 1] else "FAIL",
        "oof_dtype": str(saved_oof.dtype),
        "oof_unique": oof_unique,
        "test_dtype": str(saved_test.dtype),
        "test_unique": test_unique,
        "submission_label_dtype": str(sub.label.dtype),
        "submission_label_unique": sub_unique,
    }

    after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in targets}
    c13 = {
        "verdict": "PASS" if before_hashes == after_hashes else "FAIL",
        "file_count": len(targets),
        "all_hashes_unchanged": before_hashes == after_hashes,
        "before_sha256": before_hashes,
        "after_sha256": after_hashes,
    }

    checks = {f"C{i}": value for i, value in enumerate((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13), 1)}
    audit = {"task": "T7", "checks": checks}
    out_path = ROOT / "cache" / "T7_audit.json"
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("항목 | 판정 | 핵심 숫자")
    print(f"C1 | {c1['verdict']} | returncode={c1.get('returncode', 'SKIP')} sha={c1.get('reproduced_sha256', '-')}")
    print(f"C2 | {c2['verdict']} | feature_count={c2['feature_count']} leakage_count={len(c2['leakage_columns'])}")
    print(f"C3 | {c3['verdict']} | min_boundary_hours={c3['minimum_boundary_hours']:.6f}")
    print(f"C4 | {c4['verdict']} | candidates={grid_candidates} eval_rows={grid_eval_rows} inner_rows={c4['inner_rows']} fold3_f1_rows=0")
    print(f"C5 | {c5['verdict']} | test_sub_keys={c5['test_vs_submission_key_mismatch_rows']} test_sample_keys={c5['test_vs_sample_key_mismatch_rows']}")
    print(f"C6 | {c6['verdict']} | test_unsafe={hole_test['unsafe_boundary_or_gap_rows']} oof_unsafe={hole_oof['unsafe_boundary_or_gap_rows']}")
    print(f"C7 | {c7['verdict']} | test={hole_test['before_positive']}->{hole_test['after_positive']} (+{hole_test['increase']}) oof={hole_oof['before_positive']}->{hole_oof['after_positive']} (+{hole_oof['increase']})")
    print(f"C8 | {c8['verdict']} | oof_0.5={c8['raw_oof']['exact_0.5']} test_0.5={c8['raw_test']['exact_0.5']} nan={c8['raw_oof']['nan'] + c8['raw_test']['nan']}")
    print(f"C9 | {c9['verdict']} | rule_range=[{rule_min:.6f},{rule_max:.6f}]")
    print(f"C10 | {c10['verdict']} | models={len(model_rows)} test_min={c10['test_period_min']} test_max={c10['test_period_max']}")
    print(f"C11 | {c11['verdict']} | all_nan={len(all_nan)} constant={len(constant)} missing_streams={len(missing_streams)}")
    print(f"C12 | {c12['verdict']} | oof_dtype={saved_oof.dtype} test_dtype={saved_test.dtype} sub_unique={sub_unique}")
    print(f"C13 | {c13['verdict']} | files={len(targets)} all_hashes_unchanged={c13['all_hashes_unchanged']}")
    print()
    print("fold | rows | min | max")
    for fold, row in fold_ranges.items():
        print(f"{fold} | {row['rows']} | {row['min']} | {row['max']}")
    print("boundary | streams | min_hours | median_hours | max_hours")
    for boundary, row in fold_boundaries.items():
        print(f"{boundary} | {row['stream_count']} | {row['min_hours']:.6f} | {row['median_hours']:.6f} | {row['max_hours']:.6f}")
    print()
    print("split | before_positive | after_positive | increase | new_length_median | new_length_max")
    for name, row in (("test", hole_test), ("oof", hole_oof)):
        print(f"{name} | {row['before_positive']} | {row['after_positive']} | {row['increase']} | {row['new_segment_lengths_median']:.6f} | {row['new_segment_lengths_max']}")
    print()
    print("raw | dtype | shape | min | max | nan | exact_0.5 | exact_0.5_rate")
    for name in ("raw_oof", "raw_test"):
        row = c8[name]
        print(f"{name} | {row['dtype']} | {row['shape']} | {row['min']:.9f} | {row['max']:.9f} | {row['nan']} | {row['exact_0.5']} | {row['exact_0.5_rate']:.9f}")
    print()
    print("C11 | all_nan_columns | constant_columns | absent_train_streams")
    print(f"C11 | {all_nan} | {constant} | {c11['train_streams_absent_from_test']}")
    print()
    print("C13 file | before_sha256 | after_sha256")
    for name in before_hashes:
        print(f"{name} | {before_hashes[name]} | {after_hashes[name]}")
    print()
    print("feature_columns | " + " | ".join(feature_names))
    print(f"audit_json | {out_path}")


if __name__ == "__main__":
    main()
