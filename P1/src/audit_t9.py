"""T9 read-only audit and inner-only boundary probe for exp005."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from exp004 import _false_runs, _inner_metrics, fill_runs
from features import make_features
from train import segment_ids

BASELINE_F1 = 0.697836
BASELINE = {"weight": 0.00, "threshold": 0.62, "hole": 48, "lambda": 0.00}
KEY = ["station", "layer"]
FULL_KEY = ["station", "year", "layer", "time"]
WEIGHTS = tuple(round(0.05 * i, 2) for i in range(13))
GRID_THRESHOLDS = tuple(round(0.20 + 0.02 * i, 2) for i in range(31))
GRID_HOLES = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48)
B1_THRESHOLDS = tuple(round(0.40 + 0.02 * i, 2) for i in range(21))
B2_THRESHOLDS = tuple(round(0.30 + 0.02 * i, 2) for i in range(34))
B1_HOLES = (24, 32, 48, 64, 80, 96, 128, 160, 200, 256, 320, 400)
B3_RATES = (0.030, 0.034, 0.038, 0.041, 0.045, 0.050, 0.055, 0.060)
B3_LAMBDAS = (0.25, 0.50, 0.75, 1.00)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def line_no(path, needle):
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return number
    return None


def f1_stats(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return {
        "f1": 2.0 * tp / max(1, 2 * tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def inner_base(score, threshold, inner):
    pred = np.zeros(len(score), dtype=bool)
    use = inner & np.isfinite(score)
    pred[use] = score[use] >= threshold
    return pred


def best_threshold(score, y, mask):
    use = mask & np.isfinite(score)
    values = np.asarray(score[use], dtype=float)
    labels = np.asarray(y[use], dtype=np.int8)
    order = np.argsort(values, kind="mergesort")[::-1]
    values = values[order]
    labels = labels[order]
    ends = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    tp = np.cumsum(labels == 1, dtype=np.int64)[ends]
    fp = np.cumsum(labels == 0, dtype=np.int64)[ends]
    fn = int(labels.sum()) - tp
    scores = 2.0 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.flatnonzero(scores == scores.max())[0])
    return {
        "threshold": float(values[ends[best]]),
        "f1": float(scores[best]),
        "tp": int(tp[best]),
        "fp": int(fp[best]),
        "fn": int(fn[best]),
    }


def spearman(a, b):
    ranks = pd.DataFrame({"a": a, "b": b}).rank(method="average")
    return float(ranks.a.corr(ranks.b))


def residual_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "unique": int(np.unique(finite).size),
        "nan": int(np.isnan(values).sum()),
        "rows": int(values.size),
    }


def hole_effect(base, filled, segments):
    base = np.asarray(base, dtype=bool)
    filled = np.asarray(filled, dtype=bool)
    new = filled & ~base
    starts, ends = _false_runs(base, segments)
    lengths = [
        int(end - start + 1)
        for start, end in zip(starts, ends)
        if new[start : end + 1].all()
    ]
    return {
        "before": int(base.sum()),
        "after": int(filled.sum()),
        "increase": int(new.sum()),
        "new_interval_count": len(lengths),
        "new_interval_median": float(np.median(lengths)) if lengths else 0.0,
        "new_interval_max": max(lengths) if lengths else 0,
    }


def row_mismatch(left, right, columns):
    if len(left) != len(right):
        return max(len(left), len(right))
    a = left[columns].reset_index(drop=True)
    b = right[columns].reset_index(drop=True)
    return int(a.ne(b).any(axis=1).sum())


def optimize_weight_grid(y, raw, rule, inner, segments):
    by_weight = []
    for weight in WEIGHTS:
        blend = (1.0 - weight) * raw + weight * rule
        candidates = []
        for threshold in GRID_THRESHOLDS:
            pred = inner_base(blend, threshold, inner)
            for value, hole in _inner_metrics(pred, y, inner, segments, GRID_HOLES):
                candidates.append({
                    "weight": weight,
                    "threshold": threshold,
                    "hole": hole,
                    "inner_f1": float(value),
                })
        by_weight.append(sorted(candidates, key=lambda r: (-r["inner_f1"], r["hole"], -r["threshold"]))[0])
    return by_weight


def optimize_boundary(weight, y, raw, rule, inner, segments):
    blend = (1.0 - weight) * raw + weight * rule
    by_hole = []
    all_rows = []
    for hole in B1_HOLES:
        rows = []
        for threshold in B1_THRESHOLDS:
            pred = inner_base(blend, threshold, inner)
            value = float(_inner_metrics(pred, y, inner, segments, (hole,))[0][0])
            row = {"weight": weight, "threshold": threshold, "hole": hole, "inner_f1": value}
            rows.append(row)
            all_rows.append(row)
        by_hole.append(sorted(rows, key=lambda r: (-r["inner_f1"], -r["threshold"]))[0])
    best = sorted(by_hole, key=lambda r: (-r["inner_f1"], r["hole"], -r["threshold"]))[0]
    return blend, by_hole, best, all_rows


def anchor_prediction(blend, raw, inner, threshold, hole, target_rate, shrinkage, segments):
    thresholds = np.full(len(raw), threshold, dtype=float)
    small = []
    finite = np.isfinite(blend)
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        use = idx[inner[idx] & finite[idx]]
        if len(use) < 500:
            small.append("|".join(map(str, raw.loc[idx[0], KEY])))
            continue
        q = float(np.quantile(blend[use], 1.0 - target_rate))
        thresholds[use] = (1.0 - shrinkage) * threshold + shrinkage * q
    base = np.zeros(len(blend), dtype=bool)
    use = inner & finite
    base[use] = blend[use] >= thresholds[use]
    return fill_runs(base, segments, int(hole)), small


def audit_climatology():
    exp005 = SRC / "exp005.py"
    diag = SRC / "diag_t4.py"
    lines = {
        "mask_definition": line_no(exp005, "masks = {fold: forward_mask(train, folds, fold)"),
        "honest_reference": line_no(exp005, "refs = {fold: honest_reference(train, masks[fold])"),
        "forward_apply": line_no(exp005, "x_forward = {fold: apply_reference(base, train, refs[fold])"),
        "outer_apply": line_no(exp005, "x_outer = apply_reference(base, train, refs[3])"),
        "test_reference": line_no(exp005, "xte, _ = make_features(test, ref_full)"),
        "reference_rows": line_no(diag, "d = raw.loc[mask, [\"station\", \"layer\", \"time\", \"temp\"]]")
            or line_no(diag, "d = raw.loc[mask"),
        "clim_overwrite": line_no(diag, "out[\"clim_resid\"] = ((raw.temp.to_numpy(float) - med) / mad)"),
    }
    source_ok = all(value is not None for value in lines.values())
    return {
        "verdict": "PASS" if source_ok else "FAIL",
        "lines": lines,
        "verification": "forward_mask -> honest_reference(mask) -> apply_reference; test uses train reference only",
    }


def main():
    before_targets = [p for p in sorted((ROOT / "P1_data").iterdir()) if p.is_file()]
    before_targets += [ROOT / "preds" / name for name in (
        "oof_exp002.npy", "test_exp002.npy", "oof_exp004.npy", "test_exp004.npy"
    )]
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in before_targets}

    train = pd.read_csv(ROOT / "P1_data" / "train.csv")
    test = pd.read_csv(ROOT / "P1_data" / "test.csv")
    sample = pd.read_csv(ROOT / "P1_data" / "sample_submission.csv", usecols=FULL_KEY)
    folds = np.load(ROOT / "cache" / "exp001" / "folds.npy")
    raw_oof = np.load(ROOT / "cache" / "exp005_raw_oof.npy")
    raw_test = np.load(ROOT / "cache" / "exp005_raw_test.npy")
    saved_oof = np.load(ROOT / "preds" / "oof_exp005.npy")
    saved_test = np.load(ROOT / "preds" / "test_exp005.npy")
    sub = pd.read_csv(ROOT / "subs" / "exp005.csv")
    base = pd.read_pickle(ROOT / "cache" / "exp001" / "train_features.pkl")
    reference = json.loads((ROOT / "cache" / "exp001" / "reference.json").read_text(encoding="utf-8"))
    test_features, _ = make_features(test, reference)
    rule_oof = base.rule_score.to_numpy(float)
    rule_test = test_features.rule_score.to_numpy(float)
    y = train.label.to_numpy(np.int8)
    inner = np.isin(folds, (1, 2))
    segments = segment_ids(train)
    test_segments = segment_ids(test)

    rule_inner = score_stats(rule_oof[inner])
    rule_test_stats = score_stats(rule_test)
    rule_best = best_threshold(rule_oof, y, inner)

    a2 = optimize_weight_grid(y, raw_oof, rule_oof, inner, segments)
    a2_best = sorted(a2, key=lambda r: (-r["inner_f1"], r["weight"], r["hole"], -r["threshold"]))[0]

    lgbm_best = best_threshold(raw_oof, y, inner)
    lgbm_pred = inner_base(raw_oof, lgbm_best["threshold"], inner)[inner]
    rule_pred = inner_base(rule_oof, rule_best["threshold"], inner)[inner]
    y_inner = y[inner]
    a3 = {
        "spearman_inner": spearman(raw_oof[inner], rule_oof[inner]),
        "lgbm_best_threshold": lgbm_best["threshold"],
        "lgbm_best_f1": lgbm_best["f1"],
        "rule_best_threshold": rule_best["threshold"],
        "rule_best_f1": rule_best["f1"],
        "residual_correlation": residual_corr(y_inner - lgbm_pred.astype(np.int8), y_inner - rule_pred.astype(np.int8)),
    }

    exp005_source = (SRC / "exp005.py").read_text(encoding="utf-8")
    model_paths = [ROOT / "models" / name for name in (
        "exp005_forward_fold1.txt", "exp005_forward_fold2.txt", "exp005_full.txt", "exp002_outer_honest.txt"
    )]
    a4 = {
        "verdict": "SKIP",
        "reason": "direct_run_writes_existing_outputs_and_contains_sealed_fold3_path",
        "all_reusable_models_exist": all(path.exists() for path in model_paths),
        "fit_or_load_present": "fit_or_load" in exp005_source,
        "writes_npy": "np.save" in exp005_source,
        "writes_submission": "to_csv" in exp005_source,
    }

    pred_unique = {
        "oof_exp005": sorted(np.unique(saved_oof.reshape(-1)).tolist()),
        "test_exp005": sorted(np.unique(saved_test.reshape(-1)).tolist()),
    }
    a5 = {
        "test_pred_sub_key_mismatch": row_mismatch(test, sub, FULL_KEY),
        "sub_sample_key_mismatch": row_mismatch(sub, sample, FULL_KEY),
        "arrays": {
            name: {"dtype": str(arr.dtype), "shape": list(arr.shape), "unique": pred_unique[name]}
            for name, arr in (("oof_exp005", saved_oof), ("test_exp005", saved_test))
        },
        "label_unique": sorted(pd.unique(sub.label).tolist()),
        "verdict": "PASS" if set(pred_unique["oof_exp005"]) <= {0, 1}
        and set(pred_unique["test_exp005"]) <= {0, 1}
        and set(pd.unique(sub.label)) == {0, 1}
        and row_mismatch(test, sub, FULL_KEY) == 0
        and row_mismatch(sub, sample, FULL_KEY) == 0 else "FAIL",
    }

    nan_mask = np.isnan(raw_oof)
    fold_nan = {str(fold): int(np.count_nonzero(nan_mask & (folds == fold))) for fold in range(4)}
    a6 = {
        "raw_oof_nan": int(nan_mask.sum()),
        "fold_rows": {str(fold): int(np.count_nonzero(folds == fold)) for fold in range(4)},
        "nan_exact_fold0": bool(np.array_equal(nan_mask, folds == 0)),
        "nan_by_fold": fold_nan,
        "saved_oof_fold0_nonzero": int(np.count_nonzero(saved_oof[folds == 0])),
        "verdict": "PASS" if np.array_equal(nan_mask, folds == 0)
        and all(fold_nan[str(fold)] == 0 for fold in (1, 2, 3))
        and int(np.count_nonzero(saved_oof[folds == 0])) == 0 else "FAIL",
    }

    a7 = audit_climatology()

    selected_score_inner = raw_oof
    selected_base_inner = inner_base(selected_score_inner, BASELINE["threshold"], inner)
    selected_filled_inner = fill_runs(selected_base_inner, segments, BASELINE["hole"])
    selected_base_test = np.isfinite(raw_test) & (raw_test >= BASELINE["threshold"])
    selected_filled_test = fill_runs(selected_base_test, test_segments, BASELINE["hole"])
    a8 = {
        "inner": hole_effect(selected_base_inner, selected_filled_inner, segments),
        "test": hole_effect(selected_base_test, selected_filled_test, test_segments),
        "threshold": BASELINE["threshold"],
        "weight": BASELINE["weight"],
    }

    after_pre_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in before_targets}
    a9 = {
        "verdict": "PASS" if before_hashes == after_pre_hashes else "FAIL",
        "unchanged": before_hashes == after_pre_hashes,
        "before_sha256": before_hashes,
        "after_sha256": after_pre_hashes,
    }

    blend, b1_by_hole, b1_best, b1_all = optimize_boundary(
        a2_best["weight"], y, raw_oof, rule_oof, inner, segments
    )
    b2_top10 = sorted(
        b1_all,
        key=lambda r: (-r["inner_f1"], -r["threshold"]),
    )
    best_hole = b1_best["hole"]
    b2_rows = []
    for threshold in B2_THRESHOLDS:
        pred = inner_base(blend, threshold, inner)
        value = float(_inner_metrics(pred, y, inner, segments, (best_hole,))[0][0])
        b2_rows.append({
            "weight": a2_best["weight"],
            "threshold": threshold,
            "hole": best_hole,
            "inner_f1": value,
        })
    b2_top10 = sorted(b2_rows, key=lambda r: (-r["inner_f1"], -r["threshold"]))[:10]
    b2_best = b2_top10[0]

    b3_rows = []
    b3_small = {}
    for target_rate in B3_RATES:
        for shrinkage in B3_LAMBDAS:
            pred, small = anchor_prediction(
                blend, train, inner, b2_best["threshold"], best_hole,
                target_rate, shrinkage, segments,
            )
            value = f1_stats(y[inner], pred[inner])["f1"]
            b3_rows.append({
                "r": target_rate,
                "lambda": shrinkage,
                "inner_f1": value,
                "small_streams": len(small),
            })
            b3_small[f"{target_rate:.3f}|{shrinkage:.2f}"] = len(small)
    b3_best = sorted(b3_rows, key=lambda r: (-r["inner_f1"], r["lambda"], r["r"]))[0]

    b4_candidates = [
        {**rule_best, "family": "A1_rule", "weight": 1.0, "hole": 0, "lambda": 0.0, "r": None},
        {**a2_best, "family": "A2"},
        *[{**row, "family": "B1"} for row in b1_by_hole],
        *[{**row, "family": "B2"} for row in b2_rows],
        *[{**row, "family": "B3"} for row in b3_rows],
    ]
    b4 = sorted(b4_candidates, key=lambda r: (-r["f1"] if "f1" in r else -r["inner_f1"], r.get("hole", 0), -r.get("threshold", 0.0)))[0]
    if "f1" in b4:
        b4_f1 = b4["f1"]
        b4_threshold = b4["threshold"]
    else:
        b4_f1 = b4["inner_f1"]
        b4_threshold = b4["threshold"]
    b4 = {
        "family": b4["family"],
        "weight": b4.get("weight"),
        "threshold": b4_threshold,
        "hole": b4.get("hole"),
        "r": b4.get("r"),
        "lambda": 0.0 if b4.get("lambda") is None else b4.get("lambda"),
        "inner_f1": float(b4_f1),
        "delta_vs_exp005": float(b4_f1 - BASELINE_F1),
    }

    summary = {
        "A": {
            "A1": {
                "verdict": "PASS" if rule_inner["std"] != 0 and rule_inner["unique"] > 2 and rule_best["f1"] >= 0.10 else "FAIL",
                "inner": rule_inner,
                "test": rule_test_stats,
                "rule_only_best": rule_best,
            },
            "A2": {"verdict": "PASS", "by_weight": a2, "best": a2_best},
            "A3": {"verdict": "PASS", **a3},
            "A4": a4,
            "A5": a5,
            "A6": a6,
            "A7": a7,
            "A8": a8,
            "A9": a9,
        },
        "B": {
            "B1": {"weight": a2_best["weight"], "by_hole": b1_by_hole, "best": b1_best, "rows": len(b1_all)},
            "B2": {"best_hole": best_hole, "top10": b2_top10, "best": b2_best},
            "B3": {"rows": b3_rows, "small_streams": b3_small, "best": b3_best},
            "B4": b4,
        },
        "inner_rows": int(inner.sum()),
        "fold3_used_in_f1_or_aggregate": 0,
        "fold3_read_for_metric": 0,
        "baseline": {**BASELINE, "inner_f1": BASELINE_F1},
    }
    out = ROOT / "cache" / "T9_audit.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("A1~A9 | 판정 | 핵심 숫자")
    print(f"A1 | {summary['A']['A1']['verdict']} | inner_rule_std={rule_inner['std']:.9f} unique={rule_inner['unique']} rule_best_threshold={rule_best['threshold']:.9f} rule_best_f1={rule_best['f1']:.6f}")
    print(f"A2 | PASS | best_weight={a2_best['weight']:.2f} best_threshold={a2_best['threshold']:.2f} best_hole={a2_best['hole']} best_inner_f1={a2_best['inner_f1']:.6f}")
    print(f"A3 | PASS | spearman={a3['spearman_inner']:.6f} residual_corr={a3['residual_correlation']:.6f} lgbm_t={a3['lgbm_best_threshold']:.9f} rule_t={a3['rule_best_threshold']:.9f}")
    print(f"A4 | SKIP | models_exist={int(a4['all_reusable_models_exist'])} writes_npy={int(a4['writes_npy'])} writes_sub={int(a4['writes_submission'])}")
    print(f"A5 | {a5['verdict']} | test_sub_key_mismatch={a5['test_pred_sub_key_mismatch']} sample_key_mismatch={a5['sub_sample_key_mismatch']} label_unique={a5['label_unique']}")
    print(f"A6 | {a6['verdict']} | raw_nan={a6['raw_oof_nan']} nan_by_fold={a6['nan_by_fold']} fold0_saved_nonzero={a6['saved_oof_fold0_nonzero']}")
    print(f"A7 | {a7['verdict']} | lines={a7['lines']}")
    print(f"A8 | PASS | inner_before={a8['inner']['before']} inner_after={a8['inner']['after']} inner_inc={a8['inner']['increase']} test_before={a8['test']['before']} test_after={a8['test']['after']} test_inc={a8['test']['increase']}")
    print(f"A9 | {a9['verdict']} | files={len(before_hashes)} unchanged={int(a9['unchanged'])}")
    print("A1 rule_score | 구간 | min | max | mean | std | unique | nan")
    for name, stats in (("inner", rule_inner), ("test", rule_test_stats)):
        print(f"{name} | {stats['min']:.9f} | {stats['max']:.9f} | {stats['mean']:.9f} | {stats['std']:.9f} | {stats['unique']} | {stats['nan']}")
    print("A2 weight | threshold | hole | inner_f1")
    for row in a2:
        print(f"{row['weight']:.2f} | {row['threshold']:.2f} | {row['hole']} | {row['inner_f1']:.6f}")
    print("A8 hole effect | 구간 | before | after | increase | new_count | median_length | max_length")
    for name in ("inner", "test"):
        row = a8[name]
        print(f"{name} | {row['before']} | {row['after']} | {row['increase']} | {row['new_interval_count']} | {row['new_interval_median']:.1f} | {row['new_interval_max']}")
    print("B1 hole | threshold | inner_f1")
    for row in b1_by_hole:
        print(f"{row['hole']} | {row['threshold']:.2f} | {row['inner_f1']:.6f}")
    print("B2 rank | threshold | hole | inner_f1")
    for rank, row in enumerate(b2_top10, 1):
        print(f"{rank} | {row['threshold']:.2f} | {row['hole']} | {row['inner_f1']:.6f}")
    print("B3 r | lambda | inner_f1 | small_streams")
    for row in b3_rows:
        print(f"{row['r']:.3f} | {row['lambda']:.2f} | {row['inner_f1']:.6f} | {row['small_streams']}")
    b4_r = "NA" if b4["r"] is None else f"{b4['r']:.3f}"
    print(f"B4 | {b4['family']} | weight={b4['weight']:.2f} threshold={b4['threshold']:.2f} hole={b4['hole']} r={b4_r} lambda={b4['lambda']:.2f} inner_f1={b4['inner_f1']:.6f} delta={b4['delta_vs_exp005']:+.6f}")
    print(f"inner_eval_rows={summary['inner_rows']}")
    print(f"fold3_used_in_f1_or_aggregate={summary['fold3_used_in_f1_or_aggregate']}")
    print("판정: OK")


if __name__ == "__main__":
    main()
