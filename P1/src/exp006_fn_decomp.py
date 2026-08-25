"""T10: exp005 false-negative decomposition without model changes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_t4 import forward_mask
from exp004 import fill_runs
from train import segment_ids


SEED = 42
THRESHOLD = 0.62
HOLE = 48
SPLITS = {
    "inner": (1, 2),
    "outer_fold3": (3,),
}
KEY = ["station", "year", "layer"]


def f1_stats(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "F1": 2.0 * tp / max(1, 2 * tp + fp + fn),
    }


def _stream_orders(raw):
    times = pd.to_datetime(raw.time, utc=True)
    for stream, values in raw.groupby(KEY, sort=False, dropna=False).groups.items():
        idx = np.asarray(list(values), dtype=np.int64)
        order = times.iloc[idx].sort_values(kind="stable").index.to_numpy(np.int64)
        intervals = np.rint(
            times.iloc[order].diff().dt.total_seconds().to_numpy()[1:] / 60.0
        ).astype(np.int64)
        positive = intervals[intervals > 0]
        modal = int(pd.Series(positive).mode().iloc[0]) if len(positive) else 10
        breaks = np.r_[True, intervals != modal]
        yield stream, order, breaks


def positive_segments(raw, positive, mask=None):
    """Positive runs in sorted station-year-layer streams, split at modal gaps."""
    positive = np.asarray(positive, dtype=bool)
    mask = np.ones(len(raw), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    segments = []
    segment_id = 0
    for stream, order, breaks in _stream_orders(raw):
        z = positive[order]
        starts = np.flatnonzero(z & (np.r_[True, ~z[:-1]] | breaks))
        ends = np.flatnonzero(z & (np.r_[~z[1:], True] | np.r_[breaks[1:], True])) + 1
        for start, end in zip(starts, ends):
            run = order[start:end]
            selected = mask[run]
            if not selected.any():
                continue
            selected_starts = np.flatnonzero(selected & np.r_[True, ~selected[:-1]])
            selected_ends = np.flatnonzero(selected & np.r_[~selected[1:], True]) + 1
            for left, right in zip(selected_starts, selected_ends):
                rows = run[left:right]
                if len(rows):
                    segments.append({
                        "id": segment_id,
                        "stream": tuple(str(x) for x in stream),
                        "rows": rows.astype(np.int64),
                    })
                    segment_id += 1
    return segments


def typed_rows(segment, anomaly_types, anomaly_type):
    rows = segment["rows"]
    return rows[anomaly_types[rows] == anomaly_type]


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p10": None, "p50": None, "p90": None, "max": None}
    p10, p50, p90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "max": float(np.max(values)),
    }


def median_or_none(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    ar = pd.Series(a).rank(method="average").to_numpy(float)
    br = pd.Series(b).rank(method="average").to_numpy(float)
    return float(np.corrcoef(ar, br)[0, 1])


def best_threshold(score, target):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=bool)
    finite = np.isfinite(score)
    score = score[finite]
    target = target[finite]
    if not len(score):
        return {"threshold": None, "TP": 0, "FP": 0, "FN": int(target.sum()), "F1": 0.0}
    order = np.argsort(-score, kind="stable")
    ordered_score = score[order]
    ordered_target = target[order].astype(np.int64)
    changes = np.r_[ordered_score[1:] != ordered_score[:-1], True]
    positions = np.flatnonzero(changes)
    tp = np.cumsum(ordered_target)[positions]
    predicted = positions + 1
    fp = predicted - tp
    fn = int(target.sum()) - tp
    f1 = 2.0 * tp / np.maximum(1, 2 * tp + fp + fn)
    best_f1 = float(np.max(f1))
    candidate_indices = np.flatnonzero(np.isclose(f1, best_f1, rtol=0.0, atol=1e-15))
    best_index = int(candidate_indices[-1])
    threshold = float(ordered_score[positions[best_index]])
    stats = {
        "TP": int(tp[best_index]),
        "FP": int(fp[best_index]),
        "FN": int(fn[best_index]),
        "F1": best_f1,
    }
    return {"threshold": threshold, **stats}


def _learn_climatology_residuals(raw, folds, eval_folds):
    """Return per-row |residual|/sigma using each fold's forward training set."""
    times = pd.to_datetime(raw.time, utc=True)
    month = times.dt.month.to_numpy(np.int8)
    out = {
        "temp": np.full(len(raw), np.nan, dtype=float),
        "psal": np.full(len(raw), np.nan, dtype=float),
    }
    for fold in eval_folds:
        target = folds == fold
        learn = forward_mask(raw, folds, fold)
        target_frame = raw.loc[target, ["station", "layer"]].copy()
        target_frame["month"] = month[target]
        for column in ("temp", "psal"):
            learn_frame = raw.loc[learn, ["station", "layer", column]].copy()
            learn_frame["month"] = month[learn]
            stats = learn_frame.groupby(["station", "layer", "month"], sort=False)[column].agg(
                ["mean", "std"]
            )
            global_mean = float(learn_frame[column].mean())
            global_std = float(learn_frame[column].std())
            if not np.isfinite(global_mean):
                global_mean = 0.0
            if not np.isfinite(global_std) or global_std <= 0:
                global_std = 1.0
            joined = target_frame.merge(
                stats.reset_index(), on=["station", "layer", "month"], how="left"
            )
            value = raw.loc[target, column].to_numpy(float)
            mean = joined["mean"].fillna(global_mean).to_numpy(float)
            sigma = joined["std"].fillna(global_std).to_numpy(float)
            sigma = np.maximum(sigma, 1e-12)
            out[column][target] = np.abs(value - mean) / sigma
    return out


def _b1(raw, y, pred, anomaly_types, split_masks):
    rows = []
    summary = []
    all_types = sorted(pd.unique(anomaly_types).tolist())
    for split, mask in split_masks.items():
        stats = f1_stats(y[mask], pred[mask])
        total_pred = int(np.count_nonzero(mask & pred))
        summary.append({
            "split": split,
            "n_eval": int(mask.sum()),
            "n_rows_true": int((mask & y).sum()),
            "pred_positive": total_pred,
            **stats,
        })
        for anomaly_type in all_types:
            typed = mask & (anomaly_types == anomaly_type)
            true = typed & y
            positive = typed & pred
            rows.append({
                "split": split,
                "anomaly_type": anomaly_type,
                "n_rows_true": int(true.sum()),
                "TP": int((true & pred).sum()),
                "FN": int((true & ~pred).sum()),
                "recall": int((true & pred).sum()) / max(1, int(true.sum())),
                "pred_positive_rows": int(positive.sum()),
                "pred_positive_share": int(positive.sum()) / max(1, total_pred),
                "FP": int((positive & ~y).sum()),
            })
    return rows, summary


def _segment_rows(raw, y, pred, anomaly_types, split_masks):
    result = {}
    for split, mask in split_masks.items():
        segments = positive_segments(raw, y, mask)
        result[split] = segments
    b2 = []
    b3 = []
    for split, segments in result.items():
        flat_types = sorted({value for seg in segments for value in anomaly_types[seg["rows"]]})
        for anomaly_type in flat_types:
            selected = [
                typed_rows(segment, anomaly_types, anomaly_type)
                for segment in segments
                if anomaly_type in set(anomaly_types[segment["rows"]])
            ]
            selected = [rows for rows in selected if len(rows)]
            lengths = np.asarray([len(rows) for rows in selected], dtype=int)
            full_miss = 0
            partial = 0
            full_hit = 0
            hit_ratios = []
            for rows in selected:
                hits = int(pred[rows].sum())
                if hits == 0:
                    full_miss += 1
                elif hits == len(rows):
                    full_hit += 1
                else:
                    partial += 1
                hit_ratios.append(hits / len(rows))
            q = quantiles(lengths)
            b2.append({
                "split": split,
                "anomaly_type": anomaly_type,
                "n_segments": int(len(selected)),
                "length_p10": q["p10"],
                "length_p50": q["p50"],
                "length_p90": q["p90"],
                "length_max": q["max"],
                "full_miss": full_miss,
                "partial": partial,
                "full_hit": full_hit,
                "hit_ratio_p50": median_or_none(hit_ratios),
            })
            recall_bins = [[] for _ in range(10)]
            for rows in selected:
                hits = pred[rows]
                bins = np.minimum(9, (np.arange(len(rows)) * 10 // len(rows)).astype(int))
                for bin_id in range(10):
                    use = bins == bin_id
                    if use.any():
                        recall_bins[bin_id].append((int(hits[use].sum()), int(use.sum())))
            for bin_id, values in enumerate(recall_bins):
                n_rows = sum(item[1] for item in values)
                tp = sum(item[0] for item in values)
                b3.append({
                    "split": split,
                    "anomaly_type": anomaly_type,
                    "decile": bin_id,
                    "n_rows": n_rows,
                    "TP": tp,
                    "FN": n_rows - tp,
                    "recall": tp / max(1, n_rows),
                })
    return result, b2, b3


def _b4(raw, anomaly_types, pred, segment_rows, split_masks, folds):
    b4 = []
    for split, segments in segment_rows.items():
        residuals = _learn_climatology_residuals(raw, folds, SPLITS[split])
        types = sorted({value for seg in segments for value in anomaly_types[seg["rows"]]})
        for anomaly_type in types:
            for status in ("full_miss", "full_hit"):
                temp_values = []
                psal_values = []
                selected_count = 0
                for segment in segments:
                    rows = typed_rows(segment, anomaly_types, anomaly_type)
                    if not len(rows):
                        continue
                    hits = int(pred[rows].sum())
                    is_status = (hits == 0) if status == "full_miss" else (hits == len(rows))
                    if not is_status:
                        continue
                    selected_count += 1
                    temp_value = median_or_none(residuals["temp"][rows])
                    psal_value = median_or_none(residuals["psal"][rows])
                    if temp_value is not None:
                        temp_values.append(temp_value)
                    if psal_value is not None:
                        psal_values.append(psal_value)
                b4.append({
                    "split": split,
                    "anomaly_type": anomaly_type,
                    "segment_status": status,
                    "n_segments": selected_count,
                    "temp_resid_sigma_p50": median_or_none(temp_values),
                    "psal_resid_sigma_p50": median_or_none(psal_values),
                })
    return b4


def _b5(raw, y, rule, lgbm, anomaly_types, split_masks):
    rows = []
    summary = []
    types = sorted(set(anomaly_types[y]))
    for split, mask in split_masks.items():
        eval_rule = rule[mask]
        eval_lgbm = lgbm[mask]
        eval_y = y[mask]
        summary_rule = best_threshold(eval_rule, eval_y)
        summary_lgbm = best_threshold(eval_lgbm, eval_y)
        summary.append({
            "split": split,
            "rule_threshold": summary_rule["threshold"],
            "rule_f1": summary_rule["F1"],
            "lgbm_threshold": summary_lgbm["threshold"],
            "lgbm_f1": summary_lgbm["F1"],
            "rule_lgbm_spearman": spearman(eval_rule, eval_lgbm),
        })
        for anomaly_type in types:
            typed = mask & (anomaly_types == anomaly_type)
            target = mask & y & (anomaly_types == anomaly_type)
            rule_best = best_threshold(rule[mask], target[mask])
            lgbm_best = best_threshold(lgbm[mask], target[mask])
            rows.append({
                "split": split,
                "anomaly_type": anomaly_type,
                "n_type_rows": int(typed.sum()),
                "n_rows_true": int(target.sum()),
                "rule_threshold": rule_best["threshold"],
                "rule_f1": rule_best["F1"],
                "rule_TP": rule_best["TP"],
                "rule_FP": rule_best["FP"],
                "rule_FN": rule_best["FN"],
                "lgbm_threshold": lgbm_best["threshold"],
                "lgbm_f1": lgbm_best["F1"],
                "lgbm_TP": lgbm_best["TP"],
                "lgbm_FP": lgbm_best["FP"],
                "lgbm_FN": lgbm_best["FN"],
                "lgbm_minus_rule_f1": lgbm_best["F1"] - rule_best["F1"],
                "spearman": spearman(rule[typed], lgbm[typed]),
            })
    return rows, summary


def _b6(train, test, train_segments, test_pred, anomaly_types):
    train_rows = {}
    all_lengths = []
    for anomaly_type in sorted({value for seg in train_segments for value in anomaly_types[seg["rows"]]}):
        lengths = [
            len(typed_rows(segment, anomaly_types, anomaly_type))
            for segment in train_segments
            if anomaly_type in set(anomaly_types[segment["rows"]])
        ]
        lengths = [length for length in lengths if length]
        train_rows[anomaly_type] = {"n_segments": len(lengths), **quantiles(lengths)}
        all_lengths.extend(lengths)
    train_rows["ALL"] = {"n_segments": len(all_lengths), **quantiles(all_lengths)}
    test_segments = positive_segments(test, test_pred)
    test_lengths = [len(segment["rows"]) for segment in test_segments]
    test_stats = {"n_segments": len(test_lengths), **quantiles(test_lengths)}
    rows = []
    for anomaly_type, stats in train_rows.items():
        rows.append({
            "train_anomaly_type": anomaly_type,
            "train_n_segments": stats["n_segments"],
            "train_length_p10": stats["p10"],
            "train_length_p50": stats["p50"],
            "train_length_p90": stats["p90"],
            "train_length_max": stats["max"],
            "test_pred_n_segments": test_stats["n_segments"],
            "test_pred_length_p10": test_stats["p10"],
            "test_pred_length_p50": test_stats["p50"],
            "test_pred_length_p90": test_stats["p90"],
            "test_pred_length_max": test_stats["max"],
            "test_p50_minus_train_p50": (
                None
                if stats["p50"] is None or test_stats["p50"] is None
                else test_stats["p50"] - stats["p50"]
            ),
        })
    return rows, test_stats


def print_table(name, rows, columns):
    print(name)
    print(" | ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def _self_check():
    raw = pd.DataFrame({
        "station": ["A"] * 5,
        "year": [2024] * 5,
        "layer": [1] * 5,
        "time": [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:10:00Z",
            "2024-01-01T00:20:00Z",
            "2024-01-01T00:40:00Z",
            "2024-01-01T00:50:00Z",
        ],
    })
    segments = positive_segments(raw, [0, 1, 1, 1, 0])
    assert [len(segment["rows"]) for segment in segments] == [2, 1]
    assert positive_segments(raw, [0, 1, 1, 0, 1])[-1]["rows"].tolist() == [4]


def main():
    _self_check()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    y = train.label.to_numpy(bool)
    anomaly_types = train.anomaly_type.fillna("normal").astype(str).to_numpy()
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    saved_oof = np.load(ROOT / "preds/oof_exp005.npy").astype(bool)
    saved_test = np.load(ROOT / "preds/test_exp005.npy").astype(bool)
    raw_oof = np.load(ROOT / "cache/exp005_raw_oof.npy").astype(float)
    raw_test = np.load(ROOT / "cache/exp005_raw_test.npy").astype(float)
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    rule = base.rule_score.to_numpy(float)

    if not (len(train) == len(y) == len(folds) == len(saved_oof) == len(raw_oof)):
        raise RuntimeError("train/folds/prediction length mismatch")
    if len(test) != len(saved_test) or len(test) != len(raw_test):
        raise RuntimeError("test/prediction length mismatch")
    if np.isnan(raw_oof[folds != 0]).any():
        raise RuntimeError("fold1-3 raw OOF contains NaN")

    split_masks = {
        split: np.isin(folds, fold_values)
        for split, fold_values in SPLITS.items()
    }
    b1, summary = _b1(train, y, saved_oof, anomaly_types, split_masks)
    segment_rows, b2, b3 = _segment_rows(train, y, saved_oof, anomaly_types, split_masks)
    b4 = _b4(train, anomaly_types, saved_oof, segment_rows, split_masks, folds)
    b5, b5_summary = _b5(train, y, rule, raw_oof, anomaly_types, split_masks)

    train_segments = positive_segments(train, y)
    test_segments_threshold = positive_segments(test, raw_test >= THRESHOLD)
    test_segments_saved = positive_segments(test, saved_test)
    b6, test_stats = _b6(train, test, train_segments, saved_test, anomaly_types)

    threshold_hole_pred = fill_runs(
        raw_test >= THRESHOLD,
        segment_ids(test),
        HOLE,
    )
    test_reconstruction = {
        "threshold_positive_rows": int((raw_test >= THRESHOLD).sum()),
        "threshold_segment_count": len(test_segments_threshold),
        "saved_positive_rows": int(saved_test.sum()),
        "saved_segment_count": len(test_segments_saved),
        "threshold_hole_mismatch_rows": int(np.count_nonzero(threshold_hole_pred != saved_test)),
    }

    baseline = {
        "inner_expected_f1": 0.697836,
        "outer_expected": {"TP": 5612, "FP": 704, "FN": 1831, "F1": 0.815757},
        "computed": summary,
        "rule_inner_expected_f1": 0.174346,
    }
    metrics = {
        "seed": SEED,
        "threshold": THRESHOLD,
        "hole": HOLE,
        "inputs": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "fold_counts": {str(k): int((folds == k).sum()) for k in range(4)},
            "oof_nan_rows": int(np.isnan(raw_oof).sum()),
        },
        "baseline": baseline,
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "B4": b4,
        "B5": b5,
        "B5_summary": b5_summary,
        "B6": b6,
        "B6_test_summary": test_stats,
        "B6_reconstruction": test_reconstruction,
        "definitions": {
            "segments": "same station-year-layer, positive runs, stream modal minute interval gap break",
            "b4_climatology": "station-layer-month mean/std from exp005 forward training mask",
            "b5_threshold_grid": "all observed finite score values plus endpoints; tie uses lower threshold",
            "test_prediction": "saved exp005 test prediction; raw threshold 0.62 and hole 48 reconstruction reported",
        },
    }
    out_path = ROOT / "experiments/exp006_fn_decomp.json"
    out_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print_table("B0_summary", summary, ["split", "n_eval", "n_rows_true", "pred_positive", "TP", "FP", "FN", "F1"])
    print_table("B1", b1, ["split", "anomaly_type", "n_rows_true", "TP", "FN", "recall", "pred_positive_rows", "pred_positive_share", "FP"])
    print_table("B2", b2, ["split", "anomaly_type", "n_segments", "length_p10", "length_p50", "length_p90", "length_max", "full_miss", "partial", "full_hit", "hit_ratio_p50"])
    print_table("B3", b3, ["split", "anomaly_type", "decile", "n_rows", "TP", "FN", "recall"])
    print_table("B4", b4, ["split", "anomaly_type", "segment_status", "n_segments", "temp_resid_sigma_p50", "psal_resid_sigma_p50"])
    print_table("B5", b5, ["split", "anomaly_type", "n_type_rows", "n_rows_true", "rule_threshold", "rule_f1", "rule_TP", "rule_FP", "rule_FN", "lgbm_threshold", "lgbm_f1", "lgbm_TP", "lgbm_FP", "lgbm_FN", "lgbm_minus_rule_f1", "spearman"])
    print_table("B5_summary", b5_summary, ["split", "rule_threshold", "rule_f1", "lgbm_threshold", "lgbm_f1", "rule_lgbm_spearman"])
    print_table("B6", b6, ["train_anomaly_type", "train_n_segments", "train_length_p10", "train_length_p50", "train_length_p90", "train_length_max", "test_pred_n_segments", "test_pred_length_p10", "test_pred_length_p50", "test_pred_length_p90", "test_pred_length_max", "test_p50_minus_train_p50"])
    print_table("B6_reconstruction", [test_reconstruction], ["threshold_positive_rows", "threshold_segment_count", "saved_positive_rows", "saved_segment_count", "threshold_hole_mismatch_rows"])
    print("status | OK")


if __name__ == "__main__":
    raise SystemExit(main())
