"""T22 segment candidates and the inner two-stage detector."""
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_t4 import apply_reference, forward_mask
from features import SEED, fit_reference
from metric import f1
from q2_cross_layer import _signals
from submission_tools import save_pool_pred
from train import fill_short_gaps, segment_ids


WINDOWS = (36, 144, 432, 1008)
PURGE_HOURS = 87
FIXED_THRESHOLD = 0.62
FIXED_HOLE = 48
MIN_LENGTH = 6
STATIONS = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
TYPE_ORDER = ("offset", "drift", "noise", "flatline", "spike")


def _blocks(raw):
    d = raw[["station", "layer", "time"]].copy()
    d["_time"] = pd.to_datetime(d.time, utc=True)
    d = d.sort_values(["station", "layer", "_time"], kind="stable")
    blocks = []
    meta = []
    for (station, layer), group in d.groupby(["station", "layer"], sort=False):
        rows = group.index.to_numpy(np.int64)
        times = group["_time"].to_numpy()
        breaks = np.r_[True, np.diff(times).astype("timedelta64[m]").astype(int) != 10]
        starts = np.flatnonzero(breaks)
        ends = np.r_[starts[1:], len(rows)]
        for lo, hi in zip(starts, ends):
            block = rows[lo:hi]
            blocks.append(block)
            meta.append((str(station), int(layer), int(block[0]), int(block[-1])))
    return blocks, meta


def _robust_sigma(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    med = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - med)))
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = float(np.std(values))
    return max(sigma if np.isfinite(sigma) else 1.0, 1e-3)


def _step_difference(values, window):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    clean = np.nan_to_num(values, nan=0.0)
    cumulative = np.r_[0.0, np.cumsum(clean)]
    counts = np.r_[0, np.cumsum(good.astype(np.int64))]
    n = len(values)
    left = np.full(n, np.nan, float)
    right = np.full(n, np.nan, float)
    if n <= window:
        return right - left
    left_count = counts[window:n] - counts[: n - window]
    left_sum = cumulative[window:n] - cumulative[: n - window]
    use = left_count == window
    left[window:n][use] = left_sum[use] / window
    right_count = counts[window:] - counts[: n + 1 - window]
    right_sum = cumulative[window:] - cumulative[: n + 1 - window]
    use = right_count == window
    right[: n + 1 - window][use] = right_sum[use] / window
    return right - left


def _candidate_segments(raw, blocks, r, folds):
    n = len(raw)
    d_by_window = {window: np.full(n, np.nan, float) for window in WINDOWS}
    candidates = []
    for block_id, block in enumerate(blocks):
        values = np.asarray(r[block], float)
        sigma = _robust_sigma(values)
        boundaries = {0, len(block)}
        for window in WINDOWS:
            delta = _step_difference(values, window)
            d_by_window[window][block] = delta
            score = np.abs(delta) / sigma
            valid = np.isfinite(score)
            if len(score) >= 3:
                middle = (
                    valid[1:-1]
                    & valid[:-2]
                    & valid[2:]
                    & (score[1:-1] >= 2.0)
                    & (score[1:-1] >= score[:-2])
                    & (score[1:-1] > score[2:])
                )
                boundaries.update((np.flatnonzero(middle) + 1).tolist())
        boundaries = sorted(boundaries)
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if hi - lo < MIN_LENGTH:
                continue
            rows = block[lo:hi]
            candidates.append(
                {
                    "block_id": block_id,
                    "lo": int(lo),
                    "hi": int(hi),
                    "start_row": int(rows[0]),
                    "end_row": int(rows[-1]),
                    "fold_start": int(folds[rows[0]]),
                    "sigma": sigma,
                }
            )
    return candidates, d_by_window


def _clim_resid(raw, mask):
    reference = fit_reference(raw, mask)
    empty = pd.DataFrame(index=raw.index)
    applied = apply_reference(empty, raw, reference)
    return applied.clim_resid.to_numpy(float), reference


def _psal_resid(raw, reference):
    month = pd.to_datetime(raw.time, utc=True).dt.month
    keys = pd.DataFrame(
        {"station": raw.station, "layer": raw.layer, "month": month}
    ).astype(str).agg("|".join, axis=1)
    med = keys.map(reference.get("psal_median", {})).fillna(
        reference.get("global_psal_median", float(raw.psal.median()))
    ).to_numpy(float)
    mad = keys.map(reference.get("psal_mad", {})).fillna(0.5).to_numpy(float)
    mad = np.maximum(mad, 0.01)
    psal = raw.psal.to_numpy(float)
    return (psal - med) / mad


def _x1(raw):
    d = raw.copy()
    d["_time"] = pd.to_datetime(d.time, utc=True)
    d["_row"] = np.arange(len(d), dtype=np.int64)
    return np.asarray(_signals(d)["x1"], float)


def _slope_times_length(values):
    values = np.asarray(values, float)
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[good]
    y = values[good]
    x -= x.mean()
    return float(np.sum(x * (y - y.mean())) / np.sum(x * x) * len(values))


def _mean(values):
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def _segment_features(raw, blocks, segment, r, x1, psal_r, raw_prob, d144, arrays):
    block = blocks[segment["block_id"]]
    lo, hi = segment["lo"], segment["hi"]
    rows = block[lo:hi]
    left = block[max(0, lo - 432):lo]
    right = block[hi:min(len(block), hi + 432)]
    values = r[rows]
    left_values = r[left]
    right_values = r[right]
    sigma = segment["sigma"]
    mean_r = _mean(values)
    mean_left = _mean(left_values)
    mean_right = _mean(right_values)
    mean_x1 = _mean(x1[rows])
    mean_x1_left = _mean(x1[left])
    mean_x1_right = _mean(x1[right])
    temp = arrays["temp"][rows]
    temp_diff = np.diff(temp)
    depth = arrays["depth"][rows]
    psal = arrays["psal"][rows]
    features = {
        "length": float(len(rows)),
        "start_month": float(arrays["month"][rows[0]]),
        "station": float(arrays["station"][rows[0]]),
        "layer": float(arrays["layer"][rows[0]]),
        "depth_median": _mean(depth) if np.isfinite(depth).any() else np.nan,
        "r_mean": mean_r,
        "r_median": float(np.nanmedian(values)),
        "r_std": float(np.nanstd(values)),
        "left_level_diff": (mean_r - mean_left) / sigma if np.isfinite(mean_left) else np.nan,
        "right_level_diff": (mean_r - mean_right) / sigma if np.isfinite(mean_right) else np.nan,
        "r_slope_x_length": _slope_times_length(values),
        "d144_start": abs(float(d144[rows[0]])) if np.isfinite(d144[rows[0]]) else np.nan,
        "d144_end": abs(float(d144[rows[-1]])) if np.isfinite(d144[rows[-1]]) else np.nan,
        "x1_mean": mean_x1,
        "x1_left_diff": mean_x1 - mean_x1_left if np.isfinite(mean_x1_left) else np.nan,
        "x1_right_diff": mean_x1 - mean_x1_right if np.isfinite(mean_x1_right) else np.nan,
        "temp_diff_abs_median": float(np.median(np.abs(temp_diff))),
        "temp_diff_zero_ratio": float(np.mean(temp_diff == 0.0)),
        "psal_missing_ratio": float(np.mean(~np.isfinite(psal))),
        "psal_resid_mean": _mean(psal_r[rows]),
        "row_prob_mean": _mean(raw_prob[rows]),
        "row_prob_p90": float(np.nanpercentile(raw_prob[rows], 90)),
        "row_prob_max": float(np.nanmax(raw_prob[rows])),
    }
    return features


def _true_runs(raw, blocks, y, folds, held):
    types = raw.anomaly_type.fillna("other").astype(str).to_numpy()
    runs = []
    for block_id, block in enumerate(blocks):
        start = None
        run_type = None
        previous = None
        for local, row in enumerate(block):
            valid = int(folds[row]) == held
            current_type = types[row] if y[row] else None
            continues = (
                start is not None
                and valid
                and previous == local - 1
                and y[row] == 1
                and current_type == run_type
            )
            if not continues and start is not None:
                runs.append({"block_id": block_id, "lo": start, "hi": local, "type": run_type})
                start = None
            if valid and y[row] == 1 and start is None:
                start, run_type = local, current_type
            previous = local if valid else None
        if start is not None:
            runs.append({"block_id": block_id, "lo": start, "hi": len(block), "type": run_type})
    return runs


def _inner_candidates(candidates, blocks, folds, held):
    out = []
    for candidate in candidates:
        rows = blocks[candidate["block_id"]][candidate["lo"]:candidate["hi"]]
        if np.any(folds[rows] == held):
            out.append(candidate)
    return out


def _a_tables(raw, blocks, contexts, y, folds):
    rows = []
    oracle = np.zeros(len(raw), dtype=bool)
    type_runs = defaultdict(list)
    covered_rows = np.zeros(len(raw), dtype=bool)
    candidate_count = 0
    for context in contexts:
        held = context["held"]
        candidates = _inner_candidates(context["candidates"], blocks, folds, held)
        candidate_count += len(candidates)
        for candidate in candidates:
            idx = blocks[candidate["block_id"]][candidate["lo"]:candidate["hi"]]
            inner_rows = idx[folds[idx] == held]
            covered_rows[inner_rows] = True
            if y[idx].mean() >= 0.5:
                oracle[inner_rows] = True
        for run in _true_runs(raw, blocks, y, folds, held):
            type_runs[run["type"]].append((context, run))

    coverage = float(covered_rows[np.isin(folds, (1, 2))].mean())
    for typ in ("offset", "drift", "noise", "flatline", "other"):
        run_list = type_runs.get(typ, [])
        best_values = []
        any_count = 0
        iou_count = 0
        for context, run in run_list:
            block = blocks[run["block_id"]]
            best = 0.0
            for candidate in context["candidates"]:
                if candidate["block_id"] != run["block_id"]:
                    continue
                c_rows = block[candidate["lo"]:candidate["hi"]]
                c_rows = c_rows[folds[c_rows] == context["held"]]
                if not len(c_rows):
                    continue
                overlap_lo = max(candidate["lo"], run["lo"])
                overlap_hi = min(candidate["hi"], run["hi"])
                if overlap_hi <= overlap_lo:
                    continue
                overlap_rows = block[overlap_lo:overlap_hi]
                overlap = int(np.count_nonzero(folds[overlap_rows] == context["held"]))
                if overlap:
                    any_count = max(any_count, 0) + 1
                    union = len(c_rows) + (run["hi"] - run["lo"]) - overlap
                    best = max(best, overlap / max(1, union))
            best_values.append(best)
            if best >= 0.5:
                iou_count += 1
        rows.append(
            {
                "type": typ,
                "n_true_runs": len(run_list),
                "covered_iou50": iou_count,
                "covered_any": sum(value > 0 for value in best_values),
                "best_iou_p50": float(np.median(best_values)) if best_values else 0.0,
                "candidate_segments": candidate_count,
                "candidate_row_coverage": coverage,
            }
        )
    return rows, f1(y[np.isin(folds, (1, 2))], oracle[np.isin(folds, (1, 2))]), oracle


def _stats(y, pred, mask):
    score, tp, fp, fn = f1(y[mask], pred[mask])
    return score, tp, fp, fn, float(pred[mask].mean())


def _print_table(title, columns, rows):
    print(title)
    print(" | ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print(" | ".join(values))


def main():
    started = time.monotonic()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    y = train.label.to_numpy(np.int8)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    raw_prob = np.load(ROOT / "cache/exp005_raw_oof.npy", allow_pickle=False).astype(float)
    if len(folds) != len(train) or raw_prob.shape != (len(train),):
        raise ValueError("cached folds/raw probability shape mismatch")
    raw_prob = np.nan_to_num(raw_prob, nan=0.0, posinf=1.0, neginf=0.0)

    blocks, block_meta = _blocks(train)
    x1 = _x1(train)
    arrays = {
        "temp": train.temp.to_numpy(float),
        "psal": train.psal.to_numpy(float),
        "depth": train.depth.to_numpy(float),
        "station": train.station.map(STATIONS).to_numpy(float),
        "layer": train.layer.to_numpy(float),
        "month": pd.to_datetime(train.time, utc=True).dt.month.to_numpy(float),
    }

    contexts = []
    for held in (1, 2):
        train_mask = forward_mask(train, folds, held)
        r, reference = _clim_resid(train, train_mask)
        psal_r = _psal_resid(train, reference)
        candidates, d_by_window = _candidate_segments(train, blocks, r, folds)
        for candidate in candidates:
            candidate["features"] = _segment_features(
                train,
                blocks,
                candidate,
                r,
                x1,
                psal_r,
                raw_prob,
                d_by_window[144],
                arrays,
            )
            rows = blocks[candidate["block_id"]][candidate["lo"]:candidate["hi"]]
            candidate["label_ratio"] = float(y[rows].mean())
            candidate["label"] = int(candidate["label_ratio"] >= 0.5)
            candidate["start_time"] = pd.to_datetime(train.time.iloc[candidate["start_row"]], utc=True)
            candidate["end_time"] = pd.to_datetime(train.time.iloc[candidate["end_row"]], utc=True)
            candidate["station_name"] = block_meta[candidate["block_id"]][0]
            candidate["layer_value"] = block_meta[candidate["block_id"]][1]

        validation_starts = {}
        time_values = pd.to_datetime(train.time, utc=True)
        for key, group in train.groupby(["station", "layer"], sort=False):
            idx = group.index.to_numpy(np.int64)
            target = idx[folds[idx] == held]
            if len(target):
                validation_starts[key] = time_values.iloc[target].min() - pd.Timedelta(hours=PURGE_HOURS)

        train_ids = []
        valid_ids = []
        leak_violations = 0
        for number, candidate in enumerate(candidates):
            rows = blocks[candidate["block_id"]][candidate["lo"]:candidate["hi"]]
            key = (candidate["station_name"], candidate["layer_value"])
            if candidate["fold_start"] == held:
                valid_ids.append(number)
            cutoff = validation_starts.get(key)
            if (
                candidate["fold_start"] < held
                and train_mask[candidate["start_row"]]
                and np.all(train_mask[rows])
            ):
                train_ids.append(number)
                if cutoff is not None and candidate["end_time"] >= cutoff:
                    leak_violations += 1
        contexts.append(
            {
                "held": held,
                "r": r,
                "candidates": candidates,
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "leak_violations": leak_violations,
                "feature_names": list(candidates[0]["features"]) if candidates else [],
            }
        )

    a1, a2, _ = _a_tables(train, blocks, contexts, y, folds)
    _print_table(
        "A1",
        ["type", "n_true_runs", "covered_iou50", "covered_any", "best_iou_p50", "candidate_segments", "candidate_row_coverage"],
        a1,
    )
    _print_table("A2", ["inner_f1", "TP", "FP", "FN"], [{"inner_f1": a2[0], "TP": a2[1], "FP": a2[2], "FN": a2[3]}])

    segment_prob = np.zeros(len(train), dtype=np.float32)
    gain = defaultdict(float)
    artifact_rows = []
    for context in contexts:
        held = context["held"]
        candidates = context["candidates"]
        names = context["feature_names"]
        features = pd.DataFrame([candidate["features"] for candidate in candidates], columns=names)
        labels = np.array([candidate["label"] for candidate in candidates], dtype=np.int8)
        train_ids = context["train_ids"]
        valid_ids = context["valid_ids"]
        if train_ids and len(np.unique(labels[train_ids])) > 1:
            model = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                n_estimators=400,
                random_state=SEED,
                n_jobs=2,
                verbosity=-1,
            )
            model.fit(features.iloc[train_ids], labels[train_ids], callbacks=[lgb.log_evaluation(0)])
            importance = model.booster_.feature_importance(importance_type="gain")
            for name, value in zip(names, importance):
                gain[name] += float(value)
            probabilities = model.predict_proba(features.iloc[valid_ids])[:, 1] if valid_ids else np.empty(0)
        else:
            prior = float(labels[train_ids].mean()) if train_ids else 0.0
            probabilities = np.full(len(valid_ids), prior, dtype=float)
        for number, probability in zip(valid_ids, probabilities):
            candidate = candidates[number]
            candidate["seg_prob"] = float(probability)
            rows = blocks[candidate["block_id"]][candidate["lo"]:candidate["hi"]]
            inner_rows = rows[folds[rows] == held]
            segment_prob[inner_rows] = np.maximum(segment_prob[inner_rows], float(probability))
            artifact = {
                "fold": held,
                "station": candidate["station_name"],
                "layer": candidate["layer_value"],
                "start_time": str(candidate["start_time"]),
                "end_time": str(candidate["end_time"]),
                "row_start": candidate["start_row"],
                "row_end": candidate["end_row"],
                "station_name": candidate["station_name"],
                "layer_value": candidate["layer_value"],
                "label_ratio": candidate["label_ratio"],
                "label": candidate["label"],
                "seg_prob": float(probability),
                "seg_pred": int(probability >= 0.5),
                **candidate["features"],
            }
            artifact_rows.append(artifact)

    inner = np.isin(folds, (1, 2))
    row_segments = segment_ids(train)
    row_pred = fill_short_gaps(raw_prob >= FIXED_THRESHOLD, train, FIXED_HOLE, row_segments)
    segment_pred = fill_short_gaps(segment_prob >= 0.5, train, FIXED_HOLE, row_segments)
    combined_score = raw_prob.copy()
    combined_score[inner] = np.maximum(combined_score[inner], segment_prob[inner])
    combined_pred = fill_short_gaps(combined_score >= FIXED_THRESHOLD, train, FIXED_HOLE, row_segments)
    variants = {"row_only": row_pred, "segment_only": segment_pred, "combined": combined_pred}
    b1 = []
    for name, pred in variants.items():
        score, tp, fp, fn, rate = _stats(y, pred, inner)
        b1.append({"variant": name, "inner_f1": score, "TP": tp, "FP": fp, "FN": fn, "positive_rate": rate})
    _print_table("B1", ["variant", "inner_f1", "TP", "FP", "FN", "positive_rate"], b1)

    types = train.anomaly_type.fillna("other").astype(str).to_numpy()
    b2 = []
    for name, pred in variants.items():
        for typ in TYPE_ORDER:
            mask = inner & (types == typ) & (y == 1)
            tp = int(np.count_nonzero(mask & pred))
            positive = int(mask.sum())
            b2.append({"variant": name, "type": typ, "recall": tp / max(1, positive), "TP": tp, "FN": positive - tp, "positive": positive})
    _print_table("B2", ["variant", "type", "recall", "TP", "FN", "positive"], b2)

    b3 = [
        {"rank": rank, "feature": name, "gain": value}
        for rank, (name, value) in enumerate(sorted(gain.items(), key=lambda item: (-item[1], item[0]))[:10], 1)
    ]
    _print_table("B3", ["rank", "feature", "gain"], b3)

    cache_dir = ROOT / "cache" / "t22"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(artifact_rows).to_csv(cache_dir / "segments_inner.csv", index=False)
    save_pool_pred(ROOT / "preds" / "score_oof_t22.npy", combined_score.astype(np.float32))
    print("leakage_violations | " + str(sum(context["leak_violations"] for context in contexts)))
    print("inner_rows | " + str(int(inner.sum())))
    print("elapsed_seconds | " + f"{time.monotonic() - started:.1f}")
    print("artifacts | P1/cache/t22/segments_inner.csv | P1/preds/score_oof_t22.npy")


if __name__ == "__main__":
    main()
