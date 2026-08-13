"""T4 read-only FN diagnosis for the sealed exp002 outer evaluation."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
N_JOBS = 2
THRESHOLD = 0.44
WEIGHT = 0.15
HOLE = 2
EXPECTED = (4935, 500, 2508)
ROOT = Path(__file__).resolve().parents[1]
KEY = ["station", "layer"]
BIN_NAMES = ("1", "2-3", "4-6", "7-12", "13-36", "37-144", "145+")


def f1_stats(y, pred, mask=None):
    if mask is not None:
        y = y[mask]
        pred = pred[mask]
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
    return {"TP": tp, "FP": fp, "FN": fn, "F1": f1}


def segment_ids(raw):
    """The same original-row-order, 10-minute-gap segmentation as train.py."""
    tm = pd.to_datetime(raw.time, utc=True)
    stream = pd.factorize(
        pd.MultiIndex.from_arrays([raw.station, raw.layer]), sort=False
    )[0].astype(np.int64)
    new = np.r_[
        True,
        (stream[1:] != stream[:-1])
        | (tm.diff().dt.total_seconds().to_numpy()[1:] != 600),
    ]
    gap = pd.Series(new).groupby(stream).cumsum().to_numpy(np.int64)
    return stream * (len(raw) + 1) + gap


def fill_short_gaps(pred, segments, max_gap):
    out = np.asarray(pred, dtype=bool).copy()
    if max_gap == 0:
        return out
    for holes in range(1, max_gap + 1):
        width = holes + 1
        bridge = out[:-width] & out[width:] & (segments[:-width] == segments[width:])
        starts = np.flatnonzero(bridge)
        for offset in range(1, width):
            out[starts + offset] = True
    return out


def forward_mask(raw, folds, held_out, hours=87):
    """Only folds earlier than held_out, with the specified left purge."""
    tm = pd.to_datetime(raw.time, utc=True)
    keep = folds < held_out
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        va = idx[folds[idx] == held_out]
        if len(va) == 0:
            continue
        cutoff = tm.iloc[va].min() - pd.Timedelta(hours=hours)
        keep[idx[tm.iloc[idx].ge(cutoff).to_numpy()]] = False
    return keep


def honest_reference(raw, mask):
    d = raw.loc[mask, ["station", "layer", "time", "temp"]].copy()
    d["month"] = pd.to_datetime(d.time, utc=True).dt.month
    med = d.groupby(["station", "layer", "month"]).temp.median()
    z = d.join(med.rename("med"), on=["station", "layer", "month"])
    mad = z.assign(dev=(z.temp - z.med).abs()).groupby(
        ["station", "layer", "month"]
    ).dev.median()
    return {
        "median": {"|".join(map(str, k)): float(v) for k, v in med.items()},
        "mad": {"|".join(map(str, k)): float(v) for k, v in mad.items()},
        "global_median": float(d.temp.median()),
    }


def apply_reference(base, raw, reference):
    out = base.copy()
    month = pd.to_datetime(raw.time, utc=True).dt.month
    keys = pd.DataFrame(
        {"station": raw.station, "layer": raw.layer, "month": month}
    ).astype(str).agg("|".join, axis=1)
    med = keys.map(reference["median"]).fillna(reference["global_median"]).to_numpy(float)
    mad = keys.map(reference["mad"]).fillna(0.5).clip(lower=0.01).to_numpy(float)
    out["clim_resid"] = ((raw.temp.to_numpy(float) - med) / mad).astype(np.float32)
    return out


def score_from_models(raw, base, folds, model_dir, honest=False):
    if honest:
        train_mask = forward_mask(raw, folds, 3)
        reference = honest_reference(raw, train_mask)
        x = apply_reference(base, raw, reference)
        booster = lgb.Booster(model_file=model_dir / "exp002_outer_honest.txt")
        raw_prob = booster.predict(x)
        return (1.0 - WEIGHT) * raw_prob + WEIGHT * base.rule_score.to_numpy(float)

    score = np.zeros(len(raw), dtype=np.float64)
    for fold in range(4):
        use = folds == fold
        booster = lgb.Booster(model_file=model_dir / f"exp002_fold{fold}.txt")
        score[use] = booster.predict(base.loc[use])
    return (1.0 - WEIGHT) * score + WEIGHT * base.rule_score.to_numpy(float)


def segment_indices(segments):
    groups = defaultdict(list)
    for i, segment in enumerate(segments):
        groups[int(segment)].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def expand_runs(pred, segments, radius):
    out = np.asarray(pred, dtype=bool).copy()
    for idx in segment_indices(segments):
        z = out[idx]
        starts = np.flatnonzero(z & np.r_[True, ~z[:-1]])
        ends = np.flatnonzero(z & np.r_[~z[1:], True])
        for start, end in zip(starts, ends):
            lo = max(0, int(start) - radius)
            hi = min(len(idx), int(end) + radius + 1)
            out[idx[lo:hi]] = True
    return out


def rolling_score(score, segments, window, operation):
    out = np.empty_like(score, dtype=np.float64)
    for idx in segment_indices(segments):
        series = pd.Series(score[idx])
        rolled = series.rolling(window, center=True, min_periods=1)
        if operation == "mean":
            out[idx] = rolled.mean().to_numpy(float)
        else:
            out[idx] = rolled.max().to_numpy(float)
    return out


def hysteresis(score, segments, enter, stay):
    out = np.zeros(len(score), dtype=bool)
    for idx in segment_indices(segments):
        active = False
        for pos, value in enumerate(score[idx]):
            if active:
                active = bool(value >= stay)
            elif value >= enter:
                active = True
            out[idx[pos]] = active
    return out


def postprocess(score, segments, name):
    if name == "current_hole_2":
        return fill_short_gaps(score >= THRESHOLD, segments, 2)
    if name.startswith("hole_"):
        return fill_short_gaps(score >= THRESHOLD, segments, int(name.split("_")[1]))
    if name.startswith("expand_"):
        return expand_runs(score >= THRESHOLD, segments, int(name.split("_")[1]))
    if name.startswith("rolling_mean_"):
        window = int(name.split("_")[-1])
        return rolling_score(score, segments, window, "mean") >= THRESHOLD
    if name.startswith("rolling_max_"):
        window = int(name.split("_")[-1])
        return rolling_score(score, segments, window, "max") >= THRESHOLD
    if name.startswith("hysteresis_"):
        stay = float(name.split("_")[-1])
        return hysteresis(score, segments, THRESHOLD, stay)
    raise ValueError(name)


def make_events(raw, y, pred, score, segments, mask):
    """Return positive events in the selected interval, preserving train order."""
    indices = np.flatnonzero(mask)
    ys = y[indices].astype(bool)
    if len(indices) == 0:
        return []
    starts = ys & np.r_[
        True,
        (~ys[:-1])
        | (segments[indices[1:]] != segments[indices[:-1]])
        | (indices[1:] != indices[:-1] + 1),
    ]
    start_pos = np.flatnonzero(starts)
    end_pos = np.r_[start_pos[1:], len(indices)]
    type_values = raw.anomaly_type.fillna("normal").astype(str).to_numpy()
    events = []
    for begin, end in zip(start_pos, end_pos):
        rows = indices[begin:end]
        if not ys[begin]:
            continue
        # The positive start positions make the first row of each event a true row;
        # the end is the next positive start or the end of the selected interval.
        if np.any(~y[rows]):
            rows = rows[y[rows]]
        types = tuple(sorted(set(type_values[rows])))
        events.append(
            {
                "rows": rows,
                "length": int(len(rows)),
                "hit": bool(pred[rows].any()),
                "max_score": float(np.max(score[rows])),
                "types": types,
            }
        )
    return events


def bin_name(length):
    if length == 1:
        return "1"
    if length <= 3:
        return "2-3"
    if length <= 6:
        return "4-6"
    if length <= 12:
        return "7-12"
    if length <= 36:
        return "13-36"
    if length <= 144:
        return "37-144"
    return "145+"


def type_table(raw, y, pred, events, mask):
    type_values = raw.anomaly_type.fillna("normal").astype(str).to_numpy()
    types = sorted(set(type_values[np.flatnonzero(mask & y)]))
    rows = []
    for anomaly_type in types:
        selected = [event for event in events if anomaly_type in event["types"]]
        idx = np.concatenate(
            [event["rows"][type_values[event["rows"]] == anomaly_type] for event in selected]
        ) if selected else np.empty(0, dtype=np.int64)
        tp = int(np.count_nonzero(pred[idx]))
        fn = int(len(idx) - tp)
        rows.append(
            {
                "anomaly_type": anomaly_type,
                "event_count": int(len(selected)),
                "positive_rows": int(len(idx)),
                "TP": tp,
                "FN": fn,
                "row_recall": tp / max(1, len(idx)),
                "event_hit_rate": sum(event["hit"] for event in selected)
                / max(1, len(selected)),
            }
        )
    return rows


def station_layer_table(raw, y, pred, mask):
    type_values = raw.anomaly_type.fillna("normal").astype(str).to_numpy()
    groups = defaultdict(list)
    for idx in np.flatnonzero(mask & y):
        groups[(str(raw.station.iloc[idx]), int(raw.layer.iloc[idx]))].append(int(idx))
    rows = []
    for (station, layer), idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        positive = len(idx)
        tp = int(np.count_nonzero(pred[idx]))
        counts = Counter(type_values[idx])
        dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append(
            {
                "station": station,
                "layer": layer,
                "positive_rows": positive,
                "FN": positive - tp,
                "recall": tp / max(1, positive),
                "dominant_anomaly_type": dominant,
            }
        )
    return sorted(rows, key=lambda row: (-row["FN"], row["station"], row["layer"]))[:10]


def length_table(y, pred, events):
    grouped = {name: [] for name in BIN_NAMES}
    for event in events:
        grouped[bin_name(event["length"])].append(event)
    rows = []
    for name in BIN_NAMES:
        selected = grouped[name]
        positives = sum(event["length"] for event in selected)
        fn = sum(int(np.count_nonzero(~pred[event["rows"]])) for event in selected)
        rows.append(
            {
                "bin": name,
                "event_count": int(len(selected)),
                "positive_rows": int(positives),
                "FN": int(fn),
                "row_recall": (positives - fn) / max(1, positives),
                "event_hit_rate": sum(event["hit"] for event in selected)
                / max(1, len(selected)),
            }
        )
    return rows


def recoverability(raw, y, pred, score, events):
    type_values = raw.anomaly_type.fillna("normal").astype(str).to_numpy()
    categories = {name: [] for name in ("recoverable_inside", "near_miss", "hard_miss")}
    for event in events:
        missed = event["rows"][~pred[event["rows"]]]
        if len(missed) == 0:
            continue
        if event["hit"]:
            category = "recoverable_inside"
        elif event["max_score"] >= 0.20:
            category = "near_miss"
        else:
            category = "hard_miss"
        categories[category].extend(int(idx) for idx in missed)

    total_fn = sum(len(indices) for indices in categories.values())
    out = {"total_FN": int(total_fn), "categories": {}, "by_anomaly_type": {}}
    all_indices = np.asarray(
        [idx for indices in categories.values() for idx in indices], dtype=np.int64
    )
    all_types = sorted(set(type_values[all_indices])) if total_fn else []
    for category, indices in categories.items():
        count = len(indices)
        out["categories"][category] = {
            "count": int(count),
            "fraction_of_FN": count / max(1, total_fn),
        }
        by_type = []
        idx = np.asarray(indices, dtype=np.int64)
        for anomaly_type in all_types:
            type_count = int(np.count_nonzero(type_values[idx] == anomaly_type))
            by_type.append(
                {
                    "anomaly_type": anomaly_type,
                    "count": type_count,
                    "fraction_of_category": type_count / max(1, count),
                }
            )
        out["by_anomaly_type"][category] = by_type
    return out


def noise_or_drift_table(raw, y, score, base, events, target):
    type_values = raw.anomaly_type.fillna("normal").astype(str).to_numpy()
    selected = [event for event in events if target in event["types"]]
    event_rows = np.concatenate([event["rows"] for event in selected]) if selected else np.empty(0, dtype=np.int64)
    streams = set(zip(raw.station.iloc[event_rows], raw.layer.iloc[event_rows]))
    normal_rows = [
        idx
        for idx in np.flatnonzero(y == 0)
        if (raw.station.iloc[idx], raw.layer.iloc[idx]) in streams
    ]
    normal_rows = np.asarray(normal_rows, dtype=np.int64)

    def finite_median(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else 0.0

    event_max = [event["max_score"] for event in selected]
    event_medians = [float(np.median(score[event["rows"]])) for event in selected]
    if target == "noise":
        ratio = np.divide(
            base["std_7"].to_numpy(float),
            base["std_1009"].to_numpy(float),
            out=np.full(len(base), np.nan, dtype=float),
            where=base["std_1009"].to_numpy(float) != 0,
        )
        return {
            "event_count": int(len(selected)),
            "event_length_median": finite_median([event["length"] for event in selected]),
            "event_probability_median": finite_median(score[event_rows]),
            "event_max_probability_median": finite_median(event_max),
            "event_median_probability_median": finite_median(event_medians),
            "normal_probability_median": finite_median(score[normal_rows]),
            "event_std_7_over_std_1009_median": finite_median(ratio[event_rows]),
            "normal_std_7_over_std_1009_median": finite_median(ratio[normal_rows]),
        }

    drift = base["drift_score"].to_numpy(float)
    return {
        "event_count": int(len(selected)),
        "event_length_median": finite_median([event["length"] for event in selected]),
        "event_probability_median": finite_median(score[event_rows]),
        "event_max_probability_median": finite_median(event_max),
        "event_median_probability_median": finite_median(event_medians),
        "normal_probability_median": finite_median(score[normal_rows]),
        "event_drift_score_median": finite_median(drift[event_rows]),
        "normal_drift_score_median": finite_median(drift[normal_rows]),
    }


def config_names():
    return (
        "current_hole_2",
        "hole_6",
        "hole_12",
        "hole_36",
        "expand_3",
        "expand_12",
        "rolling_mean_7",
        "rolling_mean_37",
        "rolling_max_7",
        "hysteresis_0.15",
        "hysteresis_0.25",
    )


def evaluate_configs(score, segments, y, mask):
    baseline = None
    rows = []
    predictions = {}
    for name in config_names():
        pred = postprocess(score, segments, name)
        predictions[name] = pred
        stats = f1_stats(y, pred, mask)
        if baseline is None:
            baseline = stats["F1"]
        rows.append(
            {
                "setting": name,
                "TP": stats["TP"],
                "FP": stats["FP"],
                "FN": stats["FN"],
                "F1": stats["F1"],
                "delta_vs_current": stats["F1"] - baseline,
            }
        )
    return rows, predictions


def print_table(title, rows, columns):
    print(title)
    print("|".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print("|".join(values))


def main():
    np.random.seed(SEED)
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    saved_oof = np.load(ROOT / "preds/oof_exp002.npy")
    if len(train) != len(base) or len(train) != len(folds) or len(train) != len(saved_oof):
        raise RuntimeError("input length mismatch")

    y = train.label.to_numpy(bool)
    segments = segment_ids(train)
    outer = folds == 3
    inner = np.isin(folds, [1, 2])
    model_dir = ROOT / "models"

    # The sealed outer uses the honest fold-3 booster and fold-3-only reference.
    honest_score = score_from_models(train, base, folds, model_dir, honest=True)
    current_pred = fill_short_gaps(honest_score >= THRESHOLD, segments, HOLE)
    current_stats = f1_stats(y, current_pred, outer)
    if (
        current_stats["TP"], current_stats["FP"], current_stats["FN"]
    ) != EXPECTED:
        print(
            f"reproduction|TP={current_stats['TP']}|FP={current_stats['FP']}|"
            f"FN={current_stats['FN']}|F1={current_stats['F1']:.6f}"
        )
        return 1

    # Inner fold 1-2 uses the locked per-fold exp002 boosters.
    inner_score = score_from_models(train, base, folds, model_dir, honest=False)
    outer_events = make_events(train, y, current_pred, honest_score, segments, outer)
    a_rows = type_table(train, y, current_pred, outer_events, outer)
    b_rows = station_layer_table(train, y, current_pred, outer)
    c_rows = length_table(y, current_pred, outer_events)
    d_rows = recoverability(train, y, current_pred, honest_score, outer_events)
    e_rows, e_predictions = evaluate_configs(honest_score, segments, y, outer)
    f_rows, _ = evaluate_configs(inner_score, segments, y, inner)
    g_rows = {
        "noise": noise_or_drift_table(train, y, honest_score, base, outer_events, "noise"),
        "drift": noise_or_drift_table(train, y, honest_score, base, outer_events, "drift"),
    }

    outer_rank = [row["setting"] for row in sorted(e_rows, key=lambda row: (-row["F1"], row["setting"]))]
    inner_rank = [row["setting"] for row in sorted(f_rows, key=lambda row: (-row["F1"], row["setting"]))]
    inner_top_in_outer_top3 = "예" if inner_rank[0] in outer_rank[:3] else "아니오"

    metrics = {
        "seed": SEED,
        "n_jobs": N_JOBS,
        "alignment_key": "train.csv original row order; station-layer time order with 10-minute gap segments",
        "inputs": {
            "train_rows": int(len(train)),
            "saved_oof_rows": int(len(saved_oof)),
            "fold_counts": {str(k): int(np.count_nonzero(folds == k)) for k in range(4)},
        },
        "reproduction": current_stats,
        "A_anomaly_type": a_rows,
        "B_station_layer_top10": b_rows,
        "C_event_length": c_rows,
        "D_recoverability": d_rows,
        "E_outer_postprocessing": e_rows,
        "F_inner_postprocessing": {
            "rows": f_rows,
            "inner_top_in_outer_top3": inner_top_in_outer_top3,
        },
        "G_noise_and_drift": g_rows,
        "event_count_outer": int(len(outer_events)),
        "event_definition": "positive runs within station-layer 10-minute segments, selected on fold3",
    }
    out_path = ROOT / "cache/T4/metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print("alignment_key|train.csv original row order|station-layer time order|10-minute gap segments")
    print(
        f"reproduction|TP={current_stats['TP']}|FP={current_stats['FP']}|"
        f"FN={current_stats['FN']}|F1={current_stats['F1']:.6f}"
    )
    print_table(
        "A",
        a_rows,
        ["anomaly_type", "event_count", "positive_rows", "TP", "FN", "row_recall", "event_hit_rate"],
    )
    print_table(
        "B",
        b_rows,
        ["station", "layer", "positive_rows", "FN", "recall", "dominant_anomaly_type"],
    )
    print_table(
        "C",
        c_rows,
        ["bin", "event_count", "positive_rows", "FN", "row_recall", "event_hit_rate"],
    )
    print("D")
    print("category|count|fraction_of_FN")
    for category, row in d_rows["categories"].items():
        print(f"{category}|{row['count']}|{row['fraction_of_FN']:.6f}")
    print("D_by_anomaly_type")
    print("category|anomaly_type|count|fraction_of_category")
    for category, rows in d_rows["by_anomaly_type"].items():
        for row in rows:
            print(
                f"{category}|{row['anomaly_type']}|{row['count']}|"
                f"{row['fraction_of_category']:.6f}"
            )
    print_table("E", e_rows, ["setting", "TP", "FP", "FN", "F1", "delta_vs_current"])
    print_table("F", f_rows, ["setting", "TP", "FP", "FN", "F1", "delta_vs_current"])
    print(f"F_inner_top_in_outer_top3|{inner_top_in_outer_top3}")
    print("G")
    for target, row in g_rows.items():
        print(target)
        for key, value in row.items():
            print(f"{key}|{value:.6f}" if isinstance(value, float) else f"{key}|{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
