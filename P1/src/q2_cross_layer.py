"""Q2 read-only cross-layer and cross-station signal measurements."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from features import _rolling_mean, _rolling_median, _rolling_slope, _segments_from_time

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
BASE_TYPES = ("offset", "drift", "noise", "flatline")


def _lookup(wide, rows, columns):
    index = pd.MultiIndex.from_frame(rows[columns])
    return wide.reindex(index).to_numpy(float).reshape(-1)


def _self_excluding_layer_reference(d):
    wide = d.pivot_table(
        index=["station", "_time"], columns="layer", values="temp", aggfunc="median"
    )
    ref = np.full(len(d), np.nan, dtype=float)
    for layer in wide.columns:
        other = [column for column in wide.columns if column != layer]
        if not other:
            continue
        mask = d.layer.eq(layer)
        ref[mask.to_numpy()] = _lookup(
            wide[other].median(axis=1), d.loc[mask], ["station", "_time"]
        )
    return ref


def _self_excluding_station_reference(d):
    allowed = d.station.isin(["I-ORS", "S-ORS"])
    wide = d.loc[allowed].pivot_table(
        index=["_time", "layer"], columns="station", values="temp", aggfunc="median"
    )
    ref = np.full(len(d), np.nan, dtype=float)
    for station in ("I-ORS", "S-ORS"):
        other = [column for column in wide.columns if column != station]
        if not other:
            continue
        mask = d.station.eq(station)
        ref[mask.to_numpy()] = _lookup(
            wide[other].median(axis=1), d.loc[mask], ["_time", "layer"]
        )
    return ref


def _rolling_mad(values, window):
    def mad(a):
        a = a[np.isfinite(a)]
        if not len(a):
            return np.nan
        med = np.median(a)
        return np.median(np.abs(a - med))

    return pd.Series(values).rolling(window, min_periods=1).apply(mad, raw=True).to_numpy(float)


def _derive_signals(d, base, prefix, second_windows):
    n = len(d)
    names = [f"{prefix}1"]
    names += [f"{prefix}2_{w}" for w in second_windows]
    names += [f"{prefix}3_{w}" for w in second_windows]
    names += [f"{prefix}4"]
    if prefix == "x":
        names += ["x5"]
    out = {name: np.full(n, np.nan, dtype=float) for name in names}

    work = d.sort_values(["station", "layer", "_time"], kind="stable")
    for _, group in work.groupby(["station", "layer"], sort=False):
        rows = group._row.to_numpy(np.int64)
        for local in _segments_from_time(group._time.to_numpy()):
            pos = rows[local]
            values = base[pos]
            out[f"{prefix}1"][pos] = values
            mad432 = _rolling_mad(values, 432)
            for window in second_windows:
                centered = values - _rolling_median(values, window)
                out[f"{prefix}2_{window}"][pos] = centered
                out[f"{prefix}3_{window}"][pos] = centered / (mad432 + 1e-3)
            out[f"{prefix}4"][pos] = _rolling_mean(values, 36, False) - _rolling_mean(
                values, 432, False
            )
            if prefix == "x":
                out["x5"][pos] = _rolling_slope(values, 144)
    return out


def _signals(d):
    layer_ref = _self_excluding_layer_reference(d)
    station_ref = _self_excluding_station_reference(d)
    temp = d.temp.to_numpy(float)
    signals = {"x1": temp - layer_ref, "y1": temp - station_ref}
    signals.update(_derive_signals(d, signals["x1"], "x", (36, 144, 432)))
    signals.update(_derive_signals(d, signals["y1"], "y", (144,)))
    return signals


def _run_table(d):
    label_wide = d.pivot_table(
        index=["station", "_time"], columns="layer", values="label", aggfunc="max"
    )
    sibling = {}
    for layer in label_wide.columns:
        other = [column for column in label_wide.columns if column != layer]
        mask = d.layer.eq(layer)
        if other:
            counts = label_wide[other].notna().sum(axis=1)
            positives = label_wide[other].fillna(0).sum(axis=1)
            sibling["count", layer] = _lookup(counts, d.loc[mask], ["station", "_time"])
            sibling["positive", layer] = _lookup(positives, d.loc[mask], ["station", "_time"])
        else:
            sibling["count", layer] = np.zeros(int(mask.sum()), dtype=float)
            sibling["positive", layer] = np.zeros(int(mask.sum()), dtype=float)

    counts = np.zeros(len(d), dtype=float)
    positives = np.zeros(len(d), dtype=float)
    for layer in label_wide.columns:
        mask = d.layer.eq(layer).to_numpy()
        counts[mask] = sibling["count", layer]
        positives[mask] = sibling["positive", layer]

    rows = []
    work = d.sort_values(["station", "layer", "_time"], kind="stable")
    for _, group in work.groupby(["station", "layer"], sort=False):
        row_ids = group._row.to_numpy(np.int64)
        labels = group.label.to_numpy(np.int8)
        types = group.anomaly_type.to_numpy(object)
        for local in _segments_from_time(group._time.to_numpy()):
            segment_rows = row_ids[local]
            segment_labels = labels[local]
            flags = segment_labels == 1
            starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
            ends = np.flatnonzero(flags & ~np.r_[flags[1:], False]) + 1
            for start, end in zip(starts, ends):
                run_rows = segment_rows[start:end]
                run_types = pd.Series(types[start:end]).fillna("").astype(str)
                mode = run_types.value_counts().index[0] if len(run_types) else "other"
                overlap_den = counts[run_rows].sum()
                overlap = positives[run_rows].sum() / overlap_den if overlap_den else np.nan
                rows.append(
                    {
                        "type": mode if mode in BASE_TYPES else "other",
                        "length": len(run_rows),
                        "overlap": overlap,
                    }
                )

    runs = pd.DataFrame(rows)
    result = []
    for kind in (*BASE_TYPES, "other"):
        subset = runs.loc[runs.type.eq(kind)]
        overlap = subset.overlap.dropna().to_numpy(float)
        result.append(
            {
                "type": kind,
                "n_runs": len(subset),
                "median_len": float(subset.length.median()) if len(subset) else np.nan,
                "sibling_overlap_p50": float(np.percentile(overlap, 50)) if len(overlap) else np.nan,
                "sibling_overlap_p90": float(np.percentile(overlap, 90)) if len(overlap) else np.nan,
                "runs_with_zero_overlap": int((overlap == 0).sum()),
            }
        )
    return pd.DataFrame(result)


def _auc_and_recall(score, positive, normal):
    score = np.asarray(score, float)
    positive = np.asarray(positive, bool)
    normal = np.asarray(normal, bool)
    finite = np.isfinite(score)
    valid = finite & (positive | normal)
    y = positive[valid]
    s = np.abs(score[valid])
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if not n_pos or not n_neg:
        return np.nan, np.nan

    ranks = rankdata(s, method="average")
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    order = np.argsort(s, kind="stable")[::-1]
    sorted_pos = y[order]
    fp = np.cumsum(~sorted_pos)
    tp = np.cumsum(sorted_pos)
    max_fp = int(np.floor(0.03 * n_neg))
    allowed = fp <= max_fp
    recall = float(tp[allowed].max() / n_pos) if allowed.any() else 0.0
    return float(auc), recall


def _signal_table(d, signals):
    types = d.anomaly_type.fillna("").astype(str).to_numpy()
    normal = d.label.to_numpy(np.int8) == 0
    rows = []
    for signal, values in signals.items():
        for kind in BASE_TYPES:
            positive = types == kind
            finite_pos = np.isfinite(values)
            auc, recall = _auc_and_recall(values, positive, normal)
            rows.append(
                {
                    "signal": signal,
                    "type": kind,
                    "n_pos": int(positive.sum()),
                    "n_nan_pos": int((positive & ~finite_pos).sum()),
                    "auc": auc,
                    "recall_at_fpr_0.03": recall,
                }
            )
    return pd.DataFrame(rows)


def _station_table(d, signals, table_b):
    types = d.anomaly_type.fillna("").astype(str).to_numpy()
    stations = d.station.to_numpy(object)
    normal = d.label.to_numpy(np.int8) == 0
    ranking = (
        table_b.loc[table_b.type.isin(["offset", "drift"])]
        .groupby("signal", sort=False)
        .auc.mean()
        .sort_values(ascending=False)
    )
    top = ranking.index[:5].tolist()
    rows = []
    for station in ("I-ORS", "S-ORS", "G-ORS"):
        station_mask = stations == station
        for signal in top:
            values = signals[signal]
            for kind in ("offset", "drift"):
                positive = station_mask & (types == kind)
                eval_normal = station_mask & normal
                auc, recall = _auc_and_recall(values, positive, eval_normal)
                rows.append(
                    {
                        "station": station,
                        "signal": signal,
                        "type": kind,
                        "auc": auc,
                        "recall_at_fpr_0.03": recall,
                    }
                )
    return pd.DataFrame(rows)


def _spearman(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    left_rank = rankdata(left[valid], method="average")
    right_rank = rankdata(right[valid], method="average")
    if left_rank.std() == 0 or right_rank.std() == 0:
        return np.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _correlation_table(signals, table_b):
    ranking = (
        table_b.loc[table_b.type.isin(["offset", "drift"])]
        .groupby("signal", sort=False)
        .auc.mean()
        .sort_values(ascending=False)
    )
    selected = ranking.index[:3].tolist()
    selected += ["offset_score", "drift_score", "clim_resid", "layer_resid", "resid_1009"]
    selected = list(dict.fromkeys(selected))
    rows = []
    for i, left_name in enumerate(selected):
        for right_name in selected[i + 1 :]:
            rows.append(
                {
                    "signal_a": left_name,
                    "signal_b": right_name,
                    "spearman": _spearman(signals[left_name], signals[right_name]),
                }
            )
    return pd.DataFrame(rows)


def _print_table(name, frame, started):
    print(name)
    print(frame.to_string(index=False))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


def main():
    started = time.perf_counter()
    raw = pd.read_csv(
        ROOT / "P1_data/train.csv",
        usecols=["station", "year", "layer", "time", "temp", "label", "anomaly_type"],
    )
    raw["_time"] = pd.to_datetime(raw.time, utc=True)
    raw["_row"] = np.arange(len(raw), dtype=np.int64)

    table_a = _run_table(raw)
    _print_table("Table A", table_a, started)

    signals = _signals(raw)
    baseline = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    if len(baseline) != len(raw):
        raise ValueError("train_features.pkl row count does not match train.csv")
    for name in ("offset_score", "drift_score", "clim_resid", "layer_resid", "resid_1009"):
        signals[name] = baseline[name].to_numpy(float)
    if len(signals) < 12:
        raise AssertionError("Q2 requires at least 12 signals")

    table_b = _signal_table(raw, signals)
    _print_table("Table B", table_b, started)

    table_c = _station_table(raw, signals, table_b)
    _print_table("Table C", table_c, started)

    table_d = _correlation_table(signals, table_b)
    _print_table("Table D", table_d, started)


if __name__ == "__main__":
    main()
