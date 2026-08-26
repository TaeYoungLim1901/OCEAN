"""Q4 read-only second-signal AUC screening."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import _rolling_mean, _rolling_slope, _run_length, _segments_from_time
from q2_cross_layer import (
    _auc_and_recall,
    _rolling_mad,
    _spearman,
)
from q3_neighbor import _derive_neighbor_signals, _fit_residual, _neighbor_residuals

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
BASE_TYPES = ("offset", "drift", "noise", "flatline")
CANDIDATES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8_432", "s8_1008")
COMPARATORS = ("n1", "n3", "n4", "resid_1009", "clim_resid")


def _neighbor_median(train, test):
    wide = train.pivot_table(
        index=["station", "_time"], columns="layer", values="temp", aggfunc="median"
    )
    test_layers = {
        station: set(group.layer.unique())
        for station, group in test.groupby("station", sort=False)
    }
    out = np.full(len(train), np.nan, dtype=float)
    work = train.sort_values(["station", "layer", "_time"], kind="stable")
    for (station, layer), group in work.groupby(["station", "layer"], sort=False):
        neighbors = sorted(test_layers.get(station, set()) - {layer})
        if not neighbors:
            continue
        index = pd.MultiIndex.from_frame(group[["station", "_time"]])
        values = wide.reindex(columns=neighbors).reindex(index)
        out[group["_row"].to_numpy(np.int64)] = values.median(axis=1).to_numpy(float)
    return out


def _rolling_corr(left, right, window):
    valid = np.isfinite(left) & np.isfinite(right)
    x = np.where(valid, left, 0.0)
    y = np.where(valid, right, 0.0)
    v = valid.astype(float)

    def roll(values):
        return pd.Series(values).rolling(window, min_periods=1).sum().to_numpy(float)

    n, sx, sy = roll(v), roll(x), roll(y)
    sxx, syy, sxy = roll(x * x), roll(y * y), roll(x * y)
    cov = n * sxy - sx * sy
    den = np.sqrt(np.maximum(0.0, (n * sxx - sx * sx) * (n * syy - sy * sy)))
    out = np.full(len(left), np.nan, dtype=float)
    use = (n >= 2) & (den > 0)
    out[use] = np.clip(cov[use] / den[use], -1.0, 1.0)
    return out


def _derive_q4_signals(train, test, n1, n3, started):
    n = len(train)
    out = {name: np.full(n, np.nan, dtype=float) for name in CANDIDATES}
    work = train.sort_values(["station", "layer", "_time"], kind="stable")
    neighbor = _neighbor_median(train, test)
    temp = train.temp.to_numpy(float)
    psal = train.psal.to_numpy(float)

    for _, group in work.groupby(["station", "layer"], sort=False):
        rows = group["_row"].to_numpy(np.int64)
        psal_fit, s7 = _fit_residual(psal[rows, None], temp[rows])
        if psal_fit is not None:
            out["s7"][rows] = s7
        for local in _segments_from_time(group["_time"].to_numpy()):
            pos = rows[local]
            z = n1[pos] / (_rolling_mad(n1[pos], 4320) + 1e-3)
            n3_values = n3[pos]
            corr = _rolling_corr(temp[pos], neighbor[pos], 432)
            out["s1"][pos] = z
            out["s2"][pos] = n3_values / (_rolling_mad(n1[pos], 4320) + 1e-3)
            out["s3"][pos] = corr
            out["s4"][pos] = z * np.clip(corr, 0.0, 1.0)
            out["s8_432"][pos] = _rolling_slope(n1[pos], 432)
            out["s8_1008"][pos] = _rolling_slope(n1[pos], 1008)

            if time.perf_counter() - started < 1140:
                clipped = np.clip(np.nan_to_num(z, nan=0.0), -3.0, 3.0)
                cumulative = np.cumsum(clipped)
                out["s5"][pos] = cumulative - _rolling_mean(cumulative, 432, False)
                out["s5"][pos] = np.where(np.isfinite(z), out["s5"][pos], np.nan)
                out["s6"][pos] = _run_length(np.isfinite(z) & (np.abs(z) > 2.0))
            else:
                out.pop("s5", None)
                out.pop("s6", None)
                print("timeout_skip=s5,s6")
                return out
    return out


def _score_values(signal, values):
    return 1.0 - values if signal == "s3" else values


def _table_a(train, signals):
    types = train.anomaly_type.fillna("").astype(str).to_numpy()
    normal = train.label.to_numpy(np.int8) == 0
    rows = []
    for signal, values in signals.items():
        score = _score_values(signal, values)
        for kind in BASE_TYPES:
            positive = types == kind
            auc, recall = _auc_and_recall(score, positive, normal)
            rows.append(
                {
                    "signal": signal,
                    "type": kind,
                    "n_pos": int(positive.sum()),
                    "n_nan_pos": int((positive & ~np.isfinite(values)).sum()),
                    "auc": auc,
                    "recall_at_fpr_0.03": recall,
                }
            )
    return pd.DataFrame(rows)


def _auc_for_mask(train, signal, values, kind, mask):
    types = train.anomaly_type.fillna("").astype(str).to_numpy()
    labels = train.label.to_numpy(np.int8)
    return _auc_and_recall(
        _score_values(signal, values), mask & (types == kind), mask & (labels == 0)
    )[0]


def _table_b(train, signals):
    month = train["_time"].dt.month.to_numpy()
    rows = []
    for signal, values in signals.items():
        jan_jun = month <= 6
        jul_dec = month >= 7
        for kind in BASE_TYPES:
            first = _auc_for_mask(train, signal, values, kind, jan_jun)
            second = _auc_for_mask(train, signal, values, kind, jul_dec)
            rows.append(
                {
                    "signal": signal,
                    "type": kind,
                    "auc_jan_jun": first,
                    "auc_jul_dec": second,
                    "diff": first - second,
                }
            )
    return pd.DataFrame(rows)


def _table_c(train, signals):
    month = train["_time"].dt.month.to_numpy()
    years = train.year.to_numpy()
    rows = []
    for signal, values in signals.items():
        for kind in ("offset", "drift", "noise"):
            first = _auc_for_mask(train, signal, values, kind, (years == 2024) & (month <= 6))
            second = _auc_for_mask(train, signal, values, kind, (years == 2025) & (month <= 6))
            rows.append(
                {
                    "signal": signal,
                    "type": kind,
                    "auc_2024": first,
                    "auc_2025": second,
                    "diff": first - second,
                }
            )
    return pd.DataFrame(rows)


def _top_signals(table_a):
    ranking = (
        table_a.loc[table_a.signal.isin(CANDIDATES) & table_a.type.isin(["offset", "drift"])]
        .groupby("signal", sort=False)
        .auc.mean()
        .sort_values(ascending=False)
    )
    return ranking.index[:5].tolist()


def _table_d(signals, top):
    rows = []
    for signal in top:
        for comparator in COMPARATORS:
            rows.append(
                {
                    "signal": signal,
                    "vs": comparator,
                    "spearman": _spearman(signals[signal], signals[comparator]),
                }
            )
    return pd.DataFrame(rows)


def _print_table(name, frame, started):
    print(name)
    print(frame.to_string(index=False))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


def main():
    started = time.perf_counter()
    train = pd.read_csv(
        ROOT / "P1_data/train.csv",
        usecols=["station", "year", "layer", "time", "temp", "psal", "label", "anomaly_type"],
    )
    train["_time"] = pd.to_datetime(train.time, utc=True)
    train["_row"] = np.arange(len(train), dtype=np.int64)
    test = pd.read_csv(ROOT / "P1_data/test.csv", usecols=["station", "layer", "time"])
    test["_time"] = pd.to_datetime(test.time, utc=True)

    n1, _ = _neighbor_residuals(train, test)
    derived = _derive_neighbor_signals(train, n1)
    signals = _derive_q4_signals(train, test, n1, derived["n3"], started)
    signals.update({name: derived[name] for name in ("n1", "n3", "n4")})

    baseline = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    if len(baseline) != len(train):
        raise ValueError("train_features.pkl row count does not match train.csv")
    signals["resid_1009"] = baseline["resid_1009"].to_numpy(float)
    signals["clim_resid"] = baseline["clim_resid"].to_numpy(float)

    table_a = _table_a(train, signals)
    baseline_auc = table_a.loc[
        table_a.signal.eq("n1") & table_a.type.eq("offset"), "auc"
    ].iloc[0]
    if not np.isclose(baseline_auc, 0.903359, atol=0.001):
        raise AssertionError(f"n1 offset AUC mismatch: {baseline_auc}")
    displayed_a = table_a.loc[table_a.auc >= 0.75].copy()
    _print_table("Q4-A filter=auc>=0.75", displayed_a, started)
    _print_table("Q4-B", _table_b(train, signals), started)
    _print_table("Q4-C", _table_c(train, signals), started)
    _print_table("Q4-D", _table_d(signals, _top_signals(table_a)), started)


if __name__ == "__main__":
    main()
