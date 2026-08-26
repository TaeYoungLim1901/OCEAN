"""Q3 read-only neighbor-layer regression residual measurements."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from features import _rolling_mean, _rolling_median, _rolling_slope, _segments_from_time
from q2_cross_layer import (
    _auc_and_recall,
    _derive_signals,
    _self_excluding_layer_reference,
    _spearman,
)

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
BASE_TYPES = ("offset", "drift", "noise", "flatline")
SIGNAL_NAMES = ("n1", "n2_144", "n2_432", "n3", "n4", "x1", "x4", "resid_1009", "clim_resid")


def _impute_neighbors(values):
    """Fill missing neighbor cells with that timestamp's available mean."""
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    total = np.where(finite, values, 0.0).sum(axis=1)
    means = np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
    missing = ~finite & (count[:, None] > 0)
    out = values.copy()
    out[missing] = np.broadcast_to(means[:, None], values.shape)[missing]
    frac = float(missing.sum() / values.size) if values.size else np.nan
    return out, count > 0, frac


def _fit_huber(x, y):
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if valid.sum() < max(20, x.shape[1] + 2):
        return None
    model = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=300, tol=1e-5)
    model.fit(x[valid], y[valid])
    return model.intercept_, model.coef_, valid


def _fit_residual(x, y, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    fit = _fit_huber(x[mask], y[mask])
    if fit is None:
        return None, np.full(len(y), np.nan, dtype=float)
    intercept, coef, valid = fit
    fit_rows = np.flatnonzero(mask)[valid]
    residual = np.full(len(y), np.nan, dtype=float)
    residual[fit_rows] = y[fit_rows] - (intercept + x[fit_rows] @ coef)
    return (intercept, coef), residual


def _cosine(left, right):
    if left is None or right is None:
        return np.nan
    a = np.asarray(left[1], float)
    b = np.asarray(right[1], float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / den) if den > 0 else np.nan


def _sigma(values):
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan


def _neighbor_residuals(train, test):
    wide = train.pivot_table(
        index=["station", "_time"], columns="layer", values="temp", aggfunc="median"
    )
    test_layers = {
        station: set(group.layer.unique())
        for station, group in test.groupby("station", sort=False)
    }
    n1 = np.full(len(train), np.nan, dtype=float)
    transfer = []
    work = train.sort_values(["station", "layer", "_time"], kind="stable")
    for (station, layer), group in work.groupby(["station", "layer"], sort=False):
        neighbors = sorted(test_layers.get(station, set()) - {layer})
        rows = group["_row"].to_numpy(np.int64)
        if not neighbors:
            transfer.append(
                {
                    "station": station,
                    "layer": layer,
                    "coef_cos_sim_2024_vs_2025": np.nan,
                    "sigma_ratio": np.nan,
                    "n_neighbors": 0,
                    "imputed_frac": np.nan,
                }
            )
            continue

        index = pd.MultiIndex.from_frame(group[["station", "_time"]])
        values = wide.reindex(columns=neighbors).reindex(index).to_numpy(float)
        values, usable, imputed_frac = _impute_neighbors(values)
        target = group.temp.to_numpy(float)
        _, residual = _fit_residual(values, target, usable)
        n1[rows] = residual

        year_fits = {}
        for year in (2024, 2025):
            year_mask = group._time.dt.year.to_numpy() == year
            year_mask &= usable
            year_fits[year], year_residual = _fit_residual(values, target, year_mask)
            year_sigma = _sigma(year_residual)
            if year == 2024:
                sigma_2024 = year_sigma
            else:
                sigma_2025 = year_sigma
        transfer.append(
            {
                "station": station,
                "layer": layer,
                "coef_cos_sim_2024_vs_2025": _cosine(year_fits[2024], year_fits[2025]),
                "sigma_ratio": (
                    float(sigma_2025 / sigma_2024)
                    if np.isfinite(sigma_2024) and sigma_2024 > 0 and np.isfinite(sigma_2025)
                    else np.nan
                ),
                "n_neighbors": len(neighbors),
                "imputed_frac": imputed_frac,
            }
        )
    return n1, pd.DataFrame(transfer)


def _derive_neighbor_signals(train, n1):
    out = {name: np.full(len(train), np.nan, dtype=float) for name in ("n1", "n2_144", "n2_432", "n3", "n4")}
    work = train.sort_values(["station", "layer", "_time"], kind="stable")
    for _, group in work.groupby(["station", "layer"], sort=False):
        rows = group._row.to_numpy(np.int64)
        for local in _segments_from_time(group._time.to_numpy()):
            pos = rows[local]
            values = n1[pos]
            out["n1"][pos] = values
            out["n2_144"][pos] = values - _rolling_median(values, 144)
            out["n2_432"][pos] = values - _rolling_median(values, 432)
            out["n3"][pos] = _rolling_mean(values, 36, False) - _rolling_mean(values, 432, False)
            # features._rolling_slope returns slope * window, matching Q2 x5.
            out["n4"][pos] = _rolling_slope(values, 144)
    return out


def _table_a(train, signals):
    types = train.anomaly_type.fillna("").astype(str).to_numpy()
    normal = train.label.to_numpy(np.int8) == 0
    rows = []
    for signal in SIGNAL_NAMES:
        values = signals[signal]
        for kind in BASE_TYPES:
            positive = types == kind
            auc, recall = _auc_and_recall(values, positive, normal)
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


def _table_b(train, signals):
    types = train.anomaly_type.fillna("").astype(str).to_numpy()
    stations = train.station.to_numpy(object)
    labels = train.label.to_numpy(np.int8)
    rows = []
    for station in ("I-ORS", "S-ORS"):
        station_mask = stations == station
        normal = station_mask & (labels == 0)
        for signal in SIGNAL_NAMES:
            for kind in ("offset", "drift"):
                auc, recall = _auc_and_recall(
                    signals[signal], station_mask & (types == kind), normal
                )
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


def _table_d(signals):
    names = ("n3", "n1", "resid_1009", "clim_resid", "x4")
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rows.append(
                {
                    "signal_a": left,
                    "signal_b": right,
                    "spearman": _spearman(signals[left], signals[right]),
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
        usecols=["station", "year", "layer", "time", "temp", "label", "anomaly_type"],
    )
    train["_time"] = pd.to_datetime(train.time, utc=True)
    train["_row"] = np.arange(len(train), dtype=np.int64)
    test = pd.read_csv(ROOT / "P1_data/test.csv", usecols=["station", "layer", "time"])
    test["_time"] = pd.to_datetime(test.time, utc=True)

    n1, table_c = _neighbor_residuals(train, test)
    signals = _derive_neighbor_signals(train, n1)

    layer_ref = _self_excluding_layer_reference(train)
    x1 = train.temp.to_numpy(float) - layer_ref
    q2_signals = _derive_signals(train, x1, "x", (36, 144, 432))
    signals["x1"] = x1
    signals["x4"] = q2_signals["x4"]

    baseline = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    if len(baseline) != len(train):
        raise ValueError("train_features.pkl row count does not match train.csv")
    signals["resid_1009"] = baseline["resid_1009"].to_numpy(float)
    signals["clim_resid"] = baseline["clim_resid"].to_numpy(float)

    table_a = _table_a(train, signals)
    baseline_auc = table_a.loc[
        table_a.signal.eq("resid_1009") & table_a.type.eq("offset"), "auc"
    ].iloc[0]
    if not np.isclose(baseline_auc, 0.871459, atol=5e-4):
        raise AssertionError(f"resid_1009 offset AUC mismatch: {baseline_auc}")
    _print_table("Q3-A", table_a, started)
    _print_table("Q3-B", _table_b(train, signals), started)
    _print_table("Q3-C", table_c, started)
    _print_table("Q3-D", _table_d(signals), started)


if __name__ == "__main__":
    main()
