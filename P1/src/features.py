"""P1 공용 시계열 feature 경로. 학습과 추론이 이 파일만 사용한다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
KEY = ["station", "layer"]
STREAM_KEY = ["station", "year", "layer"]
INTEGRATED_FEATURES = tuple(
    [f"{var}_z_{mode}_{w}" for var in ("temp", "psal") for w in (12, 48, 144, 288) for mode in ("center", "trailing")]
    + [f"max_z_{mode}_{w}" for w in (12, 48, 144, 288) for mode in ("center", "trailing")]
    + [f"{var}_step{suffix}_{w}" for var in ("temp", "psal") for w in (48, 144) for suffix in ("", "_abs")]
    + [f"{var}_slope_{w}" for var in ("temp", "psal") for w in (144, 288)]
    + [f"{var}_z288_run" for var in ("temp", "psal")]
    + ["max_z288_run"]
)
WINDOWS = (7, 37, 145, 433, 1009)  # 약 1h, 6h, 1d, 3d, 7d


def _smooth(x, size):
    if len(x) == 1:
        return x.astype(np.float32)
    size = min(size, len(x) if len(x) % 2 else max(1, len(x) - 1))
    return uniform_filter1d(x.astype(float), size=max(1, size), mode="nearest").astype(np.float32)


def _finite_dict(values):
    return {
        "|".join(map(str, key if isinstance(key, tuple) else (key,))): float(value)
        for key, value in values.items()
        if np.isfinite(value)
    }


def fit_reference(raw, mask=None):
    """Fit climatology and residual scales on the selected training rows only."""
    d = raw if mask is None else raw.loc[np.asarray(mask, dtype=bool)]
    d = d.copy()
    d["_month"] = pd.to_datetime(d.time, utc=True).dt.month
    ref = {}
    for var, prefix in (("temp", ""), ("psal", "psal_")):
        med = d.groupby(["station", "layer", "_month"], sort=False)[var].median()
        joined = d.join(med.rename("_med"), on=["station", "layer", "_month"])
        mad = joined.assign(_dev=(joined[var] - joined["_med"]).abs()).groupby(
            ["station", "layer", "_month"], sort=False
        )[["_dev"]].median()['_dev']
        med_name = "median" if not prefix else f"{prefix}median"
        mad_name = "mad" if not prefix else f"{prefix}mad"
        ref[med_name] = _finite_dict(med)
        ref[mad_name] = _finite_dict(mad)
        global_median = float(d[var].median())
        ref[f"global_{prefix}median"] = global_median if np.isfinite(global_median) else 0.0

        keys = d[["station", "layer", "_month"]].astype(str).agg("|".join, axis=1)
        residual = d[var].to_numpy(float) - keys.map(ref[med_name]).fillna(ref[f"global_{prefix}median"]).to_numpy(float)
        work = d[["station", "year", "layer"]].copy()
        work["_resid"] = residual
        stream_sigma = work.groupby(["station", "year", "layer"], sort=False)[["_resid"]].std()['_resid']
        layer_sigma = work.groupby(["station", "layer"], sort=False)[["_resid"]].std()['_resid']
        global_sigma = float(np.nanstd(residual, ddof=1))
        if not np.isfinite(global_sigma) or global_sigma <= 0:
            global_sigma = 1.0
        ref[f"sigma_{prefix}stream"] = _finite_dict(stream_sigma)
        ref[f"sigma_{prefix}layer"] = _finite_dict(layer_sigma)
        ref[f"global_sigma_{prefix}"] = global_sigma
    return ref


def _mapped(d, reference, name, columns, fallback=None, default=np.nan):
    keys = d[columns].astype(str).agg("|".join, axis=1)
    values = keys.map(reference.get(name, {}))
    if fallback:
        fallback_keys = d[fallback].astype(str).agg("|".join, axis=1)
        fallback_name = name.replace("_stream", "_layer")
        values = values.fillna(fallback_keys.map(reference.get(fallback_name, {})))
    return values.fillna(default).to_numpy(float)


def _segments_from_time(times):
    t = np.asarray(times)
    breaks = np.r_[True, np.diff(t).astype("timedelta64[m]").astype(int) != 10]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(t)]
    return [np.arange(start, end, dtype=np.int64) for start, end in zip(starts, ends)]


def _rolling_mean(values, window, center):
    return pd.Series(values).rolling(window, center=center, min_periods=1).mean().to_numpy(float)


def _rolling_median(values, window, center=False):
    return pd.Series(values).rolling(window, center=center, min_periods=1).median().to_numpy(float)


def _step_score(values, window):
    before = pd.Series(values).shift(1).rolling(window, min_periods=1).median().to_numpy(float)
    reverse = pd.Series(values[::-1]).shift(1).rolling(window, min_periods=1).median().to_numpy(float)[::-1]
    return reverse - before


def _rolling_slope(values, window):
    valid = np.isfinite(values).astype(float)
    y = np.nan_to_num(values, nan=0.0)
    x = np.arange(len(values), dtype=float)

    def roll(a):
        return pd.Series(a).rolling(window, center=True, min_periods=1).sum().to_numpy(float)

    n = roll(valid)
    sx = roll(x * valid)
    sy = roll(y)
    sxx = roll(x * x * valid)
    sxy = roll(x * y)
    den = n * sxx - sx * sx
    slope = np.full(len(values), np.nan, dtype=float)
    use = (n >= 2) & (den > 0)
    slope[use] = (n[use] * sxy[use] - sx[use] * sy[use]) / den[use]
    return slope * window


def _run_length(flag):
    out = np.zeros(len(flag), dtype=np.float32)
    for i, value in enumerate(flag):
        out[i] = out[i - 1] + 1 if value and i else float(value)
    return out


def _integrated_features(raw, reference):
    d = raw.copy()
    d["_time"] = pd.to_datetime(d.time, utc=True)
    d["_pos"] = np.arange(len(d), dtype=np.int64)
    d = d.sort_values(STREAM_KEY + ["_time"], kind="stable")
    d["_row"] = np.arange(len(d), dtype=np.int64)
    month = d["_time"].dt.month
    residuals = {}
    sigmas = {}
    for var, prefix in (("temp", ""), ("psal", "psal_")):
        med_name = "median" if not prefix else f"{prefix}median"
        global_median = reference.get(f"global_{prefix}median", float(d[var].median()))
        med = _mapped(d.assign(_month=month), reference, med_name, ["station", "layer", "_month"], default=global_median)
        residuals[var] = d[var].to_numpy(float) - med
        sigma_name = f"sigma_{prefix}stream"
        global_sigma = reference.get(f"global_sigma_{prefix}", 1.0)
        sigmas[var] = np.maximum(
            _mapped(
                d,
                reference,
                sigma_name,
                STREAM_KEY,
                fallback=["station", "layer"],
                default=global_sigma,
            ),
            1e-6,
        )

    values = {name: np.full(len(d), np.nan, dtype=np.float32) for name in INTEGRATED_FEATURES}
    for _, group in d.groupby(STREAM_KEY, sort=False):
        positions = group["_row"].to_numpy(np.int64)
        for local in _segments_from_time(group["_time"].to_numpy()):
            pos = positions[local]
            z_by_var = {}
            for var in ("temp", "psal"):
                z_by_var[var] = {}
                x = residuals[var][pos]
                sigma = sigmas[var][pos]
                for window in (12, 48, 144, 288):
                    for mode, center in (("center", True), ("trailing", False)):
                        mean = _rolling_mean(x, window, center)
                        z = mean / (sigma / np.sqrt(window))
                        z_by_var[var][mode, window] = z
                        values[f"{var}_z_{mode}_{window}"][pos] = z
                for window in (48, 144):
                    step = _step_score(x, window) / sigma
                    values[f"{var}_step_{window}"][pos] = step
                    values[f"{var}_step_abs_{window}"][pos] = np.abs(step)
                for window in (144, 288):
                    values[f"{var}_slope_{window}"][pos] = _rolling_slope(x, window) / sigma
                values[f"{var}_z288_run"][pos] = _run_length(
                    np.isfinite(z_by_var[var]["center", 288])
                    & (np.abs(z_by_var[var]["center", 288]) >= 2.0)
                )
            for mode in ("center", "trailing"):
                for window in (12, 48, 144, 288):
                    values[f"max_z_{mode}_{window}"][pos] = np.fmax(
                        np.abs(z_by_var["temp"][mode, window]),
                        np.abs(z_by_var["psal"][mode, window]),
                    )
            values["max_z288_run"][pos] = _run_length(
                values["max_z_center_288"][pos] >= 2.0
            )
    return pd.DataFrame(values, index=d["_pos"].to_numpy()).sort_index().set_axis(raw.index)


def add_integrated_features(base, raw, reference):
    """Add T12 columns to an existing feature frame without changing old columns."""
    out = base.copy()
    d = raw.copy()
    d["_month"] = pd.to_datetime(d.time, utc=True).dt.month
    med = _mapped(d, reference, "median", ["station", "layer", "_month"], default=float(d.temp.median()))
    mad = _mapped(d, reference, "mad", ["station", "layer", "_month"], default=0.5)
    out["clim_resid"] = ((d.temp.to_numpy(float) - med) / np.maximum(mad, 0.01)).astype(np.float32)
    extra = _integrated_features(raw, reference)
    for column in INTEGRATED_FEATURES:
        out[column] = extra[column]
    return out


def _stream_features(g):
    """한 station-layer를 gap별로 처리한다."""
    t = g["_time"].to_numpy()
    x = g.temp.to_numpy(float)
    gap = np.r_[True, np.diff(t).astype("timedelta64[m]").astype(int) != 10]
    gid = gap.cumsum()
    out = {name: np.empty(len(g), np.float32) for name in
           ["d_prev", "d_next", "spike"] +
           [f"mean_{w}" for w in WINDOWS] + [f"std_{w}" for w in WINDOWS]}
    out["gap_start"] = gap.astype(np.float32)
    for k in np.unique(gid):
        pos = np.flatnonzero(gid == k)
        z = x[pos]
        prev = np.r_[z[0], z[:-1]]
        nxt = np.r_[z[1:], z[-1]]
        out["d_prev"][pos] = z - prev
        out["d_next"][pos] = nxt - z
        out["spike"][pos] = z - (prev + nxt) / 2
        for w in WINDOWS:
            m = _smooth(z, w)
            q = _smooth(z * z, w)
            out[f"mean_{w}"][pos] = m
            out[f"std_{w}"][pos] = np.sqrt(np.maximum(0, q - m * m))
    return pd.DataFrame(out, index=g.index)


def make_features(raw, reference=None, integrated=False):
    d = raw.copy()
    d["_time"] = pd.to_datetime(d.time, utc=True)
    d = d.sort_values(["station", "layer", "_time"], kind="stable")
    parts = [_stream_features(g) for _, g in d.groupby(KEY, sort=False)]
    f = pd.concat(parts).sort_index()
    d = d.sort_index()

    eps = 1e-4
    f["abs_d_prev"] = f.d_prev.abs()
    f["abs_d_next"] = f.d_next.abs()
    f["abs_spike"] = f.spike.abs()
    for w in WINDOWS:
        f[f"resid_{w}"] = d.temp.to_numpy(np.float32) - f[f"mean_{w}"]
    f["noise_ratio_7_145"] = f.std_7 / (f.std_145 + eps)
    f["noise_ratio_37_433"] = f.std_37 / (f.std_433 + eps)
    f["flat_ratio"] = f.std_7 / (f.std_145 + 0.01)
    f["offset_score"] = f.resid_1009.abs() / (f.std_1009 + 0.03)
    f["drift_score"] = (f.mean_37 - f.mean_433).abs() / (f.std_433 + 0.03)

    # 같은 기지·시각의 다른 층은 이상 주입이 없는 국지 기준 후보다.
    section = d.groupby(["station", "_time"], sort=False).temp.transform("median")
    spread = d.groupby(["station", "_time"], sort=False).temp.transform("std").fillna(0)
    f["layer_resid"] = d.temp.to_numpy(np.float32) - section.to_numpy(np.float32)
    f["layer_spread"] = spread.to_numpy(np.float32)
    f["layer_count"] = d.groupby(["station", "_time"], sort=False).temp.transform("size").to_numpy(np.float32)

    tm = d._time.dt
    f["station"] = d.station.map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}).to_numpy(np.float32)
    f["layer"] = d.layer.to_numpy(np.float32)
    f["month"] = tm.month.to_numpy(np.float32)
    doy = tm.dayofyear.to_numpy(float)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    f["temp"] = d.temp.to_numpy(np.float32)
    for c in ("psal", "depth"):
        f[c] = d[c].to_numpy(np.float32)
        f[c + "_missing"] = d[c].isna().to_numpy(np.float32)

    # station-layer-month robust climatology: train에서만 적합, 추론에서 고정 재사용.
    ref_key = pd.DataFrame({"station": d.station, "layer": d.layer, "month": tm.month})
    if reference is None:
        reference = fit_reference(raw)
    keys = ref_key.astype(str).agg("|".join, axis=1)
    med = keys.map(reference.get("median", {})).fillna(reference.get("global_median", d.temp.median())).to_numpy(float)
    mad = keys.map(reference.get("mad", {})).fillna(0.5).clip(lower=0.01).to_numpy(float)
    f["clim_resid"] = ((d.temp.to_numpy(float) - med) / mad).astype(np.float32)

    # 규칙 후보는 학습 라벨을 보지 않는 도메인 점수다.
    spike = f.abs_spike / (f.std_145 + 0.03)
    noise = np.maximum(f.noise_ratio_7_145 - 1.5, 0)
    flat = np.maximum(1.0 - f.flat_ratio * 5, 0) * (f.std_145 > 0.005)
    persistent = np.maximum(f.offset_score - 1.5, 0) + np.maximum(f.drift_score - 1.0, 0)
    rule = np.maximum.reduce([spike.to_numpy() / 8, noise.to_numpy() / 5,
                              flat.to_numpy(), persistent.to_numpy() / 6])
    f["rule_score"] = np.clip(rule, 0, 1).astype(np.float32)
    if integrated:
        f = add_integrated_features(f, raw, reference)
    return f.replace([np.inf, -np.inf], np.nan).fillna(-999).astype(np.float32), reference


def time_folds(raw, n_folds=4):
    """station-layer별 순차 block. 랜덤 행 분할을 만들지 않는다."""
    d = raw.assign(_time=pd.to_datetime(raw.time, utc=True)).sort_values(["station", "layer", "_time"])
    fold = np.empty(len(d), np.int8)
    for _, g in d.groupby(KEY, sort=False):
        fold[g.index] = np.minimum(n_folds - 1, np.arange(len(g)) * n_folds // len(g))
    return fold


def train_mask_with_purge(raw, folds, held_out, hours=87):
    """과거 학습 행만 남기고 검증 시작 전 87시간을 purge한다."""
    tm = pd.to_datetime(raw.time, utc=True)
    keep = folds < held_out
    groups = raw[KEY].astype(str).agg("\x1f".join, axis=1)
    starts = tm.where(folds == held_out).groupby(groups, sort=False).transform("min")
    has_validation = starts.notna().to_numpy()
    cutoff = starts - pd.Timedelta(hours=hours)
    keep[has_validation] &= tm.to_numpy()[has_validation] < cutoff.to_numpy()[has_validation]
    assert int((keep & (folds >= held_out)).sum()) == 0
    return keep
