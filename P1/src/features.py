"""P1 공용 시계열 feature 경로. 학습과 추론이 이 파일만 사용한다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
KEY = ["station", "layer"]
WINDOWS = (7, 37, 145, 433, 1009)  # 약 1h, 6h, 1d, 3d, 7d


def _smooth(x, size):
    if len(x) == 1:
        return x.astype(np.float32)
    size = min(size, len(x) if len(x) % 2 else max(1, len(x) - 1))
    return uniform_filter1d(x.astype(float), size=max(1, size), mode="nearest").astype(np.float32)


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


def make_features(raw, reference=None):
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
        z = ref_key.assign(temp=d.temp.to_numpy())
        med = z.groupby(["station", "layer", "month"]).temp.median()
        zz = z.join(med.rename("med"), on=["station", "layer", "month"])
        mad = zz.assign(dev=(zz.temp - zz.med).abs()).groupby(["station", "layer", "month"]).dev.median()
        reference = {"median": {"|".join(map(str, k)): float(v) for k, v in med.items()},
                     "mad": {"|".join(map(str, k)): float(v) for k, v in mad.items()}}
    keys = ref_key.astype(str).agg("|".join, axis=1)
    med = keys.map(reference["median"]).fillna(d.temp.median()).to_numpy(float)
    mad = keys.map(reference["mad"]).fillna(0.5).clip(lower=0.01).to_numpy(float)
    f["clim_resid"] = ((d.temp.to_numpy(float) - med) / mad).astype(np.float32)

    # 규칙 후보는 학습 라벨을 보지 않는 도메인 점수다.
    spike = f.abs_spike / (f.std_145 + 0.03)
    noise = np.maximum(f.noise_ratio_7_145 - 1.5, 0)
    flat = np.maximum(1.0 - f.flat_ratio * 5, 0) * (f.std_145 > 0.005)
    persistent = np.maximum(f.offset_score - 1.5, 0) + np.maximum(f.drift_score - 1.0, 0)
    rule = np.maximum.reduce([spike.to_numpy() / 8, noise.to_numpy() / 5,
                              flat.to_numpy(), persistent.to_numpy() / 6])
    f["rule_score"] = np.clip(rule, 0, 1).astype(np.float32)
    return f.replace([np.inf, -np.inf], np.nan).fillna(-999).astype(np.float32), reference


def time_folds(raw, n_folds=4):
    """station-layer별 순차 block. 랜덤 행 분할을 만들지 않는다."""
    d = raw.assign(_time=pd.to_datetime(raw.time, utc=True)).sort_values(["station", "layer", "_time"])
    fold = np.empty(len(d), np.int8)
    for _, g in d.groupby(KEY, sort=False):
        fold[g.index] = np.minimum(n_folds - 1, np.arange(len(g)) * n_folds // len(g))
    return fold


def train_mask_with_purge(raw, folds, held_out, hours=87):
    """검증 block 경계 양쪽에서 87시간을 학습에서 제외한다."""
    tm = pd.to_datetime(raw.time, utc=True)
    keep = folds != held_out
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx))
        val = idx[folds[idx] == held_out]
        if not len(val):
            continue
        lo, hi = tm.iloc[val].min(), tm.iloc[val].max()
        near = ((tm.iloc[idx] >= lo - pd.Timedelta(hours=hours)) &
                (tm.iloc[idx] <= hi + pd.Timedelta(hours=hours)))
        keep[idx[near.to_numpy()]] = False
    return keep

