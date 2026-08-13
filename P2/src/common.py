"""P2 데이터·피처의 단일 생산 경로."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "P2_data"
TARGET_LAYERS = [2, 3, 4]
PUBLIC_LAYERS = [1, 5, 6, 7, 8]
KEY = ["station", "layer", "time"]
SEED = 42


def load_obs():
    return pd.read_csv(DATA / "observations.csv", parse_dates=["time"])


def split_rows(obs):
    target = obs[obs.layer.isin(TARGET_LAYERS) & obs.temp.notna()].copy()
    hold = target.time.between(pd.Timestamp("2024-09-01", tz="Asia/Seoul"),
                               pd.Timestamp("2024-10-31 23:50", tz="Asia/Seoul"))
    public_n = (obs[obs.layer.isin(PUBLIC_LAYERS) & obs.temp.notna()]
                .groupby("time").size())
    valid = target.time.map(public_n).fillna(0).ge(2)
    val = target[hold & valid].copy()
    train = target[~hold].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)


def test_rows(obs):
    rows = pd.read_csv(DATA / "test_index.csv", parse_dates=["time"])
    depth = obs.loc[obs.layer.isin(TARGET_LAYERS), KEY + ["depth"]]
    return rows.merge(depth, on=KEY, how="left", validate="one_to_one")


def _wide(obs, value):
    return obs.pivot(index="time", columns="layer", values=value).sort_index()


def _fill_by_year(w):
    return w.groupby(w.index.year, group_keys=False).apply(
        lambda x: x.interpolate(method="linear", limit_direction="both"),
        include_groups=False,
    )


def _baseline(rows, temp, depth, nominal, actual, use_target_depth=True):
    raw_t = temp
    fill_t = _fill_by_year(temp)
    fill_z = _fill_by_year(depth)
    out = np.empty(len(rows), dtype=np.float64)
    for i, r in enumerate(rows.itertuples(index=False)):
        ti = r.time
        zt = getattr(r, "depth", np.nan) if actual and use_target_depth else r.nominal_depth
        if not np.isfinite(zt):
            zt = r.nominal_depth
        pts = []
        for layer in PUBLIC_LAYERS:
            if layer not in raw_t.columns:
                continue
            tv = raw_t.at[ti, layer]
            zv = fill_z.at[ti, layer] if actual else nominal.at[ti, layer]
            if np.isfinite(tv) and np.isfinite(zv):
                pts.append((zv, tv))
        pts.sort()
        lo, hi = [p for p in pts if p[0] <= zt], [p for p in pts if p[0] >= zt]
        # 센서 흔들림으로 대상층 실제 depth가 표층 센서보다 얕아지는 경우 공칭 target으로 폴백한다.
        if actual and (not lo or not hi):
            zt = r.nominal_depth
            lo, hi = [p for p in pts if p[0] <= zt], [p for p in pts if p[0] >= zt]
        if not lo or not hi:
            pts = sorted((fill_z.at[ti, layer] if actual else nominal.at[ti, layer], fill_t.at[ti, layer])
                         for layer in PUBLIC_LAYERS if layer in fill_t.columns
                         and np.isfinite(fill_t.at[ti, layer])
                         and np.isfinite(fill_z.at[ti, layer] if actual else nominal.at[ti, layer]))
            lo, hi = [p for p in pts if p[0] <= zt], [p for p in pts if p[0] >= zt]
        if actual and (not lo or not hi):
            zt = r.nominal_depth
            pts = sorted((nominal.at[ti, layer], fill_t.at[ti, layer]) for layer in PUBLIC_LAYERS
                         if layer in fill_t.columns and np.isfinite(fill_t.at[ti, layer])
                         and np.isfinite(nominal.at[ti, layer]))
            lo, hi = [p for p in pts if p[0] <= zt], [p for p in pts if p[0] >= zt]
        if not lo or not hi or hi[0][0] == lo[-1][0]:
            raise ValueError(f"보간 앵커 부족: {ti}, layer {r.layer}")
        a, b = lo[-1], hi[0]
        out[i] = a[1] + (b[1] - a[1]) * (zt - a[0]) / (b[0] - a[0])
    return out


def elapsed_hours(times):
    """Datetime storage unit에 무관한 Unix epoch 이후 시간."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (times - epoch).dt.total_seconds().to_numpy() / 3600.0


def build_features(obs, rows, baseline_override=None, tide_correct=True, use_target_depth=True):
    rows = rows.reset_index(drop=True).copy()
    temp, psal = _wide(obs, "temp"), _wide(obs, "psal")
    depth, nominal = _wide(obs, "depth"), _wide(obs, "nominal_depth")
    times = rows.time
    x = pd.DataFrame({
        "layer": rows.layer.astype(float),
        "target_nominal": rows.nominal_depth.astype(float),
        "target_actual": (rows.get("depth", rows.nominal_depth).fillna(rows.nominal_depth)
                          if use_target_depth else rows.nominal_depth).astype(float),
    })
    x["base_nom"] = _baseline(rows, temp, depth, nominal, actual=False)
    if baseline_override is not None:
        b = rows[KEY].merge(baseline_override[KEY + ["temp"]], on=KEY, how="left", validate="one_to_one")
        if b.temp.isna().any():
            raise ValueError("baseline override 키 누락")
        x["base_nom"] = b.temp.to_numpy()
    x["base_actual"] = _baseline(rows, temp, depth, nominal, actual=True,
                                 use_target_depth=use_target_depth)
    x["base_depth_delta"] = x.base_actual - x.base_nom

    for layer in PUBLIC_LAYERS:
        for name, w in (("temp", temp), ("psal", psal), ("depth", depth)):
            x[f"{name}_{layer}"] = w[layer].reindex(times).to_numpy() if layer in w else np.nan
        if layer in temp:
            s = temp[layer]
            x[f"temp_{layer}_mean6h"] = s.rolling(37, center=True, min_periods=3).mean().reindex(times).to_numpy()
            x[f"temp_{layer}_std6h"] = s.rolling(37, center=True, min_periods=3).std().reindex(times).to_numpy()
            x[f"temp_{layer}_lag6h"] = s.shift(36).reindex(times).to_numpy()
            x[f"temp_{layer}_lead6h"] = s.shift(-36).reindex(times).to_numpy()
            x[f"temp_{layer}_trend12h"] = (s.shift(-36) - s.shift(36)).reindex(times).to_numpy()
        else:
            x[f"temp_{layer}_mean6h"] = np.nan
            x[f"temp_{layer}_std6h"] = np.nan
            x[f"temp_{layer}_lag6h"] = np.nan
            x[f"temp_{layer}_lead6h"] = np.nan
            x[f"temp_{layer}_trend12h"] = np.nan

    for lower in (5, 6, 7, 8):
        x[f"strat_1_{lower}"] = x.temp_1 - x[f"temp_{lower}"]
    x["transition"] = x.temp_1_trend12h - x.temp_6_trend12h
    # E001은 pandas datetime64[us]를 ns로 가정해 1000배 작은 시간을 만들었다.
    hours = elapsed_hours(times) if tide_correct else times.astype("int64").to_numpy() / 3.6e12
    for name, period in (("m2", 12.4206), ("s2", 12.0)):
        x[f"{name}_sin"] = np.sin(2 * np.pi * hours / period)
        x[f"{name}_cos"] = np.cos(2 * np.pi * hours / period)
    doy = times.dt.dayofyear.to_numpy() + times.dt.hour.to_numpy() / 24
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return x


BASIC = [
    "layer", "target_nominal", "target_actual", "base_nom", "base_actual", "base_depth_delta",
    *[f"temp_{x}" for x in PUBLIC_LAYERS], *[f"depth_{x}" for x in PUBLIC_LAYERS],
    "strat_1_5", "strat_1_6", "strat_1_7", "strat_1_8",
]
FULL = BASIC + [
    *[f"psal_{x}" for x in PUBLIC_LAYERS],
    *[f"temp_{x}_mean6h" for x in PUBLIC_LAYERS],
    *[f"temp_{x}_trend12h" for x in PUBLIC_LAYERS],
    "transition", "m2_sin", "m2_cos", "s2_sin", "s2_cos", "doy_sin", "doy_cos",
]
ENHANCED = FULL + [
    *[f"temp_{x}_std6h" for x in PUBLIC_LAYERS],
    *[f"temp_{x}_lag6h" for x in PUBLIC_LAYERS],
    *[f"temp_{x}_lead6h" for x in PUBLIC_LAYERS],
]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


if __name__ == "__main__":
    sys = __import__("sys")
    sys.stdout.reconfigure(encoding="utf-8")
    t = pd.Series(pd.to_datetime(["2025-01-01T00:00:00+09:00", "2025-01-01T00:10:00+09:00"]))
    assert abs(np.diff(elapsed_hours(t))[0] - 1 / 6) < 1e-9
    print("common selfcheck OK: 10분=1/6시간")
