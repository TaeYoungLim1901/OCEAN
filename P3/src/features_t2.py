"""Small case-local additions for P3 T2."""
import numpy as np

from cases import COLS
from features import extract as extract_base, feature_names as base_names

WINDOW_H = (3, 6, 12, 24, 48)
IDX = {name: i for i, name in enumerate(COLS)}


def _finite_mean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _ema(x):
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan
    y = x[ok].astype(float)
    w = np.exp(np.linspace(-3.0, 0.0, len(y)))
    return float(np.dot(y, w) / w.sum())


def feature_names():
    extra = ("hs_ema", "wspd_ema", "gust_ema", "hs_energy", "wind_energy",
             "wind_wave_energy", "dir_align_cos", "hmax_hs_ratio")
    return base_names() + [f"{name}_{h}h" for h in WINDOW_H for name in extra]


def extract(ctx):
    """Use only one (289, 10) context; no clock or cross-case state exists here."""
    if ctx.shape != (289, 10):
        raise ValueError(f"context shape {ctx.shape} != (289, 10)")
    out = extract_base(ctx).tolist()
    for h in WINDOW_H:
        z = ctx[-(h * 6 + 1):].astype(float)
        hs, hmax = z[:, IDX["hs"]], z[:, IDX["hmax"]]
        wspd, gust = z[:, IDX["wspd"]], z[:, IDX["gust"]]
        wave, wind = z[:, IDX["wvdir"]], z[:, IDX["wdir"]]
        align = np.cos(np.deg2rad(wave - wind))
        ratio = np.divide(hmax, hs, out=np.full_like(hmax, np.nan),
                          where=np.isfinite(hmax) & np.isfinite(hs) & (np.abs(hs) > 1e-6))
        out.extend((_ema(hs), _ema(wspd), _ema(gust), _finite_mean(hs * hs),
                    _finite_mean(wspd * wspd), _finite_mean(hs * wspd * wspd),
                    _finite_mean(align), _finite_mean(ratio)))
    return np.asarray(out, np.float32)


if __name__ == "__main__":
    x = np.full((289, 10), np.nan, np.float32)
    x[-1] = np.arange(10)
    got = extract(x)
    assert got.shape == (len(feature_names()),)
    print(f"features_t2 selftest OK ({len(got)})")
