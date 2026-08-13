"""P3 case-local summary features."""
import numpy as np

from cases import COLS

WINDOW_H = (1, 3, 6, 12, 24, 48)
DIR_COLS = {"wvdir", "wdir"}


def _summary(x):
    ok = np.isfinite(x)
    if not ok.any():
        return [np.nan] * 5 + [1.0]
    y = x[ok].astype(float)
    t = np.flatnonzero(ok).astype(float)
    slope = 0.0 if len(y) < 2 else float(np.polyfit(t, y, 1)[0])
    return [float(y[-1]), float(y.mean()), float(y.std()),
            float(y[-1] - y[0]), slope, float(1 - ok.mean())]


def _direction(x):
    ok = np.isfinite(x)
    if not ok.any():
        return [np.nan] * 5 + [1.0]
    a = np.deg2rad(x[ok].astype(float))
    s, c = np.sin(a), np.cos(a)
    return [float(s[-1]), float(c[-1]), float(s.mean()), float(c.mean()),
            float(np.hypot(s.mean(), c.mean())), float(1 - ok.mean())]


def feature_names():
    stats = ("last", "mean", "std", "delta", "slope", "missing")
    circ = ("last_sin", "last_cos", "mean_sin", "mean_cos", "strength", "missing")
    return [f"{col}_{h}h_{s}" for h in WINDOW_H for col in COLS
            for s in (circ if col in DIR_COLS else stats)] + ["wind_wave_cos_last"]


def extract(ctx):
    """Return features using this case's (289, 10) context only."""
    if ctx.shape != (289, 10):
        raise ValueError(f"context shape {ctx.shape} != (289, 10)")
    out = []
    for h in WINDOW_H:
        z = ctx[-(h * 6 + 1):]
        for j, col in enumerate(COLS):
            out += _direction(z[:, j]) if col in DIR_COLS else _summary(z[:, j])
    wave, wind = ctx[-1, 3], ctx[-1, 6]
    out.append(float(np.cos(np.deg2rad(wave - wind))) if np.isfinite(wave + wind) else np.nan)
    return np.asarray(out, np.float32)


if __name__ == "__main__":
    x = np.full((289, 10), np.nan, np.float32)
    x[-1] = np.arange(10)
    assert extract(x).shape == (len(feature_names()),)
    print(f"features selftest OK ({len(feature_names())})")
