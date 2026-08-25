"""Wind-sea/swell energy decomposition member for T21."""
import numpy as np


SEED = 42
LEAD_MIN = np.array([180, 360, 540, 720, 1080, 1440], dtype=float)
TAU_WS_GRID = np.array([30, 60, 120, 240, 360], dtype=float)
TAU_SW_GRID = np.array([180, 360, 720, 1440, 2880], dtype=float)


def _contexts(contexts):
    x = np.asarray(contexts, dtype=np.float64)
    single = x.ndim == 2
    if single:
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (289, 10):
        raise ValueError(f"contexts must have shape (n,289,10), got {x.shape}")
    return x, single


def _last(ctx, column):
    values = ctx[:, column]
    valid = np.flatnonzero(np.isfinite(values))
    return float(values[valid[-1]]) if len(valid) else np.nan


def _design(ctx):
    wspd = _last(ctx, 4)
    tp = _last(ctx, 1)
    wvdir = _last(ctx, 3)
    wdir = _last(ctx, 6)
    speed = max(wspd, 0.0) if np.isfinite(wspd) else 0.0
    speed_log = np.log1p(speed)
    period_log = np.log1p(max(tp, 0.0)) if np.isfinite(tp) else 0.0
    if np.isfinite(wvdir) and np.isfinite(wdir):
        alignment = 0.5 * (1.0 + np.cos(np.deg2rad(wvdir - wdir)))
    else:
        alignment = 0.5
    return np.array([1.0, speed_log, period_log, alignment, speed_log * alignment], dtype=float)


def _pseudo_ratio(design):
    speed = np.expm1(np.maximum(design[:, 1], 0.0))
    period = np.expm1(np.maximum(design[:, 2], 0.0))
    alignment = np.clip(design[:, 3], 0.0, 1.0)
    wind_strength = speed / (speed + 7.0)
    period_penalty = np.exp(-np.maximum(period - 8.0, 0.0) / 12.0)
    return np.clip(0.05 + 0.85 * wind_strength * (0.5 + 0.5 * alignment) * period_penalty,
                   0.0, 1.0)


def _ridge(x, y, ridge=1e-2):
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += ridge
    return np.linalg.solve(gram, x.T @ y)


def _pred_from_parts(hs0, ratio, wspd, alignment, tau_ws, tau_sw, wind_scale):
    hs0 = np.asarray(hs0, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    wspd = np.nan_to_num(np.asarray(wspd, dtype=float), nan=0.0)
    alignment = np.nan_to_num(np.asarray(alignment, dtype=float), nan=0.5)
    wind_target = np.square(np.maximum(wind_scale * wspd * (0.5 + 0.5 * alignment), 0.0))
    ws0 = np.square(np.maximum(hs0, 0.0)) * ratio
    sw0 = np.square(np.maximum(hs0, 0.0)) * (1.0 - ratio)
    ws = wind_target[:, None] + (ws0[:, None] - wind_target[:, None]) * np.exp(-LEAD_MIN[None, :] / tau_ws)
    sw = sw0[:, None] * np.exp(-LEAD_MIN[None, :] / tau_sw)
    return np.sqrt(np.maximum(ws + sw, 0.0))


def _features(contexts):
    design = np.stack([_design(ctx) for ctx in contexts])
    hs0 = np.array([_last(ctx, 0) for ctx in contexts], dtype=float)
    wspd = np.array([_last(ctx, 4) for ctx in contexts], dtype=float)
    alignment = np.array([
        0.5 * (1.0 + np.cos(np.deg2rad(_last(ctx, 3) - _last(ctx, 6))))
        if np.isfinite(_last(ctx, 3)) and np.isfinite(_last(ctx, 6)) else 0.5
        for ctx in contexts
    ], dtype=float)
    fallback = ~np.isfinite(np.asarray(contexts)[:, :, 4]).any(axis=1)
    return design, hs0, wspd, alignment, fallback


def fit(contexts, targets=None):
    """Fit ratio coefficients and two fixed-grid component time constants."""
    x, _ = _contexts(contexts)
    design, hs0, wspd, alignment, fallback = _features(x)
    ratio_coef = _ridge(design, _pseudo_ratio(design))
    ratio = np.clip(design @ ratio_coef, 0.0, 1.0)
    finite_wind = np.isfinite(wspd)
    if finite_wind.any():
        fill_wind = float(np.nanmedian(wspd[finite_wind]))
    else:
        fill_wind = 0.0
    wind_for_fit = np.where(finite_wind, wspd, fill_wind)
    targets_array = None if targets is None else np.asarray(targets, dtype=float)
    tau_ws, tau_sw, wind_scale = 120.0, 720.0, 0.10
    if targets_array is not None:
        if targets_array.shape != (len(x), 6):
            raise ValueError(f"targets must have shape ({len(x)},6), got {targets_array.shape}")
        valid = np.isfinite(targets_array).all(axis=1) & np.isfinite(hs0)
        valid &= ~fallback
        if valid.any():
            best = (float("inf"), tau_ws, tau_sw, wind_scale)
            for scale in (0.05, 0.08, 0.10, 0.14, 0.18):
                for ws_tau in TAU_WS_GRID:
                    for sw_tau in TAU_SW_GRID:
                        prediction = _pred_from_parts(
                            hs0[valid], ratio[valid], wind_for_fit[valid], alignment[valid],
                            ws_tau, sw_tau, scale)
                        score = float(np.mean((prediction - targets_array[valid]) ** 2))
                        candidate = (score, ws_tau, sw_tau, scale)
                        if candidate < best:
                            best = candidate
            _, tau_ws, tau_sw, wind_scale = best
    return {
        "ratio_coef": ratio_coef,
        "tau_ws_min": float(tau_ws),
        "tau_sw_min": float(tau_sw),
        "wind_scale": float(wind_scale),
        "fallback_fit": int(fallback.sum()),
        "fit_cases": int(len(x)),
    }


def predict(model, contexts, return_info=False):
    x, single = _contexts(contexts)
    design, hs0, wspd, alignment, fallback = _features(x)
    ratio = np.clip(design @ np.asarray(model["ratio_coef"], dtype=float), 0.0, 1.0)
    wind_ok = np.isfinite(wspd)
    pred = np.empty((len(x), 6), dtype=float)
    active = ~fallback & np.isfinite(hs0)
    if active.any():
        pred[active] = _pred_from_parts(
            hs0[active], ratio[active], wspd[active], alignment[active],
            float(model["tau_ws_min"]), float(model["tau_sw_min"]), float(model["wind_scale"])
        )
    if (~active).any():
        safe_hs = np.nan_to_num(hs0[~active], nan=0.0, posinf=0.0, neginf=0.0)
        pred[~active] = safe_hs[:, None] * np.exp(-LEAD_MIN[None, :] /
                                                    float(model["tau_sw_min"]))
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=0.0), 0.0, 30.0)
    info = {"fallback": fallback, "fallback_ratio": float(fallback.mean()) if len(fallback) else 0.0,
            "wind_observed": int(wind_ok.sum())}
    if return_info:
        return (pred[0] if single else pred), info
    return pred[0] if single else pred


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    context = rng.normal(size=(289, 10)).astype(np.float32)
    context[:, 0] = 2.0
    model = fit(context[None], np.full((1, 6), 1.5))
    pred, info = predict(model, context, return_info=True)
    assert np.isfinite(pred).all() and info["fallback_ratio"] == 0.0
    print("OK")
