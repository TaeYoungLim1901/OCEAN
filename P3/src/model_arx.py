"""Small missing-value tolerant ARX/Kalman member for T21."""
import numpy as np


SEED = 42
ORDER = 6
LEAD_STEPS = np.array([18, 36, 54, 72, 108, 144], dtype=int)


def _contexts(contexts):
    x = np.asarray(contexts, dtype=np.float64)
    single = x.ndim == 2
    if single:
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (289, 10):
        raise ValueError(f"contexts must have shape (n,289,10), got {x.shape}")
    return x, single


def _ridge(x, y, ridge=1e-3):
    if not len(x):
        return np.zeros(x.shape[1], dtype=float)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += ridge
    return np.linalg.solve(gram, x.T @ y)


def _exog(row, use_exog):
    if not use_exog:
        return np.empty(0, dtype=float)
    return np.nan_to_num(np.asarray(row, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _base_predict(model, context):
    ctx = np.asarray(context, dtype=float)
    p = int(model["order"])
    coef = np.asarray(model["coef"], dtype=float)
    use_exog = bool(model["use_exog"])
    F = np.zeros((p, p), dtype=float)
    F[0] = coef[1:1 + p]
    if p > 1:
        F[1:, :-1] = np.eye(p - 1)
    B = coef[1 + p:] if use_exog else np.empty(0, dtype=float)

    hs = ctx[:, 0]
    state = np.zeros(p, dtype=float)
    covariance = np.eye(p, dtype=float) * 4.0
    observation_noise = float(model["observation_noise"])
    process_noise = float(model["process_noise"])
    initialized = False
    last_u = np.zeros(2, dtype=float)
    have_u = np.zeros(2, dtype=bool)

    for i, value in enumerate(hs):
        if not initialized:
            if np.isfinite(value):
                state[:] = value
                initialized = True
            if use_exog:
                for j, raw in enumerate(ctx[i, (4, 5)]):
                    if np.isfinite(raw):
                        last_u[j] = raw
                        have_u[j] = True
            continue

        u = _exog(last_u, use_exog)
        state = F @ state + (B @ u if use_exog else 0.0) * np.eye(1, p, 0).ravel()
        covariance = F @ covariance @ F.T + np.eye(p) * process_noise
        covariance = (covariance + covariance.T) * 0.5

        if use_exog:
            for j, raw in enumerate(ctx[i, (4, 5)]):
                if np.isfinite(raw):
                    last_u[j] = raw
                    have_u[j] = True

        if np.isfinite(value):
            innovation = float(value - state[0])
            variance = max(float(covariance[0, 0] + observation_noise), 1e-8)
            gain = covariance[:, 0] / variance
            state += gain * innovation
            covariance -= np.outer(gain, covariance[0])
            covariance = (covariance + covariance.T) * 0.5
            covariance[np.diag_indices(p)] = np.maximum(np.diag(covariance), 1e-8)

    if not initialized:
        finite = hs[np.isfinite(hs)]
        value = float(finite[-1]) if len(finite) else 0.0
        state[:] = value

    predictions = []
    step = 0
    for target_step in LEAD_STEPS:
        while step < target_step:
            u = _exog(last_u, use_exog)
            state = F @ state + (B @ u if use_exog else 0.0) * np.eye(1, p, 0).ravel()
            step += 1
        predictions.append(state[0])
    return np.asarray(predictions, dtype=float)


def _calibrate(model, contexts, targets):
    y = np.asarray(targets, dtype=float)
    if y.shape != (len(contexts), 6):
        raise ValueError(f"targets must have shape ({len(contexts)},6), got {y.shape}")
    raw = np.stack([_base_predict(model, ctx) for ctx in contexts])
    calibration = np.tile(np.array([0.0, 1.0]), (6, 1))
    for j in range(6):
        valid = np.isfinite(raw[:, j]) & np.isfinite(y[:, j])
        if valid.sum() < 20 or np.var(raw[valid, j]) < 1e-8:
            continue
        design = np.column_stack((np.ones(valid.sum()), raw[valid, j]))
        beta = _ridge(design, y[valid, j], ridge=1e-2)
        calibration[j] = [float(beta[0]), float(np.clip(beta[1], 0.0, 1.5))]
    model["calibration"] = calibration


def fit(contexts, targets=None, order=ORDER):
    """Fit one shared ARX transition from raw context lags."""
    x, _ = _contexts(contexts)
    p = int(order)
    if p < 1 or p > 24:
        raise ValueError("order must be in [1,24]")
    use_exog = bool(np.isfinite(x[:, :, (4, 5)]).any())
    rows, values = [], []
    for ctx in x:
        hs = ctx[:, 0]
        for i in range(p, len(hs)):
            history = hs[i - p:i][::-1]
            if not np.isfinite(hs[i]) or not np.isfinite(history).all():
                continue
            row = np.r_[1.0, history]
            if use_exog:
                row = np.r_[row, _exog(ctx[i, (4, 5)], True)]
            rows.append(row)
            values.append(hs[i])
    n_features = 1 + p + (2 if use_exog else 0)
    design = np.asarray(rows, dtype=float).reshape(-1, n_features)
    observed = np.asarray(values, dtype=float)
    if len(observed) < n_features:
        coef = np.zeros(n_features, dtype=float)
        coef[1] = 1.0
    else:
        coef = _ridge(design, observed)
    residual = observed - design @ coef if len(observed) else np.array([1.0])
    model = {
        "order": p,
        "coef": coef,
        "use_exog": use_exog,
        "observation_noise": max(float(np.nanvar(residual)), 1e-4),
        "process_noise": max(float(np.nanvar(residual)) * 0.05, 1e-5),
        "n_rows": int(len(observed)),
        "calibration": np.tile(np.array([0.0, 1.0]), (6, 1)),
    }
    if targets is not None:
        _calibrate(model, x, targets)
    return model


def predict(model, contexts):
    x, single = _contexts(contexts)
    calibration = np.asarray(model.get("calibration", np.tile([0.0, 1.0], (6, 1))))
    raw = np.stack([_base_predict(model, ctx) for ctx in x])
    pred = calibration[:, 0] + calibration[:, 1] * raw
    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=30.0, neginf=0.0), 0.0, 30.0)
    return pred[0] if single else pred


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    context = rng.normal(size=(289, 10)).astype(np.float32)
    context[:, 0] = np.linspace(1.0, 2.0, 289)
    context[10:20, 0] = np.nan
    model = fit(context[None], np.full((1, 6), 1.5))
    assert np.isfinite(predict(model, context)).all()
    print("OK")
