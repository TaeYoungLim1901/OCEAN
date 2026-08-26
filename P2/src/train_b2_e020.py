"""T21: train B2 residual models on the four season-aligned holds."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arm", choices=("A0", "A1", "A2"))
    group.add_argument("--all", action="store_true")
    return parser


if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
    _parser().parse_args()

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import depth_base_e019 as D
import gate_e004 as G
import season_e006 as S
from submission_tools import save_pool_pred
from train import estimator


WINDOWS = ("2024_sepoct",) + tuple(
    row[0] for row in S.WINDOWS if row[0] in {"S1", "S3", "S4"}
)
B2_FEATURES = [
    "b2", "b2_ts", "b2_td", "b2_z0", "b2_h", "b2_dtdz",
    "b2_fit_rmse", "b2_z0_minus_target_depth", "b2_fit_fallback",
]


def _profile_features(work, rows, b0):
    rows = rows.reset_index(drop=True)
    b0 = np.asarray(b0, dtype=float)
    _, b2, _ = D._profile_bases(work, rows, b0, use_depth=True, fit_tanh=True)
    out = pd.DataFrame(np.nan, index=rows.index, columns=B2_FEATURES)
    out["b2"] = b2
    out["b2_fit_fallback"] = 0.0
    public = work[work.layer.isin(C.PUBLIC_LAYERS) & work.temp.notna()]
    by_time = {stamp: group for stamp, group in public.groupby("time", sort=False)}
    depth = rows.depth.to_numpy(float)
    nominal = rows.nominal_depth.to_numpy(float)

    for stamp, indices in rows.groupby("time", sort=False).indices.items():
        points = D._points(by_time.get(stamp, public.iloc[0:0]), "depth")
        if points is None:
            out.loc[indices, "b2_fit_fallback"] = 1.0
            continue
        x, y = points
        z, _ = D._target_z(depth, nominal, indices, x[0], x[-1], True)
        if z is None:
            out.loc[indices, "b2_fit_fallback"] = 1.0
            continue
        try:
            slopes = np.abs(np.diff(y) / np.diff(x))
            z0 = float((x[np.argmax(slopes)] + x[np.argmax(slopes) + 1]) / 2.0)
            result = D.least_squares(
                lambda params: D._tanh(x, params) - y,
                [y[0], y[-1], z0, 3.0],
                bounds=([-np.inf, -np.inf, -np.inf, 0.5],
                        [np.inf, np.inf, np.inf, 30.0]),
                max_nfev=100,
            )
            params = np.asarray(result.x, dtype=float)
            fitted = np.asarray(D._tanh(x, params), dtype=float)
            values = np.asarray(D._tanh(z, params), dtype=float)
            valid = result.success and np.isfinite(params).all()
            valid = valid and np.isfinite(fitted).all() and np.isfinite(values).all()
            valid = valid and ((values >= -5) & (values <= 45)).all()
        except Exception:
            valid = False
        if not valid:
            out.loc[indices, "b2_fit_fallback"] = 1.0
            continue

        gradient = -(params[0] - params[1]) / (2.0 * params[3])
        gradient *= 1.0 - np.tanh((z - params[2]) / params[3]) ** 2
        values = np.column_stack([
            np.full(len(indices), params[0]),
            np.full(len(indices), params[1]),
            np.full(len(indices), params[2]),
            np.full(len(indices), params[3]),
            gradient,
            np.full(len(indices), np.sqrt(np.mean((fitted - y) ** 2))),
            params[2] - z,
            np.zeros(len(indices)),
        ])
        out.loc[indices, B2_FEATURES[1:]] = values

    if not np.isfinite(out["b2"].to_numpy()).all():
        raise ValueError("B2 비유한 피처")
    return out


def _holdout(obs, label, selected):
    if label in {"2024_sepoct", "S1"}:
        return G.holdout(obs, "2024_sepoct")
    return S._season_holdout(obs, selected[label])


def _fit_models(x, rows, columns, base_column):
    models = {}
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        model = estimator("lgb_full")
        model.fit(x.loc[use, columns], rows.loc[use, "temp"] - x.loc[use, base_column])
        models[layer] = model
    return models


def _predict(models, x, rows, columns, base_column):
    out = np.empty(len(rows), dtype=float)
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        out[use] = x.loc[use, base_column] + models[layer].predict(x.loc[use, columns])
    out = np.clip(out, -5.0, 45.0)
    if not np.isfinite(out).all():
        raise ValueError("비유한 예측")
    return out


def _run_arm(arm, obs, selected, started):
    use_b2 = arm in {"A1", "A2"}
    base_column = "b2" if arm == "A2" else "base_nom"
    columns = C.FULL + (B2_FEATURES if use_b2 else [])
    cache = C.ROOT / "cache" / "T21"
    cache.mkdir(parents=True, exist_ok=True)

    for label in WINDOWS:
        hold_started = time.perf_counter()
        work, train_rows, val = _holdout(obs, label, selected)
        x_train = C.build_features(work, train_rows, tide_correct=True, use_target_depth=True)
        x_val = C.build_features(work, val, tide_correct=True, use_target_depth=True)
        if use_b2:
            x_train = pd.concat([x_train, _profile_features(work, train_rows, x_train.base_nom)], axis=1)
            x_val = pd.concat([x_val, _profile_features(work, val, x_val.base_nom)], axis=1)
        models = _fit_models(x_train, train_rows, columns, base_column)
        pred = _predict(models, x_val, val, columns, base_column)
        path = C.ROOT / "preds" / f"oof_{arm}_{label}_E020.npy"
        save_pool_pred(path, pred)
        save_pool_pred(cache / path.name, pred)
        (cache / f"metrics_{arm}_{label}.json").write_text(
            json.dumps({"arm": arm, "window": label, "rows": len(val),
                        "seconds": time.perf_counter() - hold_started}, indent=2),
            encoding="utf-8",
        )
        print(f"{arm} | {label} | {time.perf_counter() - started:.1f}s", flush=True)

    all_rows = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].reset_index(drop=True)
    x_all = C.build_features(obs, all_rows, tide_correct=True, use_target_depth=True)
    if use_b2:
        x_all = pd.concat([x_all, _profile_features(obs, all_rows, x_all.base_nom)], axis=1)
    models = _fit_models(x_all, all_rows, columns, base_column)
    test_rows = C.test_rows(obs)
    x_test = C.build_features(obs, test_rows, tide_correct=True, use_target_depth=True)
    if use_b2:
        x_test = pd.concat([x_test, _profile_features(obs, test_rows, x_test.base_nom)], axis=1)
    test_pred = _predict(models, x_test, test_rows, columns, base_column)
    save_pool_pred(C.ROOT / "preds" / f"test_{arm}_E020.npy", test_pred)
    artifact = {"arm": arm, "models": models, "features": columns,
                "base_column": base_column, "seed": C.SEED}
    with open(C.ROOT / "models" / f"E020_{arm}.pkl", "wb") as handle:
        pickle.dump(artifact, handle)
    print(f"{arm} | final/test | {time.perf_counter() - started:.1f}s", flush=True)


def main():
    args = _parser().parse_args()
    if C.SEED != 42:
        raise ValueError(f"SEED 불일치: {C.SEED}")
    obs = C.load_obs()
    selected = {row["label"]: row for row in S._choose_windows(obs)[0]}
    arms = ("A0", "A1", "A2") if args.all else (args.arm,)
    started = time.perf_counter()
    for arm in arms:
        _run_arm(arm, obs, selected, started)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
