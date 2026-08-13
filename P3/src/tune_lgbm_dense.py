"""T13: retune the dense LGBM judge and blend it on folds 1-3 only."""
import itertools
import json
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import cases as C
import metric as M
from features import extract
from train import expand, target_delta
from train_t2 import forward_train_mask

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
N_JOBS = 2
N_LEADS = len(C.LEADS)
EXP = "p3_t13"
P3 = pathlib.Path(__file__).resolve().parent.parent
T8_CACHE = P3 / "cache" / "p3_t8"
CACHE = P3 / "cache" / EXP
PREDS = P3 / "preds"
REFERENCE_FOLD4 = 0.614009
REFERENCE_T1F_FOLD4 = 0.611959

CANDIDATES = ("A_base", "B_leaves15", "C_leaves63", "D_mcs60", "E_slow", "F_bag")
BASE_PARAMS = {
    "objective": "regression_l2",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "max_depth": -1,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_jobs": N_JOBS,
    "verbosity": -1,
}
VARIANTS = {
    "A_base": {},
    "B_leaves15": {"num_leaves": 15},
    "C_leaves63": {"num_leaves": 63},
    "D_mcs60": {"min_child_samples": 60},
    "E_slow": {"learning_rate": 0.015, "n_estimators": 1000},
}


def weighted_rmse(y, pred, weights):
    error = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.sqrt(np.sum(weights[:, None] * error ** 2) /
                         (weights.sum() * error.shape[1])))


def unweighted_rmse(cases, y, pred):
    true_long = M.wide_to_long(cases)
    pred_long = M.wide_to_long(
        cases, {f"y{lead}": pred[:, j] for j, lead in enumerate(C.LEADS)})
    checked = M.evaluate(
        pred_long, true_long, by=["lead", "station", "hs0bin"],
        hs0=cases[["case_id", "station", "hs0"]])
    return float(checked["total"])


def make_model(variant, seed=SEED):
    params = dict(BASE_PARAMS)
    params["random_state"] = seed
    if variant == "F_bag":
        params.pop("subsample")
        params.pop("colsample_bytree")
        params.update(feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1)
    else:
        params.update(VARIANTS[variant])
    return lgb.LGBMRegressor(**params)


def predict(model, x, cases):
    delta = model.predict(expand(x, cases)).reshape(len(cases), N_LEADS)
    return (cases.hs0.to_numpy(float)[:, None] + delta).clip(0, 30)


def model_iteration(model):
    value = getattr(model, "best_iteration_", None)
    return 0 if value is None else int(value)


def load_inputs():
    features_path = T8_CACHE / "features.npy"
    ids_path = T8_CACHE / "feature_ids.npy"
    dense_path = T8_CACHE / "dense_index.csv"
    weights_path = P3 / "cache" / "p3_t9" / "dense_weights.csv"
    required = (features_path, ids_path, dense_path, weights_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required T8/T9 asset: " + ", ".join(missing))

    cases = pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)
    features = np.load(features_path, allow_pickle=False)
    feature_ids = np.load(ids_path, allow_pickle=True)
    if features.shape[0] != len(cases) or features.shape[1] != 361:
        raise RuntimeError(f"feature cache shape mismatch: {features.shape}")
    if not np.array_equal(feature_ids, cases.case_id.to_numpy()):
        raise RuntimeError("feature cache IDs do not match cases.parquet")

    dense_index = pd.read_csv(dense_path)
    if dense_index.columns.tolist() != ["case_id", "station", "fold", "hs0"]:
        raise RuntimeError("dense index schema mismatch")
    dense = cases.set_index("case_id").loc[dense_index.case_id].reset_index()
    if not np.array_equal(dense.case_id.to_numpy(), dense_index.case_id.to_numpy()):
        raise RuntimeError("dense case order mismatch")
    if not np.array_equal(dense.station.to_numpy(), dense_index.station.to_numpy()):
        raise RuntimeError("dense station order mismatch")
    if not np.allclose(dense.hs0.to_numpy(float), dense_index.hs0.to_numpy(float), atol=1e-12, rtol=0):
        raise RuntimeError("dense hs0 order mismatch")
    if set(dense_index.fold) != {1, 2, 3, 4}:
        raise RuntimeError("dense index must contain folds 1..4")

    weights = pd.read_csv(weights_path)
    if weights.columns.tolist() != ["case_id", "stratum", "w"]:
        raise RuntimeError("dense weight schema mismatch")
    if not np.array_equal(weights.case_id.to_numpy(), dense.case_id.to_numpy()):
        raise RuntimeError("dense weight order mismatch")
    w = weights.w.to_numpy(float)
    if not np.isfinite(w).all() or not np.isclose(w.mean(), 1.0, atol=1e-12):
        raise RuntimeError("invalid dense weights")

    saved = {}
    for name in ("persistence", "damped", "lgbm_t1"):
        path = PREDS / f"oof_{name}_p3_t8.npy"
        saved[name] = np.load(path, allow_pickle=False)
        if saved[name].shape != (len(dense), N_LEADS) or not np.isfinite(saved[name]).all():
            raise RuntimeError(f"invalid saved OOF: {path}")

    y = np.column_stack([dense[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    return cases, features, dense, dense_index.fold.to_numpy(int), y, w, saved


def load_test():
    rows, values = [], []
    for cid, station, context in C.iter_test_contexts():
        if context.shape != (289, 10):
            raise RuntimeError(f"test context shape mismatch: {cid} {context.shape}")
        rows.append((cid, station, float(context[-1, 0])))
        values.append(extract(context))
    test_cases = pd.DataFrame(rows, columns=["case_id", "station", "hs0"])
    test_x = np.stack(values).astype(np.float32)
    index = pd.read_csv(C.DATA / "test_index.csv")
    if not np.array_equal(index.case_id.drop_duplicates().to_numpy(), test_cases.case_id.to_numpy()):
        raise RuntimeError("test context/index case order mismatch")
    if test_x.shape != (len(test_cases), 361):
        raise RuntimeError(f"test feature shape mismatch: {test_x.shape}")
    return test_cases, test_x


def fit_fold_models(variant, x, cases, val, train_mask, val_x):
    train = cases.loc[train_mask].reset_index(drop=True)
    train_x = x[train_mask]
    models = [make_model(variant, seed) for seed in (42, 43, 44)] if variant == "F_bag" else [make_model(variant)]
    predictions = []
    iterations = []
    for model in models:
        model.fit(expand(train_x, train), target_delta(train))
        predictions.append(predict(model, val_x, val))
        iterations.append(model_iteration(model))
    return np.mean(predictions, axis=0), iterations


def fit_full(variant, x, cases, test_x, test_cases):
    models = [make_model(variant, seed) for seed in (42, 43, 44)] if variant == "F_bag" else [make_model(variant)]
    predictions = []
    iterations = []
    for model in models:
        model.fit(expand(x, cases), target_delta(cases))
        predictions.append(predict(model, test_x, test_cases))
        iterations.append(model_iteration(model))
    return np.mean(predictions, axis=0), iterations


def simplex(step=0.05):
    units = round(1 / step)
    return np.array([
        np.asarray(parts, float) / units
        for parts in itertools.product(range(units + 1), repeat=3)
        if sum(parts) == units
    ])


def main():
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    cases, x, dense, fold, y, weights, saved = load_inputs()
    dense_positions = pd.Series(np.arange(len(cases)), index=cases.case_id).loc[dense.case_id].to_numpy()
    dense_x = x[dense_positions]
    oof = {name: np.full_like(y, np.nan) for name in CANDIDATES}
    iterations = {name: {} for name in CANDIDATES}
    scores = {name: {} for name in CANDIDATES}

    for name in CANDIDATES:
        for value in range(1, 5):
            mask = fold == value
            val = dense.loc[mask].reset_index(drop=True)
            train_mask = forward_train_mask(cases, val)
            pred, iters = fit_fold_models(name, x, cases, val, train_mask, dense_x[mask])
            oof[name][mask] = pred
            iterations[name][f"fold{value}"] = iters if name == "F_bag" else iters[0]
            scores[name][f"fold{value}"] = weighted_rmse(y[mask], pred, weights[mask])
            print(f"{name} fold{value} weighted={scores[name][f'fold{value}']:.6f}", flush=True)
        scores[name]["folds1_3_mean"] = float(np.mean([scores[name][f"fold{f}"] for f in (1, 2, 3)]))
        scores[name]["fold4_unweighted"] = unweighted_rmse(
            dense.loc[fold == 4].reset_index(drop=True), y[fold == 4], oof[name][fold == 4])
        np.save(PREDS / f"oof_{name}_{EXP}.npy", oof[name])

    test_cases, test_x = load_test()
    full_test = {}
    full_iterations = {}
    for name in CANDIDATES:
        full_test[name], full_iterations[name] = fit_full(name, x, cases, test_x, test_cases)
        np.save(PREDS / f"test_{name}_{EXP}.npy", full_test[name])

    ess = {f"fold{value}": float(weights[fold == value].sum() ** 2 /
                                  np.square(weights[fold == value]).sum())
           for value in range(1, 5)}
    best_iterations_ok = all(
        value != 1 and value != (1000 if name == "E_slow" else 500)
        for name in CANDIDATES
        for fold_values in iterations[name].values()
        for value in (fold_values if isinstance(fold_values, list) else [fold_values])
    )
    score_signatures = {tuple(round(scores[name][f"fold{f}"], 6) for f in range(1, 5))
                        for name in CANDIDATES}
    control_delta = scores["A_base"]["fold4"] - REFERENCE_FOLD4
    control_pass = abs(control_delta) <= 0.002
    ess_reference = {"fold1": 571.931657, "fold2": 948.602515,
                     "fold3": 1124.684355, "fold4": 649.308480}
    ess_pass = all(np.isclose(ess[key], value, atol=1e-6, rtol=0)
                   for key, value in ess_reference.items())
    task0_pass = control_pass and best_iterations_ok and len(score_signatures) == 6 and ess_pass

    report = {
        "experiment": "T13_dense_lgbm_retune",
        "candidate_order": list(CANDIDATES),
        "dense_cases": int(len(dense)),
        "fold_counts": {f"fold{f}": int((fold == f).sum()) for f in range(1, 5)},
        "ess": ess,
        "ess_reference": ess_reference,
        "ess_pass": ess_pass,
        "reference_lgbm_t1_fold4": REFERENCE_FOLD4,
        "control": {
            "A_base_fold4_reweighted": scores["A_base"]["fold4"],
            "delta": control_delta,
            "passed": control_pass,
        },
        "best_iteration": iterations,
        "best_iteration_pass": best_iterations_ok,
        "candidate_score_signatures_distinct": len(score_signatures) == 6,
        "scores": scores,
        "task0_pass": task0_pass,
        "outputs": ([f"P3/preds/oof_{name}_{EXP}.npy" for name in CANDIDATES] +
                    [f"P3/preds/test_{name}_{EXP}.npy" for name in CANDIDATES]),
    }

    if not task0_pass:
        report["task1"] = {"created": False, "reason": "task0 acceptance failed"}
        (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    means = {name: scores[name]["folds1_3_mean"] for name in CANDIDATES}
    chosen = min(CANDIDATES, key=lambda name: (means[name], name))
    blend_names = ("persistence", "damped", chosen)
    blend_oof = np.stack([saved["persistence"], saved["damped"], oof[chosen]])
    selection = np.isin(fold, [1, 2, 3])
    grid = simplex(0.05)
    grid_scores = np.array([weighted_rmse(y[selection], np.einsum("i,inj->nj", weight, blend_oof[:, selection, :]), weights[selection])
                            for weight in grid])
    weight = grid[int(np.argmin(grid_scores))]
    final_oof = np.einsum("i,inj->nj", weight, blend_oof)
    test_saved = np.stack([
        np.repeat(test_cases.hs0.to_numpy(float)[:, None], N_LEADS, axis=1),
        np.load(PREDS / "test_damped_p3_t1.npy", allow_pickle=False),
    ])
    if test_saved.shape != (2, len(test_cases), N_LEADS):
        raise RuntimeError("saved test baseline shape mismatch")
    final_test = np.einsum("i,inj->nj", weight,
                           np.concatenate([test_saved, full_test[chosen][None]], axis=0)).clip(0, 30)
    np.save(PREDS / f"oof_final_{EXP}.npy", final_oof)
    np.save(PREDS / f"test_final_{EXP}.npy", final_test)
    fold4 = fold == 4
    fold4_score = weighted_rmse(y[fold4], final_oof[fold4], weights[fold4])
    report["task1"] = {
        "created": True,
        "selected_lgbm": chosen,
        "folds1_3_candidate_mean": means,
        "blend_names": list(blend_names),
        "selection_folds": [1, 2, 3],
        "selection_fold4_used": False,
        "weights": {name: float(value) for name, value in zip(blend_names, weight)},
        "weight_sum": float(weight.sum()),
        "folds1_3_reweighted": float(grid_scores.min()),
        "fold4_reweighted": fold4_score,
        "fold4_delta_vs_t1f": fold4_score - REFERENCE_T1F_FOLD4,
        "full_fit_best_iteration": full_iterations,
        "outputs": [
            f"P3/preds/oof_final_{EXP}.npy",
            f"P3/preds/test_final_{EXP}.npy",
        ],
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
