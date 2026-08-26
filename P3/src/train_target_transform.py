"""T26 target-transform dense folds."""
import argparse
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import cases as C
from features import extract
from submission_tools import save_pool_pred
from train_dense import (case_features, dense_cases, fit_lgbm_t1,
                         load_cases, load_reference_folds, predict_lgbm_t1)
from train_t2 import forward_train_mask
from tune_lgbm_dense import weighted_rmse
from train import expand

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
EPS = 0.1
REFERENCE_FOLD4 = 0.614009
VARIANTS = ("r", "lr", "ls")
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE = P3 / "cache" / "p3_t26"
PREDS = P3 / "preds"


def targets(cases, variant):
    y = np.column_stack([cases[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    hs0 = cases.hs0.to_numpy(float)[:, None]
    if variant == "r":
        return y / (hs0 + EPS)
    if variant == "lr":
        return np.log(y / (hs0 + EPS))
    if variant == "ls":
        return np.log1p(y)
    raise ValueError(variant)


def inverse(pred, cases, variant, smearing):
    hs0 = cases.hs0.to_numpy(float)[:, None]
    if variant == "r":
        return pred * (hs0 + EPS)
    if variant == "lr":
        return np.exp(pred) * (hs0 + EPS) * smearing
    if variant == "ls":
        return np.expm1(pred) * smearing
    raise ValueError(variant)


def fit_variant(x, cases, variant, n_jobs, seed=SEED):
    target = targets(cases, variant).ravel()
    params = dict(
        objective="regression_l2", n_estimators=500,
        learning_rate=0.03, num_leaves=31, min_child_samples=30,
        max_depth=-1, reg_lambda=1.0, subsample=0.9,
        colsample_bytree=0.8, random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    if seed != SEED:
        params.update(bagging_seed=seed, feature_fraction_seed=seed,
                      data_random_seed=seed, drop_seed=seed, extra_seed=seed)
    model = lgb.LGBMRegressor(**params).fit(expand(x, cases), target)
    if variant == "r":
        return model, 1.0
    residual = target - model.predict(expand(x, cases))
    return model, float(np.mean(np.exp(residual)))


def load_inputs():
    cases = load_cases()
    dense, _ = dense_cases(cases, load_reference_folds())
    x = case_features(cases)
    weights_path = P3 / "cache" / "p3_t9" / "dense_weights.csv"
    weights = pd.read_csv(weights_path)
    if weights.columns.tolist() != ["case_id", "stratum", "w"]:
        raise AssertionError("dense weight schema mismatch")
    if not np.array_equal(weights.case_id.to_numpy(), dense.case_id.to_numpy()):
        raise AssertionError("dense weight order mismatch")
    w = weights.w.to_numpy(float)
    if not np.isfinite(w).all() or not np.isclose(w.mean(), 1.0, atol=1e-12):
        raise AssertionError("invalid dense weights")
    positions = pd.Series(np.arange(len(cases)), index=cases.case_id)
    dense_pos = positions.loc[dense.case_id].to_numpy()
    y = np.column_stack([dense[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    return cases, dense, x, dense_pos, y, w


def load_test():
    rows, values = [], []
    for cid, station, context in C.iter_test_contexts():
        if context.shape != (C.CTX_ROWS, len(C.COLS)):
            raise AssertionError(f"test context shape mismatch: {cid} {context.shape}")
        rows.append((cid, station, float(context[-1, 0])))
        values.append(extract(context))
    cases = pd.DataFrame(rows, columns=["case_id", "station", "hs0"])
    return cases, np.stack(values).astype(np.float32)


def run(max_folds, n_jobs, seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    cases, dense, x, dense_pos, y, weights = load_inputs()
    fold = dense.fold.to_numpy()
    baseline_scores = {}
    for value in range(1, max_folds + 1):
        mask = fold == value
        val = dense.loc[mask].reset_index(drop=True)
        train_mask = forward_train_mask(cases, val)
        train = cases.loc[train_mask].reset_index(drop=True)
        model = fit_lgbm_t1(x[train_mask], train, n_jobs)
        pred = predict_lgbm_t1(model, x[dense_pos[mask]], val)
        baseline_scores[value] = weighted_rmse(y[mask], pred, weights[mask])

    if max_folds == 4 and abs(baseline_scores[4] - REFERENCE_FOLD4) > 0.002:
        print("대조군 | 값 | 기대값 | 델타")
        print(f"lgbm_t1_fold4_reweighted | {baseline_scores[4]:.6f} | "
              f"{REFERENCE_FOLD4:.6f} | {baseline_scores[4] - REFERENCE_FOLD4:+.6f}")
        print("4폴드 실행 명령")
        print("python P3/src/train_target_transform.py --max-folds 4 --n-jobs 2")
        return

    variant_oof = {name: np.full_like(y, np.nan) for name in VARIANTS}
    variant_scores, variant_smearing, variant_clips, variant_elapsed = ({name: {} for name in VARIANTS}
                                                                          for _ in range(4))
    for name in VARIANTS:
        for value in range(1, max_folds + 1):
            mask = fold == value
            val = dense.loc[mask].reset_index(drop=True)
            train_mask = forward_train_mask(cases, val)
            train = cases.loc[train_mask].reset_index(drop=True)
            start = time.perf_counter()
            model, smearing = fit_variant(x[train_mask], train, name, n_jobs, seed)
            raw = inverse(model.predict(expand(x[dense_pos[mask]], val)).reshape(-1, len(C.LEADS)),
                          val, name, smearing)
            pred = np.clip(raw, 0, 30)
            variant_oof[name][mask] = pred
            variant_scores[name][value] = weighted_rmse(y[mask], pred, weights[mask])
            variant_smearing[name][value] = smearing
            variant_clips[name][value] = int(np.any(raw != pred, axis=1).sum())
            variant_elapsed[name][value] = time.perf_counter() - start

    if max_folds == 4:
        test_cases, test_x = load_test()
        suffix = "" if seed == SEED else f"_s{seed}"
        for name in VARIANTS:
            save_pool_pred(PREDS / f"oof_dense_{name}_p3_t26{suffix}.npy", variant_oof[name])
            model, smearing = fit_variant(x, cases, name, n_jobs, seed)
            raw = inverse(model.predict(expand(test_x, test_cases)).reshape(-1, len(C.LEADS)),
                          test_cases, name, smearing)
            save_pool_pred(PREDS / f"test_{name}_p3_t26{suffix}.npy", np.clip(raw, 0, 30))

    control_fold = 4 if max_folds == 4 else 1
    control_value = baseline_scores[control_fold]
    control_expected = REFERENCE_FOLD4
    control_delta = control_value - control_expected if max_folds == 4 else float("nan")
    print("대조군 | 값 | 기대값 | 델타")
    print(f"lgbm_t1_fold{control_fold}_reweighted | {control_value:.6f} | "
          f"{control_expected:.6f} | {control_delta:+.6f}" if max_folds == 4 else
          f"lgbm_t1_fold{control_fold}_reweighted | {control_value:.6f} | "
          f"{control_expected:.6f}(fold4) | NA")
    print("변형 | fold1 RMSE | smearing s | 클립행수 | 소요초")
    for name in VARIANTS:
        smearing = ", ".join(
            f"f{value}={variant_smearing[name][value]:.9f}"
            for value in range(1, max_folds + 1)
        )
        clips = ", ".join(
            f"f{value}={variant_clips[name][value]}"
            for value in range(1, max_folds + 1)
        )
        elapsed = ", ".join(
            f"f{value}={variant_elapsed[name][value]:.1f}"
            for value in range(1, max_folds + 1)
        )
        print(f"{name} | {variant_scores[name][1]:.6f} | {smearing} | {clips} | {elapsed}")
    print("4폴드 실행 명령")
    print("python P3/src/train_target_transform.py --max-folds 4 --n-jobs 2")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if not 1 <= args.max_folds <= 4:
        parser.error("--max-folds must be between 1 and 4")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    run(args.max_folds, args.n_jobs, args.seed)


if __name__ == "__main__":
    main()
