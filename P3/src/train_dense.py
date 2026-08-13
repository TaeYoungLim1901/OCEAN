"""T8 dense OOF training and deferred dense-view reports."""
import argparse
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
from train_t2 import fit_damped, forward_train_mask, predict_damped

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
EXP = "p3_t8"
N_LEADS = len(C.LEADS)
CANDIDATES = ("persistence", "damped", "lgbm_t1")
T1F_FOLD4 = 0.754739
VARIANT_ORDER = ["V1", "V3", "V2", "V4"]
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE = P3 / "cache" / EXP
CKPT = CACHE / "ckpt"
PREDS = P3 / "preds"
FEATURES = CACHE / "features.npy"
FEATURE_IDS = CACHE / "feature_ids.npy"


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def load_cases():
    return pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)


def load_reference_folds():
    path = P3 / "cache" / "p3_t2" / "folds.csv"
    folds = pd.read_csv(path, parse_dates=["t0"])
    if set(folds.fold.unique()) != set(range(5)):
        raise AssertionError("p3_t2 folds.csv must contain folds 0..4")
    return folds


def assign_dense_folds(cases, reference):
    """Apply the existing per-station fold start cutoffs to every case."""
    out = np.full(len(cases), -1, dtype=int)
    for station, group in cases.groupby("station", sort=False):
        ref = reference.loc[reference.station.eq(station)]
        starts = ref.groupby("fold", sort=True).t0.min()
        if len(starts) != 5:
            raise AssertionError(f"missing fold boundary for {station}")
        t = group.t0.to_numpy()
        fold = np.searchsorted(starts.to_numpy(), t, side="right") - 1
        if ((fold < 0) | (fold > 4)).any():
            raise AssertionError(f"dense case before fold-0 boundary for {station}")
        out[group.index.to_numpy()] = fold
    if (out < 0).any():
        raise AssertionError("unassigned dense case")
    return out


def dense_cases(cases, reference):
    fold = assign_dense_folds(cases, reference)
    valid = fold >= 1
    dense = cases.loc[valid].reset_index(drop=True).copy()
    dense["fold"] = fold[valid]
    index = dense[["case_id", "station", "fold", "hs0"]].copy()
    index.to_csv(CACHE / "dense_index.csv", index=False)

    thin = set(reference.loc[reference.fold >= 1, "case_id"])
    dense_ids = set(index.case_id)
    intersection = len(thin & dense_ids)
    if intersection != len(thin):
        raise AssertionError("thinned cases are not a subset of dense cases")
    print(f"[t8] dense_valid={len(dense)} thin_valid={len(thin)} "
          f"thin_subset_dense=True intersection={intersection}", flush=True)
    return dense, index


def case_features(cases):
    ids = cases.case_id.to_numpy()
    if FEATURES.exists() and FEATURE_IDS.exists():
        saved_ids = np.load(FEATURE_IDS, allow_pickle=True)
        if np.array_equal(saved_ids, ids):
            x = np.load(FEATURES, allow_pickle=False)
            if x.shape[0] == len(cases):
                return x
    by_id = {cid: extract(ctx) for cid, _, ctx in C.iter_train_contexts(cases)}
    x = np.stack([by_id[cid] for cid in ids]).astype(np.float32)
    np.save(FEATURES, x)
    np.save(FEATURE_IDS, ids)
    return x


def fit_lgbm_t1(x, cases, n_jobs):
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=500,
        learning_rate=0.03, num_leaves=31, min_child_samples=30,
        max_depth=-1, reg_lambda=1.0, subsample=0.9,
        colsample_bytree=0.8, random_state=SEED, n_jobs=n_jobs,
        verbosity=-1,
    )
    return model.fit(expand(x, cases), target_delta(cases))


def predict_lgbm_t1(model, x, cases):
    delta = model.predict(expand(x, cases)).reshape(len(cases), N_LEADS)
    return (cases.hs0.to_numpy(float)[:, None] + delta).clip(0, 30)


def timed_stage(fold, name, fn, timings):
    start = time.perf_counter()
    print(f"[t8] fold={fold} stage={name} start elapsed=0.0s", flush=True)
    value = fn()
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    print(f"[t8] fold={fold} stage={name} elapsed={elapsed:.1f}s", flush=True)
    return value


def checkpoint_predictions(path, mask, expected_n):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            saved_mask = saved["k"]
            if saved_mask.shape != mask.shape or not np.array_equal(saved_mask, mask):
                return None
            pred = {name: saved[name] for name in CANDIDATES}
            if not all(value.shape == (expected_n, N_LEADS) for value in pred.values()):
                return None
            if not all(np.isfinite(value).all() for value in pred.values()):
                return None
            return pred
    except (KeyError, OSError, ValueError):
        return None


def save_oof(oof):
    for name in CANDIDATES:
        np.save(PREDS / f"oof_{name}_{EXP}.npy", oof[name])


def run_train(max_folds, n_jobs):
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    cases = load_cases()
    reference = load_reference_folds()
    dense, _ = dense_cases(cases, reference)
    x = case_features(cases)
    positions = pd.Series(np.arange(len(cases)), index=cases.case_id)
    dense_pos = positions.loc[dense.case_id].to_numpy()
    y = np.column_stack([dense[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    oof = {name: np.full_like(y, np.nan) for name in CANDIDATES}
    fold_sizes = {}
    thin_counts = reference.loc[reference.fold >= 1].groupby("fold").size()
    print(f"[t8] fold1_dense={int((dense.fold == 1).sum())} "
          f"thin={int(thin_counts.loc[1])} "
          f"multiplier={((dense.fold == 1).sum() / thin_counts.loc[1]):.3f}", flush=True)

    for fold in range(1, max_folds + 1):
        mask = dense.fold.to_numpy() == fold
        if not mask.any():
            raise AssertionError(f"no dense cases in fold {fold}")
        val = dense.loc[mask].reset_index(drop=True)
        allowed = forward_train_mask(cases, val)
        fold_sizes[str(fold)] = {
            "validation_cases": int(mask.sum()),
            "training_cases": int(allowed.sum()),
        }
        checkpoint = CKPT / f"fold{fold}.npz"
        pred = checkpoint_predictions(checkpoint, mask, int(mask.sum()))
        if pred is not None:
            for name in CANDIDATES:
                oof[name][mask] = pred[name]
            print(f"[t8] fold={fold} stage=resume elapsed=0.0s", flush=True)
        else:
            fold_start = time.perf_counter()
            timings = {}
            print(f"[t8] fold={fold} stage=fit start elapsed=0.0s "
                  f"train={allowed.sum()} val={mask.sum()}", flush=True)
            train = cases.loc[allowed].reset_index(drop=True)
            damped = timed_stage(fold, "damped", lambda: fit_damped(train), timings)
            model = timed_stage(
                fold, "lgbm_t1", lambda: fit_lgbm_t1(x[allowed], train, n_jobs), timings
            )
            val_x = x[dense_pos[mask]]
            pred = {
                "persistence": np.repeat(val.hs0.to_numpy(float)[:, None], N_LEADS, axis=1),
                "damped": predict_damped(damped, val),
                "lgbm_t1": predict_lgbm_t1(model, val_x, val),
            }
            for name in CANDIDATES:
                oof[name][mask] = pred[name]
            np.savez(checkpoint, k=mask, **pred)
            print(f"[t8] fold={fold} stage=checkpoint "
                  f"elapsed={time.perf_counter() - fold_start:.1f}s "
                  f"damped={timings['damped']:.1f}s lgbm_t1={timings['lgbm_t1']:.1f}s",
                  flush=True)
        for name in CANDIDATES:
            print(f"[t8] fold={fold} candidate={name} "
                  f"val_rmse={rmse(y[mask], oof[name][mask]):.6f}", flush=True)
        save_oof(oof)

    print(f"[t8] fold_train_sizes={json.dumps(fold_sizes, ensure_ascii=False)}", flush=True)
    if max_folds < 4:
        print(f"[t8] max_folds={max_folds} stage=stop", flush=True)
    print(f"[t8] total elapsed={time.perf_counter() - started:.1f}s", flush=True)


def variant_groups(cases, variant):
    if variant == "V1":
        return np.full(len(cases), "all", dtype=object)
    if variant == "V2":
        return cases.station.to_numpy(dtype=object)
    if variant == "V3":
        return np.where(cases.hs0.to_numpy() <= 1.75, "hs0<=1.75", "hs0>1.75")
    return np.array([
        f"{station}|{'hs0<=1.75' if hs0 <= 1.75 else 'hs0>1.75'}"
        for station, hs0 in zip(cases.station, cases.hs0)
    ], dtype=object)


def simplex(step):
    units = round(1 / step)
    return np.array([
        np.array(parts, dtype=float) / units
        for parts in itertools.product(range(units + 1), repeat=len(CANDIDATES))
        if sum(parts) == units
    ])


def best_weight(preds, y, mask, grid):
    scores = [
        float(np.mean((np.einsum("i,inj->nj", weight, preds[:, mask, :]) - y[mask]) ** 2))
        for weight in grid
    ]
    return grid[int(np.argmin(scores))]


def fit_weights(preds, y, groups, selection, step):
    grid = simplex(step)
    global_weight = best_weight(preds, y, selection, grid)
    group_weights, group_info = {}, {}
    for group in np.unique(groups):
        mask = selection & (groups == group)
        count = int(mask.sum())
        fallback = count < 15
        if not fallback:
            group_weights[group] = best_weight(preds, y, mask, grid)
        group_info[group] = {
            "selection_n": count,
            "fallback_global": fallback,
            "weight": group_weights.get(group, global_weight).tolist(),
        }
    return global_weight, group_weights, group_info


def apply_weights(preds, groups, global_weight, group_weights):
    out = np.empty((len(groups), N_LEADS), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        out[mask] = np.einsum("i,inj->nj", group_weights.get(group, global_weight), preds[:, mask, :])
    return out


def view(cases, y, pred, mask):
    selected = cases.loc[mask].reset_index(drop=True)
    selected_y, selected_pred = y[mask], pred[mask]
    true_long = M.wide_to_long(selected)
    pred_long = M.wide_to_long(
        selected,
        {f"y{lead}": selected_pred[:, j] for j, lead in enumerate(C.LEADS)},
    )
    checked = M.evaluate(
        pred_long, true_long, by=["lead", "station", "hs0bin"],
        hs0=selected[["case_id", "station", "hs0"]],
    )
    if abs(checked["total"] - rmse(selected_y, selected_pred)) > 1e-12:
        raise AssertionError("metric total mismatch")
    counts = {
        "lead": {str(lead): int(len(selected)) for lead in C.LEADS},
        "station": {st: int((selected.station == st).sum()) for st in C.STATIONS
                    if (selected.station == st).any()},
    }
    bins = np.digitize(selected.hs0.to_numpy(), [1.75, 2.25, 3.0], right=True)
    labels = ["1.5-1.75", "1.75-2.25", "2.25-3", "3+"]
    counts["hs0bin"] = {label: int((bins == value).sum())
                         for value, label in enumerate(labels) if (bins == value).any()}
    return {"total": float(checked["total"]), "n": int(len(selected) * N_LEADS),
            "lead": checked["lead"], "station": checked["station"],
            "hs0bin": checked["hs0bin"], "counts": counts}


def segment_deltas(candidate, baseline):
    rows = []
    for axis in ("lead", "station", "hs0bin"):
        for key, score in candidate[axis].items():
            rows.append({
                "segment": f"{axis}:{key}",
                "n": candidate["counts"][axis][key],
                "rmse": score,
                "v1_rmse": baseline[axis][key],
                "delta": score - baseline[axis][key],
            })
    return sorted(rows, key=lambda row: row["delta"], reverse=True)


def temporal_blocks(cases, mask):
    block = np.full(len(cases), -1, dtype=int)
    selected = cases.loc[mask]
    for station, group in selected.groupby("station", sort=False):
        order = group.sort_values("t0").index.to_numpy()
        for value, part in enumerate(np.array_split(order, 5)):
            block[part] = value
    if (block[mask] < 0).any():
        raise AssertionError("unassigned temporal block")
    return block


def load_dense_data():
    index = pd.read_csv(CACHE / "dense_index.csv")
    expected = ["case_id", "station", "fold", "hs0"]
    if index.columns.tolist() != expected:
        raise AssertionError("dense index schema mismatch")
    cases = load_cases().set_index("case_id")
    dense = cases.loc[index.case_id].reset_index()
    for col in ("station", "fold", "hs0"):
        values = index[col].to_numpy()
        expected_values = dense[col].to_numpy() if col != "fold" else None
        if col == "fold":
            reference = load_reference_folds()
            assigned = assign_dense_folds(dense, reference)
            expected_values = assigned
        if col == "hs0":
            if not np.allclose(values.astype(float), expected_values.astype(float), atol=1e-12, rtol=0):
                raise AssertionError("dense index hs0 mismatch")
        elif not np.array_equal(values, expected_values):
            raise AssertionError(f"dense index {col} mismatch")
    dense["fold"] = index.fold.to_numpy()
    y = np.column_stack([dense[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    preds = {}
    for name in CANDIDATES:
        path = PREDS / f"oof_{name}_{EXP}.npy"
        pred = np.load(path, allow_pickle=False)
        if pred.shape != y.shape:
            raise AssertionError(f"{name} shape mismatch")
        preds[name] = pred
    return dense, y, preds


def load_t7_v1_weight():
    report = json.loads((P3 / "cache" / "p3_t7" / "report.json").read_text(encoding="utf-8"))
    return np.array([report["variants"]["V1"]["global_weight"][name]
                     for name in CANDIDATES], dtype=float)


def run_report():
    dense, y, saved = load_dense_data()
    names = list(CANDIDATES)
    preds = np.stack([saved[name] for name in names])
    fold = dense.fold.to_numpy()
    fold4 = fold == 4
    selection = np.isin(fold, [1, 2, 3])
    reference = load_reference_folds()
    thin_ids = set(reference.loc[reference.fold == 4, "case_id"])
    thin4 = fold4 & dense.case_id.isin(thin_ids).to_numpy()
    if int(thin4.sum()) != 53:
        raise AssertionError(f"expected 53 thinned fold4 cases, got {int(thin4.sum())}")
    if not np.isfinite(preds[:, fold4, :]).all():
        raise AssertionError("fold4 OOF is incomplete")

    task2 = {"fold4": {"thinned": {}, "dense": {}}, "dense_temporal_blocks": {}}
    for name, pos in zip(names, range(len(names))):
        task2["fold4"]["thinned"][name] = view(dense, y, preds[pos], thin4)
        task2["fold4"]["dense"][name] = view(dense, y, preds[pos], fold4)
    blocks = temporal_blocks(dense, fold4)
    for name, pos in zip(names, range(len(names))):
        block_views = []
        for value in range(5):
            block_mask = fold4 & (blocks == value)
            block_views.append({"block": value, **view(dense, y, preds[pos], block_mask)})
        scores = np.array([row["total"] for row in block_views])
        task2["dense_temporal_blocks"][name] = {
            "blocks": block_views,
            "std": float(np.std(scores)),
            "ratio_0.002_to_std": float(0.002 / np.std(scores)) if np.std(scores) else None,
        }

    oof = preds
    weights = {}
    reports = {}
    for variant in ("V1", "V2", "V3", "V4"):
        groups = variant_groups(dense, variant)
        step = 0.05 if variant == "V1" else 0.25
        global_weight, group_weights, group_info = fit_weights(
            oof, y, groups, selection, step
        )
        prediction = apply_weights(oof, groups, global_weight, group_weights)
        weights[variant] = global_weight
        reports[variant] = {
            "grid_step": step,
            "global_weight": {name: float(value) for name, value in zip(names, global_weight)},
            "groups": {str(group): info for group, info in group_info.items()},
            "selection": view(dense, y, prediction, selection),
            "fold4_dense": view(dense, y, prediction, fold4),
            "fold4_thinned": view(dense, y, prediction, thin4),
        }

    v1_dense = reports["V1"]["fold4_dense"]
    t7_weight = load_t7_v1_weight()
    control_delta = weights["V1"] - t7_weight
    control = {
        "t7_v1_global_weight": {name: float(value) for name, value in zip(names, t7_weight)},
        "dense_v1_global_weight": reports["V1"]["global_weight"],
        "delta": {name: float(value) for name, value in zip(names, control_delta)},
        "within_pm_0.15": bool(np.all(np.abs(control_delta) <= 0.15)),
    }
    for variant, report in reports.items():
        deltas = segment_deltas(report["fold4_dense"], v1_dense)
        large = [row for row in deltas if row["n"] >= 100]
        max_bad = max(large, key=lambda row: row["delta"]) if large else None
        report["fold4_dense_delta_vs_v1"] = report["fold4_dense"]["total"] - v1_dense["total"]
        report["fold4_thinned_delta_vs_t1f"] = report["fold4_thinned"]["total"] - T1F_FOLD4
        report["max_degradation_dense_n_ge_100"] = max_bad
        report["acceptance"] = {
            "dense_improvement_ge_0.003": v1_dense["total"] - report["fold4_dense"]["total"] >= 0.003,
            "thinned_not_worse_than_t1f": report["fold4_thinned"]["total"] <= T1F_FOLD4,
            "dense_max_degradation_n_ge_100_le_0.03": max_bad is not None and max_bad["delta"] <= 0.03,
        }
        report["accepted"] = bool(all(report["acceptance"].values()))

    accepted = [name for name, report in reports.items()
                if report["accepted"] and control["within_pm_0.15"]]
    if accepted:
        best = min(reports[name]["fold4_dense"]["total"] for name in accepted)
        near = [name for name in accepted
                if reports[name]["fold4_dense"]["total"] - best < 0.002]
        selected = min(near, key=VARIANT_ORDER.index)
    else:
        selected = None
    output = {
        "experiment": "T8_dense_eval",
        "candidate_order": names,
        "dense_cases": int(len(dense)),
        "fold_counts": {str(value): int((fold == value).sum()) for value in range(1, 5)},
        "task2": task2,
        "control": control,
        "variants": reports,
        "selection": {"selected": selected},
    }
    (CACHE / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "report"), default="train")
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args(argv)
    if args.stage == "report":
        run_report()
        return
    if not 1 <= args.max_folds <= 4:
        parser.error("--max-folds must be between 1 and 4")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    run_train(args.max_folds, args.n_jobs)


if __name__ == "__main__":
    main()
