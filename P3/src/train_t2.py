"""Strict-forward P3 T2 candidate training and sealed-outer decision."""
import argparse
import hashlib
import json
import pathlib
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cases as C
from features_t2 import extract, feature_names

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
EXP = "p3_t2"
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE, MODELS, PREDS = P3 / "cache" / EXP, P3 / "models" / EXP, P3 / "preds"
CKPT = CACHE / "ckpt"
N_LEADS = len(C.LEADS)
T1F_OUTER = 0.754739
CANDIDATES = ("damped", "lgbm_pooled", "lgbm_lead", "lgbm_station", "extra")


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def make_folds(ev):
    fold = np.empty(len(ev), int)
    for _, g in ev.groupby("station", sort=False):
        for f, idx in enumerate(np.array_split(g.sort_values("t0").index.to_numpy(), 5)):
            fold[idx] = f
    return fold


def forward_train_mask(cases, val):
    allowed = np.zeros(len(cases), bool)
    for st in C.STATIONS:
        start = val.loc[val.station.eq(st), "t0"].min()
        cutoff = start - pd.Timedelta(hours=C.SPACING_H)
        allowed |= cases.station.eq(st).to_numpy() & cases.t0.lt(cutoff).to_numpy()
    return allowed


def matrix(cases, test=False):
    tag = "test" if test else "train"
    path, ids_path = CACHE / f"features_{tag}.npy", CACHE / f"feature_ids_{tag}.npy"
    ids = cases.case_id.to_numpy()
    if not path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"missing cached {tag} features")
    if not np.array_equal(np.load(ids_path, allow_pickle=True), ids):
        raise RuntimeError(f"{tag} feature ids mismatch")
    x = np.load(path)
    if x.shape[0] != len(cases):
        raise RuntimeError(f"{tag} feature row count mismatch")
    return x


def test_cases_and_features():
    cases = pd.read_csv(CACHE / "test_cases.csv")
    x = np.load(CACHE / "features_test.npy")
    ids = np.load(CACHE / "feature_ids_test.npy", allow_pickle=True)
    if not np.array_equal(ids, cases.case_id.to_numpy()):
        raise RuntimeError("test feature ids mismatch")
    if x.shape[0] != len(cases):
        raise RuntimeError("test feature row count mismatch")
    return cases, x


def target(cases):
    return np.column_stack([cases[f"y{lead}"].to_numpy(float) for lead in C.LEADS])


def target_delta(cases):
    return (target(cases) - cases.hs0.to_numpy(float)[:, None]).ravel()


def expand(x, cases):
    n = len(cases)
    xx = np.repeat(x, N_LEADS, axis=0)
    leads = np.tile(np.asarray(C.LEADS), n)
    stations = np.repeat(cases.station.to_numpy(), N_LEADS)
    lead_hot = np.column_stack([(leads == lead).astype(np.float32) for lead in C.LEADS])
    station_hot = np.column_stack([(stations == st).astype(np.float32) for st in C.STATIONS])
    interaction = np.column_stack([((stations == st) & (leads == lead)).astype(np.float32)
                                   for st in C.STATIONS for lead in C.LEADS])
    return np.column_stack((xx, leads.astype(np.float32) / 24, lead_hot,
                            station_hot, interaction)).astype(np.float32)


def station_matrix(x, cases):
    return np.column_stack((x, *[(cases.station.to_numpy() == st).astype(np.float32)
                                 for st in C.STATIONS])).astype(np.float32)


def lgbm(min_child=30, n_jobs=6, seed=SEED):
    return lgb.LGBMRegressor(objective="regression_l2", n_estimators=450,
                             learning_rate=0.025, num_leaves=31, min_child_samples=min_child,
                             reg_lambda=2.0, subsample=0.85, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs,
                             deterministic=True, force_col_wise=True, verbosity=-1)


def fit_damped(cases):
    params = {}
    for st, g in cases.groupby("station"):
        x = g.hs0.to_numpy(float)
        var = float(np.var(x))
        for lead in C.LEADS:
            y = g[f"y{lead}"].to_numpy(float)
            slope = 0.0 if var == 0 else float(np.clip(np.cov(x, y, ddof=0)[0, 1] / var, 0, 1))
            params[(st, lead)] = (float(y.mean() - slope * x.mean()), slope)
    return params


def predict_damped(model, cases):
    return np.column_stack([[model[(st, lead)][0] + model[(st, lead)][1] * hs0
                             for st, hs0 in zip(cases.station, cases.hs0)]
                            for lead in C.LEADS]).clip(0, 30)


def fit_models(x, cases, n_jobs=6, fold=0, timings=None):
    seed = SEED + fold
    if timings is None:
        timings = {}

    def stage(name, fn):
        start = time.perf_counter()
        print(f"[t2] fold={fold} stage={name} start elapsed=0.0s", flush=True)
        value = fn()
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        print(f"[t2] fold={fold} stage={name} elapsed={elapsed:.1f}s", flush=True)
        return value

    models = {}
    models["lgbm_pooled"] = stage(
        "lgbm_pooled", lambda: lgbm(35, n_jobs=n_jobs, seed=seed).fit(
            expand(x, cases), target_delta(cases)))
    sx = station_matrix(x, cases)

    def fit_lead():
        return [lgbm(25, n_jobs=n_jobs, seed=seed).fit(
            sx, cases[f"y{lead}"].to_numpy(float) - cases.hs0.to_numpy(float))
                for lead in C.LEADS]

    models["lgbm_lead"] = stage("lgbm_lead", fit_lead)

    def fit_station():
        station_models = {}
        for st in C.STATIONS:
            k = cases.station.eq(st).to_numpy()
            station_models[st] = lgbm(20, n_jobs=n_jobs, seed=seed).fit(
                expand(x[k], cases.loc[k].reset_index(drop=True)),
                target_delta(cases.loc[k].reset_index(drop=True)))
        return station_models

    models["lgbm_station"] = stage("lgbm_station", fit_station)
    ex = expand(x, cases)

    def fit_extra():
        median = np.nanmedian(ex, axis=0)
        median[~np.isfinite(median)] = 0.0
        extra = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=8, max_features=0.7,
                                    random_state=seed, n_jobs=n_jobs).fit(
                                        np.where(np.isfinite(ex), ex, median), target_delta(cases))
        return {"model": extra, "median": median}

    models["extra"] = stage("extratrees", fit_extra)
    models["damped"] = stage("damped", lambda: fit_damped(cases))
    return models


def predict_models(models, x, cases):
    hs0 = cases.hs0.to_numpy(float)[:, None]
    out = {"damped": predict_damped(models["damped"], cases)}
    out["lgbm_pooled"] = (hs0 + models["lgbm_pooled"].predict(expand(x, cases)).reshape(-1, N_LEADS)).clip(0, 30)
    sx = station_matrix(x, cases)
    delta = np.column_stack([model.predict(sx) for model in models["lgbm_lead"]])
    out["lgbm_lead"] = (hs0 + delta).clip(0, 30)
    station = np.zeros((len(cases), N_LEADS))
    for st in C.STATIONS:
        k = cases.station.eq(st).to_numpy()
        station[k] = models["lgbm_station"][st].predict(
            expand(x[k], cases.loc[k].reset_index(drop=True))).reshape(-1, N_LEADS)
    out["lgbm_station"] = (hs0 + station).clip(0, 30)
    ex = expand(x, cases)
    info = models["extra"]
    delta = info["model"].predict(np.where(np.isfinite(ex), ex, info["median"])).reshape(-1, N_LEADS)
    out["extra"] = (hs0 + delta).clip(0, 30)
    return out


def regime_labels(x):
    names = feature_names()
    delta = x[:, names.index("hs_6h_delta")]
    return np.where(delta > 0.10, "rising", np.where(delta < -0.10, "falling", "stable"))


def segment_scores(cases, y, pred, regimes):
    out = {}
    bins = pd.cut(cases.hs0, [0, 1.75, 2.25, 3, 99], labels=False).to_numpy()
    for j, lead in enumerate(C.LEADS):
        out[f"lead:{lead}"] = rmse(y[:, j], pred[:, j])
    for st in C.STATIONS:
        k = cases.station.eq(st).to_numpy()
        if k.any(): out[f"station:{st}"] = rmse(y[k], pred[k])
    for b in np.unique(bins[~pd.isna(bins)]):
        k = bins == b
        out[f"hs0bin:{int(b)}"] = rmse(y[k], pred[k])
    for value in ("falling", "stable", "rising"):
        k = regimes == value
        if k.any(): out[f"regime:{value}"] = rmse(y[k], pred[k])
    return out


def choose(selection_y, preds, mask, cases, regimes):
    choices = []
    for name in CANDIDATES:
        score = rmse(selection_y, preds[name][mask])
        seg = segment_scores(cases.loc[mask].reset_index(drop=True), selection_y,
                             preds[name][mask], regimes[mask])
        choices.append((score, max(seg.values()), [(name, 1.0)]))
    for i, left in enumerate(CANDIDATES):
        for right in CANDIDATES[i + 1:]:
            for w in np.arange(0.1, 1.0, 0.1):
                p = (1 - w) * preds[left][mask] + w * preds[right][mask]
                seg = segment_scores(cases.loc[mask].reset_index(drop=True), selection_y, p, regimes[mask])
                choices.append((rmse(selection_y, p), max(seg.values()),
                                [(left, round(float(1 - w), 1)), (right, round(float(w), 1))]))
    choices.sort(key=lambda z: (z[0], len(z[2]), z[1], str(z[2])))
    score, tail, weights = choices[0]
    return {"rmse": score, "max_segment_rmse": tail,
            "weights": [{"candidate": n, "weight": w} for n, w in weights]}


def combine(preds, selection):
    return sum(item["weight"] * preds[item["candidate"]] for item in selection["weights"])


def adversarial_drift(x_train, x_test):
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(x_train), size=min(len(x_train), len(x_test) * 3), replace=False)
    x = np.vstack((x_train[take], x_test))
    y = np.r_[np.zeros(len(take), int), np.ones(len(x_test), int)]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(x, y):
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.2, max_iter=1000, random_state=SEED))
        model.fit(x[tr], y[tr])
        oof[va] = model.predict_proba(x[va])[:, 1]
    train_mean, test_mean = np.nanmean(x_train, axis=0), np.nanmean(x_test, axis=0)
    scale = np.nanstd(x_train, axis=0)
    drift = np.abs(test_mean - train_mean) / np.where(scale > 1e-8, scale, np.inf)
    top = np.argsort(np.nan_to_num(drift, nan=-1))[-10:][::-1]
    return {"auc": float(roc_auc_score(y, oof)),
            "top_standardized_mean_shift": [{"feature": feature_names()[i], "z": float(drift[i])}
                                             for i in top]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.max_folds <= 4:
        parser.error("--max-folds must be between 1 and 4")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")

    total_start = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)
    ev = cases.loc[C.eval_split(cases)].reset_index(drop=True)
    folds = make_folds(ev)
    pd.DataFrame({"case_id": ev.case_id, "fold": folds, "t0": ev.t0,
                  "station": ev.station}).to_csv(CACHE / "folds.csv", index=False)
    x = matrix(cases)
    pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    xe = x[pos.loc[ev.case_id].to_numpy()]
    y = target(ev)
    oof = {name: np.full_like(y, np.nan) for name in CANDIDATES}
    fold_train_sizes = {}
    for fold in range(1, args.max_folds + 1):
        k = folds == fold
        allowed = forward_train_mask(cases, ev.loc[k])
        fold_train_sizes[str(fold)] = int(allowed.sum())
        checkpoint = CKPT / f"fold{fold}.npz"
        fold_pred = None
        if checkpoint.exists():
            try:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    saved_k = saved["k"]
                    expected_shape = (int(k.sum()), y.shape[1])
                    if saved_k.shape == k.shape and np.array_equal(saved_k, k):
                        candidate_preds = {name: saved[name] for name in CANDIDATES}
                        if all(pred.shape == expected_shape for pred in candidate_preds.values()):
                            fold_pred = candidate_preds
            except (KeyError, OSError, ValueError):
                fold_pred = None
        if fold_pred is not None:
            for name in CANDIDATES:
                oof[name][k] = fold_pred[name]
            print(f"[t2] fold={fold} stage=resume elapsed=0.0s", flush=True)
        else:
            fold_start = time.perf_counter()
            print(f"[t2] fold={fold} stage=fit start elapsed=0.0s", flush=True)
            tr = cases.loc[allowed].reset_index(drop=True)
            models = fit_models(x[allowed], tr, n_jobs=args.n_jobs, fold=fold)
            fold_pred = predict_models(models, xe[k], ev.loc[k].reset_index(drop=True))
            for name in CANDIDATES:
                oof[name][k] = fold_pred[name]
            np.savez(checkpoint, k=k, **fold_pred)
            print(f"[t2] fold={fold} stage=checkpoint elapsed={time.perf_counter() - fold_start:.1f}s",
                  flush=True)
        for name in CANDIDATES:
            print(f"[t2] fold={fold} candidate={name} val_rmse={rmse(y[k], oof[name][k]):.6f}",
                  flush=True)

    if args.max_folds < 4:
        print(f"[t2] max_folds={args.max_folds} stage=stop", flush=True)
        print(f"[t2] total elapsed={time.perf_counter() - total_start:.1f}s", flush=True)
        return

    MODELS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    selection_mask = np.isin(folds, [1, 2, 3])
    outer_mask = folds == 4
    regimes = regime_labels(xe)
    selection = choose(y[selection_mask], oof, selection_mask, ev, regimes)
    selection_report = {"folds": [1, 2, 3], "outer_fold": 4, "selection": selection,
                        "candidate_rmse": {n: rmse(y[selection_mask], oof[n][selection_mask])
                                           for n in CANDIDATES},
                        "fold_train_sizes": fold_train_sizes}
    (CACHE / "selection_locked.json").write_text(json.dumps(selection_report, indent=2), encoding="utf-8")
    print("selection locked before outer:", json.dumps(selection, ensure_ascii=False))

    test_cases, xtest = test_cases_and_features()
    test_checkpoint = CKPT / "test_preds.npz"
    bundle_path = MODELS / "bundle.pkl"
    test_preds = None
    bundle = None
    if test_checkpoint.exists():
        try:
            with np.load(test_checkpoint, allow_pickle=False) as saved:
                expected_shape = (len(test_cases), y.shape[1])
                candidate_preds = {name: saved[name] for name in CANDIDATES}
                if all(pred.shape == expected_shape for pred in candidate_preds.values()):
                    test_preds = candidate_preds
                else:
                    raise ValueError("invalid test prediction checkpoint")
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            if not isinstance(bundle, dict) or "models" not in bundle:
                raise ValueError("invalid model bundle")
            print("[t2] fold=full stage=resume_test elapsed=0.0s", flush=True)
        except (KeyError, OSError, ValueError, EOFError, pickle.PickleError):
            test_preds = None
            bundle = None
    if test_preds is None or bundle is None:
        full_models = fit_models(x, cases, n_jobs=args.n_jobs, fold=0)
        test_preds = predict_models(full_models, xtest, test_cases)
        np.savez(test_checkpoint, **test_preds)
        bundle = {"models": full_models, "selection": selection, "features": feature_names(),
                  "leads": C.LEADS, "stations": C.STATIONS, "seed": SEED}
        with open(bundle_path, "wb") as f: pickle.dump(bundle, f)

    selected_oof = combine(oof, selection)
    old_cases = pd.read_parquet(P3 / "cache" / "p3_t1" / "oof_cases.parquet")
    if not np.array_equal(old_cases.case_id.to_numpy(), ev.case_id.to_numpy()):
        raise RuntimeError("T1F OOF case alignment mismatch")
    t1f = np.load(P3 / "cache" / "p3_t1" / "oof_final_p3_t1f.npy")
    outer_cases = ev.loc[outer_mask].reset_index(drop=True)
    outer_regime = regimes[outer_mask]
    seg_new = segment_scores(outer_cases, y[outer_mask], selected_oof[outer_mask], outer_regime)
    seg_t1f = segment_scores(outer_cases, y[outer_mask], t1f[outer_mask], outer_regime)
    degradation = {k: seg_new[k] - seg_t1f[k] for k in seg_new.keys() & seg_t1f.keys()}
    outer_new, outer_t1f = rmse(y[outer_mask], selected_oof[outer_mask]), rmse(y[outer_mask], t1f[outer_mask])
    accepted = outer_new <= T1F_OUTER - 0.002 and max(degradation.values()) <= 0.03
    for name in CANDIDATES: np.save(CACHE / f"oof_{name}_p3_t2.npy", oof[name])
    np.save(CACHE / "oof_final_p3_t2.npy", selected_oof)
    residual_corr = {n: float(np.corrcoef((y[selection_mask] - oof[n][selection_mask]).ravel(),
                                         (y[selection_mask] - selected_oof[selection_mask]).ravel())[0, 1])
                     for n in CANDIDATES}
    drift = adversarial_drift(x, xtest)
    report = {**selection_report, "selection_rmse": rmse(y[selection_mask], selected_oof[selection_mask]),
              "outer_t1f": outer_t1f, "outer_selected": outer_new,
              "outer_delta_vs_t1f": outer_new - outer_t1f,
              "outer_max_segment_degradation": max(degradation.values()),
              "outer_segment_degradation": dict(sorted(degradation.items(), key=lambda z: z[1], reverse=True)),
              "outer_segments_selected": seg_new, "outer_segments_t1f": seg_t1f,
              "residual_corr_vs_selected": residual_corr, "drift": drift, "accepted": accepted}
    (CACHE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bundle["accepted"] = accepted; bundle["report_sha256"] = hashlib.sha256((CACHE / "report.json").read_bytes()).hexdigest().upper()
    with open(MODELS / "bundle.pkl", "wb") as f: pickle.dump(bundle, f)
    print(json.dumps(report, indent=2))
    print(f"[t2] total elapsed={time.perf_counter() - total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
