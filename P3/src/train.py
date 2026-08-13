"""Train P3 T1 damped-persistence and direct-delta LightGBM models."""
import json
import pathlib
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import cases as C
import metric as M
from features import extract, feature_names

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
EXP = "p3_t1"
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE, MODELS, PREDS = P3 / "cache" / EXP, P3 / "models" / EXP, P3 / "preds"


def make_folds(eval_cases):
    fold = np.empty(len(eval_cases), int)
    for _, g in eval_cases.groupby("station", sort=False):
        for f, idx in enumerate(np.array_split(g.sort_values("t0").index.to_numpy(), 5)):
            fold[idx] = f
    return fold


def case_features(cases):
    path = CACHE / "features.npy"
    ids_path = CACHE / "feature_ids.npy"
    if path.exists() and ids_path.exists():
        ids = np.load(ids_path, allow_pickle=True)
        if np.array_equal(ids, cases.case_id.to_numpy()):
            return np.load(path)
    by_id = {cid: extract(ctx) for cid, _, ctx in C.iter_train_contexts(cases)}
    x = np.stack([by_id[cid] for cid in cases.case_id])
    np.save(path, x)
    np.save(ids_path, cases.case_id.to_numpy())
    return x


def fit_damped(train):
    params = {}
    for st, g in train.groupby("station"):
        x = g.hs0.to_numpy(float)
        for lead in C.LEADS:
            y = g[f"y{lead}"].to_numpy(float)
            slope = float(np.clip(np.cov(x, y, ddof=0)[0, 1] / np.var(x), 0, 1))
            params[(st, lead)] = (float(y.mean() - slope * x.mean()), slope)
    return params


def predict_damped(cases, params):
    return np.column_stack([
        [params[(st, lead)][0] + params[(st, lead)][1] * hs0
         for st, hs0 in zip(cases.station, cases.hs0)] for lead in C.LEADS
    ]).clip(0, 30)


def expand(x, cases):
    n, p = x.shape
    xx = np.repeat(x, len(C.LEADS), axis=0)
    leads = np.tile(np.asarray(C.LEADS, np.float32), n)
    stations = np.repeat(cases.station.to_numpy(), len(C.LEADS))
    onehot = np.column_stack([(stations == st).astype(np.float32) for st in C.STATIONS] +
                             [(leads == lead).astype(np.float32) for lead in C.LEADS])
    return np.column_stack([xx, leads / 24, onehot])


def target_delta(cases):
    return np.column_stack([cases[f"y{l}"].to_numpy() - cases.hs0.to_numpy()
                            for l in C.LEADS]).ravel()


def fit_lgbm(x, cases):
    model = lgb.LGBMRegressor(objective="regression_l2", n_estimators=500,
                              learning_rate=0.03, num_leaves=31, min_child_samples=30,
                              max_depth=-1, reg_lambda=1.0, subsample=0.9,
                              colsample_bytree=0.8, random_state=SEED, n_jobs=2,
                              verbosity=-1)
    model.fit(expand(x, cases), target_delta(cases))
    return model


def predict_lgbm(model, x, cases):
    delta = model.predict(expand(x, cases)).reshape(len(cases), len(C.LEADS))
    return (cases.hs0.to_numpy()[:, None] + delta).clip(0, 30)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def segment_rmse(cases, y, p):
    out = {}
    bins = pd.cut(cases.hs0, [0, 1.75, 2.25, 3, 99], labels=False).to_numpy()
    for j, lead in enumerate(C.LEADS):
        out[f"lead:{lead}"] = rmse(y[:, j], p[:, j])
    for st in C.STATIONS:
        k = cases.station.eq(st).to_numpy()
        out[f"station:{st}"] = rmse(y[k], p[k])
    for b in np.unique(bins):
        k = bins == b
        out[f"hs0bin:{b}"] = rmse(y[k], p[k])
    return out


def save_local_csv(cases, pred):
    true = M.wide_to_long(cases)
    pred_wide = {f"y{lead}": pred[:, j] for j, lead in enumerate(C.LEADS)}
    predicted = M.wide_to_long(cases, pred_wide)
    true[M.KEY + ["hs"]].to_csv(CACHE / "oof_true.csv", index=False)
    predicted[M.KEY + ["hs_pred"]].to_csv(CACHE / "oof_final.csv", index=False)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    cases = pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)
    keep = C.eval_split(cases)
    ev = cases.loc[keep].reset_index(drop=True)
    folds = make_folds(ev)
    pd.DataFrame({"case_id": ev.case_id, "fold": folds, "t0": ev.t0,
                  "station": ev.station}).to_csv(CACHE / "folds.csv", index=False)
    x = case_features(cases)
    pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    xe = x[pos.loc[ev.case_id].to_numpy()]
    oof_d = np.full((len(ev), len(C.LEADS)), np.nan)
    oof_l = np.full_like(oof_d, np.nan)
    skipped = []
    for f in range(5):
        val = ev[folds == f]
        allowed = np.zeros(len(cases), bool)
        for st in C.STATIONS:
            start = val.loc[val.station.eq(st), "t0"].min()
            cutoff = start - pd.Timedelta(hours=C.SPACING_H)
            allowed |= cases.station.eq(st).to_numpy() & cases.t0.lt(cutoff).to_numpy()
        tr = cases.loc[allowed].reset_index(drop=True)
        xt = x[allowed]
        print(f"fold {f}: train={len(tr)} val={len(val)}")
        if len(tr) < 50:
            skipped.append((f, len(tr)))
            print(f"fold {f}: skipped")
            continue
        d = fit_damped(tr)
        model = fit_lgbm(xt, tr)
        k = folds == f
        oof_d[k] = predict_damped(val.reset_index(drop=True), d)
        oof_l[k] = predict_lgbm(model, xe[k], val.reset_index(drop=True))
    print(f"skipped folds: {skipped}")
    y = np.column_stack([ev[f"y{l}"].to_numpy() for l in C.LEADS])
    persist = np.repeat(ev.hs0.to_numpy()[:, None], len(C.LEADS), axis=1)
    valid = np.isfinite(oof_d).all(axis=1) & np.isfinite(oof_l).all(axis=1)
    ev_valid = ev.loc[valid].reset_index(drop=True)
    y_valid = y[valid]
    seg_d = segment_rmse(ev_valid, y_valid, oof_d[valid])
    best = (rmse(y_valid, oof_d[valid]), 0.0)
    for w in np.arange(0, 1.001, 0.05):
        p = (1 - w) * oof_d + w * oof_l
        score = rmse(y_valid, p[valid])
        worsening = max(segment_rmse(ev_valid, y_valid, p[valid])[k] - seg_d[k]
                         for k in seg_d)
        if score < best[0] and worsening <= 0.05:
            best = (score, float(round(w, 2)))
    accepted = best[1] > 0 and rmse(y_valid, oof_d[valid]) - best[0] >= 0.003
    weight = best[1] if accepted else 0.0
    final = (1 - weight) * oof_d + weight * oof_l
    np.save(PREDS / "oof_damped_p3_t1.npy", oof_d)
    np.save(PREDS / "oof_lgbm_p3_t1.npy", oof_l)
    np.save(PREDS / "oof_final_p3_t1.npy", final)
    save_local_csv(ev, final)
    full_d = fit_damped(cases)
    full_l = fit_lgbm(x, cases)
    bundle = {"damped": full_d, "lgbm": full_l, "weight": weight,
              "features": feature_names(), "leads": C.LEADS, "stations": C.STATIONS}
    with open(MODELS / "bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    resid_corr = float(np.corrcoef((y_valid - oof_d[valid]).ravel(),
                                   (y_valid - oof_l[valid]).ravel())[0, 1])
    scores = {"persistence": rmse(y_valid, persist[valid]),
              "damped": rmse(y_valid, oof_d[valid]),
              "lgbm": rmse(y_valid, oof_l[valid]),
              "final": rmse(y_valid, final[valid]),
              "weight_lgbm": weight, "residual_corr": resid_corr,
              "accepted": accepted, "valid_cases": int(valid.sum()),
              "nan_cases": int((~valid).sum()),
              "skipped_folds": {str(f): n for f, n in skipped},
              "segments_final": segment_rmse(ev_valid, y_valid, final[valid]),
              "segments_damped": seg_d}
    (CACHE / "report.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    np.save(CACHE / "oof_true.npy", y)
    ev[["case_id", "station", "hs0", "t0"] + [f"y{l}" for l in C.LEADS]].to_parquet(
        CACHE / "oof_cases.parquet", index=False)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
