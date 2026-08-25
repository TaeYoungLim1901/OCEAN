"""E002: 시간 단위 수정과 2025 same-sensor pseudo-gap 교차검증."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from train import estimator
from submission_tools import save_pool_pred

HOLDS = {
    "2024": (pd.Timestamp("2024-09-01", tz="Asia/Seoul"),
             pd.Timestamp("2024-10-31 23:50", tz="Asia/Seoul"), False),
    "2025": (pd.Timestamp("2025-07-02", tz="Asia/Seoul"),
             pd.Timestamp("2025-08-31 23:50", tz="Asia/Seoul"), True),
}
CANDIDATES = {
    "bug_depth": (False, True),
    "fixed_depth": (True, True),
    "fixed_no_target_depth": (True, False),
}


def pseudo_obs(obs, start, end, transfer):
    work = obs.copy()
    if transfer:
        source = work[work.time.between(pd.Timestamp("2025-09-01", tz="Asia/Seoul"),
                                       pd.Timestamp("2025-10-31 23:50", tz="Asia/Seoul"))]
        target = work[work.time.between(start, end)]
        for layer in C.PUBLIC_LAYERS:
            src = source[source.layer.eq(layer)].sort_values("time")
            dst = target[target.layer.eq(layer)].sort_values("time")
            if len(src) != 8784 or len(dst) != 8784:
                raise ValueError(f"pseudo mask 길이: layer {layer} src={len(src)} dst={len(dst)}")
            for col in ("temp", "psal", "depth"):
                work.loc[dst.index[src[col].isna().to_numpy()], col] = np.nan
    hidden = work.time.between(start, end) & work.layer.isin(C.TARGET_LAYERS)
    work.loc[hidden, ["temp", "psal"]] = np.nan
    return work


def holdout(obs, label):
    start, end, transfer = HOLDS[label]
    truth = obs[obs.time.between(start, end) & obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].copy()
    work = pseudo_obs(obs, start, end, transfer)
    public_n = (work[work.layer.isin(C.PUBLIC_LAYERS) & work.temp.notna()]
                .groupby("time").size())
    truth = truth[truth.time.map(public_n).fillna(0).ge(2)].reset_index(drop=True)
    train = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()
                & ~obs.time.between(start, end)].reset_index(drop=True)
    return work, train, truth


def fit_models(xtrain, train):
    models = {}
    for layer in C.TARGET_LAYERS:
        use = train.layer.eq(layer)
        model = estimator("lgb_full")
        model.fit(xtrain.loc[use, C.FULL], train.loc[use, "temp"] - xtrain.loc[use, "base_nom"])
        models[layer] = model
    return models


def predict(models, x, rows):
    out = np.empty(len(rows))
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        out[use] = x.loc[use, "base_nom"] + models[layer].predict(x.loc[use, C.FULL])
    return out


def score(rows, x, pred):
    expected = [1, 5, 6, 7] if rows.time.dt.year.iat[0] == 2024 else C.PUBLIC_LAYERS
    availability = x[[f"temp_{layer}" for layer in expected]].notna().sum(axis=1)
    missing = availability.lt(len(expected)).to_numpy()
    result = {
        "overall": C.rmse(rows.temp, pred),
        "layers": {str(layer): C.rmse(rows.loc[rows.layer.eq(layer), "temp"], pred[rows.layer.eq(layer)])
                   for layer in C.TARGET_LAYERS},
        "missing_public": C.rmse(rows.loc[missing, "temp"], pred[missing]) if missing.any() else None,
        "complete_public": C.rmse(rows.loc[~missing, "temp"], pred[~missing]) if (~missing).any() else None,
        "n": len(rows), "n_missing_public": int(missing.sum()),
    }
    return result


def main():
    obs = C.load_obs()
    report, hold_data = {}, {}
    for label in HOLDS:
        work, train, val = holdout(obs, label)
        hold_data[label] = (work, train, val)
        report[label] = {}
        for name, (tide_correct, use_depth) in CANDIDATES.items():
            xtrain = C.build_features(work, train, tide_correct=tide_correct, use_target_depth=use_depth)
            xval = C.build_features(work, val, tide_correct=tide_correct, use_target_depth=use_depth)
            pred = predict(fit_models(xtrain, train), xval, val)
            save_pool_pred(C.ROOT / "preds" / f"oof_{name}_{label}_E002.npy", pred)
            report[label][name] = score(val, xval, pred)

    means = {name: np.mean([report[label][name]["overall"] for label in HOLDS])
             for name in CANDIDATES}
    fixed_ok = (means["fixed_depth"] < means["bug_depth"] and
                all(report[label]["fixed_depth"]["overall"] <= report[label]["bug_depth"]["overall"] + 0.003
                    for label in HOLDS))
    selected = "fixed_depth" if fixed_ok else "bug_depth"
    if selected == "fixed_depth" and means["fixed_no_target_depth"] - means["fixed_depth"] < 0.003:
        selected = "fixed_no_target_depth"

    tide_correct, use_depth = CANDIDATES[selected]
    all_rows = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].reset_index(drop=True)
    xall = C.build_features(obs, all_rows, tide_correct=tide_correct, use_target_depth=use_depth)
    models = fit_models(xall, all_rows)
    artifact = {"models": models, "selected": selected, "tide_correct": tide_correct,
                "use_target_depth": use_depth, "features": C.FULL, "seed": C.SEED,
                "report": report, "means": means}
    with open(C.ROOT / "models" / "E002.pkl", "wb") as f:
        pickle.dump(artifact, f)
    payload = {"selected": selected, "means": means, "scores": report}
    (C.ROOT / "cache" / "E002" / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
