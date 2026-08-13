"""E003: 네 연속 pseudo-gap, drift/segment 진단, 저랭크 profile 앙상블."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from train import estimator

HOLDS = {
    "2024_julaug": ("2024-07-02", "2024-08-31 23:50", True),
    "2024_sepoct": ("2024-09-01", "2024-10-31 23:50", False),
    "2025_mayjun": ("2025-05-02", "2025-07-01 23:50", True),
    "2025_julaug": ("2025-07-02", "2025-08-31 23:50", True),
}
WEIGHTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
MIN_GAIN, HOLD_TOL, SEGMENT_TOL, MIN_SEGMENT = 0.010, 0.010, 0.030, 200


def bounds(label):
    a, b, transfer = HOLDS[label]
    return pd.Timestamp(a, tz="Asia/Seoul"), pd.Timestamp(b, tz="Asia/Seoul"), transfer


def pseudo_obs(obs, label):
    start, end, transfer = bounds(label)
    work = obs.copy()
    if transfer:
        source = work[work.time.between(pd.Timestamp("2025-09-01", tz="Asia/Seoul"),
                                       pd.Timestamp("2025-10-31 23:50", tz="Asia/Seoul"))]
        target = work[work.time.between(start, end)]
        for layer in C.PUBLIC_LAYERS:
            src = source[source.layer.eq(layer)].sort_values("time")
            dst = target[target.layer.eq(layer)].sort_values("time")
            if dst.empty:  # 2024에는 layer 8이 없다.
                continue
            if len(src) != 8784 or len(dst) != 8784:
                raise ValueError(f"pseudo mask 길이 {label} layer={layer}: {len(src)}/{len(dst)}")
            for col in ("temp", "psal", "depth"):
                work.loc[dst.index[src[col].isna().to_numpy()], col] = np.nan
    hidden = work.time.between(start, end) & work.layer.isin(C.TARGET_LAYERS)
    work.loc[hidden, ["temp", "psal"]] = np.nan
    return work


def holdout(obs, label):
    start, end, _ = bounds(label)
    truth = obs[obs.time.between(start, end) & obs.layer.isin(C.TARGET_LAYERS)
                & obs.temp.notna()].copy()
    work = pseudo_obs(obs, label)
    public_n = (work[work.layer.isin(C.PUBLIC_LAYERS) & work.temp.notna()]
                .groupby("time").size())
    truth = truth[truth.time.map(public_n).fillna(0).ge(2)].reset_index(drop=True)
    train = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()
                & ~obs.time.between(start, end)].reset_index(drop=True)
    if len(truth) < 25000:
        raise ValueError(f"holdout truth 부족 {label}: {len(truth)}")
    return work, train, truth


def fit_layers(x, rows, features):
    models = {}
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        model = estimator("lgb_full")
        model.fit(x.loc[use, features], rows.loc[use, "temp"] - x.loc[use, "base_nom"])
        models[layer] = model
    return models


def predict_layers(models, x, rows, features):
    pred = np.empty(len(rows))
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        pred[use] = x.loc[use, "base_nom"] + models[layer].predict(x.loc[use, features])
    return pred


def profile_rows(obs, times):
    return (obs[obs.layer.eq(3) & obs.time.isin(pd.Index(times))]
            [C.KEY + ["depth", "nominal_depth"]].sort_values("time").drop_duplicates("time")
            .reset_index(drop=True))


def fit_eof(obs, work, rows, x):
    residual = rows[C.KEY].assign(resid=rows.temp.to_numpy() - x.base_nom.to_numpy())
    matrix = residual.pivot(index="time", columns="layer", values="resid").dropna()
    matrix = matrix[C.TARGET_LAYERS]
    pca = PCA(n_components=2, random_state=C.SEED)
    scores = pca.fit_transform(matrix)
    prows = profile_rows(obs, matrix.index)
    px = C.build_features(work, prows)
    order = pd.Series(np.arange(len(prows)), index=prows.time)
    take = order.loc[matrix.index].to_numpy()
    models = []
    for k in range(2):
        model = estimator("lgb_full")
        model.fit(px.iloc[take][C.ENHANCED], scores[:, k])
        models.append(model)
    return {"pca": pca, "models": models}


def predict_eof(eof, obs, work, rows):
    times = pd.Index(rows.time.unique()).sort_values()
    prows = profile_rows(obs, times)
    if len(prows) != len(times):
        raise ValueError("EOF profile row 누락")
    px = C.build_features(work, prows)
    scores = np.column_stack([model.predict(px[C.ENHANCED]) for model in eof["models"]])
    residual = eof["pca"].inverse_transform(scores)
    lookup = {(t, layer): residual[i, j] for i, t in enumerate(prows.time)
              for j, layer in enumerate(C.TARGET_LAYERS)}
    return rows.time.combine(rows.layer, lambda t, layer: lookup[(t, layer)]).to_numpy() + rows["_base"].to_numpy()


def regimes(x, rows):
    expected = [1, 5, 6, 7] if rows.time.dt.year.iloc[0] == 2024 else C.PUBLIC_LAYERS
    availability = x[[f"temp_{layer}" for layer in expected]].notna().sum(axis=1)
    strat = x.strat_1_6.abs()
    return {
        "missing_public": availability.lt(len(expected)).to_numpy(),
        "complete_public": availability.eq(len(expected)).to_numpy(),
        "strat_low": strat.lt(5).to_numpy(),
        "strat_mid": strat.between(5, 10, inclusive="left").to_numpy(),
        "strat_high": strat.ge(10).to_numpy(),
        "transition": x.transition.abs().ge(2.2).to_numpy(),
        "calm": x.transition.abs().lt(2.2).to_numpy(),
    }


def score(rows, x, pred):
    result = {"overall": C.rmse(rows.temp, pred), "n": len(rows), "segments": {}}
    masks = {f"layer_{layer}": rows.layer.eq(layer).to_numpy() for layer in C.TARGET_LAYERS}
    masks.update(regimes(x, rows))
    for name, mask in masks.items():
        result["segments"][name] = {"n": int(mask.sum()),
                                    "rmse": C.rmse(rows.loc[mask, "temp"], pred[mask]) if mask.any() else None}
    return result


def robust_blend(hold_preds, truths):
    chosen, out = {}, {}
    for hold in HOLDS:
        others = [h for h in HOLDS if h != hold]
        losses = []
        for w in WEIGHTS:
            losses.append(np.mean([C.rmse(truths[h].temp,
                                (1 - w) * hold_preds[h]["enhanced"] + w * hold_preds[h]["eof"])
                                   for h in others]))
        w = float(WEIGHTS[int(np.argmin(losses))])
        chosen[hold] = w
        out[hold] = (1 - w) * hold_preds[hold]["enhanced"] + w * hold_preds[hold]["eof"]
    return out, chosen, float(np.median(list(chosen.values())))


def gate_blend(hold_preds, truths, hold_x):
    out, chosen = {}, {}
    for hold in HOLDS:
        out[hold] = hold_preds[hold]["enhanced"].copy()
        chosen[hold] = {}
        target_regimes = regimes(hold_x[hold], truths[hold])
        for regime in ("strat_low", "strat_mid", "strat_high"):
            losses = {}
            for model in ("enhanced", "eof"):
                vals = []
                for other in HOLDS:
                    if other == hold:
                        continue
                    mask = regimes(hold_x[other], truths[other])[regime]
                    if mask.sum() >= MIN_SEGMENT:
                        vals.append(C.rmse(truths[other].loc[mask, "temp"], hold_preds[other][model][mask]))
                losses[model] = np.mean(vals) if vals else np.inf
            winner = min(losses, key=losses.get)
            chosen[hold][regime] = winner
            out[hold][target_regimes[regime]] = hold_preds[hold][winner][target_regimes[regime]]
    final = {}
    for regime in ("strat_low", "strat_mid", "strat_high"):
        losses = {}
        for model in ("enhanced", "eof"):
            vals = []
            for hold in HOLDS:
                mask = regimes(hold_x[hold], truths[hold])[regime]
                if mask.sum() >= MIN_SEGMENT:
                    vals.append(C.rmse(truths[hold].loc[mask, "temp"], hold_preds[hold][model][mask]))
            losses[model] = np.mean(vals)
        final[regime] = min(losses, key=losses.get)
    return out, chosen, final


def drift_auc(xhold, hold_rows, xtest, test_rows):
    cols = ["target_actual", *[f"temp_{x}" for x in C.PUBLIC_LAYERS],
            *[f"depth_{x}" for x in C.PUBLIC_LAYERS], "strat_1_6", "transition"]
    x = pd.concat([xhold[cols], xtest[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(xhold)), np.ones(len(xtest))]
    groups = pd.concat([hold_rows.time.astype(str), test_rows.time.astype(str)], ignore_index=True)
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=C.SEED)
    tr, va = next(split.split(x, y, groups))
    model = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(),
                          LogisticRegression(max_iter=500, random_state=C.SEED))
    model.fit(x.iloc[tr], y[tr])
    return float(roc_auc_score(y[va], model.predict_proba(x.iloc[va])[:, 1]))


def drift_summary(xhold, hold_rows, xtest, test_rows):
    return {
        "adversarial_auc": drift_auc(xhold, hold_rows, xtest, test_rows),
        "rows": len(hold_rows),
        "public": {str(layer): {
            "hold_temp_missing": float(xhold[f"temp_{layer}"].isna().mean()),
            "test_temp_missing": float(xtest[f"temp_{layer}"].isna().mean()),
            "hold_depth_mean": float(xhold[f"depth_{layer}"].mean()),
            "test_depth_mean": float(xtest[f"depth_{layer}"].mean()),
        } for layer in C.PUBLIC_LAYERS},
        "target_depth": {str(layer): {
            "hold_mean": float(hold_rows.loc[hold_rows.layer.eq(layer), "depth"].mean()),
            "test_mean": float(test_rows.loc[test_rows.layer.eq(layer), "depth"].mean()),
            "hold_missing": float(hold_rows.loc[hold_rows.layer.eq(layer), "depth"].isna().mean()),
            "test_missing": float(test_rows.loc[test_rows.layer.eq(layer), "depth"].isna().mean()),
        } for layer in C.TARGET_LAYERS},
    }


def eligible(candidate, reports):
    means = {name: np.mean([reports[h][name]["overall"] for h in HOLDS])
             for name in ("e002", candidate)}
    if means[candidate] > means["e002"] - MIN_GAIN:
        return False
    for hold in HOLDS:
        a, b = reports[hold][candidate], reports[hold]["e002"]
        if a["overall"] > b["overall"] + HOLD_TOL:
            return False
        for name, seg in b["segments"].items():
            other = a["segments"][name]
            if seg["n"] >= MIN_SEGMENT and other["n"] >= MIN_SEGMENT \
                    and other["rmse"] > seg["rmse"] + SEGMENT_TOL:
                return False
    return True


def main():
    obs = C.load_obs()
    test_rows = C.test_rows(obs)
    xtest = C.build_features(obs, test_rows)
    hold_preds, truths, hold_x, reports, drift = {}, {}, {}, {}, {}
    index_parts = []
    for label in HOLDS:
        print(f"hold {label}", flush=True)
        work, train, val = holdout(obs, label)
        xtr = C.build_features(work, train)
        xval = C.build_features(work, val)
        xtr_nd = C.build_features(work, train, use_target_depth=False)
        xval_nd = C.build_features(work, val, use_target_depth=False)
        models_e002 = fit_layers(xtr, train, C.FULL)
        models_enh = fit_layers(xtr, train, C.ENHANCED)
        models_nd = fit_layers(xtr_nd, train, C.FULL)
        eof = fit_eof(obs, work, train, xtr)
        val_eof = val.copy(); val_eof["_base"] = xval.base_nom.to_numpy()
        hold_preds[label] = {
            "e002": predict_layers(models_e002, xval, val, C.FULL),
            "enhanced": predict_layers(models_enh, xval, val, C.ENHANCED),
            "eof": predict_eof(eof, obs, work, val_eof),
            "no_depth": predict_layers(models_nd, xval_nd, val, C.FULL),
        }
        truths[label], hold_x[label] = val, xval
        reports[label] = {name: score(val, xval, pred) for name, pred in hold_preds[label].items()}
        drift[label] = drift_summary(xval, val, xtest, test_rows)
        index_parts.append(val[C.KEY + ["temp"]].assign(hold=label))

    blend, blend_weights, final_weight = robust_blend(hold_preds, truths)
    gate, gate_choices, final_gate = gate_blend(hold_preds, truths, hold_x)
    for hold in HOLDS:
        hold_preds[hold]["robust_blend"] = blend[hold]
        hold_preds[hold]["regime_gate"] = gate[hold]
        reports[hold]["robust_blend"] = score(truths[hold], hold_x[hold], blend[hold])
        reports[hold]["regime_gate"] = score(truths[hold], hold_x[hold], gate[hold])

    candidates = ["enhanced", "robust_blend", "regime_gate", "eof"]
    accepted = [name for name in candidates if eligible(name, reports)]
    means = {name: float(np.mean([reports[h][name]["overall"] for h in HOLDS]))
             for name in ["e002", "no_depth", *candidates]}
    preference = {"enhanced": 0, "robust_blend": 1, "regime_gate": 2, "eof": 3}
    selected = "E002"
    if accepted:
        best = min(means[name] for name in accepted)
        selected = min((name for name in accepted if means[name] <= best + 0.002),
                       key=lambda name: preference[name])

    all_index = pd.concat(index_parts, ignore_index=True)
    all_index.to_csv(C.ROOT / "cache" / "E003" / "oof_index.csv", index=False)
    for name in ["e002", "no_depth", *candidates]:
        np.save(C.ROOT / "preds" / f"oof_{name}_E003.npy",
                np.concatenate([hold_preds[h][name] for h in HOLDS]))
    truth_all = all_index.temp.to_numpy()
    correlations = {}
    for i, a in enumerate(["e002", "enhanced", "eof", "robust_blend", "regime_gate"]):
        pa = np.concatenate([hold_preds[h][a] for h in HOLDS])
        for b in ["e002", "enhanced", "eof", "robust_blend", "regime_gate"][i + 1:]:
            pb = np.concatenate([hold_preds[h][b] for h in HOLDS])
            correlations[f"{a}__{b}"] = float(np.corrcoef(truth_all - pa, truth_all - pb)[0, 1])

    print("fit final", flush=True)
    all_rows = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].reset_index(drop=True)
    xall = C.build_features(obs, all_rows)
    xall_nd = C.build_features(obs, all_rows, use_target_depth=False)
    final_eof_rows = all_rows.copy(); final_eof_rows["_base"] = xall.base_nom.to_numpy()
    artifact = {
        "models": {
            "e002": fit_layers(xall, all_rows, C.FULL),
            "enhanced": fit_layers(xall, all_rows, C.ENHANCED),
            "no_depth": fit_layers(xall_nd, all_rows, C.FULL),
            "eof": fit_eof(obs, obs, final_eof_rows, xall),
        },
        "selected": selected, "accepted": selected != "E002", "means": means,
        "reports": reports, "blend_weights_by_hold": blend_weights,
        "final_blend_weight": final_weight, "gate_choices_by_hold": gate_choices,
        "final_gate": final_gate, "correlations": correlations, "drift": drift,
        "seed": C.SEED,
    }
    with open(C.ROOT / "models" / "E003.pkl", "wb") as f:
        pickle.dump(artifact, f)
    payload = {key: artifact[key] for key in ("selected", "accepted", "means", "reports",
                                               "blend_weights_by_hold", "final_blend_weight",
                                               "gate_choices_by_hold", "final_gate", "correlations", "drift")}
    (C.ROOT / "cache" / "E003" / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.ROOT / "cache" / "E003" / "drift.json").write_text(
        json.dumps(drift, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "means": means, "correlations": correlations}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
