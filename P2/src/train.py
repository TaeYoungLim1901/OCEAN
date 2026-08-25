"""E001: 연속 2개월 mask CV, 잔차 모델 학습과 사전규칙 선택."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from submission_tools import save_pool_pred


def estimator(name):
    if name == "ridge_full":
        return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0))
    return LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.035,
        num_leaves=31, min_child_samples=100, reg_lambda=1.0,
        colsample_bytree=0.9, random_state=C.SEED, n_jobs=2, verbosity=-1,
    )


def fit_predict(name, columns, xtr, train, xval, val):
    pred = np.empty(len(val))
    models = {}
    for layer in C.TARGET_LAYERS:
        a, b = train.layer.eq(layer), val.layer.eq(layer)
        model = estimator(name)
        model.fit(xtr.loc[a, columns], train.loc[a, "temp"] - xtr.loc[a, "base_nom"])
        pred[b] = xval.loc[b, "base_nom"] + model.predict(xval.loc[b, columns])
        models[layer] = model
    return pred, models


def scores(val, xval, pred):
    out = {"overall": C.rmse(val.temp, pred)}
    out["layers"] = {str(layer): C.rmse(val.loc[val.layer.eq(layer), "temp"], pred[val.layer.eq(layer)])
                     for layer in C.TARGET_LAYERS}
    threshold = xval.transition.abs().quantile(0.75)
    transition = xval.transition.abs().ge(threshold).to_numpy()
    out["transition"] = C.rmse(val.loc[transition, "temp"], pred[transition])
    out["worst_layer"] = max(out["layers"].values())
    return out


def main():
    obs = C.load_obs()
    train, val = C.split_rows(obs)
    hold = obs.time.between(val.time.min(), val.time.max()) & obs.layer.isin(C.TARGET_LAYERS)
    obs_train = obs.loc[~hold].copy()
    feature_source = obs_train
    xtr, xval = C.build_features(feature_source, train), C.build_features(feature_source, val)
    predictions = {"baseline_nom": xval.base_nom.to_numpy(), "baseline_actual": xval.base_actual.to_numpy()}
    fitted = {}
    for name, columns in (("ridge_full", C.FULL), ("lgb_basic", C.BASIC), ("lgb_full", C.FULL)):
        predictions[name], fitted[name] = fit_predict(name, columns, xtr, train, xval, val)
    for a, b in (("lgb_basic", "lgb_full"), ("ridge_full", "lgb_full")):
        for w in (0.25, 0.5, 0.75):
            predictions[f"blend_{a}_{b}_{w:.2f}"] = (1 - w) * predictions[a] + w * predictions[b]

    report = {name: scores(val, xval, pred) for name, pred in predictions.items()}
    base = report["baseline_nom"]
    eligible = [name for name, s in report.items() if name not in {"baseline_nom", "baseline_actual"}
                and s["overall"] < base["overall"] and s["worst_layer"] <= base["worst_layer"]
                and s["transition"] <= base["transition"]]
    if not eligible:
        eligible = [name for name in report if name not in {"baseline_nom", "baseline_actual"}]
    best_score = min(report[name]["overall"] for name in eligible)
    near = [name for name in eligible if report[name]["overall"] <= best_score + 0.002]
    order = {"ridge_full": 0, "lgb_basic": 1, "lgb_full": 2}
    selected = min(near, key=lambda n: (n.startswith("blend_"), order.get(n, 9), report[n]["overall"]))

    for name, pred in predictions.items():
        save_pool_pred(C.ROOT / "preds" / f"oof_{name}_E001.npy", pred)
    answer = val[C.KEY + ["temp"]]
    answer.to_csv(C.ROOT / "cache" / "E001" / "val_answer.csv", index=False)
    val[C.KEY].assign(temp=predictions[selected]).to_csv(C.ROOT / "cache" / "E001" / "val_selected.csv", index=False)

    # 최종 모델은 배포된 모든 라벨을 사용한다.
    all_rows = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].reset_index(drop=True)
    feature_source = obs
    xall = C.build_features(feature_source, all_rows)
    final = {}
    for name, columns in (("ridge_full", C.FULL), ("lgb_basic", C.BASIC), ("lgb_full", C.FULL)):
        final[name] = {}
        for layer in C.TARGET_LAYERS:
            use = all_rows.layer.eq(layer)
            model = estimator(name)
            model.fit(xall.loc[use, columns], all_rows.loc[use, "temp"] - xall.loc[use, "base_nom"])
            final[name][layer] = model
    artifact = {"models": final, "selected": selected, "report": report,
                "basic": C.BASIC, "full": C.FULL, "seed": C.SEED}
    with open(C.ROOT / "models" / "E001.pkl", "wb") as f:
        pickle.dump(artifact, f)
    (C.ROOT / "cache" / "E001" / "metrics.json").write_text(
        json.dumps({"selected": selected, "scores": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "selected_score": report[selected], "baseline": base,
                      "delta": report[selected]["overall"] - base["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
