"""P1 exp001: gap-safe feature + 4-block LightGBM OOF."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features, time_folds, train_mask_with_purge, SEED
from metric import evaluate, f1


def segment_ids(raw):
    tm = pd.to_datetime(raw.time, utc=True)
    stream = pd.factorize(pd.MultiIndex.from_arrays([raw.station, raw.layer]), sort=False)[0]
    new = pd.Series(stream).ne(pd.Series(stream).shift()) | tm.diff().dt.total_seconds().ne(600)
    # stream 코드와 stream 내부 gap 번호를 결합해 다른 stream이 절대 같은 ID를 공유하지 않는다.
    gap = new.groupby(stream).cumsum().to_numpy(np.int64)
    return stream.astype(np.int64) * (len(raw) + 1) + gap


def fill_short_gaps(pred, raw, max_gap, segments=None):
    if max_gap == 0:
        return pred.copy()
    out = pred.copy()
    seg = segment_ids(raw) if segments is None else segments
    for holes in range(1, max_gap + 1):
        width = holes + 1
        bridge = out[:-width] & out[width:] & (seg[:-width] == seg[width:])
        starts = np.flatnonzero(bridge)
        for offset in range(1, width):
            out[starts + offset] = True
    return out


def main():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    cache = ROOT / "cache/exp001"
    cached = (cache / "train_features.pkl").exists() and all((cache / f"model_fold{i}.txt").exists() for i in range(4))
    if cached:
        x = pd.read_pickle(cache / "train_features.pkl")
        ref = json.loads((cache / "reference.json").read_text(encoding="utf-8"))
    else:
        x, ref = make_features(train)
    y = train.label.to_numpy(np.int8)
    folds = time_folds(train)
    np.save(ROOT / "cache/exp001/folds.npy", folds)
    (ROOT / "cache/exp001/reference.json").write_text(json.dumps(ref), encoding="utf-8")
    x.to_pickle(ROOT / "cache/exp001/train_features.pkl")
    oof_path = ROOT / "preds/oof_lgbm_exp001.npy"
    oof = np.zeros(len(train), np.float32)
    best_iters = []
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, max_depth=-1,
                  min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
                  reg_lambda=1.0, n_estimators=700, random_state=SEED,
                  n_jobs=2, verbosity=-1, scale_pos_weight=2.0)
    for fold in ([] if cached else range(4)):
        va = folds == fold
        tr = train_mask_with_purge(train, folds, fold)
        print(f"fold {fold}: train={tr.sum():,} valid={va.sum():,} purge={(~tr & ~va).sum():,}")
        model = lgb.LGBMClassifier(**params)
        model.fit(x.loc[tr], y[tr], eval_set=[(x.loc[va], y[va])],
                  callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
        oof[va] = model.predict_proba(x.loc[va], num_iteration=model.best_iteration_)[:, 1]
        best_iters.append(model.best_iteration_)
        model.booster_.save_model(ROOT / f"cache/exp001/model_fold{fold}.txt")
    if cached:
        for fold in range(4):
            va = folds == fold
            model = lgb.Booster(model_file=ROOT / f"cache/exp001/model_fold{fold}.txt")
            oof[va] = model.predict(x.loc[va], num_iteration=model.best_iteration)
            best_iters.append(model.best_iteration)
    np.save(oof_path, oof)

    # 결과를 보기 전 고정한 작은 grid: threshold, 규칙 혼합, event 내부 1~2행 hole만 비교.
    rule = x.rule_score.to_numpy()
    candidates = []
    segments = segment_ids(train)
    for weight in (0.0, 0.15, 0.30):
        score = (1 - weight) * oof + weight * rule
        for threshold in np.arange(0.10, 0.71, 0.02):
            for gap in (0, 1, 2):
                pred = fill_short_gaps(score >= threshold, train, gap, segments)
                candidates.append((*f1(y, pred), weight, float(threshold), gap))
    best = max(candidates, key=lambda z: z[0])
    score, tp, fp, fn, weight, threshold, gap = best
    final_score = (1 - weight) * oof + weight * rule
    final_pred = fill_short_gaps(final_score >= threshold, train, gap, segments)
    # 저장 점수는 inference가 0.5 threshold로 제출하도록 선택 임계값 중심으로 재척도.
    oof_submit = np.clip(final_score - threshold + 0.5, 0, 1).astype(np.float32)
    oof_submit[final_pred] = np.maximum(oof_submit[final_pred], 0.5)
    oof_submit[~final_pred] = np.minimum(oof_submit[~final_pred], np.nextafter(np.float32(0.5), np.float32(0)))
    np.save(ROOT / "preds/oof_lgbm_exp001.npy", oof_submit)
    meta = {"weight": weight, "threshold": threshold, "fill_gap": gap,
            "best_iterations": best_iters, "oof_f1": score, "tp": tp, "fp": fp, "fn": fn,
            "features": list(x.columns)}
    (ROOT / "cache/exp001/meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    type_scores = evaluate(train.assign(anomaly_type=train.anomaly_type.fillna("normal")), final_pred, ["anomaly_type"])
    sl_scores = evaluate(train, final_pred, ["station", "layer"])
    type_scores.to_csv(ROOT / "experiments/exp001_by_type.csv", index=False)
    sl_scores.to_csv(ROOT / "experiments/exp001_by_station_layer.csv", index=False)
    rule_pred = rule >= 0.5
    rule_f1 = f1(y, rule_pred)[0]
    err_corr = np.corrcoef((y - (oof >= threshold)), (y - rule_pred))[0, 1]
    print(f"OOF F1={score:.6f} TP={tp} FP={fp} FN={fn} weight={weight:.2f} threshold={threshold:.2f} gap={gap}")
    print(f"rule F1={rule_f1:.6f} error_corr={err_corr:.4f} best_iters={best_iters}")
    print(type_scores.sort_values("positive", ascending=False).head(8).to_string(index=False))
    print(sl_scores.sort_values("f1").head(3).to_string(index=False))


if __name__ == "__main__":
    main()
