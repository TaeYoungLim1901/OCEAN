"""exp002: exp001 모델 고정, folds 0~2 선택, fold3 sealed outer, exact 생산."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features
from metric import evaluate, f1
from train import fill_short_gaps, segment_ids


def model_scores(x, folds=None):
    score = np.zeros(len(x), np.float64)
    if folds is None:
        preds = []
        for k in range(4):
            m = lgb.Booster(model_file=ROOT / f"models/exp002_fold{k}.txt")
            preds.append(m.predict(x))
        return np.mean(preds, axis=0)
    for k in range(4):
        use = folds == k
        m = lgb.Booster(model_file=ROOT / f"models/exp002_fold{k}.txt")
        score[use] = m.predict(x.loc[use])
    return score


def choose(y, raw_score, rule, raw, folds):
    inner = folds < 3
    seg = segment_ids(raw)
    rows = []
    for weight in (0.0, 0.15, 0.30):
        score = (1 - weight) * raw_score + weight * rule
        for threshold in np.arange(0.10, 0.71, 0.02):
            for hole in (0, 1, 2):
                pred = fill_short_gaps(score >= threshold, raw, hole, seg)
                rows.append((f1(y[inner], pred[inner])[0], weight, float(threshold), hole))
    return max(rows, key=lambda z: z[0])


def store_score(score, pred):
    z = np.clip(score, 0, 1).astype(np.float32)
    z[pred] = np.maximum(z[pred], 0.5)
    z[~pred] = np.minimum(z[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    return z


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def reproduce():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    xtr = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    ref = json.loads((ROOT / "cache/exp001/reference.json").read_text(encoding="utf-8"))
    xte, _ = make_features(test, ref)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    meta = json.loads((ROOT / "cache/exp002_meta.json").read_text(encoding="utf-8"))
    raw_oof, raw_test = model_scores(xtr, folds), model_scores(xte)
    oof_score = (1 - meta["weight"]) * raw_oof + meta["weight"] * xtr.rule_score.to_numpy()
    test_score = (1 - meta["weight"]) * raw_test + meta["weight"] * xte.rule_score.to_numpy()
    oof_pred = fill_short_gaps(oof_score >= meta["threshold"], train, meta["hole"])
    test_pred = fill_short_gaps(test_score >= meta["threshold"], test, meta["hole"])
    expected_oof = store_score(oof_score - meta["threshold"] + 0.5, oof_pred)
    expected_test = store_score(test_score - meta["threshold"] + 0.5, test_pred)
    saved_oof, saved_test = np.load(ROOT / "preds/oof_exp002.npy"), np.load(ROOT / "preds/test_exp002.npy")
    sub = pd.read_csv(ROOT / "subs/exp002.csv")
    fails = []
    if not np.array_equal(expected_oof, saved_oof):
        fails.append(f"OOF mismatch {int((expected_oof != saved_oof).sum())}")
    if not np.array_equal(expected_test, saved_test):
        fails.append(f"test mismatch {int((expected_test != saved_test).sum())}")
    if not np.array_equal(test_pred.astype(int), sub.label.to_numpy()):
        fails.append(f"sub mismatch {int((test_pred.astype(int) != sub.label.to_numpy()).sum())}")
    if sha(ROOT / "subs/exp002.csv") != meta["submission_sha256"]:
        fails.append("submission hash mismatch")
    print(f"  exact_oof={not any('OOF' in x for x in fails)} exact_test={not any('test' in x for x in fails)} sha256={sha(ROOT / 'subs/exp002.csv')}")
    print("판정: " + ("OK" if not fails else "STOP"))
    return int(bool(fails))


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    for k in range(4):
        src, dst = ROOT / f"cache/exp001/model_fold{k}.txt", ROOT / f"models/exp002_fold{k}.txt"
        if not dst.exists():
            shutil.copyfile(src, dst)
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    xtr = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    ref = json.loads((ROOT / "cache/exp001/reference.json").read_text(encoding="utf-8"))
    xte, _ = make_features(test, ref)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    y = train.label.to_numpy(np.int8)
    raw_oof = model_scores(xtr, folds)
    raw_test = model_scores(xte)
    inner_f1, weight, threshold, hole = choose(y, raw_oof, xtr.rule_score.to_numpy(), train, folds)
    oof_score = (1 - weight) * raw_oof + weight * xtr.rule_score.to_numpy()
    test_score = (1 - weight) * raw_test + weight * xte.rule_score.to_numpy()
    oof_pred = fill_short_gaps(oof_score >= threshold, train, hole)
    test_pred = fill_short_gaps(test_score >= threshold, test, hole)
    outer = folds == 3
    outer_stats = f1(y[outer], oof_pred[outer])

    np.save(ROOT / "preds/oof_exp002.npy", store_score(oof_score - threshold + 0.5, oof_pred))
    np.save(ROOT / "preds/test_exp002.npy", store_score(test_score - threshold + 0.5, test_pred))
    sub = test[["station", "year", "layer", "time"]].assign(label=test_pred.astype(int))
    sub.to_csv(ROOT / "subs/exp002.csv", index=False)
    by_type = evaluate(train.loc[outer].assign(anomaly_type=train.loc[outer].anomaly_type.fillna("normal")), oof_pred[outer], ["anomaly_type"])
    by_station = evaluate(train.loc[outer], oof_pred[outer], ["station"])
    by_type.to_csv(ROOT / "experiments/exp002_outer_by_type.csv", index=False)
    by_station.to_csv(ROOT / "experiments/exp002_outer_by_station.csv", index=False)
    meta = {"inner_f1": inner_f1, "outer_f1": outer_stats[0], "outer_tp": outer_stats[1],
            "outer_fp": outer_stats[2], "outer_fn": outer_stats[3], "weight": weight,
            "threshold": threshold, "hole": hole, "features": list(xtr.columns),
            "submission_sha256": sha(ROOT / "subs/exp002.csv")}
    (ROOT / "cache/exp002_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    raise SystemExit(reproduce() if len(sys.argv) > 1 and sys.argv[1] == "reproduce" else main())
