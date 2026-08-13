"""P1 exp001 재현 가능한 추론과 제출 생성."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features
from train import fill_short_gaps


def main():
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    ref = json.loads((ROOT / "cache/exp001/reference.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "cache/exp001/meta.json").read_text(encoding="utf-8"))
    x, _ = make_features(test, ref)
    probs = []
    for fold in range(4):
        model = lgb.Booster(model_file=ROOT / f"cache/exp001/model_fold{fold}.txt")
        probs.append(model.predict(x[meta["features"]], num_iteration=model.best_iteration))
    model_score = np.mean(probs, axis=0)
    score = (1 - meta["weight"]) * model_score + meta["weight"] * x.rule_score.to_numpy()
    pred = fill_short_gaps(score >= meta["threshold"], test, meta["fill_gap"])
    stored = np.clip(score - meta["threshold"] + 0.5, 0, 1).astype(np.float32)
    # 후처리된 양성은 생산파일의 0.5 threshold와 정확히 맞춘다.
    stored[pred] = np.maximum(stored[pred], 0.5)
    stored[~pred] = np.minimum(stored[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    np.save(ROOT / "preds/test_lgbm_exp001.npy", stored)
    sub = test[["station", "year", "layer", "time"]].assign(label=pred.astype(int))
    sub.to_csv(ROOT / "subs/exp001.csv", index=False)
    print(f"submission={ROOT / 'subs/exp001.csv'} rows={len(sub):,} positive={pred.sum():,} rate={pred.mean():.4%}")


if __name__ == "__main__":
    main()
