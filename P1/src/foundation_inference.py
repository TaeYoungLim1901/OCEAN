"""exp003 saved-model inference; training stays in foundation.py."""
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
from foundation import sha, store_score
from train import fill_short_gaps


def main():
    meta_path = ROOT / "cache/exp003_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    ref = meta["full_reference"]
    xte, _ = make_features(test, {"median": ref["median"], "mad": ref["mad"]})
    binary = lgb.Booster(model_file=ROOT / "models/exp003_binary_full.txt").predict(
        xte, num_iteration=meta["iterations"]["binary"]
    ).astype(np.float32)
    pm = lgb.Booster(model_file=ROOT / "models/exp003_multiclass_full.txt").predict(
        xte, num_iteration=meta["iterations"]["multiclass"]
    )
    multi = (1.0 - pm[:, 0]).astype(np.float32)
    np.save(ROOT / "preds/test_binary_exp003.npy", binary)
    np.save(ROOT / "preds/test_multiclass_exp003.npy", multi)

    if meta["promoted"]:
        score = (1 - meta["multi_weight"]) * binary + meta["multi_weight"] * multi
        pred = fill_short_gaps(score >= meta["threshold"], test, meta["hole"])
        np.save(ROOT / "preds/test_exp003.npy", store_score(score - meta["threshold"] + 0.5, pred))
        sub = test[["station", "year", "layer", "time"]].assign(label=pred.astype(int))
        sub.to_csv(ROOT / "subs/exp003.csv", index=False)
        meta["submission_sha256"] = sha(ROOT / "subs/exp003.csv")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"submission={ROOT / 'subs/exp003.csv'} positive={int(pred.sum()):,} sha256={meta['submission_sha256']}")
    else:
        print("exp003 not promoted; exp002 remains final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
