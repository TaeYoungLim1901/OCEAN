"""Cache-free exp002 reproduction from source data and model weights."""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
WEIGHT = 0.15
THRESHOLD = 0.44
HOLE = 2
KEY = ["station", "year", "layer", "time"]


def legacy_reference(raw):
    d = raw.copy()
    d["_time"] = pd.to_datetime(d.time, utc=True)
    d = d.sort_values(["station", "layer", "_time"], kind="stable")
    month = d["_time"].dt.month
    z = pd.DataFrame({"station": d.station, "layer": d.layer, "month": month, "temp": d.temp.to_numpy()})
    med = z.groupby(["station", "layer", "month"]).temp.median()
    joined = z.join(med.rename("med"), on=["station", "layer", "month"])
    mad = joined.assign(dev=(joined.temp - joined.med).abs()).groupby(
        ["station", "layer", "month"]
    ).dev.median()
    return {
        "median": {"|".join(map(str, key)): float(value) for key, value in med.items()},
        "mad": {"|".join(map(str, key)): float(value) for key, value in mad.items()},
    }


def main(data_root, weights_dir, output):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import make_features, time_folds
    from train import fill_short_gaps

    data_root, weights_dir = Path(data_root), Path(weights_dir)
    train = pd.read_csv(data_root / "train.csv")
    test = pd.read_csv(data_root / "test.csv")
    xtr, reference = make_features(train, legacy_reference(train))
    xte, _ = make_features(test, reference)
    folds = time_folds(train)
    models = [lgb.Booster(model_file=weights_dir / f"exp002_fold{k}.txt") for k in range(4)]
    oof = np.zeros(len(train), dtype=float)
    for fold, model in enumerate(models):
        use = folds == fold
        oof[use] = model.predict(xtr.loc[use])
    test_score = np.mean([model.predict(xte) for model in models], axis=0)
    oof_score = (1 - WEIGHT) * oof + WEIGHT * xtr.rule_score.to_numpy()
    test_score = (1 - WEIGHT) * test_score + WEIGHT * xte.rule_score.to_numpy()
    fill_short_gaps(oof_score >= THRESHOLD, train, HOLE)
    test_pred = fill_short_gaps(test_score >= THRESHOLD, test, HOLE)
    pd.DataFrame(test[KEY].assign(label=test_pred.astype(int))).to_csv(output, index=False)
    print(f"rows={len(test_pred):,} positive={int(test_pred.sum()):,}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    raise SystemExit(main(args.data_root, args.weights_dir, args.output))
