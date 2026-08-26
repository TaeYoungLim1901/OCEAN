"""T23: season-aligned, invariant-feature, and rank-ensemble probes."""
import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp004 import fill_runs
from exp005 import store_score
from exp007_postmortem import model_params
from features import SEED, make_features
from submission_tools import digest, save_pool_pred
from train import segment_ids


N_ESTIMATORS = 700
HOLE = 48
TARGET_RATE = 0.0300
SEASON_MONTHS = (12, 1, 2, 3, 4, 5, 6)
REMOVED_FEATURES = {"month", "doy_sin", "doy_cos", "clim_resid", "layer_count"}
BASE_FEATURES = ROOT / "cache/exp001/train_features.pkl"
REFERENCE = ROOT / "cache/exp001/reference.json"


def load_base_frames(train, test):
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    x_train = pd.read_pickle(BASE_FEATURES)
    x_test = make_features(test, reference)[0]
    assert len(x_train) == len(train) and list(x_train.columns) == list(x_test.columns)
    assert len(x_train.columns) == 42
    return x_train, x_test, reference


def fit_full(x_train, y, columns, mask):
    model = lgb.LGBMClassifier(**model_params(N_ESTIMATORS))
    model.fit(x_train.loc[mask, columns], y[mask], callbacks=[lgb.log_evaluation(0)])
    return model.booster_


def postprocessed(raw, segments, threshold):
    return fill_runs(np.asarray(raw) >= threshold, segments, HOLE)


def choose_threshold(raw, segments):
    grid = np.round(np.arange(0.0, 1.0001, 0.001), 3)
    candidates = [(float(t), float(postprocessed(raw, segments, t).mean())) for t in grid]
    return min(candidates, key=lambda row: (abs(row[1] - TARGET_RATE), -row[0]))


def rank01(values):
    values = np.asarray(values, dtype=np.float64)
    ranks = rankdata(values, method="average")
    return (ranks - 1.0) / max(1, len(values) - 1)


def paths(exp):
    return {
        "raw": ROOT / f"cache/{exp}_raw_test.npy",
        "pred": ROOT / f"preds/test_{exp}.npy",
        "sub": ROOT / f"subs/{exp}.csv",
        "meta": ROOT / f"cache/{exp}_meta.json",
    }


def save_submission(exp, raw, feature_names, train_rows):
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    raw = np.asarray(raw, dtype=np.float64)
    assert raw.shape == (len(test),) and np.isfinite(raw).all()
    segments = segment_ids(test)
    threshold, _ = choose_threshold(raw, segments)
    base_pred = raw >= threshold
    test_pred = postprocessed(raw, segments, threshold)
    raw_rate = float(base_pred.mean())
    post_rate = float(test_pred.mean())
    artifact = paths(exp)

    np.save(artifact["raw"], raw)
    sub = test[["station", "year", "layer", "time"]].assign(label=test_pred.astype(int))
    sub.to_csv(artifact["sub"], index=False)
    save_pool_pred(artifact["pred"], store_score(raw, test_pred, threshold))
    artifact["meta"].write_text(
        json.dumps(
            {
                "t": threshold,
                "threshold": threshold,
                "t_high": threshold,
                "hole": HOLE,
                "raw_rate": raw_rate,
                "post_rate": post_rate,
                "n_positive": int(test_pred.sum()),
                "positive_rate": post_rate,
                "feature_count": len(feature_names),
                "features": list(feature_names),
                "train_rows": int(train_rows),
                "seed": SEED,
                "sha256": digest(artifact["sub"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("exp | n_features | train_rows | t_high | raw_rate | post_rate | n_positive | sha256")
    print(f"{exp} | {len(feature_names)} | {train_rows} | {threshold:.3f} | {raw_rate:.6f} | {post_rate:.6f} | {int(test_pred.sum())} | {digest(artifact['sub'])[:10]}")
    subprocess.run(
        [sys.executable, "P1/src/submission_tools.py", "audit", f"P1/subs/{exp}.csv", "--pred", f"P1/preds/test_{exp}.npy"],
        cwd=ROOT.parent,
        check=True,
    )
    subprocess.run([sys.executable, "P1/src/check.py", "sub", f"P1/subs/{exp}.csv"], cwd=ROOT.parent, check=True)


def train_stage(stage):
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    x_train, x_test, _ = load_base_frames(train, test)
    y = train.label.to_numpy(np.int8)

    if stage == "season":
        columns = list(x_train.columns)
        mask = np.isin(pd.to_datetime(train.time, utc=True).dt.month.to_numpy(), SEASON_MONTHS)
        assert 340_000 <= int(mask.sum()) <= 350_000
        exp = "exp013"
    else:
        columns = [name for name in x_train.columns if name not in REMOVED_FEATURES]
        assert len(columns) == 37
        mask = np.ones(len(train), dtype=bool)
        exp = "exp014"

    assert list(x_train[columns].columns) == columns
    assert list(x_test[columns].columns) == columns
    model = fit_full(x_train, y, columns, mask)
    raw = model.predict(x_test[columns]).astype(np.float64, copy=False)
    save_submission(exp, raw, columns, int(mask.sum()))


def ensemble_stage():
    exp005 = np.load(ROOT / "cache/exp005_raw_test.npy", allow_pickle=False).astype(np.float64, copy=False)
    exp012 = np.load(ROOT / "cache/exp012_raw_test.npy", allow_pickle=False).astype(np.float64, copy=False)
    assert exp005.shape == exp012.shape and np.isfinite(exp005).all() and np.isfinite(exp012).all()
    raw = ((rank01(exp005) + rank01(exp012)) / 2.0).astype(np.float64, copy=False)
    save_submission("exp015", raw, ["exp005_rank01", "exp012_rank01"], 0)


def self_check():
    assert np.allclose(rank01([3.0, 1.0, 2.0]), [1.0, 0.0, 0.5])
    assert np.array_equal(fill_runs(np.array([True, False, True, False]), np.array([0, 0, 0, 1]), 1), [True, True, True, False])
    print("self-check | OK")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"season", "invariant", "ensemble", "--self-check"}:
        raise SystemExit("usage: python P1/src/exp013_shift.py season|invariant|ensemble|--self-check")
    if sys.argv[1] == "--self-check":
        self_check()
    elif sys.argv[1] == "ensemble":
        ensemble_stage()
    else:
        train_stage(sys.argv[1])


if __name__ == "__main__":
    main()
