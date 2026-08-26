"""T20 exp012: V1 full-train scores and positive-rate anchored submission."""
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_t4 import forward_mask
from exp004 import fill_runs
from exp005 import model_audit, store_score
from exp007 import prepare_features
from exp007_postmortem import importance, model_params
from features import INTEGRATED_FEATURES, SEED, add_integrated_features, fit_reference
from metric import f1
from submission_tools import digest, save_pool_pred
from train import segment_ids


N_ESTIMATORS = 700
HOLE = 48
TARGET_RATE = 0.0302
EXPECTED_TOP = {
    "temp_step_abs_144",
    "max_z_center_144",
    "temp_step_144",
    "temp_step_abs_48",
    "psal_step_abs_144",
}
INTEGRATED_MODEL = ROOT / "models/exp007_full.txt"
MODEL_DIR = ROOT / "models"
RAW_OOF = ROOT / "cache/exp012_raw_oof.npy"
RAW_TEST = ROOT / "cache/exp012_raw_test.npy"
META = ROOT / "cache/exp012_meta.json"


def feature_spec():
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    base_names = list(base.columns)
    assert len(base_names) == 42
    imp = importance(INTEGRATED_MODEL)
    order = np.argsort(-imp["gain"], kind="stable")
    top = [str(imp["names"][i]) for i in order if imp["names"][i] in INTEGRATED_FEATURES][:5]
    assert set(top) == EXPECTED_TOP, top
    columns = base_names + top
    assert len(columns) == 47 and len(set(columns)) == 47
    return columns, top


def fit_or_load(x, y, mask, path, columns):
    if path.exists():
        return lgb.Booster(model_file=path)
    model = lgb.LGBMClassifier(**model_params(N_ESTIMATORS))
    model.fit(x.loc[mask, columns], y[mask], callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(path)
    return model.booster_


def train_stage():
    started = time.monotonic()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    columns, top = feature_spec()
    x_train, x_test, _ = prepare_features(train, test)
    assert list(x_train[columns].columns) == columns
    assert list(x_test[columns].columns) == columns

    y = train.label.to_numpy(np.int8)
    folds = np.load(ROOT / "cache/exp001/folds.npy", allow_pickle=False)
    raw_oof = np.full(len(train), np.nan, dtype=np.float64)
    MODEL_DIR.mkdir(exist_ok=True)

    for fold in (1, 2):
        valid = folds == fold
        train_mask = forward_mask(train, folds, fold)
        fold_reference = fit_reference(train, train_mask)
        x_fold = add_integrated_features(x_train, train, fold_reference)
        path = MODEL_DIR / f"exp012_forward_fold{fold}.txt"
        model = fit_or_load(x_fold, y, train_mask, path, columns)
        raw_oof[valid] = model.predict(x_fold.loc[valid, columns])
        print(f"fold={fold} elapsed_seconds={time.monotonic() - started:.1f}")

    full_mask = np.ones(len(train), dtype=bool)
    full_path = MODEL_DIR / "exp012_full.txt"
    full_model = fit_or_load(x_train, y, full_mask, full_path, columns)
    raw_test = np.asarray(full_model.predict(x_test[columns]), dtype=np.float64)
    print(f"full elapsed_seconds={time.monotonic() - started:.1f}")
    np.save(RAW_OOF, raw_oof)
    np.save(RAW_TEST, raw_test)

    audits = [
        model_audit("forward_fold1", train, forward_mask(train, folds, 1), folds == 1),
        model_audit("forward_fold2", train, forward_mask(train, folds, 2), folds == 2),
    ]
    tm_train = pd.to_datetime(train.time, utc=True)
    tm_test = pd.to_datetime(test.time, utc=True)
    audits.append({
        "model": "full_test",
        "train_rows": int(full_mask.sum()),
        "train_max_time": str(tm_train.max()),
        "predict_min_time": str(tm_test.min()),
        "future_rows": int((tm_train >= tm_test.min()).sum()),
    })
    assert all(row["future_rows"] == 0 for row in audits), audits

    print(f"feature_count={len(columns)} top_integrated={','.join(top)}")
    print("model | train_rows | train_max_time | predict_min_time | future_rows")
    for row in audits:
        print("{model} | {train_rows} | {train_max_time} | {predict_min_time} | {future_rows}".format(**row))
    print(f"raw_oof_nan_rows={int(np.isnan(raw_oof).sum())} elapsed_seconds={time.monotonic() - started:.1f}")


def postprocessed(raw, segments, threshold):
    return fill_runs(np.asarray(raw) >= threshold, segments, HOLE)


def submit_stage():
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    raw_test = np.load(RAW_TEST, allow_pickle=False).astype(np.float64, copy=False)
    assert raw_test.shape == (len(test),) and np.isfinite(raw_test).all()
    test_segments = segment_ids(test)

    if META.exists():
        threshold = float(json.loads(META.read_text(encoding="utf-8"))["threshold"])
    else:
        grid = np.round(np.arange(0.0, 1.0001, 0.001), 3)
        candidates = [(float(t), float(postprocessed(raw_test, test_segments, t).mean())) for t in grid]
        threshold, _ = min(candidates, key=lambda row: (abs(row[1] - TARGET_RATE), -row[0]))

    base_pred = raw_test >= threshold
    test_pred = postprocessed(raw_test, test_segments, threshold)
    raw_rate = float(base_pred.mean())
    post_rate = float(test_pred.mean())
    print("t_high | raw_rate | post_rate | n_positive")
    print(f"{threshold:.3f} | {raw_rate:.6f} | {post_rate:.6f} | {int(test_pred.sum())}")
    print("t_high | post_rate")
    for delta in (-0.02, -0.01, 0.0, 0.01, 0.02):
        t = round(threshold + delta, 3)
        print(f"{t:.3f} | {postprocessed(raw_test, test_segments, t).mean():.6f}")

    train = pd.read_csv(ROOT / "P1_data/train.csv")
    folds = np.load(ROOT / "cache/exp001/folds.npy", allow_pickle=False)
    raw_oof = np.load(RAW_OOF, allow_pickle=False).astype(np.float64, copy=False)
    inner = np.isin(folds, [1, 2])
    oof_pred = postprocessed(raw_oof, segment_ids(train), threshold)
    score, tp, fp, fn = f1(train.label.to_numpy(np.int8)[inner], oof_pred[inner])
    print(f"inner_f1_reference={score:.6f} TP={tp} FP={fp} FN={fn}")

    key = ["station", "year", "layer", "time"]
    sub = test[key].assign(label=test_pred.astype(int))
    template = pd.read_csv(ROOT / "subs/exp010.csv", nrows=0)
    assert list(sub.columns) == list(template.columns)
    sub_path = ROOT / "subs/exp012.csv"
    sub.to_csv(sub_path, index=False)
    save_pool_pred(ROOT / "preds/test_exp012.npy", store_score(raw_test, test_pred, threshold))

    feature_names = lgb.Booster(model_file=MODEL_DIR / "exp012_full.txt").feature_name()
    assert len(feature_names) == 47
    META.write_text(json.dumps({
        "threshold": threshold,
        "t_high": threshold,
        "hole": HOLE,
        "n_positive": int(test_pred.sum()),
        "positive_rate": post_rate,
        "feature_count": len(feature_names),
        "features": feature_names,
        "seed": SEED,
        "sha256": digest(sub_path),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, "P1/src/submission_tools.py", "audit", "P1/subs/exp012.csv", "--pred", "P1/preds/test_exp012.npy"],
        cwd=ROOT.parent, check=True,
    )
    subprocess.run([sys.executable, "P1/src/check.py", "sub", "P1/subs/exp012.csv"], cwd=ROOT.parent, check=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "submit"}:
        raise SystemExit("usage: python P1/src/exp012_v1_full.py train|submit")
    {"train": train_stage, "submit": submit_stage}[sys.argv[1]]()


if __name__ == "__main__":
    main()
