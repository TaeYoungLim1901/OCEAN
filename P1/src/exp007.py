"""T12 exp007: integrated residual features and inner-only postprocessing grid."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_t4 import forward_mask
from exp004 import fill_runs
from features import INTEGRATED_FEATURES, SEED, add_integrated_features, fit_reference, make_features
from metric import evaluate, f1
from submission_tools import save_pool_pred
from train import segment_ids


N_ESTIMATORS = 700
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.9001, 0.02), 2))
HOLES = (0, 12, 24, 48, 96)
DILATIONS = (0, 2, 6, 12, 24)
MIN_LENGTHS = (0, 3, 6, 12)
KEY = ["station", "layer"]
STREAM_KEY = ["station", "year", "layer"]


def params(n_estimators=N_ESTIMATORS):
    return dict(
        objective="binary", learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, n_estimators=n_estimators, random_state=SEED,
        n_jobs=2, verbosity=-1, scale_pos_weight=2.0,
    )


def train_model(x, y, mask, path, n_estimators=N_ESTIMATORS):
    model = lgb.LGBMClassifier(**params(n_estimators))
    model.fit(x.loc[mask], y[mask], callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(path)
    return model.booster_


def segment_indices(segments):
    groups = {}
    for i, value in enumerate(segments):
        groups.setdefault(int(value), []).append(i)
    return [np.asarray(index, dtype=np.int64) for index in groups.values()]


def dilate(pred, segments, radius):
    out = np.asarray(pred, dtype=bool).copy()
    if radius == 0:
        return out
    for index in segment_indices(segments):
        z = out[index]
        starts = np.flatnonzero(z & np.r_[True, ~z[:-1]])
        ends = np.flatnonzero(z & np.r_[~z[1:], True])
        for start, end in zip(starts, ends):
            out[index[max(0, start - radius):min(len(index), end + radius + 1)]] = True
    return out


def remove_short_runs(pred, segments, minimum):
    out = np.asarray(pred, dtype=bool).copy()
    if minimum == 0:
        return out
    for index in segment_indices(segments):
        z = out[index]
        starts = np.flatnonzero(z & np.r_[True, ~z[:-1]])
        ends = np.flatnonzero(z & np.r_[~z[1:], True]) + 1
        for start, end in zip(starts, ends):
            if end - start < minimum:
                out[index[start:end]] = False
    return out


def postprocess(score, segments, threshold, hole, dilation, minimum):
    pred = fill_runs(np.asarray(score) >= threshold, segments, hole)
    pred = dilate(pred, segments, dilation)
    return remove_short_runs(pred, segments, minimum)


def choose_grid(y, score, inner, segments):
    rows = []
    for threshold in THRESHOLDS:
        for hole in HOLES:
            for dilation in DILATIONS:
                for minimum in MIN_LENGTHS:
                    pred = postprocess(score, segments, threshold, hole, dilation, minimum)
                    score_f1 = f1(y[inner], pred[inner])[0]
                    rows.append({
                        "threshold": float(threshold),
                        "hole": int(hole),
                        "dilation": int(dilation),
                        "min_length": int(minimum),
                        "inner_f1": float(score_f1),
                    })
    return sorted(
        rows,
        key=lambda row: (
            -row["inner_f1"], row["hole"], row["dilation"],
            row["min_length"], -row["threshold"],
        ),
    )


def type_recall(train, pred, inner):
    frame = train.loc[inner].copy()
    frame["anomaly_type"] = frame.anomaly_type.fillna("normal")
    pred_inner = np.asarray(pred)[inner]
    table = evaluate(frame, pred_inner, ["anomaly_type"])
    rows = []
    for row in table.itertuples(index=False):
        positive = int(row.positive)
        typed = frame.anomaly_type.to_numpy() == str(row.anomaly_type)
        rows.append({
            "split": "inner",
            "anomaly_type": str(row.anomaly_type),
            "n_rows_true": positive,
            "TP": int(row.tp),
            "FN": int(row.fn),
            "recall": int(row.tp) / max(1, positive),
            "pred_positive_rows": int(pred_inner[typed].sum()),
            "FP": int(row.fp),
        })
    return rows


def prepare_features(train, test):
    full_reference = fit_reference(train)
    base_path = ROOT / "cache/exp001/train_features.pkl"
    base_train = pd.read_pickle(base_path) if base_path.exists() else make_features(train, full_reference)[0]
    base_test = make_features(test, full_reference)[0]
    x_train = add_integrated_features(base_train, train, full_reference)
    x_test = add_integrated_features(base_test, test, full_reference)
    return x_train, x_test, full_reference


def smoke():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    groups = list(train.groupby(STREAM_KEY, sort=False).groups.items())
    selected = next(
        (set(key for key, _ in groups[start:start + 3]) for start in range(max(1, len(groups) - 2))
         if train.loc[np.concatenate([np.asarray(index) for _, index in groups[start:start + 3]]), "label"].nunique() == 2),
        set(key for key, _ in groups[:3]),
    )
    mask = np.asarray(train.set_index(STREAM_KEY).index.isin(selected), dtype=bool)
    parts = []
    for _, group in train.loc[mask].groupby(STREAM_KEY, sort=False):
        part = group.head(600)
        if part.label.nunique() < 2 and group.label.nunique() == 2:
            part = pd.concat([part, group[group.label == 1].head(5)], ignore_index=True)
        parts.append(part)
    small_train = pd.concat(parts, ignore_index=True)
    small_test = small_train.drop(columns=["label", "anomaly_type"]).copy()
    small_test["year"] = small_test["year"] + 1
    reference = fit_reference(small_train)
    x_train = make_features(small_train, reference, integrated=True)[0]
    x_test = make_features(small_test, reference, integrated=True)[0]
    model = train_model(x_train, small_train.label.to_numpy(np.int8), np.ones(len(small_train), dtype=bool), ROOT / "cache/exp007_smoke.txt", 20)
    prediction = model.predict(x_test)
    assert prediction.shape == (len(small_test),)
    assert list(x_train.columns) == list(x_test.columns)
    print(f"smoke_streams={len(selected)} smoke_trees=20 rows={len(small_train)}")
    print(f"integrated_feature_count={len(INTEGRATED_FEATURES)} total_feature_count={len(x_train.columns)}")
    print("integrated_feature_names=" + ",".join(INTEGRATED_FEATURES))
    print(f"smoke_prediction_range={prediction.min():.6f},{prediction.max():.6f}")
    print("판정: OK")


def main():
    if "--smoke" in sys.argv:
        smoke()
        return 0

    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    y = train.label.to_numpy(np.int8)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    x_train, x_test, full_reference = prepare_features(train, test)
    (ROOT / "cache/exp007").mkdir(parents=True, exist_ok=True)
    (ROOT / "cache/exp007/reference.json").write_text(
        json.dumps(full_reference, ensure_ascii=False), encoding="utf-8"
    )

    raw_oof = np.full(len(train), np.nan, dtype=np.float64)
    for fold in (1, 2):
        inner_mask = folds == fold
        train_mask = forward_mask(train, folds, fold)
        reference = fit_reference(train, train_mask)
        x_forward = add_integrated_features(x_train, train, reference)
        model_path = ROOT / f"models/exp007_forward_fold{fold}.txt"
        model = train_model(x_forward, y, train_mask, model_path)
        raw_oof[inner_mask] = model.predict(x_forward.loc[inner_mask])
        print(f"fold={fold} train_rows={int(train_mask.sum())} valid_rows={int(inner_mask.sum())}")

    full_mask = np.ones(len(train), dtype=bool)
    full_model = train_model(x_train, y, full_mask, ROOT / "models/exp007_full.txt")
    raw_test = full_model.predict(x_test)
    np.save(ROOT / "cache/exp007_raw_oof.npy", raw_oof)
    np.save(ROOT / "cache/exp007_raw_test.npy", raw_test)

    inner = np.isin(folds, [1, 2])
    segments = segment_ids(train)
    test_segments = segment_ids(test)
    grid = choose_grid(y, raw_oof, inner, segments)
    selected = grid[0]
    oof_pred = postprocess(raw_oof, segments, selected["threshold"], selected["hole"], selected["dilation"], selected["min_length"])
    test_pred = postprocess(raw_test, test_segments, selected["threshold"], selected["hole"], selected["dilation"], selected["min_length"])
    oof_out = np.zeros(len(train), dtype=np.uint8)
    oof_out[inner] = oof_pred[inner]
    test_out = test_pred.astype(np.uint8)
    stored_oof = np.clip(np.nan_to_num(raw_oof, nan=0.0) - selected["threshold"] + 0.5, 0, 1).astype(np.float32)
    stored_oof[oof_pred] = np.maximum(stored_oof[oof_pred], 0.5)
    stored_oof[~oof_pred] = np.minimum(stored_oof[~oof_pred], np.nextafter(np.float32(0.5), np.float32(0)))
    stored_test = np.clip(raw_test - selected["threshold"] + 0.5, 0, 1).astype(np.float32)
    stored_test[test_pred] = np.maximum(stored_test[test_pred], 0.5)
    stored_test[~test_pred] = np.minimum(stored_test[~test_pred], np.nextafter(np.float32(0.5), np.float32(0)))
    save_pool_pred(ROOT / "preds/oof_exp007.npy", stored_oof)
    save_pool_pred(ROOT / "preds/test_exp007.npy", stored_test)

    key = ["station", "year", "layer", "time"]
    sub = test[key].assign(label=test_out.astype(int))
    sample = pd.read_csv(ROOT / "P1_data/sample_submission.csv", usecols=key)
    assert sub[key].reset_index(drop=True).equals(sample.reset_index(drop=True))
    sub.to_csv(ROOT / "subs/exp007.csv", index=False)

    type_rows = type_recall(train, oof_pred, inner)
    grid_output = {
        "seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "folds": [1, 2],
        "purge_hours": 87,
        "inner_f1": float(selected["inner_f1"]),
        "selected": selected,
        "feature_count": int(len(x_train.columns)),
        "integrated_feature_count": int(len(INTEGRATED_FEATURES)),
        "integrated_features": list(INTEGRATED_FEATURES),
        "type_recall": type_rows,
        "grid": grid,
    }
    (ROOT / "experiments/exp007_grid.json").write_text(
        json.dumps(grid_output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"integrated_feature_count={len(INTEGRATED_FEATURES)} total_feature_count={len(x_train.columns)}")
    print(f"grid_rows={len(grid)} selected_threshold={selected['threshold']:.2f} hole={selected['hole']} dilation={selected['dilation']} min_length={selected['min_length']} inner_f1={selected['inner_f1']:.6f}")
    print("anomaly_type | n_rows_true | TP | FN | recall | FP")
    for row in type_rows:
        print(f"{row['anomaly_type']} | {row['n_rows_true']} | {row['TP']} | {row['FN']} | {row['recall']:.6f} | {row['FP']}")
    print(f"test_positive_rows={int(test_out.sum())} test_positive_rate={test_out.mean():.6f}")
    print("판정: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
