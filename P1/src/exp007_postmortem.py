"""T18: exp007 importance, dilution probe, and raw-score correlation."""
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
from features import INTEGRATED_FEATURES, SEED, add_integrated_features, fit_reference
from metric import evaluate, f1
from submission_tools import save_pool_pred
from train import segment_ids


BASE_MODEL = ROOT / "models/exp005_full.txt"
INTEGRATED_MODEL = ROOT / "models/exp007_full.txt"
BASE_FEATURES = 42
FIXED_THRESHOLD = 0.62
FIXED_HOLE = 48
VARIANTS = {
    "V1": (5, 700),
    "V2": (12, 700),
    "V3": (39, 1400),
}


def model_params(n_estimators):
    return dict(
        objective="binary", learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, n_estimators=n_estimators, random_state=SEED,
        n_jobs=2, verbosity=-1, scale_pos_weight=2.0,
    )


def importance(path):
    booster = lgb.Booster(model_file=path)
    names = booster.feature_name()
    gain = np.asarray(booster.feature_importance("gain"), dtype=float)
    split = np.asarray(booster.feature_importance("split"), dtype=np.int64)
    order = np.argsort(-gain, kind="stable")
    rank = np.empty(len(names), dtype=np.int64)
    rank[order] = np.arange(1, len(names) + 1)
    return {
        "path": str(path.relative_to(ROOT)),
        "names": names,
        "gain": gain,
        "split": split,
        "rank": rank,
    }


def table(columns, rows, float_columns=()):
    print(" | ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if column in float_columns:
                value = f"{value:.6f}"
            values.append(str(value))
        print(" | ".join(values))


def d1():
    exp5 = importance(BASE_MODEL)
    exp7 = importance(INTEGRATED_MODEL)
    names7 = exp7["names"]
    pos7 = {name: i for i, name in enumerate(names7)}
    integrated = [name for name in INTEGRATED_FEATURES if name in pos7]

    rows = []
    for name in integrated:
        i = pos7[name]
        rows.append({
            "feature": name,
            "gain": exp7["gain"][i],
            "split": int(exp7["split"][i]),
            "gain_rank": int(exp7["rank"][i]),
        })
    table(["feature", "gain", "split", "gain_rank"], rows, ("gain",))

    total_gain = float(exp7["gain"].sum())
    total_split = int(exp7["split"].sum())
    integrated_gain = float(sum(row["gain"] for row in rows))
    integrated_split = int(sum(row["split"] for row in rows))
    top20 = set(np.argsort(-exp7["gain"], kind="stable")[:20])
    top20_integrated = sum(pos7[name] in top20 for name in integrated)
    summary = [
        {"metric": "integrated_gain_sum_pct", "value": 100.0 * integrated_gain / total_gain},
        {"metric": "integrated_split_sum_pct", "value": 100.0 * integrated_split / total_split},
        {"metric": "integrated_gain_top20_count", "value": int(top20_integrated)},
        {"metric": "integrated_gain_zero_count", "value": int(sum(row["gain"] == 0.0 for row in rows))},
        {"metric": "integrated_count", "value": len(integrated)},
        {"metric": "exp007_feature_count", "value": len(names7)},
        {"metric": "exp007_source", "value": exp7["path"]},
    ]
    table(["metric", "value"], summary)

    names5 = exp5["names"]
    pos5 = {name: i for i, name in enumerate(names5)}
    total5_gain = float(exp5["gain"].sum())
    total5_split = int(exp5["split"].sum())
    rows = []
    for name in names5:
        i5 = pos5[name]
        i7 = pos7[name]
        p5 = 100.0 * exp5["gain"][i5] / total5_gain
        p7 = 100.0 * exp7["gain"][i7] / total_gain
        s5 = 100.0 * exp5["split"][i5] / total5_split
        s7 = 100.0 * exp7["split"][i7] / total_split
        rows.append({
            "feature": name,
            "exp005_gain": exp5["gain"][i5],
            "exp005_split": int(exp5["split"][i5]),
            "exp005_gain_rank": int(exp5["rank"][i5]),
            "exp007_gain": exp7["gain"][i7],
            "exp007_split": int(exp7["split"][i7]),
            "exp007_gain_rank": int(exp7["rank"][i7]),
            "exp005_gain_pct": p5,
            "exp007_gain_pct": p7,
            "gain_pct_delta": p7 - p5,
            "exp005_split_pct": s5,
            "exp007_split_pct": s7,
        })
    table(
        [
            "feature", "exp005_gain", "exp005_split", "exp005_gain_rank",
            "exp007_gain", "exp007_split", "exp007_gain_rank",
            "exp005_gain_pct", "exp007_gain_pct", "gain_pct_delta",
            "exp005_split_pct", "exp007_split_pct",
        ],
        rows,
        (
            "exp005_gain", "exp007_gain", "exp005_gain_pct", "exp007_gain_pct",
            "gain_pct_delta", "exp005_split_pct", "exp007_split_pct",
        ),
    )
    comparison = [
        {
            "model": "exp005",
            "feature_set": "existing_42",
            "feature_count": len(names5),
            "gain_sum_pct": 100.0,
            "split_sum_pct": 100.0,
            "gain_top20_count": int(sum(exp5["rank"] <= 20)),
            "gain_zero_count": int(np.count_nonzero(exp5["gain"] == 0.0)),
        },
        {
            "model": "exp007",
            "feature_set": "existing_42",
            "feature_count": len(names5),
            "gain_sum_pct": 100.0 * sum(exp7["gain"][pos7[n]] for n in names5) / total_gain,
            "split_sum_pct": 100.0 * sum(exp7["split"][pos7[n]] for n in names5) / total_split,
            "gain_top20_count": int(sum(exp7["rank"][pos7[n]] <= 20 for n in names5)),
            "gain_zero_count": int(sum(exp7["gain"][pos7[n]] == 0.0 for n in names5)),
        },
    ]
    table(
        ["model", "feature_set", "feature_count", "gain_sum_pct", "split_sum_pct", "gain_top20_count", "gain_zero_count"],
        comparison,
        ("gain_sum_pct", "split_sum_pct"),
    )


def spearman(left, right):
    left = pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    right = pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def d3():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    left = np.load(ROOT / "cache/exp005_raw_oof.npy")
    right = np.load(ROOT / "cache/exp007_raw_oof.npy")
    valid = np.isfinite(left) & np.isfinite(right)
    types = train.anomaly_type.fillna("normal").astype(str).to_numpy()
    rows = [{
        "scope": "all",
        "n": int(valid.sum()),
        "spearman": spearman(left[valid], right[valid]),
    }]
    for value in sorted(np.unique(types[valid])):
        use = valid & (types == value)
        rows.append({
            "scope": value,
            "n": int(use.sum()),
            "spearman": spearman(left[use], right[use]),
        })
    table(["scope", "n", "spearman"], rows, ("spearman",))


def fit_variant(x, y, mask, n_estimators):
    model = lgb.LGBMClassifier(**model_params(n_estimators))
    model.fit(x.loc[mask], y[mask], callbacks=[lgb.log_evaluation(0)])
    return model.booster_


def type_rows(train, pred, inner):
    frame = train.loc[inner].copy()
    frame["anomaly_type"] = frame.anomaly_type.fillna("normal")
    table_data = evaluate(frame, pred[inner], ["anomaly_type"])
    rows = []
    for row in table_data.itertuples(index=False):
        positive = int(row.positive)
        rows.append({
            "anomaly_type": str(row.anomaly_type),
            "n_rows_true": positive,
            "TP": int(row.tp),
            "FN": int(row.fn),
            "recall": int(row.tp) / max(1, positive),
            "FP": int(row.fp),
        })
    return rows


def d2():
    started = time.monotonic()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    y = train.label.to_numpy(np.int8)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    inner = np.isin(folds, [1, 2])
    segments = segment_ids(train)
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    base_names = list(base.columns)

    exp7 = importance(INTEGRATED_MODEL)
    order = np.argsort(-exp7["gain"], kind="stable")
    top_integrated = [
        exp7["names"][i] for i in order if exp7["names"][i] in INTEGRATED_FEATURES
    ]
    fold_features = {}
    for fold in (1, 2):
        mask = forward_mask(train, folds, fold)
        reference = fit_reference(train, mask)
        fold_features[fold] = (
            mask,
            add_integrated_features(base, train, reference),
        )

    results = []
    for variant, (count, n_estimators) in VARIANTS.items():
        selected = top_integrated[:count]
        columns = base_names + selected
        raw = np.full(len(train), np.nan, dtype=np.float64)
        for fold in (1, 2):
            mask, full_features = fold_features[fold]
            booster = fit_variant(full_features[columns], y, mask, n_estimators)
            raw[folds == fold] = booster.predict(full_features.loc[folds == fold, columns])
            print(f"{variant}_fold={fold} elapsed_seconds={time.monotonic() - started:.1f}")
        save_pool_pred(ROOT / f"preds/score_oof_{variant}.npy", np.nan_to_num(raw, nan=0.0))
        pred = fill_runs(raw >= FIXED_THRESHOLD, segments, FIXED_HOLE)
        score, tp, fp, fn = f1(y[inner], pred[inner])
        results.append({
            "variant": variant,
            "n_estimators": n_estimators,
            "feature_count": len(columns),
            "inner_f1": score,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "selected": ",".join(selected),
        })
        table(["variant", "anomaly_type", "n_rows_true", "TP", "FN", "recall", "FP"], [
            {"variant": variant, **row} for row in type_rows(train, pred, inner)
        ], ("recall",))

    table(
        ["variant", "n_estimators", "feature_count", "inner_f1", "TP", "FP", "FN"],
        results,
        ("inner_f1",),
    )
    table(
        ["variant", "selected_integrated_features"],
        [{"variant": row["variant"], "selected_integrated_features": row["selected"]} for row in results],
    )
    print(f"elapsed_seconds={time.monotonic() - started:.1f}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"d1", "d2", "d3"}:
        raise SystemExit("usage: python P1/src/exp007_postmortem.py d1|d2|d3")
    {"d1": d1, "d2": d2, "d3": d3}[sys.argv[1]]()


if __name__ == "__main__":
    main()
