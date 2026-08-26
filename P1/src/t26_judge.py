"""T26 year-holdout judge for the five already-defined P1 variants."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp004 import fill_runs
from exp007_postmortem import model_params
from exp012_v1_full import feature_spec
from exp016_neighbor import _append_neighbor_features, _add_ids
from exp019_standardized import (
    _add_ids as _add_standardized_ids,
    _append_neighbor_features as _append_standardized_features,
)
from features import SEED, add_integrated_features, fit_reference, make_features
from metric import evaluate, f1
from train import segment_ids


VARIANTS = (
    "base42", "neighbor46", "integrated47", "invariant37", "season42",
    "standardized49", "hp_trees1400", "hp_leaves127", "hp_posweight1",
)
SEASON_MONTHS = (12, 1, 2, 3, 4, 5, 6)
REMOVED_FEATURES = {"month", "doy_sin", "doy_cos", "clim_resid", "layer_count"}
PUBLIC_VARIANT = {
    "base42": "exp010",
    "integrated47": "exp012",
    "season42": "exp013",
    "neighbor46": "exp016",
}
VAL_START = pd.Timestamp("2025-01-01", tz="Asia/Seoul")
VAL_END = pd.Timestamp("2025-07-01", tz="Asia/Seoul")
PURGE = pd.Timedelta(hours=87)
TARGET_RATE = 0.0300
HOLE = 48
N_ESTIMATORS = 700
CACHE = ROOT / "cache" / "t26"


def _load_data():
    train = pd.read_csv(ROOT / "P1_data" / "train.csv")
    tm = pd.to_datetime(train.time, utc=True)
    cutoff = VAL_START - PURGE
    history_mask = tm < cutoff
    val_mask = (tm >= VAL_START) & (tm < VAL_END)
    history = train.loc[history_mask].reset_index(drop=True)
    val_indices = np.flatnonzero(val_mask.to_numpy()).astype(np.int64)
    val = train.loc[val_mask].reset_index(drop=True)
    if history.empty or val.empty:
        raise ValueError("year holdout has no history or validation rows")
    return history, val, cutoff, val_indices


def _base_frames(history, val):
    reference = fit_reference(history)
    x_history, _ = make_features(history, reference)
    x_val, _ = make_features(val, reference)
    if list(x_history.columns) != list(x_val.columns) or len(x_history.columns) != 42:
        raise ValueError("base feature columns mismatch")
    return x_history, x_val, reference


def _variant_frames(variant, history, val):
    x_history, x_val, reference = _base_frames(history, val)
    if variant == "base42":
        return x_history, x_val, list(x_history.columns)
    if variant == "invariant37":
        columns = [name for name in x_history.columns if name not in REMOVED_FEATURES]
        if len(columns) != 37:
            raise ValueError(f"invariant feature count={len(columns)}")
        return x_history, x_val, columns
    if variant == "season42":
        return x_history, x_val, list(x_history.columns)
    if variant == "integrated47":
        x_history = add_integrated_features(x_history, history, reference)
        x_val = add_integrated_features(x_val, val, reference)
        columns, _ = feature_spec()
        if len(columns) != 47:
            raise ValueError(f"integrated feature count={len(columns)}")
        return x_history, x_val, columns
    if variant in {"neighbor46", "hp_trees1400", "hp_leaves127", "hp_posweight1"}:
        train_ids = _add_ids(history)
        val_ids = _add_ids(val)
        x_history, x_val, columns, _, _ = _append_neighbor_features(
            x_history, x_val, train_ids, val_ids
        )
        if len(columns) != 46:
            raise ValueError(f"neighbor feature count={len(columns)}")
        return x_history, x_val, columns
    if variant == "standardized49":
        train_ids = _add_standardized_ids(history)
        val_ids = _add_standardized_ids(val)
        x_history, x_val, columns, *_ = _append_standardized_features(
            x_history, x_val, train_ids, val_ids
        )
        if len(columns) != 49:
            raise ValueError(f"standardized feature count={len(columns)}")
        return x_history, x_val, columns
    raise ValueError(f"unknown variant: {variant}")


def _threshold_for_rate(score, segments):
    """Choose only by postprocessed positive-rate distance; labels are unused."""
    score = np.asarray(score, dtype=np.float64)
    best = None
    for threshold in np.round(np.arange(0.0, 1.0001, 0.001), 3):
        pred = fill_runs(score >= threshold, segments, HOLE)
        key = (abs(float(pred.mean()) - TARGET_RATE), -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), pred)
    return best[1], best[2]


def _type_rows(frame, pred):
    typed = frame.assign(anomaly_type=frame.anomaly_type.fillna("normal"))
    table = evaluate(typed, pred, by=["anomaly_type"])
    wanted = ("offset", "drift", "noise", "flatline", "spike")
    rows = []
    for kind in wanted:
        hit = table.loc[table.anomaly_type.eq(kind)]
        tp = int(hit.tp.iloc[0]) if len(hit) else 0
        fn = int(hit.fn.iloc[0]) if len(hit) else 0
        rows.append({
            "anomaly_type": kind,
            "TP": tp,
            "FN": fn,
            "recall": tp / max(1, tp + fn),
        })
    return rows


def _fit(variant):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{variant}.json"
    raw_path = CACHE / f"{variant}_val_raw.npy"
    index_path = CACHE / f"{variant}_val_index.npy"
    model_path = ROOT / "models" / f"t26_{variant}.txt"
    if path.exists() and raw_path.exists() and index_path.exists() and model_path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"cached | {variant} | {result['judge_F1']:.6f}")
        return result

    history, val, cutoff, val_indices = _load_data()
    x_history, x_val, columns = _variant_frames(variant, history, val)
    if list(x_history[columns].columns) != columns or list(x_val[columns].columns) != columns:
        raise ValueError(f"{variant} feature columns mismatch")
    y_history = history.label.to_numpy(np.int8)
    history_tm = pd.to_datetime(history.time, utc=True)
    if variant == "season42":
        train_mask = np.isin(history_tm.dt.month.to_numpy(), SEASON_MONTHS)
    else:
        train_mask = np.ones(len(history), dtype=bool)
    leakage_violations = int((history_tm.loc[train_mask] >= cutoff).sum())
    if leakage_violations:
        raise AssertionError(f"leakage violations={leakage_violations}")

    params = model_params(N_ESTIMATORS)
    if variant == "hp_trees1400":
        params.update(n_estimators=1400, learning_rate=0.025)
    elif variant == "hp_leaves127":
        params.update(num_leaves=127, min_child_samples=40)
    elif variant == "hp_posweight1":
        params.update(scale_pos_weight=1.0)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_history.loc[train_mask, columns],
        y_history[train_mask],
        callbacks=[lgb.log_evaluation(0)],
    )
    raw = model.predict_proba(x_val[columns])[:, 1].astype(np.float64, copy=False)
    unique = np.unique(raw)
    if unique.size < 1000 or np.isin(unique, (0.0, 1.0)).all():
        raise ValueError(f"{raw_path}: invalid raw probabilities, unique_values={unique.size}")
    np.save(raw_path, raw)
    np.save(index_path, val_indices)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    threshold, pred = _threshold_for_rate(raw, segment_ids(val))
    judge_f1, tp, fp, fn = f1(val.label.to_numpy(np.int8), pred)
    type_recall = _type_rows(val, pred)
    result = {
        "variant": variant,
        "n_features": len(columns),
        "features": columns,
        "train_rows": int(train_mask.sum()),
        "history_rows": len(history),
        "val_rows": len(val),
        "train_max_time": str(history_tm.loc[train_mask].max()),
        "cutoff": str(cutoff),
        "leakage_violations": leakage_violations,
        "t_high": threshold,
        "val_positive_rate": float(pred.mean()),
        "judge_F1": float(judge_f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "type_recall": type_recall,
        "seed": SEED,
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"leakage_violations | {variant} | {leakage_violations}")
    print(
        "J1 | variant | n_features | train_rows | t_high | val_positive_rate | "
        "judge_F1 | TP | FP | FN"
    )
    print(
        f"J1 | {variant} | {len(columns)} | {int(train_mask.sum())} | {threshold:.3f} | "
        f"{pred.mean():.6f} | {judge_f1:.6f} | {tp} | {fp} | {fn}"
    )
    return result


def _public_scores():
    scores = pd.read_csv(ROOT / "experiments" / "public_scores.csv")
    scores["public"] = pd.to_numeric(scores["public"], errors="coerce")
    return {
        str(row.file).removesuffix(".csv"): float(row.public)
        for row in scores.itertuples()
        if np.isfinite(row.public)
    }


def _report():
    paths = [CACHE / f"{variant}.json" for variant in VARIANTS]
    if not all(path.exists() for path in paths):
        missing = ",".join(path.stem for path in paths if not path.exists())
        raise SystemExit(f"missing cache: {missing}")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    print("J1 | variant | n_features | train_rows | t_high | val_positive_rate | judge_F1 | TP | FP | FN")
    for row in results:
        print(
            f"J1 | {row['variant']} | {row['n_features']} | {row['train_rows']} | "
            f"{row['t_high']:.3f} | {row['val_positive_rate']:.6f} | "
            f"{row['judge_F1']:.6f} | {row['TP']} | {row['FP']} | {row['FN']}"
        )
    print("leakage_violations | " + " | ".join(
        f"{row['variant']}={row['leakage_violations']}" for row in results
    ))

    public = _public_scores()
    paired = []
    print("J2 | variant | judge_F1 | public_F1 | diff")
    for row in results:
        exp = PUBLIC_VARIANT.get(row["variant"])
        if exp not in public:
            continue
        public_f1 = public[exp]
        diff = public_f1 - row["judge_F1"]
        paired.append((row["judge_F1"], public_f1))
        print(f"J2 | {row['variant']} | {row['judge_F1']:.6f} | {public_f1:.6f} | {diff:.6f}")

    if len(paired) >= 2:
        judge = np.asarray([row[0] for row in paired], dtype=float)
        public_values = np.asarray([row[1] for row in paired], dtype=float)
        spearman = float(spearmanr(judge, public_values).statistic)
        a, b = np.polyfit(judge, public_values, 1)
        residual_max = float(np.max(np.abs(public_values - (a * judge + b))))
    else:
        spearman = a = b = residual_max = float("nan")
    print("J2-summary | pairs | spearman | a | b | residual_max")
    print(f"J2-summary | {len(paired)} | {spearman:.6f} | {a:.6f} | {b:.6f} | {residual_max:.6f}")

    print("J3 | variant | anomaly_type | TP | FN | recall")
    for row in results:
        for typed in row["type_recall"]:
            print(
                f"J3 | {row['variant']} | {typed['anomaly_type']} | {typed['TP']} | "
                f"{typed['FN']} | {typed['recall']:.6f}"
            )


def _self_check():
    segments = np.array([0, 0, 0, 0, 1, 1])
    threshold, pred = _threshold_for_rate(np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1]), segments)
    assert 0.0 <= threshold <= 1.0
    assert pred.dtype == bool and pred.shape == (6,)
    assert _public_scores()["exp010"] == 0.741274
    print("self-check | OK")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "report":
        _report()
        return
    if len(sys.argv) == 2 and sys.argv[1] == "--self-check":
        _self_check()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "fit" and sys.argv[2] in VARIANTS:
        _fit(sys.argv[2])
        return
    raise SystemExit(
        "usage: python P1/src/t26_judge.py fit "
        "base42|neighbor46|integrated47|invariant37|season42|standardized49|"
        "hp_trees1400|hp_leaves127|hp_posweight1 | report | --self-check"
    )


if __name__ == "__main__":
    main()
