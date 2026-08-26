"""T27: rank-based judge over saved validation probabilities."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp004 import fill_runs
from metric import f1
from train import segment_ids


SEED = 42
VARIANTS = (
    "base42", "neighbor46", "integrated47", "invariant37", "season42",
    "standardized49", "hp_trees1400", "hp_leaves127", "hp_posweight1",
)
PUBLIC_VARIANT = {
    "base42": 0.741274,
    "neighbor46": 0.776056,
    "integrated47": 0.717975,
    "season42": 0.582477,
}
RATES = (0.020, 0.025, 0.030)
HOLE = 48
VAL_START = pd.Timestamp("2025-01-01", tz="Asia/Seoul")
VAL_END = pd.Timestamp("2025-07-01", tz="Asia/Seoul")


def _load_val():
    train = pd.read_csv(ROOT / "P1_data" / "train.csv")
    tm = pd.to_datetime(train.time, utc=True)
    mask = (tm >= VAL_START) & (tm < VAL_END)
    val = train.loc[mask].reset_index(drop=True)
    if val.empty:
        raise ValueError("empty validation window")
    return train, val, np.flatnonzero(mask.to_numpy())


def _raw_candidates(variant):
    return (
        ROOT / "cache" / "t26" / f"{variant}_val_raw.npy",
        ROOT / "cache" / "t27" / f"{variant}_raw_val.npy",
        ROOT / "cache" / "t27" / f"raw_val_{variant}.npy",
        ROOT / "cache" / "t26" / f"{variant}_raw_val.npy",
        ROOT / "cache" / "t26" / f"raw_val_{variant}.npy",
        ROOT / "cache" / f"{variant}_raw_val.npy",
        ROOT / "preds" / f"val_{variant}_raw.npy",
    )


def _model_candidates(variant):
    return (
        ROOT / "models" / f"t27_{variant}.txt",
        ROOT / "models" / f"t26_{variant}.txt",
        ROOT / "cache" / "t27" / f"{variant}.txt",
        ROOT / "cache" / "t26" / f"{variant}.txt",
    )


def _load_raw_file(path, train_rows, val_indices):
    raw = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    unique = np.unique(raw)
    if unique.size < 1000 or np.isin(unique, (0.0, 1.0)).all():
        raise ValueError(f"{path}: invalid raw probabilities, unique_values={unique.size}")
    if raw.ndim != 1 or raw.size not in (len(val_indices), len(train_rows)):
        raise ValueError(f"{path}: shape={raw.shape}, expected={len(val_indices)} or {len(train_rows)}")
    raw = raw if raw.size == len(val_indices) else raw[val_indices]
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: non-finite raw probabilities")
    return raw, f"raw:{path.relative_to(ROOT)}", None


def _load_model_raw(variant, val, train_rows):
    paths = [path for path in _model_candidates(variant) if path.exists()]
    if not paths:
        raise FileNotFoundError(
            f"{variant}: saved validation raw probabilities or exact variant booster not found"
        )

    import lightgbm as lgb
    from t26_judge import _variant_frames

    history, _, cutoff = _history_and_val(train_rows)
    x_history, x_val, columns = _variant_frames(variant, history, val)
    del x_history
    path = paths[0]
    before = path.stat().st_mtime_ns
    booster = lgb.Booster(model_file=path)
    if booster.feature_name() != columns:
        raise ValueError(f"{path}: feature columns do not match {variant}")
    raw = np.asarray(booster.predict(x_val[columns]), dtype=np.float64)
    if raw.shape != (len(val),) or not np.isfinite(raw).all():
        raise ValueError(f"{path}: invalid validation prediction")
    if pd.to_datetime(history.time, utc=True).max() >= cutoff:
        raise AssertionError(f"{variant}: history crosses purge cutoff")
    return raw, f"model:{path.relative_to(ROOT)}", (path, before)


def _history_and_val(train):
    cutoff = VAL_START - pd.Timedelta(hours=87)
    tm = pd.to_datetime(train.time, utc=True)
    history = train.loc[tm < cutoff].reset_index(drop=True)
    if history.empty:
        raise ValueError("empty history window")
    return history, None, cutoff


def _load_raw(variant, train, val, val_indices):
    for path in _raw_candidates(variant):
        if path.exists():
            return _load_raw_file(path, train, val_indices)
    return _load_model_raw(variant, val, train)


def _rank_prediction(raw, segments, rate):
    raw = np.asarray(raw, dtype=np.float64)
    k = int(len(raw) * rate)
    order = np.lexsort((np.arange(len(raw), dtype=np.int64), -raw))
    base = np.zeros(len(raw), dtype=bool)
    if k:
        base[order[:k]] = True
        cutoff = raw[order[k - 1]]
        n_ties_top = int(np.count_nonzero(raw == cutoff))
    else:
        n_ties_top = 0
    return fill_runs(base, segments, HOLE), k, n_ties_top


def _auc(y, raw):
    return float(roc_auc_score(y, raw)) if np.unique(y).size == 2 else float("nan")


def _auc_rows(val, raw):
    y = val.label.to_numpy(np.int8)
    types = val.anomaly_type.fillna("normal").astype(str).to_numpy()
    rows = {"all": _auc(y, raw)}
    for kind in ("offset", "drift", "noise", "flatline", "spike"):
        use = (types == kind) | (types == "normal")
        rows[kind] = _auc(y[use], raw[use])
    return rows


def _k2_row(variant, rate, val, raw, segments):
    pred, k, ties = _rank_prediction(raw, segments, rate)
    score, tp, fp, fn = f1(val.label.to_numpy(np.int8), pred)
    return {
        "variant": variant,
        "r": rate,
        "k": k,
        "post_rate": float(pred.mean()),
        "judge_F1": float(score),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "n_ties_top": ties,
    }


def _regression(values):
    judge = np.asarray(values, dtype=float)
    public = np.asarray([PUBLIC_VARIANT[v] for v in ("base42", "neighbor46", "integrated47", "season42")])
    rho = float(spearmanr(judge, public).statistic)
    a, b = np.polyfit(judge, public, 1)
    residual = float(np.max(np.abs(public - (a * judge + b))))
    return rho, float(a), float(b), residual


def _self_check():
    pred, k, ties = _rank_prediction(
        np.array([0.9, 0.1, 0.9, 0.1]), np.arange(4, dtype=np.int64), 0.5
    )
    assert k == 2 and ties == 2 and pred.dtype == bool
    assert np.array_equal(np.flatnonzero(pred), np.array([0, 2]))
    assert _auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    print("self-check | OK")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-check":
        _self_check()
        return
    if len(sys.argv) != 1:
        raise SystemExit("usage: python P1/src/t27_judge_fix.py | --self-check")

    train, val, val_indices = _load_val()
    segments = segment_ids(val)
    results = {}
    mtimes = {}
    for variant in VARIANTS:
        raw, source, stamp = _load_raw(variant, train, val, val_indices)
        results[variant] = {
            "source": source,
            "raw": raw,
            "auc": _auc_rows(val, raw),
            "k2": [_k2_row(variant, rate, val, raw, segments) for rate in RATES],
        }
        if stamp is not None:
            mtimes[variant] = stamp

    print("K1 | variant | auc_all | auc_offset | auc_drift | auc_noise | auc_flatline | n_ties_top")
    for variant in VARIANTS:
        row = results[variant]
        ties = row["k2"][RATES.index(0.030)]["n_ties_top"]
        print(
            f"K1 | {variant} | {row['auc']['all']:.6f} | {row['auc']['offset']:.6f} | "
            f"{row['auc']['drift']:.6f} | {row['auc']['noise']:.6f} | "
            f"{row['auc']['flatline']:.6f} | {ties}"
        )

    print("K2 | variant | r | k | post_rate | judge_F1 | TP | FP | FN")
    for variant in VARIANTS:
        for row in results[variant]["k2"]:
            print(
                f"K2 | {variant} | {row['r']:.3f} | {row['k']} | {row['post_rate']:.6f} | "
                f"{row['judge_F1']:.6f} | {row['TP']} | {row['FP']} | {row['FN']}"
            )

    print("K3 | metric | spearman | a | b | residual_max")
    k3 = {
        "auc_all": [results[v]["auc"]["all"] for v in ("base42", "neighbor46", "integrated47", "season42")],
        "auc_offset": [results[v]["auc"]["offset"] for v in ("base42", "neighbor46", "integrated47", "season42")],
        "judge_F1_r030": [results[v]["k2"][RATES.index(0.030)]["judge_F1"] for v in ("base42", "neighbor46", "integrated47", "season42")],
    }
    for metric, values in k3.items():
        rho, a, b, residual = _regression(values)
        print(f"K3 | {metric} | {rho:.6f} | {a:.6f} | {b:.6f} | {residual:.6f}")

    print("source | variant | path")
    for variant in VARIANTS:
        print(f"source | {variant} | {results[variant]['source']}")
    for variant in VARIANTS:
        if variant not in mtimes:
            print(f"mtime | {variant} | unchanged=1 | model=not_used")
            continue
        path, before = mtimes[variant]
        after = path.stat().st_mtime_ns
        print(f"mtime | {variant} | unchanged={int(before == after)} | before={before} | after={after}")
    print(f"val_rows | {len(val)} | seed={SEED} | hole={HOLE}")


if __name__ == "__main__":
    main()
