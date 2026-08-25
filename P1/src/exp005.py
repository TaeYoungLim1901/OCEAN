"""P1 exp005: strictly-forward scores, expanded grid, stream-rate anchoring."""
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_t4 import apply_reference, forward_mask, honest_reference
from exp004 import _inner_metrics, fill_runs
from features import SEED, make_features
from metric import evaluate, f1
from submission_tools import save_pool_pred
from train import fill_short_gaps, segment_ids


N_ESTIMATORS = 700
WEIGHTS = tuple(round(0.05 * i, 2) for i in range(13))
THRESHOLDS = tuple(round(0.20 + 0.02 * i, 2) for i in range(31))
HOLES = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48)
TARGET_RATES = (0.030, 0.034, 0.038, 0.041, 0.045, 0.050, 0.055)
LAMBDAS = (0.00, 0.25, 0.50, 0.75, 1.00)
KEY = ["station", "layer"]
EXPECTED_EXP004 = (5352, 440, 2091)
EXPECTED_EXP004_F1 = 2.0 * EXPECTED_EXP004[0] / (2.0 * EXPECTED_EXP004[0] + EXPECTED_EXP004[1] + EXPECTED_EXP004[2])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def store_score(score, pred, threshold):
    z = np.clip(np.nan_to_num(score, nan=0.0) - threshold + 0.5, 0, 1).astype(np.float32)
    z[pred] = np.maximum(z[pred], 0.5)
    z[~pred] = np.minimum(z[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    return z


def params():
    return dict(
        objective="binary", learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, n_estimators=N_ESTIMATORS, random_state=SEED,
        n_jobs=2, verbosity=-1, scale_pos_weight=2.0,
    )


def fit_or_load(x, y, train_mask, path):
    if path.exists():
        return lgb.Booster(model_file=path)
    model = lgb.LGBMClassifier(**params())
    model.fit(x.loc[train_mask], y[train_mask], callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(path)
    return lgb.Booster(model_file=path)


def future_rows(raw, train_mask, target_mask):
    """Count training rows at/after their own prediction stream's interval."""
    tm = pd.to_datetime(raw.time, utc=True)
    future = np.zeros(len(raw), dtype=bool)
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        va = idx[target_mask[idx]]
        if len(va):
            cutoff = tm.iloc[va].min()
            future[idx[train_mask[idx] & tm.iloc[idx].ge(cutoff).to_numpy()]] = True
    return int(future.sum())


def model_audit(name, raw, train_mask, target_mask):
    tm = pd.to_datetime(raw.time, utc=True)
    return {
        "model": name,
        "train_rows": int(train_mask.sum()),
        "train_max_time": str(tm[train_mask].max()),
        "predict_min_time": str(tm[target_mask].min()),
        "future_rows": future_rows(raw, train_mask, target_mask),
    }


def quantile_row(name, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {"array": name, **{f"p{p:g}": float(v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)}}


def print_table(rows, columns, float_columns=()):
    print(" | ".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if column in float_columns:
                value = f"{value:.6f}"
            values.append(str(value))
        print(" | ".join(values))


def grid_2a(y, raw_score, rule, raw, folds, segments):
    inner = np.isin(folds, [1, 2])
    rows = []
    for weight in WEIGHTS:
        score = (1.0 - weight) * raw_score + weight * rule
        for threshold in THRESHOLDS:
            base = score >= threshold
            for score_value, hole in _inner_metrics(base, y, inner, segments, HOLES):
                rows.append({
                    "weight": weight,
                    "threshold": threshold,
                    "hole": hole,
                    "inner_f1": float(score_value),
                })
    return sorted(rows, key=lambda r: (-r["inner_f1"], r["hole"], -r["threshold"], r["weight"]))


def stream_thresholds(score, raw, interval, target_rate, shrinkage, threshold):
    thresholds = np.full(len(raw), threshold, dtype=float)
    small_streams = 0
    finite = np.isfinite(score)
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=np.int64)
        use = idx[interval[idx] & finite[idx]]
        if len(use) < 500:
            small_streams += 1
            continue
        q = float(np.quantile(score[use], 1.0 - target_rate))
        thresholds[use] = (1.0 - shrinkage) * threshold + shrinkage * q
    return thresholds, small_streams


def anchored_prediction(score, raw, interval, target_rate, shrinkage, threshold, hole, segments):
    thresholds, small_streams = stream_thresholds(
        score, raw, interval, target_rate, shrinkage, threshold
    )
    base = score >= thresholds
    pred = fill_runs(base, segments, int(hole))
    return pred, small_streams


def grid_2b(y, score, rule, raw, folds, segments, selected):
    inner = np.isin(folds, [1, 2])
    blend = (1.0 - selected["weight"]) * score + selected["weight"] * rule
    rows = []
    small = {}
    for target_rate in TARGET_RATES:
        for shrinkage in LAMBDAS:
            thresholds, small_count = stream_thresholds(
                blend, raw, inner, target_rate, shrinkage, selected["threshold"]
            )
            pred = fill_runs(blend >= thresholds, segments, selected["hole"])
            value = f1(y[inner], pred[inner])[0]
            rows.append({
                "r": target_rate,
                "lambda": shrinkage,
                "inner_f1": float(value),
            })
            small[f"{target_rate:.3f}|{shrinkage:.2f}"] = small_count
    rows = sorted(rows, key=lambda r: (-r["inner_f1"], r["lambda"], selected["hole"], -selected["threshold"], r["r"]))
    return rows, small, blend


def breakdown(train, pred, outer):
    frame = train.loc[outer].copy()
    typed = frame.assign(anomaly_type=frame.anomaly_type.fillna("normal"))
    by_type = evaluate(typed, pred[outer], ["anomaly_type"])
    type_rows = []
    for row in by_type.itertuples(index=False):
        type_rows.append({
            "type": str(row.anomaly_type),
            "positive_rows": int(row.positive),
            "tp": int(row.tp),
            "fn": int(row.fn),
            "row_recall": int(row.tp) / max(1, int(row.positive)),
            "f1": float(row.f1),
        })
    by_station = evaluate(frame, pred[outer], ["station"])
    station_rows = [{"station": str(row.station), "f1": float(row.f1)} for row in by_station.itertuples(index=False)]
    return type_rows, station_rows


def stream_rates(raw, pred):
    d = raw.assign(_pred=np.asarray(pred, dtype=np.uint8))
    grouped = d.groupby(KEY, sort=False)._pred.agg(["sum", "count"])
    grouped["rate"] = grouped["sum"] / grouped["count"]
    rows = []
    for (station, layer), row in grouped.iterrows():
        rows.append({
            "stream": f"{station}|{layer}",
            "positive": int(row["sum"]),
            "rows": int(row["count"]),
            "rate": float(row["rate"]),
        })
    return rows


def main():
    started = time.monotonic()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    ref_full = json.loads((ROOT / "cache/exp001/reference.json").read_text(encoding="utf-8"))
    xte, _ = make_features(test, ref_full)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    y = train.label.to_numpy(np.int8)
    segments = segment_ids(train)
    test_segments = segment_ids(test)

    masks = {fold: forward_mask(train, folds, fold) for fold in (1, 2, 3)}
    refs = {fold: honest_reference(train, masks[fold]) for fold in (1, 2, 3)}
    x_forward = {fold: apply_reference(base, train, refs[fold]) for fold in (1, 2)}
    x_outer = apply_reference(base, train, refs[3])

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    models = {}
    for fold in (1, 2):
        path = model_dir / f"exp005_forward_fold{fold}.txt"
        models[fold] = fit_or_load(x_forward[fold], y, masks[fold], path)

    outer_path = model_dir / "exp002_outer_honest.txt"
    outer_meta = json.loads((ROOT / "experiments/exp002_outer_honest.json").read_text(encoding="utf-8"))
    reuse_outer = masks[3].sum() == int(outer_meta["train_rows"]) and outer_path.exists()
    if reuse_outer:
        models[3] = lgb.Booster(model_file=outer_path)
    else:
        path = model_dir / "exp005_sealed_fold3.txt"
        models[3] = fit_or_load(x_outer, y, masks[3], path)

    full_mask = np.ones(len(train), dtype=bool)
    full_model_path = model_dir / "exp005_full.txt"
    models[4] = fit_or_load(base, y, full_mask, full_model_path)
    if time.monotonic() - started > 1200:
        print(f"training_elapsed_seconds={time.monotonic() - started:.1f}")
        print("판정: STOP")
        return 1

    raw_oof = np.full(len(train), np.nan, dtype=np.float64)
    for fold in (1, 2):
        va = folds == fold
        raw_oof[va] = models[fold].predict(x_forward[fold].loc[va])
    outer = folds == 3
    raw_oof[outer] = models[3].predict(x_outer.loc[outer])
    raw_test = models[4].predict(xte)
    np.save(ROOT / "cache/exp005_raw_oof.npy", raw_oof)
    np.save(ROOT / "cache/exp005_raw_test.npy", raw_test)

    audits = [
        model_audit("forward_fold1", train, masks[1], folds == 1),
        model_audit("forward_fold2", train, masks[2], folds == 2),
        model_audit("sealed_fold3", train, masks[3], folds == 3),
    ]
    tm_train = pd.to_datetime(train.time, utc=True)
    tm_test = pd.to_datetime(test.time, utc=True)
    audits.append({
        "model": "full_test",
        "train_rows": int(full_mask.sum()),
        "train_max_time": str(tm_train[full_mask].max()),
        "predict_min_time": str(tm_test.min()),
        "future_rows": int((tm_train[full_mask] >= tm_test.min()).sum()),
    })
    quantiles = [
        quantile_row("sealed_fold3", raw_oof[outer]),
        quantile_row("full_test", raw_test),
        quantile_row("exp004_raw_test", np.load(ROOT / "cache/exp004/raw_test.npy")),
    ]

    rule_oof = base.rule_score.to_numpy(float)
    rule_test = xte.rule_score.to_numpy(float)
    rows_2a = grid_2a(y, raw_oof, rule_oof, train, folds, segments)
    selected_2a = rows_2a[0]
    check_row = selected_2a
    check_score = (1.0 - check_row["weight"]) * raw_oof + check_row["weight"] * rule_oof
    check_base = check_score >= check_row["threshold"]
    check_fast = fill_runs(check_base, segments, check_row["hole"])
    check_exact = fill_short_gaps(check_base, train, check_row["hole"], segments)
    hole_mismatch = int(np.count_nonzero(check_fast != check_exact))

    rows_2b, small_inner, blend_oof = grid_2b(y, raw_oof, rule_oof, train, folds, segments, selected_2a)
    selected_2b = rows_2b[0]
    selected = {
        **selected_2a,
        "target_rate": selected_2b["r"],
        "lambda": selected_2b["lambda"],
        "inner_f1": selected_2b["inner_f1"],
    }
    inner = np.isin(folds, [1, 2])
    blend_test = (1.0 - selected["weight"]) * raw_test + selected["weight"] * rule_test
    preliminary = {
        "weight": selected["weight"],
        "threshold": selected["threshold"],
        "hole": selected["hole"],
        "target_rate": selected["target_rate"],
        "lambda": selected["lambda"],
        "inner_f1": selected["inner_f1"],
        "inner_2a_f1": selected_2a["inner_f1"],
        "outer": None,
        "raw_oof": "cache/exp005_raw_oof.npy",
        "raw_test": "cache/exp005_raw_test.npy",
        "model_audit": audits,
        "quantiles": quantiles,
        "hole_optimization_mismatch_rows": hole_mismatch,
        "inner_small_streams": small_inner,
        "selection_stage": "pre_outer",
    }
    meta_path = ROOT / "cache/exp005_meta.json"
    meta_path.write_text(json.dumps(preliminary, indent=2), encoding="utf-8")

    repro_raw = np.load(ROOT / "cache/exp004/raw_oof.npy").astype(float)
    repro_raw[outer] = raw_oof[outer]
    repro_score = 0.60 * repro_raw + 0.40 * rule_oof
    repro_base = repro_score >= 0.54
    repro_pred = fill_short_gaps(repro_base, train, 24, segments)
    repro_stats = f1(y[outer], repro_pred[outer])
    if repro_stats[1:] != EXPECTED_EXP004:
        print(f"exp004_reproduction TP={repro_stats[1]} FP={repro_stats[2]} FN={repro_stats[3]} F1={repro_stats[0]:.6f}")
        print("판정: STOP")
        return 1

    outer_pred, small_outer = anchored_prediction(
        blend_oof, train, outer, selected["target_rate"], selected["lambda"],
        selected["threshold"], selected["hole"], segments,
    )
    inner_pred, _ = anchored_prediction(
        blend_oof, train, inner, selected["target_rate"], selected["lambda"],
        selected["threshold"], selected["hole"], segments,
    )
    test_pred, small_test = anchored_prediction(
        blend_test, test, np.ones(len(test), dtype=bool), selected["target_rate"], selected["lambda"],
        selected["threshold"], selected["hole"], test_segments,
    )
    oof_pred = np.zeros(len(train), dtype=np.uint8)
    oof_pred[inner] = inner_pred[inner].astype(np.uint8)
    oof_pred[outer] = outer_pred[outer].astype(np.uint8)
    test_out = test_pred.astype(np.uint8)

    outer_stats = f1(y[outer], outer_pred[outer])
    type_rows, station_rows = breakdown(train, outer_pred, outer)
    test_stream_rows = stream_rates(test, test_out)
    test_rate = float(test_out.mean())

    save_pool_pred(ROOT / "preds/oof_exp005.npy", store_score(blend_oof, oof_pred, selected["threshold"]))
    save_pool_pred(ROOT / "preds/test_exp005.npy", store_score(blend_test, test_pred, selected["threshold"]))
    key = ["station", "year", "layer", "time"]
    sub = test[key].assign(label=test_out.astype(int))
    sample = pd.read_csv(ROOT / "P1_data/sample_submission.csv", usecols=key)
    assert sub[key].reset_index(drop=True).equals(sample.reset_index(drop=True))
    sub.to_csv(ROOT / "subs/exp005.csv", index=False)

    final_meta = {
        **preliminary,
        "selection_stage": "post_outer",
        "outer_f1": outer_stats[0],
        "outer_tp": outer_stats[1],
        "outer_fp": outer_stats[2],
        "outer_fn": outer_stats[3],
        "delta_vs_exp004": outer_stats[0] - EXPECTED_EXP004_F1,
        "outer_by_type": type_rows,
        "outer_by_station": station_rows,
        "test_positive_rate": test_rate,
        "test_stream_rates": test_stream_rows,
        "outer_small_streams": small_outer,
        "test_small_streams": small_test,
        "exp004_reproduction": {"tp": repro_stats[1], "fp": repro_stats[2], "fn": repro_stats[3], "f1": repro_stats[0]},
        "submission_sha256": sha(ROOT / "subs/exp005.csv"),
    }
    meta_path.write_text(json.dumps(final_meta, indent=2), encoding="utf-8")

    print(f"n_estimators={N_ESTIMATORS}")
    print(f"exp002_outer_honest_reuse={'YES' if reuse_outer else 'NO'} train_rows={int(masks[3].sum())} expected_rows={int(outer_meta['train_rows'])}")
    print_table(audits, ["model", "train_rows", "train_max_time", "predict_min_time", "future_rows"])
    print_table(quantiles, ["array", "p1", "p5", "p25", "p50", "p75", "p95", "p99"], ("p1", "p5", "p25", "p50", "p75", "p95", "p99"))
    print(f"hole_optimization_mismatch_rows={hole_mismatch}")
    print_table(rows_2a[:10], ["weight", "threshold", "hole", "inner_f1"], ("weight", "threshold", "inner_f1"))
    print_table(rows_2b[:10], ["r", "lambda", "inner_f1"], ("r", "lambda", "inner_f1"))
    print(f"selected weight={selected['weight']:.2f} threshold={selected['threshold']:.2f} hole={selected['hole']} r={selected['target_rate']:.3f} lambda={selected['lambda']:.2f} inner_f1={selected['inner_f1']:.6f}")
    print(f"exp004_reproduction TP={repro_stats[1]} FP={repro_stats[2]} FN={repro_stats[3]} F1={repro_stats[0]:.6f}")
    print(f"outer TP={outer_stats[1]} FP={outer_stats[2]} FN={outer_stats[3]} F1={outer_stats[0]:.6f} delta={outer_stats[0] - EXPECTED_EXP004_F1:+.6f}")
    print_table(type_rows, ["type", "positive_rows", "tp", "fn", "row_recall", "f1"], ("row_recall", "f1"))
    print_table(station_rows, ["station", "f1"], ("f1",))
    inner_small_count = next(iter(small_inner.values()), 0)
    print(f"test_positive_rate={test_rate:.6f} small_streams_inner={inner_small_count} small_streams_outer={small_outer} small_streams_test={small_test}")
    print("test_stream_bottom3")
    print_table(sorted(test_stream_rows, key=lambda r: (r["rate"], r["stream"]))[:3], ["stream", "positive", "rows", "rate"], ("rate",))
    print("test_stream_top3")
    print_table(sorted(test_stream_rows, key=lambda r: (-r["rate"], r["stream"]))[:3], ["stream", "positive", "rows", "rate"], ("rate",))
    print("판정: OK")
    print(f"submission_sha256={final_meta['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
