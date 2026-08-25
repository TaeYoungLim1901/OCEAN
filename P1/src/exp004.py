"""exp004: 저장 모델의 raw score로 사후처리만 재탐색."""
import hashlib
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
from metric import evaluate, f1
from submission_tools import save_pool_pred
from train import fill_short_gaps, segment_ids
from diag_t4 import apply_reference, forward_mask, honest_reference


SEED = 42
WEIGHTS = (0.0, 0.10, 0.15, 0.20, 0.30, 0.40)
THRESHOLDS = tuple(round(0.20 + 0.02 * i, 2) for i in range(26))
HOLES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)
BASELINE = (0.15, 0.44, 2)
EXPECTED_OUTER = (4935, 500, 2508)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def store_score(score, pred, threshold):
    z = np.clip(np.nan_to_num(score, nan=0.0) - threshold + 0.5, 0, 1).astype(np.float32)
    z[pred] = np.maximum(z[pred], 0.5)
    z[~pred] = np.minimum(z[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    return z


def model_scores(x, folds=None, raw=None, honest_outer=False):
    models = [lgb.Booster(model_file=ROOT / f"cache/exp001/model_fold{k}.txt") for k in range(4)]
    if folds is None:
        return np.mean([m.predict(x) for m in models], axis=0)
    score = np.zeros(len(x), np.float64)
    for k, model in enumerate(models[:3]):
        use = folds == k
        score[use] = model.predict(x.loc[use])
    if honest_outer:
        if raw is None:
            raise ValueError("raw is required for honest outer scoring")
        mask = forward_mask(raw, folds, 3)
        reference = honest_reference(raw, mask)
        honest_x = apply_reference(x, raw, reference)
        model = lgb.Booster(model_file=ROOT / "models/exp002_outer_honest.txt")
        score[folds == 3] = model.predict(honest_x.loc[folds == 3])
    else:
        use = folds == 3
        score[use] = models[3].predict(x.loc[use])
    return score


def _false_runs(pred, segments):
    false = ~pred
    prev_false = np.r_[False, false[:-1]]
    next_false = np.r_[false[1:], False]
    starts = np.flatnonzero(false & (~prev_false | (segments != np.r_[segments[0], segments[:-1]])))
    ends = np.flatnonzero(false & (~next_false | (segments != np.r_[segments[1:], segments[-1]])))
    if len(starts) != len(ends):
        raise RuntimeError("false run boundary mismatch")
    left = np.clip(starts - 1, 0, len(pred) - 1)
    right = np.clip(ends + 1, 0, len(pred) - 1)
    valid = (
        (starts > 0)
        & (ends + 1 < len(pred))
        & pred[left]
        & pred[right]
        & (segments[left] == segments[starts])
        & (segments[right] == segments[ends])
    )
    return starts[valid], ends[valid]


def fill_runs(pred, segments, max_gap):
    """Equivalent short-gap fill used for the grid and the self-check."""
    out = pred.copy()
    if max_gap == 0:
        return out
    starts, ends = _false_runs(pred, segments)
    for start, end in zip(starts, ends):
        if end - start + 1 <= max_gap:
            out[start : end + 1] = True
    return out


def _inner_metrics(pred, y, inner, segments, holes):
    base_y = y[inner]
    base_p = pred[inner]
    base_tp = int(((base_y == 1) & base_p).sum())
    base_fp = int(((base_y == 0) & base_p).sum())
    base_fn = int(((base_y == 1) & ~base_p).sum())
    starts, ends = _false_runs(pred, segments)
    if not len(starts):
        return [(2 * base_tp / max(1, 2 * base_tp + base_fp + base_fn), h) for h in holes]
    iy = y.astype(bool) & inner
    cum_y = np.r_[0, np.cumsum(iy, dtype=np.int64)]
    cum_n = np.r_[0, np.cumsum(inner & ~y.astype(bool), dtype=np.int64)]
    gain_tp = cum_y[ends + 1] - cum_y[starts]
    gain_fp = cum_n[ends + 1] - cum_n[starts]
    lengths = ends - starts + 1
    rows = []
    for hole in holes:
        use = lengths <= hole
        tp = base_tp + int(gain_tp[use].sum())
        fp = base_fp + int(gain_fp[use].sum())
        fn = base_fn - int(gain_tp[use].sum())
        rows.append((2 * tp / max(1, 2 * tp + fp + fn), hole))
    return rows


def _self_check():
    rng = np.random.default_rng(SEED)
    pred = rng.random(257) > 0.63
    segments = np.repeat(np.arange(9), [31, 28, 34, 22, 29, 27, 30, 25, 31])
    for hole in range(7):
        assert np.array_equal(fill_runs(pred, segments, hole), fill_short_gaps(pred, segments, hole, segments))


def _grid(y, raw_score, rule, raw, folds):
    inner = np.isin(folds, [1, 2])
    segments = segment_ids(raw)
    rows = []
    for weight in WEIGHTS:
        score = (1 - weight) * raw_score + weight * rule
        for threshold in THRESHOLDS:
            rows.extend(
                {
                    "inner_f1": score_value,
                    "weight": weight,
                    "threshold": threshold,
                    "hole": hole,
                }
                for score_value, hole in _inner_metrics(score >= threshold, y, inner, segments, HOLES)
            )
    return sorted(rows, key=lambda r: (-r["inner_f1"], r["hole"], -r["threshold"], r["weight"]))


def _outer_breakdown(train, pred, outer):
    frame = train.loc[outer].copy()
    typed = frame.assign(anomaly_type=frame.anomaly_type.fillna("normal"))
    by_type = evaluate(typed, pred[outer], ["anomaly_type"])
    type_rows = []
    for row in by_type.itertuples(index=False):
        positive = int(row.positive)
        type_rows.append(
            {
                "type": str(row.anomaly_type),
                "positive_rows": positive,
                "tp": int(row.tp),
                "fn": int(row.fn),
                "row_recall": int(row.tp) / max(1, positive),
                "f1": float(row.f1),
            }
        )
    by_station = evaluate(frame, pred[outer], ["station"])
    station_rows = [{"station": str(row.station), "f1": float(row.f1)} for row in by_station.itertuples(index=False)]
    return type_rows, station_rows


def _report_table(rows, columns):
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(row[c]) for c in columns))


def main():
    _self_check()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    xtr = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    ref = json.loads((ROOT / "cache/exp001/reference.json").read_text(encoding="utf-8"))
    xte, _ = make_features(test, ref)
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    y = train.label.to_numpy(np.int8)
    rule_tr = xtr.rule_score.to_numpy()
    rule_te = xte.rule_score.to_numpy()

    raw_oof = model_scores(xtr, folds, train, honest_outer=True)
    raw_test = model_scores(xte)
    (ROOT / "cache/exp004").mkdir(parents=True, exist_ok=True)
    np.save(ROOT / "cache/exp004/raw_oof.npy", raw_oof)
    np.save(ROOT / "cache/exp004/raw_test.npy", raw_test)

    segments = segment_ids(train)
    baseline_score = (1 - BASELINE[0]) * raw_oof + BASELINE[0] * rule_tr
    baseline_pred = fill_short_gaps(baseline_score >= BASELINE[1], train, BASELINE[2], segments)
    baseline_outer = f1(y[folds == 3], baseline_pred[folds == 3])
    if baseline_outer[1:] != EXPECTED_OUTER:
        print(f"reproduction TP={baseline_outer[1]} FP={baseline_outer[2]} FN={baseline_outer[3]} F1={baseline_outer[0]:.6f}")
        print("판정: STOP")
        return 1

    rows = _grid(y, raw_oof, rule_tr, train, folds)
    baseline_inner = next(
        r["inner_f1"] for r in rows if (r["weight"], r["threshold"], r["hole"]) == BASELINE
    )
    selected = rows[0]
    if selected["inner_f1"] < baseline_inner:
        print(f"inner_baseline={baseline_inner:.6f} inner_selected={selected['inner_f1']:.6f}")
        print("판정: STOP")
        return 1

    preliminary = {
        "weight": selected["weight"],
        "threshold": selected["threshold"],
        "hole": selected["hole"],
        "inner_f1": selected["inner_f1"],
        "inner_baseline_f1": baseline_inner,
        "outer": None,
        "raw_oof": "cache/exp004/raw_oof.npy",
        "raw_test": "cache/exp004/raw_test.npy",
    }
    (ROOT / "cache/exp004_meta.json").write_text(json.dumps(preliminary, indent=2), encoding="utf-8")

    selected_score = (1 - selected["weight"]) * raw_oof + selected["weight"] * rule_tr
    selected_test_score = (1 - selected["weight"]) * raw_test + selected["weight"] * rule_te
    selected_key = (selected["weight"], selected["threshold"], selected["hole"])
    if selected_key == BASELINE:
        oof_pred = baseline_pred
    else:
        oof_pred = fill_short_gaps(selected_score >= selected["threshold"], train, selected["hole"], segments)
    test_pred = fill_short_gaps(selected_test_score >= selected["threshold"], test, selected["hole"])
    assert np.array_equal(fill_runs(selected_score >= selected["threshold"], segments, selected["hole"]), oof_pred)

    outer = folds == 3
    outer_stats = f1(y[outer], oof_pred[outer])
    type_rows, station_rows = _outer_breakdown(train, oof_pred, outer)
    oof_out = oof_pred.astype(np.uint8)
    test_out = test_pred.astype(np.uint8)
    save_pool_pred(ROOT / "preds/oof_exp004.npy", store_score(selected_score, oof_pred, selected["threshold"]))
    save_pool_pred(ROOT / "preds/test_exp004.npy", store_score(selected_test_score, test_pred, selected["threshold"]))

    key = ["station", "year", "layer", "time"]
    sub = test[key].assign(label=test_out.astype(int))
    sample = pd.read_csv(ROOT / "P1_data/sample_submission.csv", usecols=key)
    assert sub[key].reset_index(drop=True).equals(sample.reset_index(drop=True))
    sub.to_csv(ROOT / "subs/exp004.csv", index=False)
    delta = outer_stats[0] - 0.7664233576642335
    meta = {
        "weight": selected["weight"],
        "threshold": selected["threshold"],
        "hole": selected["hole"],
        "inner_f1": selected["inner_f1"],
        "inner_baseline_f1": baseline_inner,
        "outer_f1": outer_stats[0],
        "outer_tp": outer_stats[1],
        "outer_fp": outer_stats[2],
        "outer_fn": outer_stats[3],
        "delta_vs_exp002": delta,
        "outer_by_type": type_rows,
        "outer_by_station": station_rows,
        "test_positive_rate": float(test_out.mean()),
        "train_positive_rate": float(y.mean()),
        "submission_sha256": sha(ROOT / "subs/exp004.csv"),
    }
    (ROOT / "cache/exp004_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("oof_exp002_blended_after_blend=yes threshold_baked=yes")
    print("weight_search=yes")
    print(f"reproduction TP={baseline_outer[1]} FP={baseline_outer[2]} FN={baseline_outer[3]} F1={baseline_outer[0]:.6f}")
    _report_table(
        [
            {
                "weight": r["weight"],
                "threshold": r["threshold"],
                "hole": r["hole"],
                "inner F1": f"{r['inner_f1']:.6f}",
            }
            for r in rows[:10]
        ],
        ["weight", "threshold", "hole", "inner F1"],
    )
    print(f"selected weight={selected['weight']:.2f} threshold={selected['threshold']:.2f} hole={selected['hole']} inner_F1={selected['inner_f1']:.6f}")
    print(f"outer TP={outer_stats[1]} FP={outer_stats[2]} FN={outer_stats[3]} F1={outer_stats[0]:.6f} delta={delta:+.6f}")
    _report_table(type_rows, ["type", "positive_rows", "tp", "fn", "row_recall", "f1"])
    _report_table(station_rows, ["station", "f1"])
    print(f"test_positive_rate={test_out.mean():.6f} train_positive_rate={y.mean():.6f}")
    print(f"submission_sha256={meta['submission_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
