"""T14: exp005 원시 확률 후처리 격자."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from submission_tools import save_pool_pred

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
KEY = ["station", "year", "layer"]
FULL_KEY = KEY + ["time"]
BASELINE_F1 = 0.697836
BASELINE_TEST_POSITIVE = 5880


def segments(frame):
    tm = pd.to_datetime(frame.time, utc=True)
    stream = pd.factorize(frame[KEY].astype(str).agg("\x1f".join, axis=1), sort=False)[0]
    new = np.r_[True, (stream[1:] != stream[:-1]) | (tm.diff().dt.total_seconds().to_numpy()[1:] != 600)]
    sid = np.cumsum(new, dtype=np.int64) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:] - 1, len(frame) - 1]
    return sid, starts[sid], ends[sid]


def cv_folds(frame):
    d = frame.assign(_time=pd.to_datetime(frame.time, utc=True)).sort_values(["station", "layer", "_time"])
    out = np.empty(len(frame), dtype=np.int8)
    for _, group in d.groupby(["station", "layer"], sort=False):
        out[group.index.to_numpy()] = np.minimum(3, np.arange(len(group)) * 4 // len(group))
    return out


def run_bounds(mask, sid):
    mask = np.asarray(mask, dtype=bool)
    seg_start = np.r_[True, sid[1:] != sid[:-1]]
    seg_end = np.r_[sid[:-1] != sid[1:], True]
    start = np.flatnonzero(mask & (seg_start | np.r_[True, mask[:-1] == False]))
    end = np.flatnonzero(mask & (seg_end | np.r_[mask[1:] == False, True]))
    return start, end


def hysteresis(base, score, low, sid):
    low_mask = np.isfinite(score) & (score >= low)
    starts, _ = run_bounds(low_mask, sid)
    if not len(starts):
        return np.zeros(len(score), dtype=bool)
    run_id = np.cumsum(low_mask & (np.r_[True, sid[1:] != sid[:-1]] | np.r_[True, low_mask[:-1] == False])) - 1
    has_base = np.zeros(len(starts), dtype=bool)
    np.logical_or.at(has_base, run_id[low_mask], base[low_mask])
    return low_mask & has_base[run_id]


def fill_holes(pred, sid, hole):
    pred = np.asarray(pred, dtype=bool)
    if hole <= 0:
        return pred.copy()
    starts, ends = run_bounds(~pred, sid)
    valid = (starts > 0) & (ends + 1 < len(pred)) & (ends - starts + 1 <= hole)
    if valid.any():
        bounded_starts, bounded_ends = starts[valid], ends[valid]
        valid[valid] = (
            pred[bounded_starts - 1] & pred[bounded_ends + 1]
            & (sid[bounded_starts - 1] == sid[bounded_ends + 1])
        )
    out = pred.copy()
    for left, right in zip(starts[valid], ends[valid]):
        out[left:right + 1] = True
    return out


def dilate(pred, sid, seg_start, seg_end, width):
    pred = np.asarray(pred, dtype=bool)
    if width <= 0:
        return pred.copy()
    starts, ends = run_bounds(pred, sid)
    if not len(starts):
        return pred.copy()
    left = np.maximum(starts - width, seg_start[starts])
    right = np.minimum(ends + width, seg_end[ends])
    delta = np.zeros(len(pred) + 1, dtype=np.int32)
    np.add.at(delta, left, 1)
    np.add.at(delta, right + 1, -1)
    return np.cumsum(delta[:-1]) > 0


def min_length(pred, sid, length):
    pred = np.asarray(pred, dtype=bool)
    if length <= 0:
        return pred.copy()
    starts, ends = run_bounds(pred, sid)
    out = pred.copy()
    for left, right in zip(starts, ends):
        if right - left + 1 < length:
            out[left:right + 1] = False
    return out


def apply_post(score, high, low, hole, width, length, sid, seg_start, seg_end):
    base = np.isfinite(score) & (score >= high)
    pred = base if low == high else hysteresis(base, score, low, sid)
    pred = fill_holes(pred, sid, hole)
    pred = dilate(pred, sid, seg_start, seg_end, width)
    return min_length(pred, sid, length)


def f1_stats(y, pred, mask):
    y, pred = np.asarray(y, dtype=bool)[mask], np.asarray(pred, dtype=bool)[mask]
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return {"TP": tp, "FP": fp, "FN": fn, "inner_f1": 2.0 * tp / max(1, 2 * tp + fp + fn)}


def grid_plain(score, y, inner, sid, seg_start, seg_end):
    rows = []
    for high in np.round(np.arange(0.30, 0.9001, 0.02), 2):
        base = np.isfinite(score) & (score >= high)
        for hole in (0, 12, 24, 48, 96):
            shaped = fill_holes(base, sid, hole)
            for dilation in (0, 1, 2, 4, 8, 16):
                widened = dilate(shaped, sid, seg_start, seg_end, dilation)
                for length in (0, 3, 6, 12):
                    pred = min_length(widened, sid, length)
                    rows.append({
                        "t_high": float(high), "t_low": float(high), "hole": hole,
                        "dilation": dilation, "min_length": length,
                        **f1_stats(y, pred, inner),
                    })
    return rows


def grid_hysteresis(score, y, inner, sid, seg_start, seg_end):
    rows = []
    lows = (0.002, 0.01, 0.05, 0.15, 0.30)
    for high in np.round(np.arange(0.40, 0.9001, 0.04), 2):
        base = np.isfinite(score) & (score >= high)
        for low in lows:
            if low > high:
                continue
            connected = base if low == high else hysteresis(base, score, low, sid)
            for hole in (0, 24, 48):
                shaped = fill_holes(connected, sid, hole)
                for dilation in (0, 2, 8):
                    widened = dilate(shaped, sid, seg_start, seg_end, dilation)
                    for length in (0, 6):
                        pred = min_length(widened, sid, length)
                        rows.append({
                            "t_high": float(high), "t_low": float(low), "hole": hole,
                            "dilation": dilation, "min_length": length,
                            **f1_stats(y, pred, inner),
                        })
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["inner_f1"], -row["hole"], -row["dilation"], -row["min_length"], row["t_high"], row["t_low"]))


def type_table(train, pred, inner):
    y = train.label.to_numpy(bool)
    types = train.anomaly_type.fillna("normal").to_numpy()
    rows = []
    for name in sorted(pd.unique(types).tolist()):
        true = inner & y & (types == name)
        tp = int(np.count_nonzero(true & pred))
        fn = int(np.count_nonzero(true & ~pred))
        rows.append({"anomaly_type": str(name), "TP": tp, "FN": fn, "recall": tp / max(1, tp + fn)})
    return rows


def write_submission(template, pred, path):
    out = template.copy()
    out["label"] = np.asarray(pred, dtype=np.int8)
    out.to_csv(path, index=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def selfcheck():
    sid = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.7, 0.1, 0.7, 0.1])
    assert fill_holes(score >= 0.5, sid, 1).tolist() == [False, True, False, True, False]
    assert dilate(score >= 0.5, sid, np.array([0, 0, 0, 3, 3]), np.array([2, 2, 2, 4, 4]), 1).tolist() == [True, True, True, True, True]
    assert hysteresis(score >= 0.6, score, 0.05, sid).tolist() == [True, True, True, True, True]


def main():
    selfcheck()
    train = pd.read_csv(ROOT / "P1_data/train.csv", usecols=FULL_KEY + ["label", "anomaly_type"])
    test = pd.read_csv(ROOT / "P1_data/test.csv", usecols=FULL_KEY)
    raw_oof = np.load(ROOT / "cache/exp005_raw_oof.npy")
    raw_test = np.load(ROOT / "cache/exp005_raw_test.npy")
    if raw_oof.shape != (len(train),) or raw_test.shape != (len(test),):
        raise ValueError("raw score shape mismatch")
    save_pool_pred(ROOT / "preds/score_oof_exp005.npy", np.nan_to_num(raw_oof, nan=0.0))
    save_pool_pred(ROOT / "preds/score_test_exp005.npy", raw_test)
    sid_oof, start_oof, end_oof = segments(train)
    sid_test, start_test, end_test = segments(test)
    # raw_oof also contains sealed fold3; fixed inner score is forward fold1·2.
    inner = np.isin(cv_folds(train), (1, 2))
    if not np.isfinite(raw_oof[inner]).all():
        raise ValueError("inner raw score contains NaN")
    y = train.label.to_numpy(np.int8)
    template = pd.read_csv(ROOT / "subs/exp005.csv")
    target_rate = 0.041362

    anchor_rows = []
    for high in np.round(np.arange(0.02, 0.9501, 0.005), 3):
        pred = apply_post(raw_test, float(high), float(high), 48, 0, 0, sid_test, start_test, end_test)
        anchor_rows.append({
            "t_high": float(high), "pre_positive_rate": float(np.mean(raw_test >= high)),
            "post_positive_rate": float(pred.mean()), "positive": int(pred.sum()),
            "distance": abs(float(pred.mean()) - target_rate),
        })
    anchor = min(anchor_rows, key=lambda row: (row["distance"], row["t_high"]))
    anchor_pred_oof = apply_post(raw_oof, anchor["t_high"], anchor["t_high"], 48, 0, 0, sid_oof, start_oof, end_oof)
    anchor_pred_test = apply_post(raw_test, anchor["t_high"], anchor["t_high"], 48, 0, 0, sid_test, start_test, end_test)
    anchor_metrics = f1_stats(y, anchor_pred_oof, inner)
    write_submission(template, anchor_pred_test, ROOT / "subs/exp006b.csv")

    plain_rows = grid_plain(raw_oof, y, inner, sid_oof, start_oof, end_oof)
    plain_best = best(plain_rows)
    plain_pred_test = apply_post(raw_test, plain_best["t_high"], plain_best["t_low"], plain_best["hole"], plain_best["dilation"], plain_best["min_length"], sid_test, start_test, end_test)
    write_submission(template, plain_pred_test, ROOT / "subs/exp008.csv")
    hyst_rows = grid_hysteresis(raw_oof, y, inner, sid_oof, start_oof, end_oof)
    hyst_best = best(hyst_rows)
    hyst_pred_test = apply_post(raw_test, hyst_best["t_high"], hyst_best["t_low"], hyst_best["hole"], hyst_best["dilation"], hyst_best["min_length"], sid_test, start_test, end_test)
    write_submission(template, hyst_pred_test, ROOT / "subs/exp009.csv")

    grid = {
        "exp006b": {"target_rate": target_rate, "selected": anchor, "inner": anchor_metrics},
        "exp008": {"selected": plain_best, "rows": plain_rows},
        "exp009": {"selected": hyst_best, "rows": hyst_rows},
    }
    (ROOT / "experiments/exp008_009_grid.json").write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")

    print("A | t_high | pre_rate | post_rate | positive | delta_vs_exp005 | inner_f1 | delta_f1")
    print(f"A | {anchor['t_high']:.3f} | {anchor['pre_positive_rate']:.6f} | {anchor['post_positive_rate']:.6f} | {anchor['positive']} | {anchor['positive'] - BASELINE_TEST_POSITIVE:+d} | {anchor_metrics['inner_f1']:.6f} | {anchor_metrics['inner_f1'] - BASELINE_F1:+.6f}")
    print("B | t_high | hole | dilation | min_length | inner_f1 | delta_f1")
    print(f"B | {plain_best['t_high']:.2f} | {plain_best['hole']} | {plain_best['dilation']} | {plain_best['min_length']} | {plain_best['inner_f1']:.6f} | {plain_best['inner_f1'] - BASELINE_F1:+.6f}")
    print("C | t_high | t_low | hole | dilation | min_length | inner_f1 | delta_f1")
    print(f"C | {hyst_best['t_high']:.2f} | {hyst_best['t_low']:.3f} | {hyst_best['hole']} | {hyst_best['dilation']} | {hyst_best['min_length']} | {hyst_best['inner_f1']:.6f} | {hyst_best['inner_f1'] - BASELINE_F1:+.6f}")
    for name, path, pred in (("exp006b", ROOT / "subs/exp006b.csv", anchor_pred_test), ("exp008", ROOT / "subs/exp008.csv", plain_pred_test), ("exp009", ROOT / "subs/exp009.csv", hyst_pred_test)):
        print(f"{name} | rows={len(pred)} | positive={int(pred.sum())} | rate={pred.mean():.6f} | sha256={sha(path)}")
    for name, pred in (("exp008", apply_post(raw_oof, plain_best["t_high"], plain_best["t_low"], plain_best["hole"], plain_best["dilation"], plain_best["min_length"], sid_oof, start_oof, end_oof)), ("exp009", apply_post(raw_oof, hyst_best["t_high"], hyst_best["t_low"], hyst_best["hole"], hyst_best["dilation"], hyst_best["min_length"], sid_oof, start_oof, end_oof))):
        print(f"D | {name} | anomaly_type | TP | FN | recall")
        for row in type_table(train, pred, inner):
            print(f"D | {name} | {row['anomaly_type']} | {row['TP']} | {row['FN']} | {row['recall']:.6f}")


if __name__ == "__main__":
    main()
