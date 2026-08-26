"""T32: apply the duration prior to exp016 without retraining."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from submission_tools import digest, save_pool_pred


SEED = 42
THRESHOLD = 0.737
HOLE = 48
CAP = 519
KEY = ["station", "year", "layer", "time"]
BUCKETS = (("1", 1, 1), ("2-11", 2, 11), ("12-47", 12, 47),
           ("48-119", 48, 119), ("120-519", 120, 519), ("520+", 520, None))


def segment_ids(raw):
    tm = pd.to_datetime(raw.time, utc=True)
    stream = pd.factorize(pd.MultiIndex.from_arrays([raw.station, raw.layer]), sort=False)[0]
    new = pd.Series(stream).ne(pd.Series(stream).shift()) | tm.diff().dt.total_seconds().ne(600)
    gap = new.groupby(stream).cumsum().to_numpy(np.int64)
    return stream.astype(np.int64) * (len(raw) + 1) + gap


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
    valid = ((starts > 0) & (ends + 1 < len(pred)) & pred[left] & pred[right]
             & (segments[left] == segments[starts]) & (segments[right] == segments[ends]))
    return starts[valid], ends[valid]


def fill_runs(pred, segments, max_gap):
    out = pred.copy()
    if max_gap == 0:
        return out
    starts, ends = _false_runs(pred, segments)
    for start, end in zip(starts, ends):
        if end - start + 1 <= max_gap:
            out[start:end + 1] = True
    return out


def positive_runs(pred, segments):
    pred = np.asarray(pred, dtype=bool)
    starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    ends = np.r_[starts[1:], len(pred)]
    for seg_start, seg_end in zip(starts, ends):
        on = np.flatnonzero(pred[seg_start:seg_end])
        if not len(on):
            continue
        split = np.diff(on) > 1
        for lo, hi in zip(on[np.r_[True, split]], on[np.r_[split, True]] + 1):
            yield int(seg_start + lo), int(seg_start + hi)


def bucket(length):
    for name, low, high in BUCKETS:
        if length >= low and (high is None or length <= high):
            return name
    raise ValueError(f"invalid run length: {length}")


def f1_stats(y, pred, mask):
    y, pred = np.asarray(y, dtype=bool)[mask], np.asarray(pred, dtype=bool)[mask]
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return 2.0 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def store_score(score, pred, threshold):
    z = np.clip(np.nan_to_num(score, nan=0.0) - threshold + 0.5, 0, 1).astype(np.float32)
    z[pred] = np.maximum(z[pred], 0.5)
    z[~pred] = np.minimum(z[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    return z


def cap_bounds(raw, lo, hi, limit=CAP):
    """Trim the lower raw-score end until a run is at most limit rows."""
    lo, hi = int(lo), int(hi)
    while hi - lo > limit:
        if raw[lo] < raw[hi - 1]:
            lo += 1
        else:
            hi -= 1
    return lo, hi


def apply_prior(base, raw, segments):
    out = np.asarray(base, dtype=bool).copy()
    runs = list(positive_runs(out, segments))
    for lo, hi in runs:
        length = hi - lo
        if 2 <= length <= 11:
            out[lo:hi] = False
        elif length >= 520:
            keep_lo, keep_hi = cap_bounds(raw, lo, hi)
            out[lo:keep_lo] = False
            out[keep_hi:hi] = False
    return out, runs


def distribution(runs):
    rows = []
    for name, _, _ in BUCKETS:
        selected = [(lo, hi) for lo, hi in runs if bucket(hi - lo) == name]
        rows.append({
            "len_bucket": name,
            "n_runs": len(selected),
            "n_rows": sum(hi - lo for lo, hi in selected),
            "r1_removed": sum(hi - lo for lo, hi in selected if 2 <= hi - lo <= 11),
            "r2_removed": sum(hi - lo - CAP for lo, hi in selected if hi - lo >= 520),
        })
    return rows


def measure_r2_oof():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    folds = np.load(ROOT / "cache/exp001/folds.npy", allow_pickle=False)
    raw = np.load(ROOT / "cache/exp005_raw_oof.npy", allow_pickle=False).astype(float)
    if raw.shape != (len(train),) or folds.shape != (len(train),):
        raise ValueError("OOF prediction/fold shape mismatch")
    segments = segment_ids(train)
    inner = np.isin(folds, (1, 2))
    if not np.isfinite(raw[inner]).all():
        raise ValueError("inner OOF raw score contains non-finite values")
    y = train.label.to_numpy(np.int8)
    threshold = float(json.loads((ROOT / "cache/exp005_meta.json").read_text(encoding="utf-8"))["threshold"])
    base = fill_runs(raw >= threshold, segments, HOLE)
    runs = [run for run in positive_runs(base, segments)
            if bucket(run[1] - run[0]) == "520+" and inner[run[0]:run[1]].any()]
    inner_rows = sum(int(inner[lo:hi].sum()) for lo, hi in runs)
    tp_rows = sum(int(np.count_nonzero(y[lo:hi][inner[lo:hi]])) for lo, hi in runs)
    r2 = base.copy()
    for lo, hi in runs:
        keep_lo, keep_hi = cap_bounds(raw, lo, hi)
        r2[lo:keep_lo] = False
        r2[keep_hi:hi] = False
    return {
        "n_runs": len(runs),
        "inner_rows": inner_rows,
        "inner_tp_rows": tp_rows,
        "tp_rate": tp_rows / max(1, inner_rows),
        "base_inner_f1": f1_stats(y, base, inner)[0],
        "r2_inner_f1": f1_stats(y, r2, inner)[0],
    }


def verify():
    tails = []
    commands = (
        [sys.executable, "P1/src/submission_tools.py", "audit",
         "P1/subs/exp020.csv", "--pred", "P1/preds/test_exp020.npy"],
        [sys.executable, "P1/src/check.py", "sub", "P1/subs/exp020.csv"],
    )
    for command in commands:
        run = subprocess.run(command, cwd=ROOT.parent, capture_output=True,
                             text=True, encoding="utf-8")
        tail = run.stdout.strip().splitlines()[-3:]
        tails.extend(tail)
        if run.returncode:
            print("\n".join(tail))
            raise RuntimeError("verification failed")
    return tails


def main():
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    raw = np.load(ROOT / "cache/exp016_raw_test.npy", allow_pickle=False).astype(float)
    if raw.shape != (len(test),):
        raise ValueError("exp016 test prediction shape mismatch")
    if not np.isfinite(raw).all():
        raise ValueError("test raw score contains non-finite values")
    segments = segment_ids(test)
    base = fill_runs(raw >= THRESHOLD, segments, HOLE)
    pred, runs = apply_prior(base, raw, segments)
    if sum(bucket(hi - lo) == "1" for lo, hi in positive_runs(pred, segments)) != sum(bucket(hi - lo) == "1" for lo, hi in runs):
        raise AssertionError("length-1 runs changed")
    if any(bucket(hi - lo) in ("2-11", "520+") for lo, hi in positive_runs(pred, segments)):
        raise AssertionError("duration prior violation remains")

    sub_path = ROOT / "subs/exp020.csv"
    pred_path = ROOT / "preds/test_exp020.npy"
    meta_path = ROOT / "cache/exp020_meta.json"
    sub = test[KEY].assign(label=pred.astype(np.uint8))
    sub.to_csv(sub_path, index=False)
    save_pool_pred(pred_path, store_score(raw, pred, THRESHOLD))
    meta = {
        "t": THRESHOLD,
        "threshold": THRESHOLD,
        "t_high": THRESHOLD,
        "hole": HOLE,
        "raw_rate": float(base.mean()),
        "post_rate": float(pred.mean()),
        "n_positive": int(pred.sum()),
        "positive_rate": float(pred.mean()),
        "seed": SEED,
        "source": "cache/exp016_raw_test.npy",
        "rules": ["R1_delete_2_11", "R2_cap_520_plus_to_519_low_end_first_tie_right"],
        "sha256": digest(sub_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p1 = distribution(runs)
    p2 = measure_r2_oof()
    tails = verify()
    length1 = sum(bucket(hi - lo) == "1" for lo, hi in positive_runs(pred, segments))
    base_length1 = sum(bucket(hi - lo) == "1" for lo, hi in runs)
    print("P1")
    print("len_bucket | n_runs | n_rows | R1_removed_rows | R2_removed_rows")
    for row in p1:
        print(f"{row['len_bucket']} | {row['n_runs']} | {row['n_rows']} | {row['r1_removed']} | {row['r2_removed']}")
    print("P2")
    print("source | n_runs | inner_rows | inner_TP_rows | TP_rate | base_inner_F1 | R2_inner_F1")
    print(f"exp005 OOF | {p2['n_runs']} | {p2['inner_rows']} | {p2['inner_tp_rows']} | {p2['tp_rate']:.6f} | {p2['base_inner_f1']:.6f} | {p2['r2_inner_f1']:.6f}")
    print("P3")
    print("exp | n_positive | positive_rate | delta_rows_vs_exp016 | length1_runs | sha10 | audit")
    print(f"exp016 | {int(base.sum())} | {base.mean():.6f} | +0 | {base_length1} | - | -")
    print(f"exp020 | {int(pred.sum())} | {pred.mean():.6f} | {int(pred.sum() - base.sum()):+d} | {length1} | {meta['sha256'][:10]} | OK")
    print("verification_tail")
    print("\n".join(tails))


if __name__ == "__main__":
    main()
