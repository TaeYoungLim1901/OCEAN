"""T25: apply the stated anomaly run-length prior without retraining."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp004 import fill_runs
from exp005 import store_score
from submission_tools import digest, save_pool_pred
from train import segment_ids


SEED = 42
THRESHOLD = 0.775
HOLE = 48
KEY = ["station", "year", "layer", "time"]
BUCKETS = (("1", 1, 1), ("2-11", 2, 11), ("12-47", 12, 47),
           ("48-119", 48, 119), ("120-519", 120, 519), ("520+", 520, None))


def positive_runs(pred, segments):
    """Return (start, stop) half-open positive runs within gap-safe segments."""
    pred = np.asarray(pred, dtype=bool)
    segments = np.asarray(segments)
    starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    ends = np.r_[starts[1:], len(pred)]
    for seg_start, seg_end in zip(starts, ends):
        on = np.flatnonzero(pred[seg_start:seg_end])
        if not len(on):
            continue
        run_start = np.r_[True, np.diff(on) > 1]
        run_end = np.r_[np.diff(on) > 1, True]
        for lo, hi in zip(on[run_start], on[run_end] + 1):
            yield int(seg_start + lo), int(seg_start + hi)


def f1_stats(y, pred, mask):
    y = np.asarray(y, dtype=bool)[mask]
    pred = np.asarray(pred, dtype=bool)[mask]
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    return 2.0 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def self_check():
    pred = np.array([False, True, True, False, True, False])
    segments = np.zeros(len(pred), dtype=np.int64)
    assert list(positive_runs(pred, segments)) == [(1, 3), (4, 5)]
    assert expand_interval(5, 7, 0, 20) == (0, 12)


def bucket(length):
    for name, low, high in BUCKETS:
        if length >= low and (high is None or length <= high):
            return name
    raise ValueError(f"invalid run length: {length}")


def deletion_stats(pred, runs, y, inner):
    rows = []
    for name, _, _ in BUCKETS:
        selected = [(lo, hi) for lo, hi in runs if bucket(hi - lo) == name]
        removed = np.zeros(len(y), dtype=bool)
        for lo, hi in selected:
            removed[lo:hi] = True
        n_rows = int(sum(inner[lo:hi].sum() for lo, hi in selected))
        tp_rows = int(sum((y[lo:hi] * inner[lo:hi]).sum() for lo, hi in selected))
        rows.append({"len_bucket": name, "n_runs": len(selected), "n_rows": n_rows,
                     "TP_rows": tp_rows, "precision": tp_rows / max(1, n_rows),
                     "inner_f1": f1_stats(y, np.asarray(pred, dtype=bool) & ~removed, inner)[0]})
    return rows


def expand_interval(lo, hi, seg_lo, seg_hi, target=12):
    extra = max(0, target - (hi - lo))
    left = extra // 2
    right = extra - left
    new_lo = max(seg_lo, lo - left)
    new_hi = min(seg_hi, hi + right)
    if new_hi - new_lo < target:
        need = target - (new_hi - new_lo)
        take = min(need, new_lo - seg_lo)
        new_lo -= take
        need -= take
        new_hi = min(seg_hi, new_hi + need)
    if new_hi - new_lo < target:
        need = target - (new_hi - new_lo)
        take = min(need, seg_hi - new_hi)
        new_hi += take
        need -= take
        new_lo = max(seg_lo, new_lo - need)
    return new_lo, new_hi


def expanded_prediction(pred, segments, runs):
    out = np.asarray(pred, dtype=bool).copy()
    seg_starts = np.flatnonzero(np.r_[True, segments[1:] != segments[:-1]])
    seg_ends = np.r_[seg_starts[1:], len(pred)]
    for lo, hi in runs:
        if 2 <= hi - lo <= 11:
            seg_index = np.searchsorted(seg_starts, lo, side="right") - 1
            new_lo, new_hi = expand_interval(
                lo, hi, int(seg_starts[seg_index]), int(seg_ends[seg_index])
            )
            out[new_lo:new_hi] = True
    return out


def m2_stats(pred, expanded, runs, y, inner):
    short = [(lo, hi) for lo, hi in runs if 2 <= hi - lo <= 11]
    added = expanded & ~pred & inner
    added_tp = int(np.count_nonzero(added & y.astype(bool)))
    return [{"n_runs": len(short), "added_rows": int(added.sum()),
             "added_TP_rows": added_tp,
             "added_precision": added_tp / max(1, int(added.sum())),
             "inner_f1": f1_stats(y, expanded, inner)[0]}]


def distribution(source, pred, segments):
    runs = list(positive_runs(pred, segments))
    counts = {name: {"n_runs": 0, "n_rows": 0} for name, _, _ in BUCKETS}
    for lo, hi in runs:
        row = counts[bucket(hi - lo)]
        row["n_runs"] += 1
        row["n_rows"] += hi - lo
    return [{"source": source, "len_bucket": name, **counts[name]} for name, _, _ in BUCKETS]


def save_variant(name, test, raw, pred, source_meta):
    sub_path = ROOT / f"subs/{name}.csv"
    pred_path = ROOT / f"preds/test_{name}.npy"
    meta_path = ROOT / f"cache/{name}_meta.json"
    sub = test[KEY].assign(label=np.asarray(pred, dtype=np.uint8))
    sub.to_csv(sub_path, index=False)
    save_pool_pred(pred_path, store_score(raw, pred, THRESHOLD))
    meta = {
        "t": THRESHOLD, "threshold": THRESHOLD, "t_high": THRESHOLD, "hole": HOLE,
        "raw_rate": float(np.asarray(raw >= THRESHOLD).mean()),
        "post_rate": float(np.asarray(pred).mean()),
        "n_positive": int(np.asarray(pred).sum()),
        "positive_rate": float(np.asarray(pred).mean()), "seed": SEED,
        "source": "cache/exp005_raw_test.npy", "source_meta": source_meta,
        "run_prior": "delete_2_11" if name == "exp017" else "expand_2_11_to_12",
        "sha256": digest(sub_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sub_path, pred_path, meta


def verify(name):
    outputs = []
    for command in (
        [sys.executable, "P1/src/submission_tools.py", "audit", f"P1/subs/{name}.csv", "--pred", f"P1/preds/test_{name}.npy"],
        [sys.executable, "P1/src/check.py", "sub", f"P1/subs/{name}.csv"],
    ):
        run = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8")
        tail = run.stdout.strip().splitlines()[-3:]
        outputs.extend(tail)
        if run.returncode:
            print("\n".join(tail))
            raise RuntimeError(f"verification failed: {' '.join(command)}")
    return outputs


def main():
    self_check()
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    folds = np.load(ROOT / "cache/exp001/folds.npy", allow_pickle=False)
    raw_oof = np.load(ROOT / "cache/exp005_raw_oof.npy", allow_pickle=False).astype(float)
    raw_test = np.load(ROOT / "cache/exp005_raw_test.npy", allow_pickle=False).astype(float)
    if raw_oof.shape != (len(train),) or raw_test.shape != (len(test),) or folds.shape != (len(train),):
        raise ValueError("prediction/fold shape mismatch")
    if not np.isfinite(raw_test).all():
        raise ValueError("test raw score contains non-finite values")

    train_segments = segment_ids(train)
    test_segments = segment_ids(test)
    y = train.label.to_numpy(np.int8)
    inner = np.isin(folds, (1, 2))

    inner_base = fill_runs(raw_oof >= 0.62, train_segments, HOLE)
    inner_runs = [run for run in positive_runs(inner_base, train_segments)
                  if inner[run[0]:run[1]].any()]
    m1 = deletion_stats(inner_base, inner_runs, y, inner)
    inner_expanded = expanded_prediction(inner_base, train_segments, inner_runs)
    m2 = m2_stats(inner_base, inner_expanded, inner_runs, y, inner)

    test_base = fill_runs(raw_test >= THRESHOLD, test_segments, HOLE)
    test_runs = list(positive_runs(test_base, test_segments))
    test_deleted = test_base.copy()
    short_test = [(lo, hi) for lo, hi in test_runs if 2 <= hi - lo <= 11]
    for lo, hi in short_test:
        test_deleted[lo:hi] = False
    test_expanded = expanded_prediction(test_base, test_segments, test_runs)

    source_meta = json.loads((ROOT / "cache/exp005_meta.json").read_text(encoding="utf-8"))
    variants = {}
    for name, pred in (("exp017", test_deleted), ("exp018", test_expanded)):
        _, _, meta = save_variant(name, test, raw_test, pred, {
            "threshold": source_meta.get("threshold"), "hole": source_meta.get("hole")
        })
        variants[name] = {"pred": pred, "meta": meta, "verify": verify(name)}

    m3 = distribution("exp010", test_base, test_segments)
    exp016_raw_path = ROOT / "cache/exp016_raw_test.npy"
    if exp016_raw_path.exists():
        exp016_raw = np.load(exp016_raw_path, allow_pickle=False).astype(float)
        exp016_meta = json.loads((ROOT / "cache/exp016_meta.json").read_text(encoding="utf-8"))
        exp016_pred = fill_runs(exp016_raw >= float(exp016_meta["threshold"]), test_segments, int(exp016_meta["hole"]))
        m3.extend(distribution("exp016", exp016_pred, test_segments))

    baseline_positive = int(test_base.sum())
    baseline_sub = pd.read_csv(ROOT / "subs/exp010.csv")
    if baseline_positive != 5042 or int(baseline_sub.label.sum()) != baseline_positive:
        raise ValueError("exp010 baseline reproduction mismatch")

    s1 = []
    for name in ("exp017", "exp018"):
        pred = variants[name]["pred"]
        s1.append({"exp": name, "n_positive": int(pred.sum()),
                   "positive_rate": float(pred.mean()),
                   "delta_rows": int(pred.sum() - baseline_positive),
                   "affected_runs": len(short_test),
                   "sha10": variants[name]["meta"]["sha256"][:10],
                   "audit": "OK" if variants[name]["verify"] else "STOP"})

    print("M1")
    print("len_bucket | n_runs | n_rows | TP_rows | precision | inner_F1_if_deleted")
    for row in m1:
        print(f"{row['len_bucket']} | {row['n_runs']} | {row['n_rows']} | {row['TP_rows']} | {row['precision']:.6f} | {row['inner_f1']:.6f}")
    print("M2")
    print("n_runs | added_rows | added_TP_rows | added_precision | inner_F1")
    for row in m2:
        print(f"{row['n_runs']} | {row['added_rows']} | {row['added_TP_rows']} | {row['added_precision']:.6f} | {row['inner_f1']:.6f}")
    print("M3")
    print("source | len_bucket | n_runs | n_rows")
    for row in m3:
        print(f"{row['source']} | {row['len_bucket']} | {row['n_runs']} | {row['n_rows']}")
    print("length_1_preservation")
    print("source | n_runs")
    print(f"baseline | {sum(bucket(hi - lo) == '1' for lo, hi in test_runs)}")
    for name in ("exp017", "exp018"):
        print(f"{name} | {sum(bucket(hi - lo) == '1' for lo, hi in positive_runs(variants[name]['pred'], test_segments))}")
    print("S1")
    print("exp | n_positive | positive_rate | delta_rows | affected_runs | sha10 | audit")
    for row in s1:
        print(f"{row['exp']} | {row['n_positive']} | {row['positive_rate']:.6f} | {row['delta_rows']:+d} | {row['affected_runs']} | {row['sha10']} | {row['audit']}")
    print("verification_tail")
    for name in ("exp017", "exp018"):
        print(name)
        print("\n".join(variants[name]["verify"]))


if __name__ == "__main__":
    main()
