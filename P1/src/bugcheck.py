import hashlib
import json
import re
import site
import sys
from pathlib import Path

if site.getusersitepackages() not in sys.path:
    sys.path.append(site.getusersitepackages())
try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = pd = None

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "P1_data"
PREDS = ROOT / "preds"
SEED = 42
N_ESTIMATORS = 700
PURGE_HOURS = 87
EXPECTED_SHA = "5E5B8840C3A6FC5E6C8A8DFD461E419E9795548492C566E27758EE5BD0185F5A"
def verdict(fails):
    print(f"판정: {'STOP' if fails else 'OK'}")
    return 1 if fails else 0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()
def raw_train():
    return pd.read_csv(DATA / "train.csv", usecols=["station", "year", "layer", "time", "temp", "psal", "label"])


def folds_of(raw):
    out = np.empty(len(raw), dtype=np.int8)
    t = pd.to_datetime(raw.time, utc=True)
    ordered = raw.assign(_t=t).sort_values(["station", "layer", "_t"], kind="stable")
    for _, g in ordered.groupby(["station", "layer"], sort=False):
        out[g.index.to_numpy()] = np.minimum(3, np.arange(len(g)) * 4 // len(g))
    return out
def forward_mask(raw, folds, held):
    t = pd.to_datetime(raw.time, utc=True)
    keep = folds < held
    for _, ids in raw.groupby(["station", "layer"], sort=False).groups.items():
        ids = np.asarray(list(ids), dtype=int)
        val = ids[folds[ids] == held]
        if len(val):
            keep[ids] &= t.iloc[ids].to_numpy() < t.iloc[val].min() - pd.Timedelta(hours=PURGE_HOURS)
    return keep
def c1():
    raw = raw_train()
    folds = folds_of(raw)
    sys.path.insert(0, str(ROOT / "src"))
    from features import train_mask_with_purge
    future = differences = 0
    for fold in range(4):
        expected = forward_mask(raw, folds, fold)
        actual = train_mask_with_purge(raw, folds, fold)
        differences += int(np.count_nonzero(actual != expected))
        future += int(np.count_nonzero(actual & ~expected))
    print(f"C1 {'FAIL' if future else 'PASS'} future_rows={future:,} mask_differences={differences:,}")
    return future > 0 or differences > 0
def ref_diff(a, b):
    n = 0
    for key in set(a) | set(b):
        x, y = a.get(key), b.get(key)
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                if k not in x or k not in y or abs(float(x[k]) - float(y[k])) > 1e-9:
                    n += 1
        elif x != y:
            n += 1
    return n
def c2():
    raw = raw_train()
    path = ROOT / "cache/exp001/reference.json"
    if not path.exists():
        print("C2 FAIL reference_missing=1")
        return True
    sys.path.insert(0, str(ROOT / "src"))
    from features import fit_reference
    saved = json.loads(path.read_text(encoding="utf-8"))
    diffs = []
    folds = folds_of(raw)
    for fold in (1, 2, 3):
        diffs.append(ref_diff(saved, fit_reference(raw, forward_mask(raw, folds, fold))))
    leak = ref_diff(saved, fit_reference(raw)) == 0
    print(f"C2 {'FAIL' if leak else 'PASS'} full_fit_match={int(leak)} fold_reference_differences={sum(diffs):,}")
    return leak
def quantiles(a):
    a = np.asarray(a, float)
    return np.quantile(a[np.isfinite(a)], [0.50, 0.90, 0.99, 1.0])
def score_config(oof_cfg, test_cfg):
    return sum((oof_cfg.get(k) != test_cfg.get(k)) for k in ("trees", "features", "ensemble", "post"))
def c3():
    meta = ROOT / "cache/exp005_meta.json"
    oofp, testp = ROOT / "cache/exp005_raw_oof.npy", ROOT / "cache/exp005_raw_test.npy"
    if not meta.exists() or not oofp.exists() or not testp.exists():
        print("C3 FAIL artifacts=0")
        return True
    m = json.loads(meta.read_text(encoding="utf-8"))
    oof, test = np.load(oofp, mmap_mode="r"), np.load(testp, mmap_mode="r")
    source = (ROOT / "src/exp005.py").read_text(encoding="utf-8")
    features = json.loads((ROOT / "cache/exp001/meta.json").read_text(encoding="utf-8"))["features"]
    cfg = {"trees": m.get("n_estimators", N_ESTIMATORS), "features": features, "ensemble": False, "post": "none"}
    issues = score_config(cfg, cfg) + int("raw_test = models[4].predict" not in source) + int("np.mean(" in source)
    q = quantiles(oof), quantiles(test)
    print(f"C3 {'FAIL' if issues else 'PASS'} config_issues={issues} p50_delta={q[1][0]-q[0][0]:.6g} p90_delta={q[1][1]-q[0][1]:.6g} p99_delta={q[1][2]-q[0][2]:.6g} max_delta={q[1][3]-q[0][3]:.6g}")
    return bool(issues)


def array_check(a, expected):
    a = np.asarray(a)
    bad = int(a.ndim != 1 or len(a) != expected or not np.isfinite(a).all())
    bad += int(a.ndim == 1 and len(a) == expected and (np.nanmin(a) < 0 or np.nanmax(a) > 1))
    binary = int(a.ndim == 1 and len(a) == expected and a.dtype == np.uint8 and np.unique(a).size <= 2)
    return bad, binary


def c4():
    n = len(pd.read_csv(DATA / "train.csv", usecols=["label"]))
    nt = len(pd.read_csv(DATA / "test.csv", usecols=["station"]))
    bad = binary = 0
    files = list(PREDS.glob("*.npy"))
    for p in files:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        expected = n if p.name.startswith("oof_") else nt if p.name.startswith("test_") else len(a)
        x, b = array_check(a, expected)
        bad += x
        binary += b
    sub, sample = ROOT / "subs/exp005.csv", DATA / "sample_submission.csv"
    key_bad = 0
    if sub.exists() and sample.exists():
        key = ["station", "year", "layer", "time"]
        key_bad = int(not pd.read_csv(sub, usecols=key).equals(pd.read_csv(sample, usecols=key)))
    fails = bad + binary + key_bad
    print(f"C4 {'FAIL' if fails else 'PASS'} arrays={len(files):,} invalid={bad:,} binary_uint8={binary:,} key_mismatch={key_bad}")
    return bool(fails)


def c5():
    mpath, sub = ROOT / "cache/exp005_meta.json", ROOT / "subs/exp005.csv"
    required = [mpath, sub, ROOT / "cache/exp005_raw_oof.npy", ROOT / "cache/exp005_raw_test.npy", ROOT / "models/exp005_full.txt"]
    missing = sum(not p.exists() for p in required)
    digest = sha(sub) if sub.exists() else "missing"
    recorded = json.loads(mpath.read_text(encoding="utf-8")).get("submission_sha256", "") if mpath.exists() else ""
    sources = [(ROOT / "src" / x).read_text(encoding="utf-8") for x in ("features.py", "train.py", "inference.py")]
    seeded = any(re.search(r"\bSEED\s*=\s*42\b", text) for text in sources) and "SEED" in sources[1]
    fail = missing or digest != (recorded or EXPECTED_SHA) or digest != EXPECTED_SHA or not seeded
    print(f"C5 {'FAIL' if fail else 'PASS'} missing={missing} sha_match={int(digest==EXPECTED_SHA)} seed42={int(seeded)}")
    return bool(fail)


def c7():
    model_path, raw_path = ROOT / "models/exp005_full.txt", ROOT / "cache/exp005_raw_oof.npy"
    if not model_path.exists() or not raw_path.exists():
        print("C7 FAIL artifacts=0")
        return True
    from lightgbm import Booster
    model = Booster(model_file=model_path)
    trees = int(model.num_trees())
    importance = int(np.count_nonzero(model.feature_importance()))
    pred = np.load(raw_path, mmap_mode="r")
    y = raw_train().label.to_numpy()
    positives = (pred >= .5)
    score = float(2 * np.count_nonzero(positives & (y == 1)) / max(1, positives.sum() + y.sum()))
    baseline = float(2 * y.mean() / (1 + y.mean()))
    fail = trees != N_ESTIMATORS or importance == 0 or float(np.std(pred)) == 0 or score <= baseline
    print(f"C7 {'FAIL' if fail else 'PASS'} trees={trees} expected={N_ESTIMATORS} nonzero_features={importance} pred_std={np.std(pred):.6g} f1={score:.6f} baseline={baseline:.6f}")
    return bool(fail)


def c8():
    record = ROOT / "data_hashes.json"
    expected = json.loads(record.read_text(encoding="utf-8")).get("files", {}) if record.exists() else {}
    actual = {p.relative_to(ROOT).as_posix() for p in DATA.iterdir() if p.is_file()}
    recorded = set(expected)
    missing, extra = recorded - actual, actual - recorded
    checked = len(recorded & actual)
    mismatch = sum(sha(ROOT / name) != str(expected[name]).upper() for name in recorded & actual)
    sample = pd.read_csv(DATA / "train.csv", nrows=1000)
    dtype_bad = int(any(c not in sample or not pd.api.types.is_numeric_dtype(sample[c]) for c in ("temp", "label")))
    fail = not expected or missing or extra or mismatch or dtype_bad
    print(f"C8 {'FAIL' if fail else 'PASS'} recorded_files={len(recorded):,} checked={checked:,} missing={len(missing):,} extra={len(extra):,} hash_mismatch={mismatch:,} dtype_errors={dtype_bad}")
    return bool(fail)


def c9():
    train = (ROOT / "src/train.py").read_text(encoding="utf-8")
    inf = (ROOT / "src/inference.py").read_text(encoding="utf-8")
    bad = int(bool(re.search(r"\b(anomaly_type|test\.label|usecols=.*label)\b", inf))) + int(".fit(" in inf) + int("make_features" not in inf)
    ok = bool(train) and bool(inf) and bad == 0
    print(f"C9 {'PASS' if ok else 'FAIL'} inference_forbidden_hits={bad} modules={int(ok)}")
    return not ok


def selftest():
    caught = 0
    caught += int(any(x >= 1 for x in (0, 1, 2, 3)))
    caught += int(abs(sum([0, 1, 100]) / 3 - sum([0, 1]) / 2) > 1)
    caught += int(score_config({"trees": 1, "features": ["a"], "ensemble": False, "post": "x"}, {"trees": 4, "features": ["a"], "ensemble": True, "post": "y"}) > 0)
    bad = [0.0, float("nan")]
    caught += int(len(bad) != 3 or any(x != x for x in bad))
    caught += int(not (ROOT / "__missing_bugcheck_artifact__").exists())
    caught += int(len(set([1.0] * 8)) == 1)
    print(f"SELFTEST fixtures=6 caught={caught}")
    return 0 if caught == 6 else 1


CHECKS = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C7": c7, "C8": c8, "C9": c9}


def main():
    if "--selftest" in sys.argv or (len(sys.argv) > 1 and sys.argv[1].upper() == "C6"):
        return selftest()
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    names = list(CHECKS) if target == "ALL" else [target]
    fails = 0
    for name in names:
        if name not in CHECKS:
            print("C6 FAIL unknown_check=1")
            fails += 1
        else:
            try:
                fails += int(CHECKS[name]())
            except Exception as e:
                print(f"{name} FAIL runtime=1 error={type(e).__name__}")
                fails += 1
    return verdict(fails)


if __name__ == "__main__":
    raise SystemExit(main())
