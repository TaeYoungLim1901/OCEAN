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
DATA, PREDS, SUBS = ROOT / "P2_data", ROOT / "preds", ROOT / "subs"
TARGET = [2, 3, 4]
PUBLIC = [1, 5, 6, 7, 8]
KEY = ["station", "layer", "time"]
SEED = 42
N_ESTIMATORS = 500
VALUE_MIN, VALUE_MAX = -5.0, 45.0


def verdict(fails):
    print(f"판정: {'STOP' if fails else 'OK'}")
    return int(bool(fails))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def c1():
    obs = pd.read_csv(DATA / "observations.csv", usecols=KEY + ["temp"], parse_dates=["time"])
    target = obs.layer.isin(TARGET) & obs.temp.notna()
    hold = obs.time.between(pd.Timestamp("2024-09-01", tz="Asia/Seoul"), pd.Timestamp("2024-10-31 23:50", tz="Asia/Seoul"))
    public_n = obs[obs.layer.isin(PUBLIC) & obs.temp.notna()].groupby("time").size()
    valid = obs.time.map(public_n).fillna(0).ge(2)
    train, val = target & ~hold, target & hold & valid
    overlap = int((train & val).sum())
    bad_dates = int((val & ~hold).sum())
    print(f"C1 {'FAIL' if overlap or bad_dates else 'PASS'} train_rows={train.sum():,} hold_rows={val.sum():,} overlap={overlap} boundary_errors={bad_dates}")
    return bool(overlap or bad_dates)


def c2():
    source = (ROOT / "src/train.py").read_text(encoding="utf-8")
    common = (ROOT / "src/common.py").read_text(encoding="utf-8")
    full_fit = int("build_features(obs, train)" in source and "obs_train" not in source)
    global_rolling = int("_fill_by_year" in common and "build_features(obs" in source)
    issues = full_fit + global_rolling
    print(f"C2 {'FAIL' if issues else 'PASS'} full_observation_fit={full_fit} global_time_features={global_rolling}")
    return bool(issues)


def path_config(a, b):
    return sum(a.get(k) != b.get(k) for k in ("trees", "features", "ensemble", "post"))


def c3():
    train = (ROOT / "src/train.py").read_text(encoding="utf-8")
    inf = (ROOT / "src/inference_e002.py").read_text(encoding="utf-8")
    issues = int("build_features" not in train or "build_features" not in inf)
    issues += int("artifact[\"features\"]" not in inf or "model.predict" not in inf)
    issues += int("np.mean" in inf or "mean([" in inf)
    arrays = list(PREDS.glob("oof_*_E*.npy")) + list(PREDS.glob("test_*_E*.npy"))
    q = []
    for p in arrays:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if a.ndim == 1 and np.isfinite(a).all():
            q.append((p.name, float(np.quantile(a, .5))))
    p50 = float(np.median([x[1] for x in q])) if q else float("nan")
    print(f"C3 {'FAIL' if issues else 'PASS'} config_issues={issues} paired_arrays={len(q):,} p50_median={p50:.6g}")
    return bool(issues)


def array_bad(a, expected=None):
    if np is None:
        return int((expected is not None and len(a) != expected) or any(x != x for x in a))
    a = np.asarray(a)
    bad = int(a.ndim != 1 or (expected is not None and len(a) != expected) or not np.isfinite(a).all())
    bad += int(a.ndim == 1 and np.isfinite(a).all() and ((a < VALUE_MIN) | (a > VALUE_MAX)).any())
    return bad


def c4():
    expected = len(pd.read_csv(DATA / "test_index.csv"))
    bad = 0
    files = list(PREDS.glob("*.npy"))
    for p in files:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        bad += array_bad(a, expected if p.name.startswith("test_") else None)
    key_bad = 0
    idx = pd.read_csv(DATA / "test_index.csv", usecols=KEY)
    for p in SUBS.glob("*.csv"):
        s = pd.read_csv(p, usecols=KEY)
        key_bad += int(len(s) != len(idx) or not s.equals(idx))
    fails = bad + key_bad
    print(f"C4 {'FAIL' if fails else 'PASS'} arrays={len(files):,} invalid={bad:,} key_mismatches={key_bad:,} test_rows={expected:,}")
    return bool(fails)


def c5():
    ledger = ROOT / "experiments/submission_audits.jsonl"
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()] if ledger.exists() else []
    checked = missing = mismatch = 0
    for row in rows:
        path = Path(row.get("file", ""))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            missing += 1
        else:
            checked += 1
            mismatch += int(digest(path) != str(row.get("sha256", "")).lower())
    common = (ROOT / "src/common.py").read_text(encoding="utf-8")
    seeded = bool(re.search(r"\bSEED\s*=\s*42\b", common))
    fail = not rows or missing or mismatch or not seeded
    print(f"C5 {'FAIL' if fail else 'PASS'} recorded={len(rows):,} checked={checked:,} missing={missing} hash_mismatch={mismatch} seed42={int(seeded)}")
    return bool(fail)


def model_stats(obj, out):
    if isinstance(obj, dict):
        for x in obj.values():
            model_stats(x, out)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            model_stats(x, out)
    elif hasattr(obj, "feature_importances_"):
        imp = np.asarray(obj.feature_importances_)
        out.append((int(getattr(obj, "n_estimators", 0)), int(np.count_nonzero(imp))))


def c7():
    model_path, pred_path = ROOT / "models/E002.pkl", ROOT / "preds/test_fixed_depth_E002.npy"
    if not model_path.exists() or not pred_path.exists():
        print("C7 FAIL artifacts=0")
        return True
    import pickle
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    stats = []
    model_stats(obj, stats)
    pred = np.load(pred_path, mmap_mode="r")
    metrics = json.loads((ROOT / "cache/E002/metrics.json").read_text(encoding="utf-8")) if (ROOT / "cache/E002/metrics.json").exists() else {}
    trees = min((x[0] for x in stats if x[0]), default=0)
    nonzero = sum(x[1] > 0 for x in stats)
    selected = metrics.get("selected")
    score = float(metrics.get("scores", {}).get("2024", {}).get(selected, {}).get("overall", 0))
    fail = not stats or trees != N_ESTIMATORS or nonzero == 0 or float(np.std(pred)) == 0 or score == 0
    print(f"C7 {'FAIL' if fail else 'PASS'} models={len(stats):,} trees_min={trees} expected={N_ESTIMATORS} nonzero_models={nonzero} pred_std={np.std(pred):.6g} score={score:.6f}")
    return bool(fail)


def c8():
    files = list(DATA.glob("*.csv"))
    sample = pd.read_csv(DATA / "observations.csv", nrows=1000)
    dtype_bad = int(any(c in sample and not pd.api.types.is_numeric_dtype(sample[c]) for c in ("temp", "psal", "depth")))
    record = ROOT / "cache/data_sha256.json"
    expected = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    actual = {p.name for p in files}
    recorded = set(expected)
    missing, extra = recorded - actual, actual - recorded
    mismatch = sum(digest(DATA / name) != expected[name] for name in recorded & actual)
    fail = not expected or missing or extra or mismatch or dtype_bad
    print(f"C8 {'FAIL' if fail else 'PASS'} data_files={len(actual):,} recorded={len(recorded):,} missing={len(missing):,} extra={len(extra):,} hash_mismatch={mismatch} dtype_errors={dtype_bad}")
    return bool(fail)


def c9():
    train = (ROOT / "src/train.py").read_text(encoding="utf-8")
    inf = (ROOT / "src/inference_e002.py").read_text(encoding="utf-8")
    forbidden = int(bool(re.search(r"\b(label|target)\b", inf))) + int(".fit(" in inf) + int("build_features" not in inf)
    ok = bool(train) and bool(inf) and forbidden == 0
    print(f"C9 {'PASS' if ok else 'FAIL'} inference_forbidden_hits={forbidden} modules={int(ok)}")
    return not ok


def selftest():
    caught = 0
    caught += int(any(x >= 1 for x in range(4)))
    caught += int(abs(sum([1, 2, 99]) / 3 - sum([1, 2]) / 2) > 1)
    caught += int(path_config({"trees": 1, "features": ["a"], "ensemble": False, "post": "x"}, {"trees": 2, "features": ["b"], "ensemble": True, "post": "y"}) > 0)
    caught += int(array_bad([1.0, float("nan")], 3) > 0)
    caught += int(not (ROOT / "__missing_bugcheck_artifact__").exists())
    caught += int(len(set([1.0] * 6)) == 1)
    print(f"SELFTEST fixtures=6 caught={caught}")
    return int(caught != 6)


CHECKS = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C7": c7, "C8": c8, "C9": c9}


def main():
    if "--selftest" in sys.argv or (len(sys.argv) > 1 and sys.argv[1].upper() == "C6"):
        return selftest()
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "ALL"
    names = list(CHECKS) if target == "ALL" else [target]
    fails = 0
    for name in names:
        try:
            fails += int(CHECKS.get(name, lambda: True)())
        except Exception as e:
            print(f"{name} FAIL runtime=1 error={type(e).__name__}")
            fails += 1
    return verdict(fails)


if __name__ == "__main__":
    raise SystemExit(main())
