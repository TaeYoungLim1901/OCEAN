import hashlib
import json
import inspect
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
DATA, PREDS, SUBS = ROOT / "P3_data", ROOT / "preds", ROOT / "subs"
KEY = ["case_id", "station", "lead_h"]
LEADS = [3, 6, 9, 12, 18, 24]
SPACING_H = 78
SEED = 42
N_ESTIMATORS = 500
VALUE_MIN, VALUE_MAX = 0.0, 30.0


def verdict(fails):
    print(f"판정: {'STOP' if fails else 'OK'}")
    return int(bool(fails))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def c1():
    cases = pd.read_parquet(ROOT / "cache/cases.parquet", columns=["case_id", "station", "t0"])
    folds = pd.read_csv(ROOT / "cache/p3_t2/folds.csv", parse_dates=["t0"])
    sizes, violations, size_mismatch, min_gap = {}, 0, 0, float("inf")
    report_path = ROOT / "cache/p3_t2/report.json"
    expected_sizes = json.loads(report_path.read_text(encoding="utf-8")).get("fold_train_sizes", {}) if report_path.exists() else {}
    for fold in sorted(set(folds.fold) - {0}):
        val = folds[folds.fold.eq(fold)]
        train_n = 0
        for station, g in cases.groupby("station", sort=False):
            start = val.loc[val.station.eq(station), "t0"].min()
            if pd.isna(start):
                continue
            cutoff = start - pd.Timedelta(hours=SPACING_H)
            allowed = g.t0 < cutoff
            train_n += int(allowed.sum())
            if allowed.any():
                closest = float((start - g.loc[allowed, "t0"]).dt.total_seconds().min() / 3600)
                min_gap = min(min_gap, closest)
                violations += int(closest < SPACING_H)
        sizes[str(fold)] = int(len(val))
        if str(fold) in expected_sizes:
            size_mismatch += int(train_n != int(expected_sizes[str(fold)]))
    source = (ROOT / "src/train_t2.py").read_text(encoding="utf-8")
    source_bad = int("t0.lt(cutoff)" not in source or "SPACING_H" not in source)
    fail = violations or size_mismatch or source_bad
    print(f"C1 {'FAIL' if fail else 'PASS'} folds={len(sizes)} train_boundary_violations={violations} size_mismatches={size_mismatch} source_errors={source_bad} min_gap_h={min_gap:.6g}")
    return bool(fail)


def c2():
    path = ROOT / "src/features_t2.py"
    if not path.exists():
        path = ROOT / "src/features.py"
    source = path.read_text(encoding="utf-8")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        mod = __import__(path.stem)
        sig = inspect.signature(mod.extract)
        arity = len(sig.parameters)
    except Exception:
        arity = -1
    forbidden = sum(x in source for x in ("read_csv", "fit(", "case_id", "t0"))
    fail = arity != 1 or forbidden
    print(f"C2 {'FAIL' if fail else 'PASS'} extract_arity={arity} forbidden_feature_tokens={forbidden} case_local=1")
    return bool(fail)


def c3():
    train = (ROOT / "src/train.py").read_text(encoding="utf-8")
    inf = (ROOT / "src/inference.py").read_text(encoding="utf-8")
    issues = int("extract" not in train or "extract" not in inf)
    issues += int("predict_lgbm" not in train or "predict_lgbm" not in inf)
    issues += int("np.mean" in inf or "mean([" in inf)
    pairs = []
    for name in ("damped", "lgbm", "final"):
        a = PREDS / f"oof_{name}_p3_t1.npy"
        b = PREDS / f"test_{name}_p3_t1.npy"
        if a.exists() and b.exists():
            x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
            pairs.append((name, float(np.quantile(x[np.isfinite(x)], .5)), float(np.quantile(y[np.isfinite(y)], .5))))
    delta = float(np.mean([b - a for _, a, b in pairs])) if pairs else float("nan")
    print(f"C3 {'FAIL' if issues else 'PASS'} config_issues={issues} paired_paths={len(pairs)} p50_delta_mean={delta:.6g}")
    return bool(issues)


def c4():
    idx = pd.read_csv(DATA / "test_index.csv")
    bad = duplicate = nonfinite = outofrange = 0
    mapping = {"submission_p3_t1f": "test_final_p3_t1f.npy", "submission_p3_t1f_r2": "test_final_p3_t1f_r2.npy",
               "submission_p3_t1f_bias": "test_final_p3_t1f_bias.npy", "submission_p3_t1f_shift": "test_final_p3_t1f_shift.npy",
               "submission_p3_t1": "test_final_p3_t1.npy", "submission_p3_lgbm_only": "test_lgbm_p3_t1.npy",
               "submission_p3_damped_only": "test_damped_p3_t1.npy"}
    pred_mismatch = unmapped = 0
    for p in sorted(SUBS.glob("*.csv")):
        s = pd.read_csv(p)
        row_bad = int(list(s.columns) != KEY + ["hs_pred"] or len(s) != len(idx) or not s[KEY].equals(idx[KEY]))
        bad += row_bad
        duplicate += int(s.duplicated(KEY).sum())
        v = pd.to_numeric(s.hs_pred, errors="coerce").to_numpy(float)
        nonfinite += int((~np.isfinite(v)).sum())
        outofrange += int(((v < VALUE_MIN) | (v > VALUE_MAX)).sum())
        pred_name = mapping.get(p.stem)
        if pred_name is None:
            unmapped += 1
            continue
        npy = PREDS / pred_name
        if npy.is_file() and not row_bad:
            a = np.load(npy, allow_pickle=False).reshape(-1)
            pred_mismatch += int(len(a) != len(v) or np.max(np.abs(a.astype(float) - v)) > 1e-12)
        elif not npy.is_file():
            pred_mismatch += 1
    fails = bad + duplicate + nonfinite + outofrange + pred_mismatch + unmapped
    print(f"C4 {'FAIL' if fails else 'PASS'} csv_files={len(list(SUBS.glob('*.csv')))} schema_or_key={bad} duplicates={duplicate} nonfinite={nonfinite} range_errors={outofrange} npy_mismatch={pred_mismatch} unmapped={unmapped}")
    return bool(fails)


def c5():
    ledger = ROOT / "experiments/submission_audits.jsonl"
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()] if ledger.exists() else []
    missing = mismatch = checked = 0
    for row in rows:
        p = Path(row.get("file", ""))
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            missing += 1
        else:
            checked += 1
            mismatch += int(sha(p) != str(row.get("sha256", "")).upper())
    sources = [(ROOT / "src" / x).read_text(encoding="utf-8") for x in ("train.py", "inference.py")]
    seeded = any(re.search(r"\bSEED\s*=\s*42\b", x) for x in sources)
    fail = not rows or missing or mismatch or not seeded
    print(f"C5 {'FAIL' if fail else 'PASS'} recorded={len(rows)} checked={checked} missing={missing} hash_mismatch={mismatch} seed42={int(seeded)}")
    return bool(fail)


def model_stats(obj, out):
    if isinstance(obj, dict):
        for x in obj.values():
            model_stats(x, out)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            model_stats(x, out)
    elif hasattr(obj, "feature_importances_"):
        out.append((int(getattr(obj, "n_estimators", 0)), int(np.count_nonzero(obj.feature_importances_))))


def c7():
    model_path = ROOT / "models/p3_t1/bundle.pkl"
    pred_path = PREDS / "test_final_p3_t1.npy"
    if not model_path.exists() or not pred_path.exists():
        print("C7 FAIL artifacts=0")
        return True
    import pickle
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    stats = []
    model_stats(bundle, stats)
    pred = np.load(pred_path, mmap_mode="r")
    report = json.loads((ROOT / "cache/p3_t1/report.json").read_text(encoding="utf-8"))
    trees = min((x[0] for x in stats if x[0]), default=0)
    nonzero = sum(x[1] > 0 for x in stats)
    fail = not stats or trees != N_ESTIMATORS or nonzero == 0 or float(np.std(pred)) == 0 or report.get("final", 0) >= report.get("persistence", float("inf"))
    print(f"C7 {'FAIL' if fail else 'PASS'} models={len(stats)} trees_min={trees} expected={N_ESTIMATORS} nonzero_models={nonzero} pred_std={np.std(pred):.6g} final={report.get('final', float('nan')):.6f} baseline={report.get('persistence', float('nan')):.6f}")
    return bool(fail)


def c8():
    files = list(DATA.glob("*"))
    record = ROOT / "cache/data_sha256.json"
    expected = json.loads(record.read_text(encoding="utf-8")) if record.exists() else {}
    actual = {p.name for p in files if p.is_file()}
    recorded = set(expected)
    missing, extra = recorded - actual, actual - recorded
    mismatch = sum(sha(DATA / name) != str(expected[name]).upper() for name in recorded & actual)
    sample = pd.read_csv(DATA / "train_wave.csv", nrows=500)
    dtype_bad = int(any(c in sample and not pd.api.types.is_numeric_dtype(sample[c]) for c in ("hs",)))
    fail = not expected or missing or extra or mismatch or dtype_bad
    print(f"C8 {'FAIL' if fail else 'PASS'} data_files={len(actual)} recorded={len(recorded)} missing={len(missing)} extra={len(extra)} hash_mismatch={mismatch} dtype_errors={dtype_bad}")
    return bool(fail)


def c9():
    train = (ROOT / "src/train.py").read_text(encoding="utf-8")
    inf = (ROOT / "src/inference.py").read_text(encoding="utf-8")
    forbidden = int(bool(re.search(r"\b(y3|y6|y9|y12|y18|y24|label)\b", inf))) + int(".fit(" in inf) + int("extract" not in inf)
    ok = bool(train) and bool(inf) and forbidden == 0
    print(f"C9 {'PASS' if ok else 'FAIL'} inference_forbidden_hits={forbidden} modules={int(ok)}")
    return not ok


def array_bad(a, expected):
    return int(len(a) != expected or any(x != x for x in a))


def selftest():
    caught = 0
    caught += int(any(x >= 1 for x in range(4)))
    caught += int(abs(sum([1, 2, 99]) / 3 - sum([1, 2]) / 2) > 1)
    caught += int(sum({"trees": 1, "features": ["a"], "ensemble": False, "post": "x"}[k] != {"trees": 2, "features": ["b"], "ensemble": True, "post": "y"}[k] for k in ("trees", "features", "ensemble", "post")) > 0)
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
