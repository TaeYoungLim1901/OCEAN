"""P2 사전 게이트: data | pipeline | cases | sub <csv> | hashes | final <csv>."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

HASHES = {
    "observations.csv": "CC5D6FD9EA398A613E485B34FD530C7DC064FA040FA675127A12318856AB178A",
    "test_index.csv": "2397AC4A240E92CD1F75194A92A3ADBFFB7F9C681739C6F24CAC481AB8C26FF0",
    "baseline_interp.csv": "AE8097BE1F0F1BA8C4D61DE9741162106262C826B42636AE963C6C9EEBD1F036",
    "sample_submission.csv": "3804E83D703E2642351C1191CC8CD3C1D0F485795D7F843A096CB6855360362B",
}


def verdict(fails):
    for msg in fails:
        print("  FAIL " + msg)
    print("판정: " + ("OK" if not fails else "STOP"))
    return int(bool(fails))


def data():
    fails = []
    for name, expected in HASHES.items():
        got = hashlib.sha256((C.DATA / name).read_bytes()).hexdigest().upper()
        if got != expected:
            fails.append(f"원본 해시 {name}")
    d = C.load_obs()
    if list(d.columns) != ["station", "year", "layer", "time", "temp", "psal", "depth", "nominal_depth"]:
        fails.append("observations 스키마")
    if len(d) != 789408 or d.duplicated(C.KEY).any():
        fails.append("observations 행 수/키")
    mask = d.time.between(pd.Timestamp("2025-09-01", tz="Asia/Seoul"), pd.Timestamp("2025-10-31 23:50", tz="Asia/Seoul"))
    hidden = d[mask & d.layer.isin(C.TARGET_LAYERS)]
    if hidden.temp.notna().any() or hidden.psal.notna().any() or len(hidden) != 26352:
        fails.append("2025 target mask")
    print(f"  rows={len(d):,} hidden={len(hidden):,} target_depth={hidden.depth.notna().sum():,}")
    return verdict(fails)


def pipeline():
    fails = []
    obs = C.load_obs()
    rows = C.test_rows(obs)
    base = pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
    x = C.build_features(obs, rows, base)
    delta = np.max(np.abs(x.base_nom.to_numpy() - base.temp.to_numpy()))
    print(f"  baseline 생산경로 최대차={delta:.10f} n={len(x):,}")
    if delta != 0 or not np.isfinite(x.to_numpy(dtype=float)).any(axis=1).all():
        fails.append("baseline 생산경로/피처")
    return verdict(fails)


def cases():
    fails = []
    obs = C.load_obs()
    train, val = C.split_rows(obs)
    if val.time.min() != pd.Timestamp("2024-09-01", tz="Asia/Seoul") or val.time.max() != pd.Timestamp("2024-10-31 23:50", tz="Asia/Seoul"):
        fails.append("holdout 경계")
    if set(val.layer) != set(C.TARGET_LAYERS) or train.time.between(val.time.min(), val.time.max()).any():
        fails.append("동시 3층 mask/학습 겹침")
    counts = val.layer.value_counts().sort_index().to_dict()
    print(f"  train={len(train):,} val={len(val):,} layers={counts}")
    return verdict(fails)


def sub(path):
    fails = []
    s = pd.read_csv(path)
    idx = pd.read_csv(C.DATA / "test_index.csv")
    if list(s.columns) != C.KEY + ["temp"]:
        fails.append(f"열 순서 {list(s.columns)}")
        return verdict(fails)
    m = idx[C.KEY].merge(s, on=C.KEY, how="outer", indicator=True, validate="one_to_one")
    p = pd.to_numeric(m.temp, errors="coerce").to_numpy()
    if len(s) != 26061 or not m._merge.eq("both").all():
        fails.append("키/행 수")
    if not np.isfinite(p).all() or not ((p >= -5) & (p <= 45)).all():
        fails.append("temp 범위/유한")
    print(f"  n={len(s):,} mean={np.nanmean(p):.4f} range=[{np.nanmin(p):.3f},{np.nanmax(p):.3f}]")
    return verdict(fails)


def hashes():
    record = C.ROOT / "cache" / "data_sha256.json"
    payload = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(C.DATA.glob("*.csv"))}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  recorded={len(payload):,} path={record}")
    return verdict([])


def _relative(path):
    return Path(path).resolve().relative_to(C.ROOT.parent.resolve()).as_posix()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def final(path):
    csv_path = Path(path).resolve()
    model_path = C.ROOT / "models" / "E002.pkl"
    code_paths = [C.ROOT / "src" / name for name in
                  ("check.py", "common.py", "inference_e002.py", "submission_tools.py")]
    if not csv_path.exists() or not model_path.exists() or any(not p.exists() for p in code_paths):
        return verdict(["final 입력/코드/가중치 누락"])
    pred_path = C.ROOT / "preds" / "test_fixed_depth_E002.npy"
    with tempfile.TemporaryDirectory(prefix="p2_final_") as td_name:
        isolated = Path(td_name) / "P2"
        (isolated / "src").mkdir(parents=True)
        (isolated / "models").mkdir()
        (isolated / "preds").mkdir()
        (isolated / "subs").mkdir()
        shutil.copytree(C.DATA, isolated / "P2_data")
        for source in code_paths:
            shutil.copy2(source, isolated / "src" / source.name)
        shutil.copy2(model_path, isolated / "models" / model_path.name)
        run = subprocess.run(
            [sys.executable, str(isolated / "src" / "inference_e002.py")],
            cwd=isolated, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        reproduced = isolated / "subs" / csv_path.name
        generated_pred = isolated / "preds" / pred_path.name
        if run.returncode or not reproduced.exists() or not generated_pred.exists():
            return verdict([f"final inference failed rc={run.returncode}: {(run.stdout + run.stderr).strip()[-240:]}"])
        expected, got = _sha256(csv_path), _sha256(reproduced)
        pred_match = generated_pred.read_bytes() == pred_path.read_bytes()
        print(f"  csv_sha256={got} expected={expected} match={got == expected} pred_bytes_match={pred_match}")
        if got != expected or not pred_match:
            return verdict(["final 재현 CSV/NPY 불일치"])
        manifest = {
            "csv": _relative(csv_path),
            "csv_sha256": expected,
            "pred_npy": _relative(pred_path),
            "pred_sha256": _sha256(pred_path),
            "code_files": {_relative(p): _sha256(p) for p in code_paths},
            "weights": {_relative(model_path): _sha256(model_path)},
            "seed": C.SEED,
            "created": datetime.now(timezone.utc).isoformat(),
        }
    manifest_path = csv_path.with_name(csv_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  manifest={_relative(manifest_path)}")
    return verdict([])


def reproduce():
    from inference_e002 import calculate
    pred, sub, artifact = calculate()
    saved = np.load(C.ROOT / "preds" / f"test_{artifact['selected']}_E002.npy")
    disk = pd.read_csv(C.ROOT / "subs" / "E002_final.csv")
    d_npy = float(np.max(np.abs(pred - saved)))
    d_sub = float(np.max(np.abs(sub.temp.to_numpy() - disk.temp.to_numpy())))
    keys = sub[C.KEY].equals(disk[C.KEY])
    print(f"  model→npy 최대차={d_npy:.12g} model→sub 최대차={d_sub:.12g} keys={keys}")
    return verdict([] if d_npy == 0 and d_sub < 1e-12 and keys else ["E002 재현 불일치"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in {"data", "pipeline", "cases", "reproduce", "hashes"}:
        raise SystemExit(globals()[cmd]())
    if cmd == "sub" and len(sys.argv) == 3:
        raise SystemExit(sub(sys.argv[2]))
    if cmd == "final" and len(sys.argv) == 3:
        raise SystemExit(final(sys.argv[2]))
    raise SystemExit(__doc__)
