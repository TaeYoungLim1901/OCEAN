"""P1 사전/생산 게이트. `판정: OK`가 아니면 진행 금지."""
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "P1_data"
KEY = ["station", "year", "layer", "time"]


def verdict(fails):
    for msg in fails:
        print(f"  FAIL {msg}")
    print("판정: " + ("OK" if not fails else "STOP"))
    return 1 if fails else 0


def data():
    fails = []
    expected = {
        "train.csv": (776706, KEY + ["temp", "psal", "depth", "label", "anomaly_type"]),
        "test.csv": (169011, KEY + ["temp", "psal", "depth"]),
    }
    for name, (rows, cols) in expected.items():
        d = pd.read_csv(DATA / name)
        if list(d.columns) != cols:
            fails.append(f"{name} 열 {list(d.columns)}")
        if len(d) != rows:
            fails.append(f"{name} 행 {len(d):,} != {rows:,}")
        if d[KEY].isna().any().any() or d.duplicated(KEY).any():
            fails.append(f"{name} 키 결측/중복")
        if d.temp.isna().any():
            fails.append(f"{name} temp 결측")
        t = pd.to_datetime(d.time, utc=True)
        backward = 0
        for _, idx in d.groupby(["station", "year", "layer"], sort=False).groups.items():
            backward += int((t.loc[idx].diff().dropna() <= pd.Timedelta(0)).sum())
        if backward:
            fails.append(f"{name} stream 내 시각 비증가 {backward}")
        print(f"  {name} {len(d):,}행 temp결측={int(d.temp.isna().sum())}")
    tr = pd.read_csv(DATA / "train.csv", usecols=KEY + ["label"])
    if not tr.label.isin([0, 1]).all():
        fails.append("train label은 0/1이 아님")
    print(f"  train 양성률 {tr.label.mean():.6f}")
    return verdict(fails)


def sub(path):
    fails = []
    s = pd.read_csv(path)
    idx = pd.read_csv(DATA / "sample_submission.csv", usecols=KEY)
    if list(s.columns) not in [KEY + ["label"], KEY + ["label", "anomaly_type"]]:
        fails.append(f"열 이름/순서 {list(s.columns)}")
        return verdict(fails)
    if len(s) != len(idx) or s.duplicated(KEY).any() or s[KEY].isna().any().any():
        fails.append("행 수 또는 키 결측/중복")
    a = idx.merge(s[KEY], on=KEY, how="outer", indicator=True)
    if not a._merge.eq("both").all():
        fails.append(f"sample 키 불일치 {int(a._merge.ne('both').sum())}")
    y = pd.to_numeric(s.label, errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        fails.append("label은 정수 0/1이어야 함")
    print(f"  제출 {len(s):,}행 양성 {int(y.sum()):,} ({y.mean():.4%})")
    return verdict(fails)


def pipeline(sub_path, oof_path, test_path):
    fails = []
    oof, test = np.load(oof_path), np.load(test_path)
    if oof.shape != (776706,):
        fails.append(f"OOF shape {oof.shape}")
    if test.shape != (169011,):
        fails.append(f"test shape {test.shape}")
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        fails.append("예측에 비유한 값")
    s = pd.read_csv(sub_path)
    if len(test) == len(s) and not np.array_equal((test >= 0.5).astype(int), s.label.to_numpy()):
        fails.append("test 예측 threshold 0.5와 제출 label 불일치")
    print(f"  OOF/test 범위 [{oof.min():.4f},{oof.max():.4f}] / [{test.min():.4f},{test.max():.4f}]")
    return verdict(fails)


def hashes():
    files = {}
    for path in sorted(DATA.iterdir()):
        if path.is_file():
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    (ROOT / "data_hashes.json").write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
    for name, value in files.items():
        print(f"  {name} {value}")
    return verdict([])


def final(path):
    path = Path(path).resolve()
    stem = path.stem
    pred = ROOT / "preds" / f"test_{stem}.npy"
    weights = sorted((ROOT / "models").glob(f"{stem}_fold*.txt"))
    if stem != "exp002" or not pred.exists() or len(weights) != 4:
        return verdict([f"final artifact set missing for {stem}"])
    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        code = ROOT / "src/final_inference.py"
        (td / "final_inference.py").write_bytes(code.read_bytes())
        (td / "features.py").write_bytes((ROOT / "src/features.py").read_bytes())
        (td / "train.py").write_bytes((ROOT / "src/train.py").read_bytes())
        weight_dir = td / "weights"
        weight_dir.mkdir()
        for weight in weights:
            (weight_dir / weight.name).write_bytes(weight.read_bytes())
        reproduced = td / "reproduced.csv"
        run = subprocess.run(
            [sys.executable, str(td / "final_inference.py"), "--data-root", str(DATA),
             "--weights-dir", str(weight_dir), "--output", str(reproduced)],
            cwd=td, capture_output=True, text=True, encoding="utf-8"
        )
        if run.returncode or not reproduced.exists():
            return verdict([f"final inference failed rc={run.returncode}: {run.stderr.strip()[-240:]}"])
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
        expected_sha, actual_sha = sha(path), sha(reproduced)
        if actual_sha != expected_sha:
            return verdict([f"final sha256 mismatch expected={expected_sha} actual={actual_sha}"])
        manifest = {
            "csv": str(path.relative_to(ROOT.parent)).replace("\\", "/"),
            "csv_sha256": expected_sha,
            "pred_npy": str(pred.relative_to(ROOT.parent)).replace("\\", "/"),
            "pred_sha256": sha(pred),
            "code_files": {f"P1/src/{name}": sha(ROOT / "src" / name) for name in ("final_inference.py", "features.py", "train.py")},
            "weights": {str(w.relative_to(ROOT.parent)).replace("\\", "/"): sha(w) for w in weights},
            "seed": 42,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        (path.parent / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"  reproduced_sha256={actual_sha}")
        return verdict([])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "data":
        sys.exit(data())
    if cmd == "sub":
        sys.exit(sub(sys.argv[2]))
    if cmd == "pipeline":
        sys.exit(pipeline(*sys.argv[2:5]))
    if cmd == "hashes":
        sys.exit(hashes())
    if cmd == "final" and len(sys.argv) == 3:
        sys.exit(final(sys.argv[2]))
    if cmd == "reproduce" and len(sys.argv) == 3 and sys.argv[2] == "exp002":
        from exp002 import reproduce
        sys.exit(reproduce())
    raise SystemExit(__doc__)
