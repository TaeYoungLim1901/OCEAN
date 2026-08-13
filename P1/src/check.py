"""P1 사전/생산 게이트. `판정: OK`가 아니면 진행 금지."""
import sys
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "data":
        sys.exit(data())
    if cmd == "sub":
        sys.exit(sub(sys.argv[2]))
    if cmd == "pipeline":
        sys.exit(pipeline(*sys.argv[2:5]))
    if cmd == "reproduce" and len(sys.argv) == 3 and sys.argv[2] == "exp002":
        from exp002 import reproduce
        sys.exit(reproduce())
    raise SystemExit(__doc__)
