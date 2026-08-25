import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy")
    parser.add_argument("out")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out}")

    keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
    pred = np.load(args.npy, allow_pickle=False)
    if pred.ndim != 1:
        raise ValueError(f"예측 배열이 1차원이 아닙니다: {pred.shape}")
    if len(pred) != len(keys):
        raise ValueError(f"예측 길이 불일치: {len(pred)} != {len(keys)}")
    if not np.isfinite(pred).all():
        raise ValueError("예측에 비유한값이 있습니다")
    if not ((pred >= -5) & (pred <= 45)).all():
        raise ValueError("예측 범위를 벗어난 값이 있습니다")

    keys.assign(temp=pred).to_csv(out, index=False)


if __name__ == "__main__":
    main()
