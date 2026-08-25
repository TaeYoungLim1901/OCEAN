import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from submission_tools import save_pool_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--delta-from", required=True)
    parser.add_argument("--delta-to", required=True)
    parser.add_argument("--w", required=True, type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out}")

    arrays = [np.load(path, allow_pickle=False) for path in
              (args.base, args.delta_from, args.delta_to)]
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError(f"예측 shape 오류: {[array.shape for array in arrays]}")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("입력 예측에 비유한값이 있습니다")

    pred = arrays[0] + args.w * (arrays[2] - arrays[1])
    if not np.isfinite(pred).all():
        raise ValueError("출력 예측에 비유한값이 있습니다")
    if not ((pred >= -5) & (pred <= 45)).all():
        raise ValueError("출력 예측 범위를 벗어난 값이 있습니다")
    save_pool_pred(out, pred)
    print(f"out={out} rows={len(pred):,} w={args.w:g}")


if __name__ == "__main__":
    main()
