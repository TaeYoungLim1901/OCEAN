import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from gate_e004 import regimes
from submission_tools import save_pool_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", required=True)
    parser.add_argument("--complete", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-npy")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out}")
    out_npy = Path(args.out_npy) if args.out_npy else None
    if out_npy is not None and out_npy.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out_npy}")

    keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
    obs = C.load_obs()
    test_rows = C.test_rows(obs)
    flags = regimes(C.build_features(obs, test_rows), test_rows)["missing_public"]
    if int(flags.sum()) != 7766:
        raise ValueError(f"missing_public 행 수 오류: {int(flags.sum())} != 7766")

    missing = np.load(args.missing, allow_pickle=False)
    complete = np.load(args.complete, allow_pickle=False)
    expected = len(keys)
    for name, pred in (("missing", missing), ("complete", complete)):
        if pred.ndim != 1 or len(pred) != expected:
            raise ValueError(f"{name} 예측 shape 오류: {pred.shape}, expected=({expected},)")

    pred = np.where(flags, missing, complete)
    if not np.isfinite(pred).all():
        raise ValueError("예측에 비유한값이 있습니다")
    if not ((pred >= -5) & (pred <= 45)).all():
        raise ValueError("예측 범위를 벗어난 값이 있습니다")
    if out_npy is not None:
        save_pool_pred(out_npy, pred)
    keys.assign(temp=pred).to_csv(out, index=False)
    print(f"out={out} rows={len(pred):,} missing={int(flags.sum()):,} complete={int((~flags).sum()):,}")


if __name__ == "__main__":
    main()
