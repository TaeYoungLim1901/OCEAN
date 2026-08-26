"""Score saved dense OOF predictions with the T21 weighting path."""
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python P3/src/judge_dense.py <oof_npy> [<oof_npy> ...]")
        return

    import numpy as np
    import pandas as pd
    import run_t21 as R

    _, dense, _ = R.load_dense()
    y = R.target_matrix(dense)
    fold = dense.fold.to_numpy(int)
    weights = pd.read_csv(P3 / "cache" / "p3_t9" / "dense_weights.csv")
    if not np.array_equal(weights.case_id.to_numpy(), dense.case_id.to_numpy()):
        raise AssertionError("dense weight order mismatch")
    w = weights.w.to_numpy(float)

    print("변형 | 시드 | fold1 | fold2 | fold3 | fold4")
    for name in paths:
        path = pathlib.Path(name)
        pred = np.load(path, allow_pickle=False)
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"invalid OOF {path}: {pred.shape}")
        scores = [R.weighted_rmse(y[fold == f], pred[fold == f], w[fold == f])
                  for f in range(1, 5)]
        stem = path.stem
        variant = stem.removeprefix("oof_dense_").split("_p3_t26", 1)[0]
        seed = "42" if stem.endswith("p3_t26") else stem.rsplit("_s", 1)[-1]
        print(f"{variant} | {seed} | " + " | ".join(f"{score:.6f}" for score in scores))


if __name__ == "__main__":
    main()
