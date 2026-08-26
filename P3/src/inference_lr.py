"""Reproduce the saved T26 log-ratio submission."""
import pathlib
import sys

import numpy as np
import pandas as pd

import cases as C
import train_target_transform as T

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent


def main():
    T.main(["--max-folds", "4", "--n-jobs", "2"])
    pred = np.load(P3 / "preds" / "test_lr_p3_t26.npy", allow_pickle=False)
    test_cases = list(C.iter_test_contexts())
    if pred.shape != (len(test_cases), len(C.LEADS)) or not np.isfinite(pred).all():
        raise ValueError(f"prediction shape/finite mismatch: {pred.shape}")
    if ((pred < 0) | (pred > 30)).any():
        raise ValueError("prediction outside [0, 30]")
    values = {(cid, station, lead): float(pred[i, j])
              for i, (cid, station, _) in enumerate(test_cases)
              for j, lead in enumerate(C.LEADS)}
    index = pd.read_csv(C.DATA / "test_index.csv")
    index["hs_pred"] = [str(values[tuple(row)])
                         for row in index.itertuples(index=False, name=None)]
    out = P3 / "subs" / "submission_p3_lr.csv"
    index.to_csv(out, index=False, lineterminator="\r\n")
    print(f"submission={out} rows={len(index)} pred={P3 / 'preds' / 'test_lr_p3_t26.npy'}")


if __name__ == "__main__":
    main()
