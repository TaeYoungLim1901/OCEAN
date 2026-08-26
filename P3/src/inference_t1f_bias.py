"""Reproduce T1F with the sealed global bias correction."""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

import cases as C
from features import extract
from submission_tools import save_pool_pred
from train import predict_damped, predict_lgbm

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
WEIGHT_LGBM = 0.65
C_BIAS = -0.041550308


def main():
    with open(P3 / "models" / "p3_t1" / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    rows, features = [], []
    for cid, station, context in C.iter_test_contexts():
        rows.append((cid, station, float(context[-1, 0])))
        features.append(extract(context))
    cases = pd.DataFrame(rows, columns=["case_id", "station", "hs0"])
    x = np.stack(features)
    damped = predict_damped(cases, bundle["damped"])
    lgbm = predict_lgbm(bundle["lgbm"], x, cases)
    final = np.clip((1 - WEIGHT_LGBM) * damped + WEIGHT_LGBM * lgbm + C_BIAS, 0, 30)
    index = pd.read_csv(C.DATA / "test_index.csv")
    values = {(cid, station, lead): float(final[i, j])
              for i, (cid, station, _) in cases.iterrows()
              for j, lead in enumerate(C.LEADS)}
    index["hs_pred"] = [str(float(values[tuple(row)]))
                         for row in index.itertuples(index=False, name=None)]
    pred_path = save_pool_pred(P3 / "preds" / "test_final_p3_t1f_bias.npy",
                               index["hs_pred"].to_numpy(float))
    out = P3 / "subs" / "submission_p3_t1f_bias.csv"
    index.to_csv(out, index=False, lineterminator="\r\n")
    print(f"submission={out} rows={len(index)} bias={C_BIAS:+.9f} pred={pred_path}")


if __name__ == "__main__":
    main()
