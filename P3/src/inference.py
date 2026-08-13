"""Reproduce P3 T1 test predictions and submission from saved models."""
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

import cases as C
from features import extract
from train import predict_damped, predict_lgbm

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
EXP = "p3_t1"


def main():
    with open(P3 / "models" / EXP / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    rows, feats = [], []
    for cid, st, ctx in C.iter_test_contexts():
        rows.append((cid, st, float(ctx[-1, 0])))
        feats.append(extract(ctx))
    cases = pd.DataFrame(rows, columns=["case_id", "station", "hs0"])
    x = np.stack(feats)
    damped = predict_damped(cases, bundle["damped"])
    lgbm = predict_lgbm(bundle["lgbm"], x, cases)
    final = (1 - bundle["weight"]) * damped + bundle["weight"] * lgbm
    pred_dir = P3 / "preds"
    sub_dir = P3 / "subs"
    pred_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)
    np.save(pred_dir / "test_damped_p3_t1.npy", damped)
    np.save(pred_dir / "test_lgbm_p3_t1.npy", lgbm)
    np.save(pred_dir / "test_final_p3_t1.npy", final)
    values = {(cid, st, lead): float(final[i, j]) for i, (cid, st, _) in cases.iterrows()
              for j, lead in enumerate(C.LEADS)}
    index = pd.read_csv(C.DATA / "test_index.csv")
    index["hs_pred"] = [values[tuple(row)] for row in index.itertuples(index=False, name=None)]
    index["hs_pred"] = index.hs_pred.clip(0, 30)
    out = sub_dir / "submission_p3_t1.csv"
    index.to_csv(out, index=False)
    print(f"submission={out} rows={len(index)} weight_lgbm={bundle['weight']:.2f}")


if __name__ == "__main__":
    main()
