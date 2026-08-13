"""Reproduce accepted P3 T1F blend from the saved T1 component models."""
import json
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


def main():
    report = json.loads((P3 / "cache" / "p3_t1" / "t1f_report.json").read_text(encoding="utf-8"))
    if not report["outer_pass"]:
        raise SystemExit("outer rule failed: T1F is not eligible")
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
    for name, fresh in (("damped", damped), ("lgbm", lgbm)):
        saved = np.load(P3 / "preds" / f"test_{name}_p3_t1.npy")
        delta = float(np.max(np.abs(fresh - saved)))
        if delta != 0:
            raise ValueError(f"{name} component mismatch: {delta}")
    weight = report["selected_weight_lgbm"]
    final = (1 - weight) * damped + weight * lgbm
    np.save(P3 / "preds" / "test_final_p3_t1f.npy", final)
    values = {(cid, station, lead): float(final[i, j])
              for i, (cid, station, _) in cases.iterrows()
              for j, lead in enumerate(C.LEADS)}
    index = pd.read_csv(C.DATA / "test_index.csv")
    index["hs_pred"] = [values[tuple(row)] for row in index.itertuples(index=False, name=None)]
    out = P3 / "subs" / "submission_p3_t1f.csv"
    index.to_csv(out, index=False)
    print(f"submission={out} rows={len(index)} weight_lgbm={weight:.2f} component_max_delta=0")


if __name__ == "__main__":
    main()
