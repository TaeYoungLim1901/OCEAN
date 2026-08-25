"""Deterministic inference for all saved P3 T2 candidates."""
import hashlib
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

import cases as C
from features_t2 import extract, feature_names
from submission_tools import save_pool_pred
from train_t2 import CANDIDATES, combine, predict_models

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
EXP = "p3_t2"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    with open(P3 / "models" / EXP / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    if bundle["features"] != feature_names():
        raise RuntimeError("feature contract mismatch")
    rows, feats = [], []
    for case_id, station, ctx in C.iter_test_contexts():
        rows.append((case_id, station, float(ctx[-1, 0])))
        feats.append(extract(ctx))
    cases = pd.DataFrame(rows, columns=["case_id", "station", "hs0"])
    x = np.stack(feats)
    preds = predict_models(bundle["models"], x, cases)
    for name in CANDIDATES:
        save_pool_pred(P3 / "preds" / f"test_{name}_p3_t2.npy", preds[name])
    final = combine(preds, bundle["selection"])
    final = np.load(save_pool_pred(P3 / "preds" / "test_final_p3_t2.npy", final), allow_pickle=False)
    info = {"accepted": bool(bundle["accepted"]), "selection": bundle["selection"],
            "test_pred_sha256": sha256(P3 / "preds" / "test_final_p3_t2.npy")}
    if bundle["accepted"]:
        index = pd.read_csv(P3 / "P3_data" / "test_index.csv")
        if index.case_id.drop_duplicates().tolist() != cases.case_id.tolist():
            raise RuntimeError("canonical case order mismatch")
        sub = index.copy()
        sub["hs_pred"] = final.ravel()
        path = P3 / "subs" / "submission_p3_t2.csv"
        sub.to_csv(path, index=False)
        info.update({"submission": str(path), "submission_sha256": sha256(path)})
    (P3 / "cache" / EXP / "inference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
