"""E003 저장 모델에서 후보 npy 및 통과 시 제출 CSV 생성."""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from train_e003 import predict_eof, predict_layers, regimes
from submission_tools import save_pool_pred


def calculate():
    with open(C.ROOT / "models" / "E003.pkl", "rb") as f:
        artifact = pickle.load(f)
    obs = C.load_obs()
    rows = C.test_rows(obs)
    base = pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
    x = C.build_features(obs, rows, base)
    xnd = C.build_features(obs, rows, base, use_target_depth=False)
    pred = {
        "e002": predict_layers(artifact["models"]["e002"], x, rows, C.FULL),
        "enhanced": predict_layers(artifact["models"]["enhanced"], x, rows, C.ENHANCED),
        "no_depth": predict_layers(artifact["models"]["no_depth"], xnd, rows, C.FULL),
    }
    erows = rows.copy(); erows["_base"] = x.base_nom.to_numpy()
    pred["eof"] = predict_eof(artifact["models"]["eof"], obs, obs, erows)
    w = artifact["final_blend_weight"]
    pred["robust_blend"] = (1 - w) * pred["enhanced"] + w * pred["eof"]
    pred["regime_gate"] = pred["enhanced"].copy()
    masks = regimes(x, rows)
    for regime, model in artifact["final_gate"].items():
        pred["regime_gate"][masks[regime]] = pred[model][masks[regime]]
    return artifact, rows, pred


def main():
    artifact, rows, predictions = calculate()
    for name, pred in predictions.items():
        save_pool_pred(C.ROOT / "preds" / f"test_{name}_E003.npy", pred)
    if not artifact["accepted"]:
        print(f"selected=E002; E003 submission not created; candidates={','.join(predictions)}")
        return
    pred = predictions[artifact["selected"]]
    save_pool_pred(C.ROOT / "preds" / "test_E003.npy", pred)
    keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
    sub = keys.assign(temp=np.clip(pred, -5, 45))
    path = C.ROOT / "subs" / "E003_final.csv"
    sub.to_csv(path, index=False)
    disk = pd.read_csv(path, float_precision="round_trip")
    if not disk[C.KEY].equals(keys) or not np.array_equal(disk.temp.to_numpy(), pred):
        raise RuntimeError("model→npy→submission exact gate 실패")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"selected={artifact['selected']} n={len(sub):,} exact=True sha256={digest}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
