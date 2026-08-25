"""E002 저장 모델로 npy와 제출을 생성."""
import hashlib
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from submission_tools import save_pool_pred


def calculate():
    with open(C.ROOT / "models" / "E002.pkl", "rb") as f:
        artifact = pickle.load(f)
    obs = C.load_obs()
    rows = C.test_rows(obs)
    base = pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
    x = C.build_features(obs, rows, base, tide_correct=artifact["tide_correct"],
                         use_target_depth=artifact["use_target_depth"])
    pred = np.empty(len(rows))
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        model = artifact["models"][layer]
        pred[use] = x.loc[use, "base_nom"] + model.predict(x.loc[use, artifact["features"]])
    keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
    return pred, keys.assign(temp=np.clip(pred, -5, 45)), artifact


def main():
    pred, sub, artifact = calculate()
    save_pool_pred(C.ROOT / "preds" / f"test_{artifact['selected']}_E002.npy", pred)
    path = C.ROOT / "subs" / "E002_final.csv"
    sub.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"selected={artifact['selected']} n={len(sub):,} sha256={digest}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
