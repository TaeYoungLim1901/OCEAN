"""E001 모델 추론과 제출 생성."""
import pickle
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from submission_tools import save_pool_pred


def predict_one(name, artifact, x, rows):
    columns = artifact["basic"] if name == "lgb_basic" else artifact["full"]
    pred = np.empty(len(rows))
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        model = artifact["models"][name][layer]
        pred[use] = x.loc[use, "base_nom"] + model.predict(x.loc[use, columns])
    return pred


def main():
    with open(C.ROOT / "models" / "E001.pkl", "rb") as f:
        artifact = pickle.load(f)
    obs, base = C.load_obs(), pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
    rows = C.test_rows(obs)
    x = C.build_features(obs, rows, base)
    singles = {name: predict_one(name, artifact, x, rows) for name in ("ridge_full", "lgb_basic", "lgb_full")}
    for name, pred in singles.items():
        save_pool_pred(C.ROOT / "preds" / f"test_{name}_E001.npy", pred)
    selected = artifact["selected"]
    if selected.startswith("blend_"):
        match = re.fullmatch(r"blend_(ridge_full|lgb_basic)_(lgb_full)_(0\.25|0\.50|0\.75)", selected)
        if not match:
            raise ValueError(f"알 수 없는 blend: {selected}")
        a, b, w = match.group(1), match.group(2), float(match.group(3))
        pred = (1 - w) * singles[a] + w * singles[b]
    else:
        pred = singles[selected]
    save_pool_pred(C.ROOT / "preds" / f"test_{selected}_E001.npy", pred)
    # 채점기는 time 문자열까지 키로 비교하므로 배포 index의 원문 키를 그대로 보존한다.
    sub = pd.read_csv(C.DATA / "test_index.csv")[C.KEY].assign(temp=np.clip(pred, -5, 45))
    path = C.ROOT / "subs" / "E001_lgb_full.csv"
    sub.to_csv(path, index=False)
    print(f"selected={selected} n={len(sub):,} mean={sub.temp.mean():.4f} path={path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
