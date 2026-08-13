"""Wrap saved T1 component predictions as canonical submission CSVs."""
import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

P3 = pathlib.Path(__file__).resolve().parent.parent
KEY = ["case_id", "station", "lead_h"]
COLS = KEY + ["hs_pred"]
LEADS = [3, 6, 9, 12, 18, 24]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def make(name, index):
    pred_path = P3 / "preds" / f"test_{name}_p3_t1.npy"
    pred = np.load(pred_path, allow_pickle=False)
    cases = list(dict.fromkeys(zip(index.case_id, index.station)))
    if pred.shape != (len(cases), len(LEADS)):
        raise ValueError(f"{name}: shape {pred.shape} != {(len(cases), len(LEADS))}")
    if not np.isfinite(pred).all():
        raise ValueError(f"{name}: non-finite prediction")

    case_pos = {key: i for i, key in enumerate(cases)}
    lead_pos = {lead: i for i, lead in enumerate(LEADS)}
    values = np.array([pred[case_pos[(cid, station)], lead_pos[lead]]
                       for cid, station, lead in index[KEY].itertuples(index=False, name=None)])
    clipped = np.clip(values, 0.0, 30.0)
    clipped_rows = int(np.count_nonzero(clipped != values))
    out = P3 / "subs" / f"submission_p3_{name}_only.csv"
    pd.DataFrame(dict(zip(KEY, (index[c] for c in KEY))) | {"hs_pred": clipped}).to_csv(out, index=False)
    return out, clipped_rows


def main():
    index = pd.read_csv(P3 / "P3_data" / "test_index.csv")
    if list(index.columns) != KEY or len(index) != 1200:
        raise ValueError("test_index schema/row count mismatch")
    if any(g.lead_h.tolist() != LEADS for _, g in index.groupby(["case_id", "station"], sort=False)):
        raise ValueError("test_index case lead order mismatch")
    for name in ("lgbm", "damped"):
        out, clipped_rows = make(name, index)
        print(f"created={out} rows={len(index)} clipped_rows={clipped_rows} sha256={sha256(out)}")


if __name__ == "__main__":
    main()
