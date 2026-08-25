"""Rejudge T2 and preserve its saved prediction pool without training."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import cases as C
from submission_tools import save_pool_pred
from train_t2 import regime_labels

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
PREDS = P3 / "preds"
CACHE = P3 / "cache" / "p3_t5"
T2_CACHE = P3 / "cache" / "p3_t2"


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def segment_scores(cases, y, pred, regimes):
    out = {}
    bins = pd.cut(cases.hs0, [0, 1.75, 2.25, 3, 99], labels=False).to_numpy()
    for j, lead in enumerate(C.LEADS):
        out[f"lead:{lead}"] = {"rmse": rmse(y[:, j], pred[:, j]), "n": int(len(cases))}
    for station in C.STATIONS:
        k = cases.station.eq(station).to_numpy()
        if k.any():
            out[f"station:{station}"] = {"rmse": rmse(y[k], pred[k]), "n": int(k.sum())}
    for value in np.unique(bins[~pd.isna(bins)]):
        k = bins == value
        out[f"hs0bin:{int(value)}"] = {"rmse": rmse(y[k], pred[k]), "n": int(k.sum())}
    for regime in ("falling", "stable", "rising"):
        k = regimes == regime
        if k.any():
            out[f"regime:{regime}"] = {"rmse": rmse(y[k], pred[k]), "n": int(k.sum())}
    return out


def compare(cases, y, pred, baseline, regimes, mask):
    c = cases.loc[mask].reset_index(drop=True)
    a = segment_scores(c, y[mask], pred[mask], regimes[mask])
    b = segment_scores(c, y[mask], baseline[mask], regimes[mask])
    delta = {
        name: {
            "t2": a[name]["rmse"],
            "t1f": b[name]["rmse"],
            "delta": a[name]["rmse"] - b[name]["rmse"],
            "n": a[name]["n"],
        }
        for name in a
    }
    return {
        "cases": int(mask.sum()),
        "t2_rmse": rmse(y[mask], pred[mask]),
        "t1f_rmse": rmse(y[mask], baseline[mask]),
        "delta": rmse(y[mask], pred[mask]) - rmse(y[mask], baseline[mask]),
        "lead_rmse": {
            str(lead): {
                "t2": a[f"lead:{lead}"]["rmse"],
                "t1f": b[f"lead:{lead}"]["rmse"],
                "delta": delta[f"lead:{lead}"]["delta"],
                "n": a[f"lead:{lead}"]["n"],
            }
            for lead in C.LEADS
        },
        "segment_delta": dict(sorted(delta.items(), key=lambda item: item[1]["delta"], reverse=True)),
        "top5_worsening": [
            {"segment": name, **values}
            for name, values in list(sorted(delta.items(), key=lambda item: item[1]["delta"], reverse=True))[:5]
        ],
        "max_degradation_n_ge_20": max(
            values["delta"] for values in delta.values() if values["n"] >= 20
        ),
    }


def t1_leak_check(all_cases, val):
    """Reproduce train.py's mask and compare it with T2 strict-forward."""
    allowed_abs = np.ones(len(all_cases), bool)
    for station, group in val.groupby("station"):
        station_mask = all_cases.station.eq(station).to_numpy()
        for vt in group.t0:
            hours = (all_cases.t0 - vt).abs().dt.total_seconds().to_numpy() / 3600
            allowed_abs &= ~(station_mask & (hours < C.SPACING_H))
    allowed_strict = np.zeros(len(all_cases), bool)
    for station in C.STATIONS:
        start = val.loc[val.station.eq(station), "t0"].min()
        cutoff = start - pd.Timedelta(hours=C.SPACING_H)
        allowed_strict |= all_cases.station.eq(station).to_numpy() & all_cases.t0.lt(cutoff).to_numpy()
    source = (P3 / "src" / "train.py").read_text(encoding="utf-8")
    non_strict_mask = "abs()" in source and "t0.lt(cutoff)" not in source
    return {
        "strict_forward": False,
        "reason": "train.py uses symmetric abs(hours) mask, not strict t0 < fold_start-78h",
        "train_py_fold4_rows": int(allowed_abs.sum()),
        "strict_forward_fold4_rows": int(allowed_strict.sum()),
        "mask_difference_rows": int((allowed_abs != allowed_strict).sum()),
        "symmetric_abs_mask_detected": non_strict_mask,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t1_cases = pd.read_parquet(P3 / "cache" / "p3_t1" / "oof_cases.parquet")
    folds = pd.read_csv(T2_CACHE / "folds.csv")
    if not np.array_equal(t1_cases.case_id.to_numpy(), folds.case_id.to_numpy()):
        raise RuntimeError("T1F/T2 fold case alignment mismatch")
    cases = t1_cases.reset_index(drop=True)
    y = np.column_stack([cases[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    baseline = np.load(P3 / "cache" / "p3_t1" / "oof_final_p3_t1f.npy", allow_pickle=False)
    damped = np.load(P3 / "cache" / "p3_t2" / "oof_damped_p3_t2.npy", allow_pickle=False)
    extra = np.load(P3 / "cache" / "p3_t2" / "oof_extra_p3_t2.npy", allow_pickle=False)
    t2 = (damped + extra) / 2
    valid = np.isfinite(t2).all(axis=1)
    if int(valid.sum()) != 216 or not np.array_equal(valid, folds.fold.isin([1, 2, 3, 4]).to_numpy()):
        raise RuntimeError("unexpected T2 OOF validity mask")

    feature_ids = np.load(T2_CACHE / "feature_ids_train.npy", allow_pickle=True)
    features = np.load(T2_CACHE / "features_train.npy", allow_pickle=False)
    feature_pos = pd.Series(np.arange(len(feature_ids)), index=feature_ids)
    regimes = regime_labels(features[feature_pos.loc[cases.case_id].to_numpy()])
    view_a = compare(cases, y, t2, baseline, regimes, valid)
    view_b = compare(cases, y, t2, baseline, regimes, folds.fold.eq(4).to_numpy())
    exp1 = {
        "weights": {"damped": 0.5, "extra": 0.5},
        "view_a_folds1_4": view_a,
        "view_b_fold4": view_b,
        "accepted": bool(view_a["delta"] <= -0.002
                           and view_a["max_degradation_n_ge_20"] <= 0.03
                           and view_b["delta"] < 0),
        "criteria": {
            "view_a_delta_le_-0.002": bool(view_a["delta"] <= -0.002),
            "view_a_max_degradation_n_ge_20_le_0.03": bool(view_a["max_degradation_n_ge_20"] <= 0.03),
            "view_b_delta_lt_0": bool(view_b["delta"] < 0),
        },
    }

    test_pool = {}
    with np.load(T2_CACHE / "ckpt" / "test_preds.npz", allow_pickle=False) as saved:
        for name in ("damped", "extra", "lgbm_pooled", "lgbm_lead", "lgbm_station"):
            target = PREDS / f"test_{name}_p3_t2.npy"
            save_pool_pred(target, saved[name])
            test_pool[name] = {"path": str(target), "sha256": sha256(target), "shape": list(saved[name].shape)}

    leak = t1_leak_check(pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True),
                         cases.loc[folds.fold.eq(4)])
    exp2 = {"leak_check": leak, "accepted": False, "scores": "not_run"}
    report = {
        "experiment": "T5",
        "intersection_cases": int(valid.sum()),
        "experiment1": exp1,
        "experiment2": exp2,
        "test_pool": test_pool,
        "final": {"created": False, "reason": "experiment1=false; experiment2=leak_stop"},
        "submission": {"created": False, "reason": "no accepted experiment"},
    }
    (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
