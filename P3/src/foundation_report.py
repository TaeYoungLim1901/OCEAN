"""Cheap, model-free checks for the P3 forecasting foundation."""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import cases as C
from train_t2 import (P3, adversarial_drift, forward_train_mask, make_folds,
                      regime_labels, segment_scores, target, test_cases_and_features)

sys.stdout.reconfigure(encoding="utf-8")
CACHE = P3 / "cache" / "p3_t2"
PREDS = P3 / "preds"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    cases = pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)
    ev = cases.loc[C.eval_split(cases)].reset_index(drop=True)
    folds = make_folds(ev)
    x = np.load(CACHE / "features_train.npy")
    ids = np.load(CACHE / "feature_ids_train.npy", allow_pickle=True)
    if not np.array_equal(ids, cases.case_id.to_numpy()):
        raise RuntimeError("train feature ids mismatch")
    pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    xe = x[pos.loc[ev.case_id].to_numpy()]
    _, xtest = test_cases_and_features()
    y = target(ev)
    regimes = regime_labels(xe)
    forward = {}
    for fold in range(1, 5):
        val = ev.loc[folds == fold]
        mask = forward_train_mask(cases, val)
        checks = {}
        for st in C.STATIONS:
            cutoff = val.loc[val.station.eq(st), "t0"].min() - pd.Timedelta(hours=C.SPACING_H)
            train_times = cases.loc[mask & cases.station.eq(st).to_numpy(), "t0"]
            checks[st] = {"cutoff": str(cutoff), "max_train_t0": str(train_times.max()),
                          "strict": bool((train_times < cutoff).all())}
        forward[str(fold)] = {"train": int(mask.sum()), "val": int((folds == fold).sum()),
                              "station_checks": checks}
    pred_paths = {"damped": P3 / "cache" / "p3_t1" / "oof_damped_p3_t1.npy",
                  "lgbm": P3 / "cache" / "p3_t1" / "oof_lgbm_p3_t1.npy",
                  "t1f": P3 / "cache" / "p3_t1" / "oof_final_p3_t1f.npy"}
    preds = {name: np.load(path) for name, path in pred_paths.items()}
    selection = np.isin(folds, [1, 2, 3])
    residual_corr = {}
    for i, left in enumerate(preds):
        for right in list(preds)[i + 1:]:
            residual_corr[f"{left}:{right}"] = float(np.corrcoef(
                (y[selection] - preds[left][selection]).ravel(),
                (y[selection] - preds[right][selection]).ravel())[0, 1])
    outer = folds == 4
    segments = segment_scores(ev.loc[outer].reset_index(drop=True), y[outer], preds["t1f"][outer], regimes[outer])
    report = {"strict_forward": forward, "selection_folds": [1, 2, 3], "sealed_outer_fold": 4,
              "feature_count": int(x.shape[1]), "drift_diagnostic_only": adversarial_drift(x, xtest),
              "legacy_pool_note": "T1 OOF folds1-3 are correlation diagnostics only; not strict-forward selection inputs",
              "legacy_residual_corr_selection": residual_corr, "t1f_outer_segments": segments,
              "blend_pool": {name: {"path": str(path), "sha256": sha256(path)}
                             for name, path in pred_paths.items()},
              "t2_training": "stopped_by_runtime_cap_before_any OOF/model/report; T1F retained"}
    (CACHE / "foundation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
