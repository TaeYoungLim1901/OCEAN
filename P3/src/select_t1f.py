"""Select T1F blend on folds 0-3 and evaluate once on sealed fold 4."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE = P3 / "cache" / "p3_t1"


def rmse(y, p, mask):
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def main():
    y = np.load(CACHE / "oof_true.npy")
    damped = np.load(P3 / "preds" / "oof_damped_p3_t1.npy")
    lgbm = np.load(P3 / "preds" / "oof_lgbm_p3_t1.npy")
    cases = pd.read_parquet(CACHE / "oof_cases.parquet")
    fold_map = pd.read_csv(CACHE / "folds.csv").set_index("case_id").fold
    folds = fold_map.loc[cases.case_id].to_numpy()
    persistence = np.repeat(cases.hs0.to_numpy()[:, None], y.shape[1], axis=1)
    valid = np.isfinite(y).all(axis=1)
    valid &= np.isfinite(damped).all(axis=1) & np.isfinite(lgbm).all(axis=1)
    selection, outer, all_rows = (folds < 4) & valid, (folds == 4) & valid, valid
    grid = []
    for weight in np.arange(0, 1.001, 0.05):
        pred = (1 - weight) * damped + weight * lgbm
        grid.append({"weight_lgbm": round(float(weight), 2),
                     "selection_rmse": rmse(y, pred, selection)})
    chosen = min(grid, key=lambda row: (row["selection_rmse"], row["weight_lgbm"]))
    # The thinned 53-case judge inverts the damped/LGBM ranking (f.thinned_judge_inverts).
    # Dense reweighted fold comparison had W_new winning 2/4, below the preregistered 3/4 threshold.
    # Evidence: P3/cache/p3_t11/report.json.
    fixed_weight_lgbm = 0.65
    searched_weight = chosen["weight_lgbm"]
    if searched_weight != fixed_weight_lgbm:
        print(f"WARNING: grid weight_lgbm={searched_weight:.2f} != fixed weight_lgbm={fixed_weight_lgbm:.2f}")
    weight = fixed_weight_lgbm
    blend = (1 - weight) * damped + weight * lgbm
    np.save(P3 / "preds" / "oof_final_p3_t1f.npy", blend)
    report = {"selected_weight_lgbm": weight,
              "grid_weight_lgbm": searched_weight,
              "fixed_weight_lgbm": fixed_weight_lgbm,
              "grid": grid,
              "valid_cases": int(valid.sum()), "nan_cases": int((~valid).sum()),
              "skipped_folds": {str(f): int((folds == f).sum())
                                for f in np.unique(folds[~valid])}}
    for name, mask in (("selection", selection), ("outer", outer), ("all_oof", all_rows)):
        report[name] = {"n_cases": int(mask.sum()),
                        "persistence": rmse(y, persistence, mask),
                        "damped": rmse(y, damped, mask),
                        "lgbm": rmse(y, lgbm, mask),
                        "blend": rmse(y, blend, mask)}
    report["outer_pass"] = all(report["outer"]["blend"] < report["outer"][name]
                               for name in ("persistence", "damped", "lgbm"))
    (CACHE / "t1f_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))
    if not report["outer_pass"]:
        raise SystemExit("outer rule failed: keep T1")
    from inference_t1f import main as inference_main
    inference_main()


if __name__ == "__main__":
    main()
