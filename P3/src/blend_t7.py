"""T7: select simplex blend weights from saved OOF predictions only."""
import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import cases as C
import metric as M
from submission_tools import save_pool_pred

sys.stdout.reconfigure(encoding="utf-8")

P3 = pathlib.Path(__file__).resolve().parent.parent
PREDS = P3 / "preds"
CACHE = P3 / "cache" / "p3_t7"
OOF_PATHS = {
    "oof_damped_p3_t1.npy": P3 / "cache" / "p3_t1" / "oof_damped_p3_t1.npy",
    "oof_lgbm_p3_t1.npy": P3 / "cache" / "p3_t1" / "oof_lgbm_p3_t1.npy",
    "oof_eqrelax_p3_t6.npy": P3 / "cache" / "p3_t6" / "oof_eqrelax_p3_t6.npy",
    "oof_final_p3_t1f.npy": P3 / "cache" / "p3_t1" / "oof_final_p3_t1f.npy",
}
LEADS = C.LEADS
N_CANDIDATES = 4
T1F_VIEW_A = 0.761125519778535
T1F_FOLD4 = 0.754739
VARIANT_ORDER = ["V1", "V3", "V2", "V4"]
VARIANT_GROUPS = {
    "V1": lambda cases: np.full(len(cases), "all", dtype=object),
    "V2": lambda cases: cases.station.to_numpy(dtype=object),
    "V3": lambda cases: np.where(cases.hs0.to_numpy() <= 1.75, "hs0<=1.75", "hs0>1.75"),
    "V4": lambda cases: np.array([
        f"{station}|{'hs0<=1.75' if hs0 <= 1.75 else 'hs0>1.75'}"
        for station, hs0 in zip(cases.station, cases.hs0)
    ], dtype=object),
}


def load(name):
    path = OOF_PATHS.get(name, PREDS / name)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def simplex(step):
    units = round(1 / step)
    return np.array([
        np.array(parts, dtype=float) / units
        for parts in itertools.product(range(units + 1), repeat=N_CANDIDATES)
        if sum(parts) == units
    ])


def best_weight(preds, y, mask, grid):
    scores = [
        float(np.mean((np.einsum("i,inj->nj", weight, preds[:, mask, :]) - y[mask]) ** 2))
        for weight in grid
    ]
    return grid[int(np.argmin(scores))]


def fit_weights(preds, y, groups, selection, step):
    grid = simplex(step)
    global_weight = best_weight(preds, y, selection, grid)
    group_weights, group_info = {}, {}
    for group in np.unique(groups):
        mask = selection & (groups == group)
        count = int(mask.sum())
        fallback = count < 15
        if not fallback:
            group_weights[group] = best_weight(preds, y, mask, grid)
        group_info[group] = {
            "selection_n": count,
            "fallback_global": fallback,
            "weight": group_weights.get(group, global_weight).tolist(),
        }
    return global_weight, group_weights, group_info


def apply_weights(preds, groups, global_weight, group_weights):
    out = np.empty((len(groups), len(LEADS)), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        weight = group_weights.get(group, global_weight)
        out[mask] = np.einsum("i,inj->nj", weight, preds[:, mask, :])
    return out


def segment_scores(cases, y, pred):
    out = {"lead": {}, "station": {}, "hs0bin": {}}
    for j, lead in enumerate(LEADS):
        out["lead"][str(lead)] = {
            "n": int(len(cases)),
            "rmse": float(np.sqrt(np.mean((pred[:, j] - y[:, j]) ** 2))),
        }
    for station in C.STATIONS:
        mask = cases.station.to_numpy() == station
        if mask.any():
            out["station"][station] = {
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))),
            }
    bins = np.digitize(cases.hs0.to_numpy(), [1.75, 2.25, 3.0], right=True)
    for value in range(4):
        mask = bins == value
        if mask.any():
            out["hs0bin"][str(value)] = {
                "n": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((pred[mask] - y[mask]) ** 2))),
            }
    return out


def view(cases, y, pred, mask):
    selected = cases.loc[mask].reset_index(drop=True)
    selected_y, selected_pred = y[mask], pred[mask]
    true_long = M.wide_to_long(selected)
    pred_long = M.wide_to_long(
        selected,
        {f"y{lead}": selected_pred[:, j] for j, lead in enumerate(LEADS)},
    )
    checked = M.evaluate(
        pred_long,
        true_long,
        by=["lead", "station", "hs0bin"],
        hs0=selected[["case_id", "station", "hs0"]],
    )
    segments = segment_scores(selected, selected_y, selected_pred)
    if abs(checked["total"] - np.sqrt(np.mean((selected_pred - selected_y) ** 2))) > 1e-12:
        raise AssertionError("metric total mismatch")
    return {"total": float(checked["total"]), "n": int(len(selected) * len(LEADS)), **segments}


def weights_json(weight):
    return {name: float(value) for name, value in zip(
        ["persistence", "damped", "lgbm_t1", "eqrelax"], weight
    )}


def delta_segments(candidate, baseline):
    rows = []
    for axis in ("lead", "station", "hs0bin"):
        for key, values in candidate[axis].items():
            rows.append({
                "segment": f"{axis}:{key}",
                "n": values["n"],
                "rmse": values["rmse"],
                "t1f_rmse": baseline[axis][key]["rmse"],
                "delta": values["rmse"] - baseline[axis][key]["rmse"],
            })
    return sorted(rows, key=lambda row: row["delta"], reverse=True)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    required = [
        "oof_damped_p3_t1.npy", "test_damped_p3_t1.npy",
        "oof_lgbm_p3_t1.npy", "test_lgbm_p3_t1.npy",
        "oof_eqrelax_p3_t6.npy", "test_eqrelax_p3_t6.npy",
        "oof_final_p3_t1f.npy", "test_final_p3_t1f.npy",
    ]
    saved = {name: load(name) for name in required}
    cases = pd.read_parquet(P3 / "cache" / "p3_t1" / "oof_cases.parquet").reset_index(drop=True)
    folds = pd.read_csv(P3 / "cache" / "p3_t1" / "folds.csv").set_index("case_id")
    if not np.array_equal(cases.case_id.to_numpy(), folds.loc[cases.case_id].index.to_numpy()):
        raise AssertionError("case/fold alignment mismatch")
    fold = folds.loc[cases.case_id, "fold"].to_numpy()
    y = np.column_stack([cases[f"y{lead}"].to_numpy(float) for lead in LEADS])
    oof = np.stack([
        np.repeat(cases.hs0.to_numpy(float)[:, None], len(LEADS), axis=1),
        saved["oof_damped_p3_t1.npy"],
        saved["oof_lgbm_p3_t1.npy"],
        saved["oof_eqrelax_p3_t6.npy"],
    ])
    t1f = saved["oof_final_p3_t1f.npy"]
    valid = fold >= 1
    selection = (fold >= 1) & (fold <= 3)
    fold4 = fold == 4
    if valid.sum() != 216 or selection.sum() != 163 or fold4.sum() != 53:
        raise AssertionError("unexpected fold counts")
    if not np.isfinite(oof[:, valid, :]).all() or not np.isfinite(t1f[valid]).all():
        raise AssertionError("non-finite valid OOF prediction")

    save_pool_pred(PREDS / "oof_persistence_p3_t7.npy", oof[0])
    test_rows = list(C.iter_test_contexts())
    test_cases = pd.DataFrame(
        [(cid, station, float(context[-1, 0])) for cid, station, context in test_rows],
        columns=["case_id", "station", "hs0"],
    )
    test_ids = test_cases.case_id.to_numpy()
    index = pd.read_csv(P3 / "P3_data" / "test_index.csv")
    if not np.array_equal(index.case_id.drop_duplicates().to_numpy(), test_ids):
        raise AssertionError("test context/index case order mismatch")
    test = np.stack([
        np.repeat(test_cases.hs0.to_numpy(float)[:, None], len(LEADS), axis=1),
        saved["test_damped_p3_t1.npy"],
        saved["test_lgbm_p3_t1.npy"],
        saved["test_eqrelax_p3_t6.npy"],
    ])
    if not np.isfinite(test).all() or test.shape[1:] != (len(test_cases), len(LEADS)):
        raise AssertionError("unexpected test prediction shape")
    save_pool_pred(PREDS / "test_persistence_p3_t7.npy", test[0])

    baseline_a = view(cases, y, t1f, valid)
    baseline_4 = view(cases, y, t1f, fold4)
    report = {
        "experiment": "T7",
        "candidate_order": ["persistence", "damped", "lgbm_t1", "eqrelax"],
        "candidate_files": {name: f"P3/preds/{name}" for name in required},
        "folds": {"selection_1_3": int(selection.sum()), "view_a_1_4": int(valid.sum()), "fold4": int(fold4.sum())},
        "baseline_t1f": {
            "view_a": baseline_a,
            "fold4": baseline_4,
            "view_a_reference": T1F_VIEW_A,
            "fold4_reference": T1F_FOLD4,
        },
        "variants": {},
    }
    for variant, group_fn in VARIANT_GROUPS.items():
        groups = group_fn(cases)
        global_weight, group_weights, group_info = fit_weights(
            oof, y, groups, selection, 0.05 if variant == "V1" else 0.25
        )
        prediction = apply_weights(oof, groups, global_weight, group_weights)
        view_a = view(cases, y, prediction, valid)
        selection_view = view(cases, y, prediction, selection)
        fold4_view = view(cases, y, prediction, fold4)
        deltas = delta_segments(view_a, baseline_a)
        max_bad = max((row for row in deltas if row["n"] >= 20), key=lambda row: row["delta"])
        acceptance = {
            "view_a_improvement_ge_0.003": T1F_VIEW_A - view_a["total"] >= 0.003,
            "view_a_max_degradation_n_ge_20_le_0.03": max_bad["delta"] <= 0.03,
            "fold4_not_worse": fold4_view["total"] <= T1F_FOLD4,
        }
        report["variants"][variant] = {
            "grid_step": 0.05 if variant == "V1" else 0.25,
            "global_weight": weights_json(global_weight),
            "groups": {
                name: {**info, "weight": weights_json(np.asarray(info["weight"]))}
                for name, info in group_info.items()
            },
            "selection": selection_view,
            "view_a": view_a,
            "fold4": fold4_view,
            "view_a_delta_vs_t1f": view_a["total"] - T1F_VIEW_A,
            "fold4_delta_vs_t1f": fold4_view["total"] - T1F_FOLD4,
            "segment_delta_vs_t1f": deltas,
            "max_degradation_n_ge_20": max_bad,
            "acceptance": acceptance,
            "accepted": bool(all(acceptance.values())),
        }

    v1 = report["variants"]["V1"]
    report["control"] = {
        "v1_view_a": v1["view_a"]["total"],
        "v1_delta_vs_t1f": v1["view_a"]["total"] - T1F_VIEW_A,
        "passed": abs(v1["view_a"]["total"] - T1F_VIEW_A) <= 0.005,
    }
    if not report["control"]["passed"]:
        report["selection"] = {"selected": None, "reason": "V1 control failed"}
        report["submission"] = {"created": False, "reason": "V1 control failed"}
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    accepted = [name for name in VARIANT_GROUPS if report["variants"][name]["accepted"]]
    if accepted:
        best = min(report["variants"][name]["view_a"]["total"] for name in accepted)
        near = [name for name in accepted if report["variants"][name]["view_a"]["total"] - best < 0.002]
        selected = min(near, key=VARIANT_ORDER.index)
        reason = "complexity tie rule" if len(near) > 1 else "best viewA among accepted"
    else:
        selected, reason = None, "no variant met all acceptance criteria"
    report["selection"] = {"selected": selected, "reason": reason}

    if selected is None:
        report["submission"] = {"created": False, "reason": "no accepted variant"}
    else:
        chosen = report["variants"][selected]
        groups = VARIANT_GROUPS[selected](cases)
        _, group_weights, _ = fit_weights(oof, y, groups, selection, chosen["grid_step"])
        test_pred = apply_weights(test, VARIANT_GROUPS[selected](test_cases),
                                  np.array([chosen["global_weight"][name] for name in
                                            ["persistence", "damped", "lgbm_t1", "eqrelax"]]),
                                  {name: np.array([info["weight"][candidate] for candidate in
                                                   ["persistence", "damped", "lgbm_t1", "eqrelax"]])
                                   for name, info in chosen["groups"].items()
                                   if not info["fallback_global"]})
        test_pred = np.clip(test_pred, 0, 30)
        np.save(CACHE / "oof_final_p3_t7.npy", apply_weights(
            oof, groups,
            np.array([chosen["global_weight"][name] for name in
                      ["persistence", "damped", "lgbm_t1", "eqrelax"]]),
            {name: np.array([info["weight"][candidate] for candidate in
                             ["persistence", "damped", "lgbm_t1", "eqrelax"]])
             for name, info in chosen["groups"].items() if not info["fallback_global"]},
        ))
        test_pred = np.load(save_pool_pred(PREDS / "test_final_p3_t7.npy", test_pred), allow_pickle=False)
        sub = index.copy()
        sub["hs_pred"] = test_pred.reshape(-1)
        sub_path = P3 / "subs" / "submission_p3_t7.csv"
        sub.to_csv(sub_path, index=False)
        report["submission"] = {"created": True, "path": str(sub_path), "pred_shape": list(test_pred.shape)}

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
