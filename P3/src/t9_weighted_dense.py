"""T9: reweight dense OOF cases to the thinned evaluation distribution."""
import hashlib
import itertools
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

import cases as C
import metric as M

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
ROOT = P3.parent
CACHE = P3 / "cache" / "p3_t9"
PRED = P3 / "preds"
LEADS = C.LEADS
CANDIDATES = ["persistence", "damped", "lgbm_t1"]
JUDGES = ["thinned", "dense_unweighted", "dense_reweighted"]


def load(path):
    return np.load(path, allow_pickle=False)


def weighted_rmse(y, pred, weights):
    return float(np.sqrt(np.sum(weights[:, None] * (y - pred) ** 2) /
                        (weights.sum() * y.shape[1])))


def masks(cases):
    out = {}
    for lead in LEADS:
        out[f"lead:{lead}"] = np.ones(len(cases), dtype=bool)
    for station in C.STATIONS:
        out[f"station:{station}"] = cases.station.to_numpy() == station
    bins = np.digitize(cases.hs0.to_numpy(float), [1.75, 2.25, 3.0], right=True)
    labels = ["1.5-1.75", "1.75-2.25", "2.25-3", "3+"]
    for i, label in enumerate(labels):
        out[f"hs0bin:{label}"] = bins == i
    return out


def score(cases, y, pred, weights, unweighted=False):
    if unweighted:
        true_long = M.wide_to_long(cases)
        pred_long = M.wide_to_long(
            cases, {f"y{lead}": pred[:, j] for j, lead in enumerate(LEADS)})
        checked = M.evaluate(
            pred_long, true_long, by=["lead", "station", "hs0bin"],
            hs0=cases[["case_id", "station", "hs0"]])
        total = checked["total"]
        lead = {f"lead:{k}": float(v) for k, v in checked["lead"].items()}
        station = {f"station:{k}": float(v) for k, v in checked["station"].items()}
        hs0bin = {f"hs0bin:{k}": float(v) for k, v in checked["hs0bin"].items()}
        return {"total": float(total), "n": int(len(cases)),
                "segments": {**lead, **station, **hs0bin},
                "segment_n": {k: int(v.sum()) for k, v in masks(cases).items()}}
    seg = masks(cases)
    segments = {
        f"lead:{lead}": weighted_rmse(y[:, j:j + 1], pred[:, j:j + 1], weights)
        for j, lead in enumerate(LEADS)
    }
    nonlead = {k: m for k, m in seg.items() if not k.startswith("lead:")}
    segments.update({k: weighted_rmse(y[m], pred[m], weights[m])
                     for k, m in nonlead.items() if m.any()})
    return {
        "total": weighted_rmse(y, pred, weights),
        "n": int(len(cases)),
        "segments": segments,
        "segment_n": {k: int(m.sum()) for k, m in seg.items() if m.any()},
    }


def simplex(step=0.05):
    units = round(1 / step)
    return [np.array([p, d, units - p - d], dtype=float) / units
            for p in range(units + 1)
            for d in range(units - p + 1)]


def blend(preds, weights):
    return np.einsum("i,inj->nj", weights, preds)


def ess(weights):
    return float(weights.sum() ** 2 / np.square(weights).sum())


def rank(scores):
    return sorted(scores, key=lambda name: (scores[name], name))


def changed_pairs(table):
    rows = []
    for a, b in itertools.combinations(CANDIDATES + ["T1F_dense"], 2):
        orders = {}
        for judge in JUDGES:
            av, bv = table[judge][a], table[judge][b]
            orders[judge] = f"{a}<{b}" if av < bv else f"{b}<{a}" if bv < av else "tie"
        if len(set(orders.values())) > 1:
            rows.append({"pair": f"{a}/{b}", "orders": orders})
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    t1f_report_path = P3 / "cache" / "p3_t1" / "t1f_report.json"
    t1f_report = json.loads(t1f_report_path.read_text(encoding="utf-8"))
    lgbm_weight = float(t1f_report["selected_weight_lgbm"])
    t1f_weights = np.array([0.0, 1.0 - lgbm_weight, lgbm_weight])
    assert np.all(t1f_weights >= 0) and abs(t1f_weights.sum() - 1) < 1e-12

    dense_index = pd.read_csv(P3 / "cache" / "p3_t8" / "dense_index.csv")
    all_cases = pd.read_parquet(P3 / "cache" / "cases.parquet")
    all_by_id = all_cases.set_index("case_id")
    dense_cases = all_by_id.loc[dense_index.case_id].reset_index()
    assert np.array_equal(dense_cases.case_id.to_numpy(), dense_index.case_id.to_numpy())
    y = np.column_stack([dense_cases[f"y{lead}"].to_numpy(float) for lead in LEADS])
    fold = dense_index.fold.to_numpy(int)

    pred = np.stack([
        load(PRED / "oof_persistence_p3_t8.npy"),
        load(PRED / "oof_damped_p3_t8.npy"),
        load(PRED / "oof_lgbm_t1_p3_t8.npy"),
    ])
    assert pred.shape == (3, len(dense_cases), len(LEADS))
    assert np.isfinite(pred).all() and np.isfinite(y).all()
    t1f_dense = blend(pred, t1f_weights)

    eval_mask = C.eval_split(all_cases)
    eval_cases = all_cases.loc[eval_mask].reset_index(drop=True)
    assert len(eval_cases) == 272
    eval_ids = set(eval_cases.case_id)
    thin_fold4 = np.array([cid in eval_ids and f == 4
                           for cid, f in zip(dense_index.case_id, fold)])
    assert thin_fold4.sum() == 53
    edges = eval_cases.hs0.quantile([0.25, 0.5, 0.75]).to_numpy(float)
    labels = np.array(["q1", "q2", "q3", "q4"])

    def strata(frame):
        q = np.digitize(frame.hs0.to_numpy(float), edges, right=True)
        return np.array([f"{s}:{labels[i]}" for s, i in zip(frame.station, q)], dtype=object)

    target_stratum = strata(eval_cases)
    dense_stratum = strata(dense_cases)
    target_count = pd.Series(target_stratum).value_counts().to_dict()
    dense_count = pd.Series(dense_stratum).value_counts().to_dict()
    target_n, dense_n = len(eval_cases), len(dense_cases)
    raw_by_stratum, clipped_by_stratum = {}, {}
    clipped_strata = []
    for stratum in sorted(set(target_count) | set(dense_count)):
        if not dense_count.get(stratum, 0):
            continue
        raw = (target_count.get(stratum, 0) / target_n) / (dense_count[stratum] / dense_n)
        clipped = min(raw, 10.0)
        raw_by_stratum[stratum] = float(raw)
        clipped_by_stratum[stratum] = float(clipped)
        if raw > 10.0:
            clipped_strata.append(stratum)
    clipped_case = np.array([clipped_by_stratum[s] for s in dense_stratum], dtype=float)
    weights = clipped_case / clipped_case.mean()
    assert np.isfinite(weights).all() and abs(weights.mean() - 1.0) < 1e-12
    pd.DataFrame({"case_id": dense_index.case_id, "stratum": dense_stratum, "w": weights}).to_csv(
        CACHE / "dense_weights.csv", index=False)

    strata_report = {}
    for stratum in sorted(set(target_count) | set(dense_count)):
        strata_report[stratum] = {
            "target_count": int(target_count.get(stratum, 0)),
            "target_fraction": float(target_count.get(stratum, 0) / target_n),
            "dense_count": int(dense_count.get(stratum, 0)),
            "dense_fraction": float(dense_count.get(stratum, 0) / dense_n),
            "raw_weight": raw_by_stratum.get(stratum),
            "clipped_weight": clipped_by_stratum.get(stratum),
            "normalized_weight": (float(clipped_by_stratum[stratum] / clipped_case.mean())
                                   if stratum in clipped_by_stratum else None),
            "used": stratum in clipped_by_stratum,
            "clipped": stratum in clipped_strata,
        }

    fold_ess = {f"fold{f}": ess(weights[fold == f]) for f in range(1, 5)}
    judges = {
        "persistence": pred[0], "damped": pred[1],
        "lgbm_t1": pred[2], "T1F_dense": t1f_dense,
    }
    judge_table = {judge: {} for judge in JUDGES}
    judge_scores = {judge: {} for judge in JUDGES}
    for judge_name, judge_pred in judges.items():
        for judge, mask, w, uw in (
            ("thinned", thin_fold4, np.ones(int(thin_fold4.sum())), True),
            ("dense_unweighted", fold == 4, np.ones(int((fold == 4).sum())), True),
            ("dense_reweighted", fold == 4, weights[fold == 4], False),
        ):
            c = dense_cases.loc[mask].reset_index(drop=True)
            s = score(c, y[mask], judge_pred[mask], w, unweighted=uw)
            judge_table[judge][judge_name] = s["total"]
            judge_scores[judge][judge_name] = s

    grid_rows = []
    selection = (fold >= 1) & (fold <= 3)
    for w_new in simplex():
        p = blend(pred, w_new)
        grid_rows.append({
            "weight": w_new.tolist(),
            "rmse": weighted_rmse(y[selection], p[selection], weights[selection]),
        })
    chosen = min(grid_rows, key=lambda row: (row["rmse"], row["weight"]))
    w_new = np.array(chosen["weight"], dtype=float)
    new_pred = blend(pred, w_new)
    fold4 = fold == 4
    dense_base = score(dense_cases.loc[fold4].reset_index(drop=True), y[fold4],
                       t1f_dense[fold4], weights[fold4])
    dense_new = score(dense_cases.loc[fold4].reset_index(drop=True), y[fold4],
                      new_pred[fold4], weights[fold4])
    thin_cases = dense_cases.loc[thin_fold4].reset_index(drop=True)
    thin_new = score(thin_cases, y[thin_fold4], new_pred[thin_fold4],
                     np.ones(int(thin_fold4.sum())), unweighted=True)
    segment_rows = []
    for segment, n in dense_new["segment_n"].items():
        if n >= 100:
            segment_rows.append({
                "segment": segment, "n": n,
                "rmse": dense_new["segments"][segment],
                "t1f_dense_rmse": dense_base["segments"][segment],
                "delta": dense_new["segments"][segment] - dense_base["segments"][segment],
            })
    segment_rows.sort(key=lambda row: row["delta"], reverse=True)
    max_degradation = segment_rows[0]
    acceptance = {
        "fold4_reweighted_ess_ge_300": fold_ess["fold4"] >= 300.0,
        "weights_in_simplex": bool(np.all((w_new >= 0) & (w_new <= 1)) and
                                    abs(w_new.sum() - 1.0) <= 1e-6),
        "dense_reweighted_improvement_ge_0.005": dense_base["total"] - dense_new["total"] >= 0.005,
        "thinned_not_worse_than_t1f_plus_0.002": thin_new["total"] - 0.754739 <= 0.002,
        "max_segment_degradation_n_ge_100_le_0.03": max_degradation["delta"] <= 0.03,
    }
    accepted = bool(all(acceptance.values()))

    submission = {"created": False, "reason": "acceptance=false"}
    if accepted:
        test_rows = list(C.iter_test_contexts())
        test_cases = pd.DataFrame(
            [(cid, st, float(ctx[-1, 0])) for cid, st, ctx in test_rows],
            columns=["case_id", "station", "hs0"])
        test_index = pd.read_csv(P3 / "P3_data" / "test_index.csv")
        assert np.array_equal(test_index.case_id.drop_duplicates().to_numpy(),
                              test_cases.case_id.to_numpy())
        test_pred = np.stack([
            np.repeat(test_cases.hs0.to_numpy(float)[:, None], len(LEADS), axis=1),
            load(PRED / "test_damped_p3_t1.npy"),
            load(PRED / "test_lgbm_p3_t1.npy"),
        ])
        assert test_pred.shape == (3, len(test_cases), len(LEADS))
        final = np.clip(blend(test_pred, w_new), 0.0, 30.0)
        pred_path = PRED / "test_final_p3_t9.npy"
        sub_path = P3 / "subs" / "submission_p3_t9.csv"
        np.save(pred_path, final)
        sub = test_index.copy()
        sub["hs_pred"] = final.reshape(-1)
        sub.to_csv(sub_path, index=False)
        audit = subprocess.run(
            [sys.executable, str(P3 / "src" / "submission_tools.py"), "audit",
             str(sub_path), "--pred", str(pred_path)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        static = subprocess.run(
            [sys.executable, str(P3 / "src" / "context_static_check.py")],
            cwd=P3 / "src", capture_output=True, text=True, encoding="utf-8")
        preflight = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "preflight.py"), "sub", str(sub_path),
             "--index", str(P3 / "P3_data" / "test_index.csv"),
             "--base", str(P3 / "P3_data" / "baseline_persistence.csv")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        gates_ok = audit.returncode == static.returncode == preflight.returncode == 0
        submission = {
            "created": True,
            "path": str(sub_path.relative_to(ROOT)).replace("\\", "/"),
            "pred_path": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
            "audit": "OK" if audit.returncode == 0 else "FAIL",
            "context_static_check": "OK" if static.returncode == 0 else "FAIL",
            "preflight": "OK" if preflight.returncode == 0 else "FAIL",
            "gates_ok": gates_ok,
            "sha256": sha256(sub_path),
        }
        if not gates_ok:
            raise SystemExit("submission gate failed")

    report = {
        "experiment": "T9_weighted_dense_judge",
        "candidate_order": CANDIDATES,
        "inputs": {
            "dense_cases": int(len(dense_cases)),
            "fold_counts": {f"fold{f}": int((fold == f).sum()) for f in range(1, 5)},
            "thinned_eval_cases": int(len(eval_cases)),
            "thinned_fold4_cases": int(thin_fold4.sum()),
        },
        "task1": {
            "t1f_weights": {
                "persistence": float(t1f_weights[0]),
                "damped": float(t1f_weights[1]),
                "lgbm_t1": float(t1f_weights[2]),
            },
            "source_file": "P3/cache/p3_t1/t1f_report.json",
            "source_field": "selected_weight_lgbm",
            "t1f_dense_fold4": {
                "dense_unweighted": float(judge_table["dense_unweighted"]["T1F_dense"]),
                "dense_reweighted": float(dense_base["total"]),
                "thinned": float(judge_table["thinned"]["T1F_dense"]),
            },
        },
        "task2": {
            "quartile_edges": edges.tolist(),
            "strata": strata_report,
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "clipped_strata_count": len(clipped_strata),
            "clipped_strata": clipped_strata,
            "ess": fold_ess,
        },
        "task3": {
            "fold4": judge_table,
            "rankings": {judge: rank(judge_table[judge]) for judge in JUDGES},
            "rank_changed_pairs": changed_pairs(judge_table),
        },
        "experiment4": {
            "grid_step": 0.05,
            "w_new": dict(zip(CANDIDATES, w_new.tolist())),
            "selection_folds1_3_reweighted_rmse": float(chosen["rmse"]),
            "fold4_dense_reweighted_w_new": float(dense_new["total"]),
            "fold4_dense_reweighted_t1f_delta": float(dense_new["total"] - dense_base["total"]),
            "fold4_thinned_w_new": float(thin_new["total"]),
            "fold4_thinned_t1f_0.754739_delta": float(thin_new["total"] - 0.754739),
            "fold4_dense_reweighted_lead": {
                k.split(":", 1)[1]: v for k, v in dense_new["segments"].items()
                if k.startswith("lead:")
            },
            "fold4_dense_reweighted_segments_n_ge_100": segment_rows,
            "max_degradation_n_ge_100": max_degradation,
        },
        "acceptance": acceptance,
        "accepted": accepted,
        "outputs": {
            "dense_weights": "P3/cache/p3_t9/dense_weights.csv",
            "report": "P3/cache/p3_t9/report.json",
        },
        "submission": submission,
    }
    (CACHE / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": accepted,
        "t1f_weights": report["task1"]["t1f_weights"],
        "ess": fold_ess,
        "w_new": report["experiment4"]["w_new"],
        "fold4_dense_reweighted": dense_new["total"],
        "fold4_thinned": thin_new["total"],
        "submission": submission,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
