"""T6 case-local wind-equilibrium relaxation experiment."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import cases as C
import metric as M
from submission_tools import save_pool_pred

sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE = P3 / "cache" / "p3_t6"
PREDS = P3 / "preds"
LEADS = np.asarray(C.LEADS, float)
WSPD_COL = C.COLS.index("wspd")
HS_COL = C.COLS.index("hs")


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def train_mask(cases, val):
    allowed = np.zeros(len(cases), bool)
    for st in C.STATIONS:
        start = val.loc[val.station.eq(st), "t0"].min()
        cutoff = start - pd.Timedelta(hours=C.SPACING_H)
        allowed |= cases.station.eq(st).to_numpy() & cases.t0.lt(cutoff).to_numpy()
    return allowed


def context_values(ctx):
    hs = ctx[-(6 * 6 + 1):, HS_COL].astype(float)
    wspd = ctx[-(6 * 6 + 1):, WSPD_COL].astype(float)
    ok = np.isfinite(wspd)
    u_eff = float(np.median(wspd[ok])) if ok.any() else np.nan
    hs_ok = hs[np.isfinite(hs)]
    hs_delta = float(hs_ok[-1] - hs_ok[0]) if len(hs_ok) else np.nan
    return u_eff, hs_delta


def context_values_train(cases):
    u_eff, hs_delta = {}, {}
    for cid, _, ctx in C.iter_train_contexts(cases):
        u_eff[cid], hs_delta[cid] = context_values(ctx)
    return u_eff, hs_delta


def fit_params(cases, y, u_eff, allowed):
    params = {}
    usable = allowed & np.isfinite(u_eff)
    for st in C.STATIONS:
        k = usable & cases.station.eq(st).to_numpy()
        if not k.any():
            params[st] = np.array([0.0246, 0.0, 12.0])
            continue
        u2 = u_eff[k] ** 2
        hs0 = cases.loc[k, "hs0"].to_numpy(float)
        target = y[k]

        def residual(q):
            eq = q[0] * u2 + q[1]
            p = eq[:, None] + (hs0[:, None] - eq[:, None]) * np.exp(-LEADS[None, :] / q[2])
            return (p - target).ravel()

        fitted = least_squares(residual, [0.0246, 0.0, 12.0],
                               bounds=([-np.inf, -np.inf, 1.0], [np.inf, np.inf, 200.0]))
        params[st] = fitted.x.astype(float)
    return params


def predict(cases, u_eff, params, fallback):
    out = np.asarray(fallback, float).copy()
    for st in C.STATIONS:
        k = cases.station.eq(st).to_numpy() & np.isfinite(u_eff)
        if not k.any():
            continue
        a, b, tau = params[st]
        hs0 = cases.loc[k, "hs0"].to_numpy(float)
        eq = a * u_eff[k] ** 2 + b
        out[k] = eq[:, None] + (hs0[:, None] - eq[:, None]) * np.exp(-LEADS[None, :] / tau)
    return np.clip(out, 0.0, 30.0)


def long_score(cases, y, pred, hs0):
    true = M.wide_to_long(cases)
    predicted = M.wide_to_long(cases, {f"y{lead}": pred[:, j]
                                       for j, lead in enumerate(C.LEADS)})
    return M.evaluate(predicted, true, by=["lead", "station", "hs0bin"],
                      hs0=cases[["case_id", "station", "hs0"]])


def segment_scores(cases, y, pred, regime):
    out = {}
    bins = pd.cut(cases.hs0, [0, 1.75, 2.25, 3.0, 99], labels=False).to_numpy()
    groups = [(f"lead:{lead}", np.ones(len(cases), bool), j) for j, lead in enumerate(C.LEADS)]
    groups += [(f"station:{st}", cases.station.eq(st).to_numpy(), None) for st in C.STATIONS]
    groups += [(f"hs0bin:{int(b)}", bins == b, None) for b in range(4)]
    groups += [(f"regime:{name}", regime == name, None)
               for name in ("falling", "stable", "rising")]
    for name, k, j in groups:
        if not k.any():
            continue
        yy, pp = (y[k, j], pred[k, j]) if j is not None else (y[k], pred[k])
        out[name] = {"n": int(k.sum()), "rmse": rmse(yy, pp)}
    return out


def regime_labels(hs_delta):
    return np.where(hs_delta > 0.10, "rising",
                    np.where(hs_delta < -0.10, "falling", "stable"))


def work0(cases, y, t1f, damped, persist, regime, mask):
    c = cases.loc[mask].reset_index(drop=True)
    yy = y[mask]
    rr = regime[mask]
    t1f_segments = segment_scores(c, yy, t1f[mask], rr)
    persist_segments = segment_scores(c, yy, persist[mask], rr)
    return {
        "n": int(mask.sum()),
        "t1f": {"total": rmse(yy, t1f[mask]), "segments": t1f_segments},
        "persistence": {"total": rmse(yy, persist[mask]), "segments": persist_segments},
        "damped": {"total": rmse(yy, damped[mask]), "segments": segment_scores(c, yy, damped[mask], rr)},
        "t1f_worse_than_persistence": [k for k in t1f_segments
                                        if t1f_segments[k]["rmse"] > persist_segments[k]["rmse"]],
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(exist_ok=True)
    all_cases = pd.read_parquet(P3 / "cache" / "cases.parquet").reset_index(drop=True)
    ev = all_cases.loc[C.eval_split(all_cases)].reset_index(drop=True)
    folds = pd.read_csv(P3 / "cache" / "p3_t2" / "folds.csv", parse_dates=["t0"])
    if not np.array_equal(folds.case_id.to_numpy(), ev.case_id.to_numpy()):
        raise RuntimeError("fixed fold case order mismatch")
    fold_values = folds.fold.to_numpy(int)
    y = np.column_stack([ev[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    u_map, delta_map = context_values_train(all_cases)
    u = ev.case_id.map(u_map).to_numpy(float)
    regime = regime_labels(ev.case_id.map(delta_map).to_numpy(float))
    t1f = np.load(P3 / "cache" / "p3_t1" / "oof_final_p3_t1f.npy", allow_pickle=False)
    damped = np.load(P3 / "cache" / "p3_t2" / "oof_damped_p3_t2.npy", allow_pickle=False)
    if t1f.shape != y.shape or damped.shape != y.shape:
        raise RuntimeError(f"OOF shape mismatch t1f={t1f.shape} damped={damped.shape} y={y.shape}")
    persist = np.repeat(ev.hs0.to_numpy(float)[:, None], len(C.LEADS), axis=1)
    view_a = np.isin(fold_values, [1, 2, 3, 4])
    selection = np.isin(fold_values, [1, 2, 3])
    work0_report = work0(ev, y, t1f, damped, persist, regime, view_a)

    oof = np.full_like(y, np.nan)
    fold_train_sizes = {}
    for fold in [1, 2, 3, 4]:
        k = fold_values == fold
        allowed = train_mask(all_cases, ev.loc[k])
        fold_train_sizes[str(fold)] = int(allowed.sum())
        params = fit_params(all_cases, np.column_stack([
            all_cases[f"y{lead}"].to_numpy(float) for lead in C.LEADS]),
                            all_cases.case_id.map(u_map).to_numpy(float), allowed)
        fallback = damped[k]
        oof[k] = predict(ev.loc[k].reset_index(drop=True), u[k], params, fallback)
        print(f"fold={fold} train={allowed.sum()} val={k.sum()}")

    weights = []
    for w in np.arange(0.0, 1.0001, 0.05):
        p = (1.0 - w) * t1f[selection] + w * oof[selection]
        weights.append((rmse(y[selection], p), round(float(w), 2)))
    selection_rmse, eq_weight = min(weights, key=lambda z: (z[0], z[1]))
    blend = (1.0 - eq_weight) * t1f + eq_weight * oof
    view_cases = ev.loc[view_a].reset_index(drop=True)
    view_y = y[view_a]
    view_t1f = t1f[view_a]
    view_eq = oof[view_a]
    view_blend = blend[view_a]
    eq_score = long_score(view_cases, view_y, view_eq, view_cases.hs0)
    blend_score = long_score(view_cases, view_y, view_blend, view_cases.hs0)
    t1f_segments = segment_scores(view_cases, view_y, view_t1f, regime[view_a])
    blend_segments = segment_scores(view_cases, view_y, view_blend, regime[view_a])
    eligible = {k: blend_segments[k]["rmse"] - t1f_segments[k]["rmse"]
                for k in blend_segments if blend_segments[k]["n"] >= 20}
    max_worsening_name, max_worsening = max(eligible.items(), key=lambda z: z[1])

    outer = fold_values == 4
    fold4_blend = blend[outer]
    fold4_t1f = t1f[outer]
    fold4_score = rmse(y[outer], fold4_blend)
    fold4_t1f_score = rmse(y[outer], fold4_t1f)
    full_y = np.column_stack([all_cases[f"y{lead}"].to_numpy(float) for lead in C.LEADS])
    full_u = all_cases.case_id.map(u_map).to_numpy(float)
    full_params = fit_params(all_cases, full_y, full_u,
                             ~all_cases.case_id.isin(ev.loc[outer, "case_id"]).to_numpy())
    test_cases = pd.read_csv(P3 / "cache" / "p3_t2" / "test_cases.csv")
    test_u_map = {}
    for cid, _, ctx in C.iter_test_contexts():
        test_u_map[cid] = context_values(ctx)[0]
    if not np.array_equal(test_cases.case_id.to_numpy(), np.asarray(list(test_u_map))):
        raise RuntimeError("test case order mismatch")
    test_u = test_cases.case_id.map(test_u_map).to_numpy(float)
    test_damped = np.load(PREDS / "test_damped_p3_t2.npy", allow_pickle=False)
    test_eq = predict(test_cases, test_u, full_params, test_damped)
    np.save(CACHE / "oof_eqrelax_p3_t6.npy", oof)
    save_pool_pred(PREDS / "test_eqrelax_p3_t6.npy", test_eq)

    params_json = {st: {"a": float(q[0]), "b": float(q[1]), "T": float(q[2])}
                   for st, q in full_params.items()}
    all_t = np.array([q[2] for q in full_params.values()])
    t_ok = bool(np.all((all_t >= 1.0) & (all_t <= 200.0)))
    fallback_view = int((~np.isfinite(u[view_a])).sum())
    fallback_test = int((~np.isfinite(test_u)).sum())
    accepted = bool(eq_score["total"] <= 0.85 and t_ok and
                    blend_score["total"] <= work0_report["t1f"]["total"] - 0.003 and
                    max_worsening <= 0.03)
    submission = {"created": False, "reason": "acceptance criteria not met"}
    if accepted:
        test_t1f = np.load(PREDS / "test_final_p3_t1f.npy", allow_pickle=False)
        test_final = (1.0 - eq_weight) * test_t1f + eq_weight * test_eq
        np.save(CACHE / "oof_final_p3_t6.npy", blend)
        test_final = np.load(save_pool_pred(PREDS / "test_final_p3_t6.npy", test_final), allow_pickle=False)
        index = pd.read_csv(P3 / "P3_data" / "test_index.csv")
        values = {(cid, st, lead): float(test_final[i, j])
                  for i, (cid, st) in enumerate(zip(test_cases.case_id, test_cases.station))
                  for j, lead in enumerate(C.LEADS)}
        index["hs_pred"] = [values[tuple(row)] for row in index.itertuples(index=False, name=None)]
        sub = P3 / "subs" / "submission_p3_t6.csv"
        index.to_csv(sub, index=False)
        submission = {"created": True, "path": str(sub),
                      "pred_path": str(PREDS / "test_final_p3_t6.npy")}
    report = {
        "experiment": "T6",
        "work0": work0_report,
        "mask": {"fold4_train_rows": fold_train_sizes["4"], "expected": 6828},
        "fit_rows": fold_train_sizes,
        "selection_rmse": float(selection_rmse),
        "params": params_json,
        "T_on_boundary": [st for st, q in full_params.items()
                          if np.isclose(q[2], 1.0) or np.isclose(q[2], 200.0)],
        "wspd_all_missing_fallback_cases": {"view_a": fallback_view, "test": fallback_test},
        "view_a": {"cases": int(view_a.sum()), "eqrelax": eq_score,
                   "blend": blend_score, "blend_weight_t1f": float(1.0 - eq_weight),
                   "blend_weight_eqrelax": float(eq_weight),
                   "eqrelax_residual_corr_vs_t1f": float(np.corrcoef(
                       (view_y - view_eq).ravel(), (view_y - view_t1f).ravel())[0, 1]),
                   "eqrelax_residual_corr_vs_damped": float(np.corrcoef(
                       (view_y - view_eq).ravel(), (view_y - damped[view_a]).ravel())[0, 1]),
                   "max_worsening_n_ge_20": float(max_worsening),
                   "max_worsening_segment_n_ge_20": max_worsening_name,
                   "segment_delta": {k: blend_segments[k]["rmse"] - t1f_segments[k]["rmse"]
                                     for k in blend_segments}},
        "fold4": {"blend": fold4_score, "t1f": fold4_t1f_score,
                   "delta_vs_t1f": fold4_score - fold4_t1f_score},
        "accepted": accepted,
        "acceptance": {"eqrelax_le_0.85": bool(eq_score["total"] <= 0.85),
                        "T_in_1_200": t_ok,
                        "blend_improvement_ge_0.003": bool(
                            blend_score["total"] <= work0_report["t1f"]["total"] - 0.003),
                        "max_worsening_n_ge_20_le_0.03": bool(max_worsening <= 0.03)},
        "outputs": [str(CACHE / "oof_eqrelax_p3_t6.npy"), str(PREDS / "test_eqrelax_p3_t6.npy")],
        "submission": submission,
    }
    (CACHE / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
