"""T21 dense OOF/test runner for the two non-tree members."""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

import cases as C
import train_dense as D


sys.stdout.reconfigure(encoding="utf-8")
SEED = 42
N_LEADS = len(C.LEADS)
P3 = pathlib.Path(__file__).resolve().parent.parent
CACHE = P3 / "cache" / "p3_t21"
PREDS = P3 / "preds"
T8 = P3 / "cache" / "p3_t8"
T9 = P3 / "cache" / "p3_t9"
MEMBERS = ("arx", "wsswell")
SIMPLEX_UNITS = 20
T25_BASELINE = 0.611959


def load_dense():
    cases = D.load_cases()
    reference = D.load_reference_folds()
    assigned = D.assign_dense_folds(cases, reference)
    index = pd.read_csv(T8 / "dense_index.csv")
    expected = cases.set_index("case_id").loc[index.case_id].reset_index()
    if not np.array_equal(expected.case_id.to_numpy(), index.case_id.to_numpy()):
        raise AssertionError("dense index case order mismatch")
    fold_by_id = pd.Series(assigned, index=cases.case_id)
    if not np.array_equal(fold_by_id.loc[index.case_id].to_numpy(), index.fold.to_numpy()):
        raise AssertionError("dense fold assignment mismatch")
    expected["fold"] = index.fold.to_numpy(int)
    return cases, expected, index


def contexts_for(frame):
    by_id = {cid: ctx for cid, _, ctx in C.iter_train_contexts(frame)}
    missing = [cid for cid in frame.case_id if cid not in by_id]
    if missing:
        raise AssertionError(f"missing contexts: {missing[:3]}")
    return np.stack([by_id[cid] for cid in frame.case_id]).astype(np.float32)


def target_matrix(frame):
    return np.column_stack([frame[f"y{lead}"].to_numpy(float) for lead in C.LEADS])


def weighted_rmse(y, pred, weights):
    valid = np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    if not valid.any():
        return float("nan")
    w = np.asarray(weights, dtype=float)[valid]
    error = (np.asarray(y)[valid] - np.asarray(pred)[valid]) ** 2
    return float(np.sqrt(np.sum(w[:, None] * error) / (w.sum() * error.shape[1])))


def bias_stats(error, mask, weights):
    e = np.asarray(error, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    if e.ndim == 1:
        weighted = np.sum(w * e) / w.sum()
    else:
        weighted = np.sum(w[:, None] * e) / (w.sum() * e.shape[1])
    return float(weighted), float(e.mean()), int(mask.sum())


def bootstrap_bias(error, weights, reps=2000):
    case_bias = np.asarray(error, dtype=float).mean(axis=1)
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        index = rng.integers(len(case_bias), size=len(case_bias))
        samples[i] = np.sum(weights[index] * case_bias[index]) / weights[index].sum()
    return float(samples.std(ddof=1))


def residual_corr(y, pred, baseline, mask):
    valid = mask & np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    valid &= np.isfinite(baseline).all(axis=1)
    if valid.sum() < 2:
        return float("nan")
    left = (y[valid] - pred[valid]).ravel()
    right = (y[valid] - baseline[valid]).ravel()
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compositions(k, units=SIMPLEX_UNITS, prefix=()):
    if k == 1:
        yield np.asarray(prefix + (units,), dtype=float) / SIMPLEX_UNITS
        return
    for value in range(units + 1):
        yield from compositions(k - 1, units - value, prefix + (value,))


def choose_blend(preds, y, weights, selection):
    best = None
    for weight in compositions(preds.shape[0]):
        pred = np.einsum("i,inj->nj", weight, preds[:, selection, :])
        score = weighted_rmse(y[selection], pred, weights[selection])
        key = (score, tuple(weight.tolist()))
        if best is None or key < best[0]:
            best = (key, weight)
    return best[1], float(best[0][0])


def load_baselines(n):
    names = ("persistence", "damped", "lgbm_t1")
    arrays = {}
    for name in names:
        path = PREDS / f"oof_{name}_p3_t8.npy"
        pred = np.load(path, allow_pickle=False)
        if pred.shape != (n, N_LEADS) or not np.isfinite(pred).all():
            raise AssertionError(f"invalid dense baseline {path.name}: {pred.shape}")
        arrays[name] = pred.astype(float)
    return arrays


def load_t21_outputs(n, test_n):
    oof = {}
    test = {}
    for name in MEMBERS:
        oof_path = PREDS / f"oof_dense_{name}_p3_t21.npy"
        test_path = PREDS / f"test_{name}_p3_t21.npy"
        oof_pred = np.load(oof_path, allow_pickle=False)
        test_pred = np.load(test_path, allow_pickle=False)
        if oof_pred.shape != (n, N_LEADS) or not np.isfinite(oof_pred).all():
            raise AssertionError(f"invalid T21 OOF {oof_path.name}: {oof_pred.shape}")
        if test_pred.shape != (test_n, N_LEADS) or not np.isfinite(test_pred).all():
            raise AssertionError(f"invalid T21 test {test_path.name}: {test_pred.shape}")
        oof[name] = oof_pred.astype(float)
        test[name] = test_pred.astype(float)
    return oof, test


def fit_bias(error, mask, weights, stations, variant):
    selected_error = error[mask]
    selected_weights = weights[mask]
    if variant == "g":
        return float(np.sum(selected_weights[:, None] * selected_error) /
                     (selected_weights.sum() * selected_error.shape[1]))
    if variant == "l":
        return np.array([np.sum(selected_weights * selected_error[:, j]) /
                         selected_weights.sum() for j in range(N_LEADS)])
    values = []
    for station in C.STATIONS:
        station_mask = mask & (stations == station)
        station_error = error[station_mask]
        station_weights = weights[station_mask]
        if not station_weights.size:
            values.append(np.nan if variant == "s" else [np.nan] * N_LEADS)
            continue
        if variant == "s":
            values.append(np.sum(station_weights[:, None] * station_error) /
                          (station_weights.sum() * N_LEADS))
        else:
            values.append([np.sum(station_weights * station_error[:, j]) /
                           station_weights.sum() for j in range(N_LEADS)])
    return np.asarray(values)


def structured_bias(dense, y, t1f, fold, weights):
    fold4 = fold == 4
    fit_mask = np.isin(fold, [1, 2, 3])
    stations = dense.station.to_numpy()
    baseline = weighted_rmse(y[fold4], t1f[fold4], weights[fold4])
    print(f"T25_CONTROL none df=0 fold4_rmse={baseline:.6f} delta={baseline - T25_BASELINE:+.6f}")
    if abs(baseline - T25_BASELINE) > 5e-7:
        print(f"T25_STOP baseline={baseline:.9f} expected={T25_BASELINE:.9f} delta={baseline - T25_BASELINE:+.9f}")
        raise SystemExit(1)

    rows = []
    fitted = {}
    for name, df in (("G", 1), ("L", N_LEADS), ("S", len(C.STATIONS)),
                     ("SL", len(C.STATIONS) * N_LEADS)):
        variant = name.lower()
        c = fit_bias(y - t1f, fit_mask, weights, stations, variant)
        fitted[variant] = c
        shift = np.asarray(c)
        if variant == "g":
            corrected = t1f + shift
        elif variant == "l":
            corrected = t1f + shift[None, :]
        else:
            station_shift = shift if variant == "s" else shift.mean(axis=1)
            corrected = t1f + station_shift[np.searchsorted(C.STATIONS, stations), None]
            if variant == "sl":
                corrected = t1f + shift[np.searchsorted(C.STATIONS, stations)]
        clipped = np.clip(corrected, 0, 30)
        np.save(PREDS / f"oof_dense_t1f_corr_{variant}_p3_t25.npy", clipped)
        rows.append((name, df, weighted_rmse(y[fold4], clipped[fold4], weights[fold4]),
                     int(np.count_nonzero(clipped != corrected))))

    g_rmse = rows[0][2]
    print("변형 | 자유도 | fold4 RMSE | vs 0.611959 델타 | vs G 델타 | 클립행수")
    for name, df, score, clipped_n in rows:
        print(f"{name} | {df} | {score:.6f} | {score - T25_BASELINE:+.6f} | {score - g_rmse:+.6f} | {clipped_n}")

    print("단위 | folds1-3 적합 c | fold4 적합 c | 차이")
    for station in C.STATIONS:
        fit = fitted["s"][C.STATIONS.index(station)]
        test_mask = fold4 & (stations == station)
        test = fit_bias(y - t1f, test_mask, weights, stations, "s")[C.STATIONS.index(station)]
        print(f"S:{station} | {fit:+.9f} | {test:+.9f} | {test - fit:+.9f}")
    for j, lead in enumerate(C.LEADS):
        fit = fitted["l"][j]
        test_mask = fold4
        test = fit_bias(y - t1f, test_mask, weights, stations, "l")[j]
        print(f"L:{lead} | {fit:+.9f} | {test:+.9f} | {test - fit:+.9f}")
    sl_fit = fitted["sl"]
    sl_test = fit_bias(y - t1f, fold4, weights, stations, "sl")
    diffs = []
    for i, station in enumerate(C.STATIONS):
        for j, lead in enumerate(C.LEADS):
            diffs.append((abs(sl_test[i, j] - sl_fit[i, j]), station, lead,
                          sl_fit[i, j], sl_test[i, j]))
    for _, station, lead, fit, test in sorted(diffs, reverse=True)[:5]:
        print(f"SL:{station} x {lead} | {fit:+.9f} | {test:+.9f} | {test - fit:+.9f}")

    print("대조군 | 값 | 기대값 | 델타")
    for label, value, expected in (("none_fold4_rmse", baseline, T25_BASELINE),
                                   ("G_fit_c", fitted["g"], -0.035651675)):
        print(f"{label} | {value:+.9f} | {expected:+.9f} | {value - expected:+.9f}")
    if abs(fitted["g"] + 0.035651675) > 0.002:
        print(f"T25_STOP G_fit_c={fitted['g']:+.9f} expected=-0.035651675 delta={fitted['g'] + 0.035651675:+.9f}")
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args(argv)
    if args.folds not in (1, 2, 3, 4):
        parser.error("--folds must be between 1 and 4")

    CACHE.mkdir(parents=True, exist_ok=True)
    all_cases, dense, dense_index = load_dense()
    n_dense = len(dense)
    y = target_matrix(dense)
    fold = dense.fold.to_numpy(int)
    weights_frame = pd.read_csv(T9 / "dense_weights.csv")
    if not np.array_equal(weights_frame.case_id.to_numpy(), dense.case_id.to_numpy()):
        raise AssertionError("dense weight order mismatch")
    dense_weights = weights_frame.w.to_numpy(float)

    test_records = list(C.iter_test_contexts())
    test_ids = np.asarray([cid for cid, _, _ in test_records])
    oof, test_pred = load_t21_outputs(n_dense, len(test_ids))
    if args.folds < 4:
        active = fold <= args.folds
        for predictions in (oof,):
            for name in MEMBERS:
                predictions[name][~active] = np.nan

    fallback = ~np.isfinite(contexts_for(dense)[:, :, 4]).any(axis=1)
    test_fallback = ~np.isfinite(
        np.stack([ctx for _, _, ctx in test_records]).astype(np.float32)[:, :, 4]
    ).any(axis=1)
    test_info = {"fallback_ratio": float(test_fallback.mean())}
    for current_fold in range(1, args.folds + 1):
        val_mask = fold == current_fold
        print(f"fold={current_fold} val={int(val_mask.sum())} "
              f"arx_rmse={weighted_rmse(y[val_mask], oof['arx'][val_mask], dense_weights[val_mask]):.6f} "
              f"wsswell_rmse={weighted_rmse(y[val_mask], oof['wsswell'][val_mask], dense_weights[val_mask]):.6f} "
              f"wsswell_fallback={fallback[val_mask].mean():.6f}")

    baselines = load_baselines(n_dense)
    if args.folds < 4:
        active = fold <= args.folds
        for name in baselines:
            baselines[name][~active] = np.nan
    control_pred = {
        "persistence": baselines["persistence"],
        "damped": baselines["damped"],
        "lgbm_t1": baselines["lgbm_t1"],
    }
    control_pred["t1f"] = 0.35 * control_pred["damped"] + 0.65 * control_pred["lgbm_t1"]
    lgbm = baselines["lgbm_t1"]
    selection = (fold >= 1) & (fold <= min(3, args.folds))
    corr = {name: residual_corr(y, oof[name], lgbm, selection) for name in MEMBERS}
    included = [name for name in MEMBERS if np.isfinite(corr[name]) and corr[name] < 0.90]
    pool = ("persistence", "damped", "lgbm_t1", *included)
    pool_pred = np.stack([baselines[name] if name in baselines else oof[name] for name in pool])
    valid_selection = selection & np.isfinite(pool_pred).all(axis=(0, 2))
    blend_weight, blend_selection_rmse = choose_blend(
        pool_pred, y, dense_weights, valid_selection)
    fold4 = fold == 4
    blend_fold4 = np.full((n_dense, N_LEADS), np.nan)
    if np.isfinite(pool_pred[:, fold4]).all():
        blend_fold4[fold4] = np.einsum("i,inj->nj", blend_weight, pool_pred[:, fold4])
    blend_fold4_rmse = weighted_rmse(y[fold4], blend_fold4[fold4], dense_weights[fold4])

    rows = []
    oof_valid = np.isfinite(oof["wsswell"]).all(axis=1)
    oof_fallback_ratio = float(fallback[oof_valid].mean()) if oof_valid.any() else float("nan")
    for name in MEMBERS:
        pred = oof[name]
        rows.append({
            "member": name,
            **{f"fold{f}_rmse": weighted_rmse(y[fold == f], pred[fold == f],
                                                dense_weights[fold == f]) for f in range(1, 5)},
            "residual_corr_lgbm_t1": corr[name],
            "fallback_ratio": (oof_fallback_ratio if name == "wsswell" else 0.0),
            "test_fallback_ratio": (test_info["fallback_ratio"] if name == "wsswell" else 0.0),
            "oof_nan_count": int(np.isnan(pred).sum()),
            "test_nan_count": int(np.isnan(test_pred[name]).sum()),
            "nan_count": int(np.isnan(pred).sum() + np.isnan(test_pred[name]).sum()),
            "blend_included": int(name in included),
            "blend_selection_rmse": blend_selection_rmse,
            "blend_fold4_rmse": blend_fold4_rmse,
        })
    for name, pred in control_pred.items():
        rows.append({
            "member": name,
            **{f"fold{f}_rmse": weighted_rmse(y[fold == f], pred[fold == f],
                                                dense_weights[fold == f]) for f in range(1, 5)},
            "residual_corr_lgbm_t1": np.nan,
            "fallback_ratio": np.nan,
            "test_fallback_ratio": np.nan,
            "oof_nan_count": int(np.isnan(pred).sum()),
            "test_nan_count": 0,
            "nan_count": int(np.isnan(pred).sum()),
            "blend_included": 0,
            "blend_selection_rmse": blend_selection_rmse,
            "blend_fold4_rmse": blend_fold4_rmse,
        })
    report = pd.DataFrame(rows)
    report.to_csv(CACHE / "report.csv", index=False)
    print(report.to_string(index=False))
    print("blend_pool=" + ",".join(pool))
    print("blend_weights=" + ",".join(f"{name}:{value:.2f}" for name, value in zip(pool, blend_weight)))
    print(f"test_cases={len(test_ids)} test_wsswell_fallback={test_info['fallback_ratio']:.6f} "
          f"report={CACHE / 'report.csv'}")

    expected = {
        "lgbm_t1": 0.614009,
        "t1f": 0.611959,
        "damped": 0.663402,
        "persistence": 0.769115,
    }
    for name in ("lgbm_t1", "t1f", "damped", "persistence"):
        score = weighted_rmse(y[fold4], control_pred[name][fold4], dense_weights[fold4])
        print(f"CONTROL {name}_fold4={score:.6f} expected={expected[name]:.6f} "
              f"delta={score - expected[name]:+.6f}")

    t1f_error = control_pred["t1f"] - y
    weighted, unweighted, n = bias_stats(t1f_error, np.ones(n_dense, dtype=bool), dense_weights)
    bias_weighted = weighted
    print(f"BIAS overall weighted={weighted:.9f} unweighted={unweighted:.9f} n={n}")
    for current_fold in range(1, 5):
        weighted, unweighted, n = bias_stats(t1f_error, fold == current_fold, dense_weights)
        print(f"BIAS fold f={current_fold} weighted={weighted:.9f} "
              f"unweighted={unweighted:.9f} n={n}")
    for j, lead in enumerate(C.LEADS):
        weighted, unweighted, n = bias_stats(t1f_error[:, j], np.ones(n_dense, dtype=bool), dense_weights)
        print(f"BIAS lead h={lead} weighted={weighted:.9f} "
              f"unweighted={unweighted:.9f} n={n}")
    for station in C.STATIONS:
        weighted, unweighted, n = bias_stats(
            t1f_error, dense.station.to_numpy() == station, dense_weights
        )
        print(f"BIAS station s={station} weighted={weighted:.9f} "
              f"unweighted={unweighted:.9f} n={n}")

    shift = -bias_weighted
    before = weighted_rmse(y[fold4], control_pred["t1f"][fold4], dense_weights[fold4])
    after = weighted_rmse(y[fold4], (control_pred["t1f"] + shift)[fold4], dense_weights[fold4])
    print(f"SHIFTGAIN c={shift:+.9f} fold4_before={before:.6f} "
          f"fold4_after={after:.6f} delta={after - before:+.6f}")
    sd = bootstrap_bias(t1f_error, dense_weights)
    z = (0.041550308 - bias_weighted) / sd
    print(f"BIASSE sd={sd:.9f} z_vs_public={z:+.6f}")
    if args.folds == 4:
        structured_bias(dense, y, control_pred["t1f"], fold, dense_weights)


if __name__ == "__main__":
    main()
