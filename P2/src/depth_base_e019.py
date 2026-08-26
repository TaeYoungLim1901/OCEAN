"""T19: same-time depth-profile bases; no training."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.stats import kurtosis

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gate_e004 as G
import season_e006 as S
from submission_tools import save_pool_pred

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("2024_sepoct", "S1", "S3", "S4")
EXPECTED_B0 = {
    "2024_sepoct": 0.970325,
    "S1": 0.970325,
    "S3": 0.862944,
    "S4": 0.428320,
}


def _points(group, zcol):
    values = group.loc[group[zcol].notna(), [zcol, "temp"]].to_numpy(float)
    if len(values) < 3:
        return None
    values = values[np.argsort(values[:, 0])]
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values[:, 0]) > 0):
        return None
    return values[:, 0], values[:, 1]


def _target_z(depth, nominal, indices, lo, hi, use_depth):
    nominal = nominal[indices]
    if not use_depth:
        z = nominal
        return z if np.all(np.isfinite(z)) and np.all((z >= lo) & (z <= hi)) else None, 0
    z = depth[indices].copy()
    fallback = ~np.isfinite(z) | (z < lo) | (z > hi)
    z[fallback] = nominal[fallback]
    if not np.all(np.isfinite(z)) or not np.all((z >= lo) & (z <= hi)):
        return None, int(fallback.sum())
    return z, int(fallback.sum())


def _tanh(z, params):
    ts, td, z0, h = params
    return td + (ts - td) / 2.0 * (1.0 - np.tanh((z - z0) / h))


def _profile_bases(work, rows, b0, use_depth=True, fit_tanh=True):
    b1 = b0.copy()
    b2 = b0.copy()
    zcol = "depth" if use_depth else "nominal_depth"
    public = work[work.layer.isin(C.PUBLIC_LAYERS) & work.temp.notna()]
    by_time = {time: group for time, group in public.groupby("time", sort=False)}
    depth = rows.depth.to_numpy(float)
    nominal = rows.nominal_depth.to_numpy(float)
    counts = {
        "sparse_times": 0,
        "b1_fallback_times": 0,
        "b2_fit_fallback_times": 0,
        "target_depth_fallback_rows": 0,
    }
    for time, indices in rows.groupby("time", sort=False).indices.items():
        points = _points(by_time.get(time, public.iloc[0:0]), zcol)
        if points is None:
            counts["sparse_times"] += 1
            continue
        x, y = points
        z, depth_fallback = _target_z(depth, nominal, indices, x[0], x[-1], use_depth)
        counts["target_depth_fallback_rows"] += depth_fallback
        if z is None:
            counts["b1_fallback_times"] += 1
            if fit_tanh:
                counts["b2_fit_fallback_times"] += 1
            continue

        b1_ok = False
        try:
            b1_values = np.asarray(PchipInterpolator(x, y)(z), dtype=float)
            b1_ok = np.isfinite(b1_values).all() and ((b1_values >= -5) & (b1_values <= 45)).all()
        except Exception:
            b1_values = None
        if b1_ok:
            b1[indices] = b1_values
        else:
            counts["b1_fallback_times"] += 1

        if not fit_tanh:
            continue
        try:
            slopes = np.abs(np.diff(y) / np.diff(x))
            z0 = float((x[np.argmax(slopes)] + x[np.argmax(slopes) + 1]) / 2.0)
            result = least_squares(
                lambda params: _tanh(x, params) - y,
                [y[0], y[-1], z0, 3.0],
                bounds=([-np.inf, -np.inf, -np.inf, 0.5],
                        [np.inf, np.inf, np.inf, 30.0]),
                max_nfev=100,
            )
            b2_values = np.asarray(_tanh(z, result.x), dtype=float)
            b2_ok = result.success and np.isfinite(b2_values).all()
            b2_ok = b2_ok and ((b2_values >= -5) & (b2_values <= 45)).all()
        except Exception:
            b2_values, b2_ok = None, False
        if b2_ok:
            b2[indices] = b2_values
        else:
            counts["b2_fit_fallback_times"] += 1
            b2[indices] = b1[indices] if b1_ok else b0[indices]
    return b1, b2, counts


def _window_data(obs):
    out = {}
    work, _, rows = G.holdout(obs, "2024_sepoct")
    out["2024_sepoct"] = (work, rows)
    selected = {window["label"]: window for window in S._choose_windows(obs)[0]}
    out["S1"] = out["2024_sepoct"]
    for label in ("S3", "S4"):
        work, _, rows = S._season_holdout(obs, selected[label])
        out[label] = (work, rows)
    return out


def _maskaug(window):
    label = "S1" if window in ("2024_sepoct", "S1") else window
    path = ROOT / "preds" / f"oof_maskaug_{label}_E006.npy"
    return np.load(path, allow_pickle=False)


def _save_oof(name, values, label):
    save_pool_pred(ROOT / "preds" / f"oof_{name}_{label}_E019.npy", values)
    save_pool_pred(ROOT / "cache" / "T19" / f"oof_{name}_{label}_E019.npy", values)
    if label != "2024_sepoct":
        save_pool_pred(ROOT / "preds" / f"oof_{name}_{label}_E006.npy", values)


def _save_test(name, values):
    save_pool_pred(ROOT / "preds" / f"test_{name}_E019.npy", values)
    save_pool_pred(ROOT / "preds" / f"test_{name}.npy", values)
    save_pool_pred(ROOT / "cache" / "T19" / f"test_{name}_E019.npy", values)


def _report_row(reports, name, label):
    return reports[name][label]["overall"]


def main():
    obs = C.load_obs()
    cache = ROOT / "cache" / "T19"
    cache.mkdir(parents=True, exist_ok=True)
    data = _window_data(obs)
    bases = {"B0": {}, "B1": {}, "B2": {}}
    nominal_b1 = {}
    reports = {name: {} for name in ("B0", "B1", "B2", "maskaug")}
    counts = {}
    residuals = {name: [] for name in ("B0", "B1", "B2")}

    for label in WINDOWS:
        work, rows = data[label]
        x = C.build_features(work, rows)
        b0 = x.base_nom.to_numpy(float)
        b1, b2, actual_counts = _profile_bases(work, rows, b0, use_depth=True)
        b1_nominal, _, _ = _profile_bases(work, rows, b0, use_depth=False, fit_tanh=False)
        bases["B0"][label], bases["B1"][label], bases["B2"][label] = b0, b1, b2
        nominal_b1[label] = b1_nominal
        counts[label] = actual_counts
        for name, values in (("B0", b0), ("B1", b1), ("B2", b2)):
            reports[name][label] = G.score(rows, x, values)
            residuals[name].append(rows.temp.to_numpy(float) - values)
        maskaug = _maskaug(label)
        if len(maskaug) != len(rows):
            raise ValueError(f"maskaug shape: {label} {len(maskaug)} != {len(rows)}")
        reports["maskaug"][label] = G.score(rows, x, maskaug)
        for name in ("B1", "B2"):
            _save_oof("base" + name, bases[name][label], label)

    for label, expected in EXPECTED_B0.items():
        actual = reports["B0"][label]["overall"]
        if f"{actual:.6f}" != f"{expected:.6f}":
            raise RuntimeError(f"B0 mismatch {label}: {actual:.9f} != {expected:.6f}")

    for name in ("B1", "B2"):
        joined = np.concatenate([bases[name][label] for label in WINDOWS])
        save_pool_pred(ROOT / "preds" / f"oof_base{name}_E019.npy", joined)
        save_pool_pred(cache / f"oof_base{name}_E019.npy", joined)

    test_rows = C.test_rows(obs)
    test_x = C.build_features(obs, test_rows)
    test_b0 = test_x.base_nom.to_numpy(float)
    test_b1, test_b2, test_counts = _profile_bases(obs, test_rows, test_b0, use_depth=True)
    _save_test("baseB1", test_b1)
    _save_test("baseB2", test_b2)
    counts["test"] = test_counts

    with (cache / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "reports": reports,
            "counts": counts,
            "test_n": len(test_rows),
            "test_finite": bool(np.isfinite(test_b1).all() and np.isfinite(test_b2).all()),
        }, handle, ensure_ascii=False, indent=2)

    print("표1 | base | 2024_sepoct | S1 | S3 | S4")
    for name in ("B0", "B1", "B2", "maskaug"):
        print(" | ".join([name] + [f"{_report_row(reports, name, label):.6f}" for label in WINDOWS]))

    x_2024 = C.build_features(data["2024_sepoct"][0], data["2024_sepoct"][1])
    print("표2 | base | 전체 | 결측 | 완전 | 층2 | 층3 | 층4")
    for name in ("B0", "B1", "B2"):
        report = reports[name]["2024_sepoct"]
        values = [report["overall"], report["segments"]["missing_public"]["rmse"],
                  report["segments"]["complete_public"]["rmse"]]
        values.extend(report["segments"][f"layer_{layer}"]["rmse"] for layer in C.TARGET_LAYERS)
        print(" | ".join([name] + [f"{value:.6f}" for value in values]))

    print("표3 | B1 | depth | nominal_depth | depth-nominal")
    for label in WINDOWS:
        depth_value = reports["B1"][label]["overall"]
        nominal_value = G.score(data[label][1], C.build_features(data[label][0], data[label][1]),
                                nominal_b1[label])["overall"]
        print(f"{label} | {depth_value:.6f} | {nominal_value:.6f} | {depth_value - nominal_value:+.6f}")

    print("표4 | base | RMSE | residual_std | kurtosis_excess")
    for name in ("B0", "B1", "B2"):
        residual = np.concatenate(residuals[name])
        print(f"{name} | {np.sqrt(np.mean(residual ** 2)):.6f} | "
              f"{np.std(residual):.6f} | {kurtosis(residual, fisher=True, bias=False):.6f}")

    print("대체 건수 | 창 | sparse_public_times | B1_fallback_times | B2_fit_fallback_times | target_depth_fallback_rows")
    for label in (*WINDOWS, "test"):
        row = counts[label]
        print(f"{label} | {row['sparse_times']} | {row['b1_fallback_times']} | "
              f"{row['b2_fit_fallback_times']} | {row['target_depth_fallback_rows']}")
    print(f"test | n={len(test_rows)} | B1_finite={np.isfinite(test_b1).all()} | B2_finite={np.isfinite(test_b2).all()}")


if __name__ == "__main__":
    main()
