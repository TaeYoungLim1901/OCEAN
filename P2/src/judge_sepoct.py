"""E013: Sep-Oct aligned OOF judgment and leave-one-window-out stacking."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gate_e004 as G
import season_e006 as S
from stack_e008 import _fit_simplex
from submission_tools import save_pool_pred

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("2024_sepoct", "S1", "S3", "S4")
REGIMES = ("missing_public", "complete_public")


def _window_data(obs):
    out = {}
    work, _, rows = G.holdout(obs, "2024_sepoct")
    x = C.build_features(work, rows)
    out["2024_sepoct"] = {"rows": rows, "x": x, "flags": G.regimes(x, rows)}
    selected = {window["label"]: window for window in S._choose_windows(obs)[0]}
    for label in ("S1", "S3", "S4"):
        if label not in selected:
            continue
        work, _, rows = S._season_holdout(obs, selected[label])
        x = C.build_features(work, rows)
        out[label] = {"rows": rows, "x": x, "flags": G.regimes(x, rows)}
    return out


def _logical_name(stem):
    return re.sub(r"_S[1-4]_E006$", "", stem)


def _candidate_pool(data, obs):
    candidates = {}
    for path in sorted((ROOT / "preds").glob("oof_*.npy")):
        name = _logical_name(path.stem[4:])
        candidates.setdefault(name, {})
        try:
            pred = np.load(path, allow_pickle=False)
            if pred.ndim != 1 or not np.isfinite(pred).all():
                continue
            for label in WINDOWS:
                if len(pred) != len(data[label]["rows"]):
                    continue
                current = candidates[name].get(label)
                preferred = f"oof_{name}_{label}_E006.npy"
                if current is None or path.name == preferred:
                    candidates[name][label] = pred.astype(float, copy=False)
        except Exception:
            continue

    test_rows = C.test_rows(obs)
    test_x = C.build_features(obs, test_rows)
    candidates["base_nom"] = {
        label: data[label]["x"].base_nom.to_numpy(float) for label in WINDOWS
    }
    full = {name: values for name, values in candidates.items()
            if all(label in values for label in WINDOWS)}
    return full, sorted(set(candidates) - set(full)), test_rows, test_x


def _candidate_report(data, pool):
    rows = []
    reports = {}
    for name, values in pool.items():
        reports[name] = {}
        for label in WINDOWS:
            item = data[label]
            reports[name][label] = G.score(item["rows"], item["x"], values[label])
        row = {
            "name": name,
            "2024_sepoct": reports[name]["2024_sepoct"]["overall"],
            "missing": reports[name]["2024_sepoct"]["segments"]["missing_public"]["rmse"],
            "complete": reports[name]["2024_sepoct"]["segments"]["complete_public"]["rmse"],
        }
        row.update({label: reports[name][label]["overall"] for label in ("S1", "S3", "S4")})
        row["sepoct_mean"] = float(np.mean([row[label] for label in WINDOWS]))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["sepoct_mean"], row["name"])), reports


def _fit_lowo(data, pool, left):
    fit_labels = [label for label in WINDOWS if label != left]
    x_fit = np.vstack([np.column_stack([pool[name][label] for name in sorted(pool)])
                       for label in fit_labels])
    y_fit = np.concatenate([data[label]["rows"].temp.to_numpy(float) for label in fit_labels])
    x_valid = np.column_stack([pool[name][left] for name in sorted(pool)])
    global_weights = _fit_simplex(x_fit, y_fit)
    global_pred = x_valid @ global_weights

    regime_pred = np.empty(len(x_valid), dtype=float)
    fallback = global_weights
    for regime in REGIMES:
        fit_flags = np.concatenate([data[label]["flags"][regime] for label in fit_labels])
        weights = _fit_simplex(x_fit[fit_flags], y_fit[fit_flags], fallback)
        valid_flags = data[left]["flags"][regime]
        regime_pred[valid_flags] = x_valid[valid_flags] @ weights
    return {"stack_global": global_pred, "stack_regime": regime_pred}


def _lowo_report(data, pool):
    reports = {name: {} for name in ("maskaug", "stack_global", "stack_regime")}
    for label in WINDOWS:
        maskaug = pool["maskaug"][label]
        reports["maskaug"][label] = G.score(data[label]["rows"], data[label]["x"], maskaug)
        candidates = _fit_lowo(data, pool, label)
        for name, pred in candidates.items():
            reports[name][label] = G.score(data[label]["rows"], data[label]["x"], pred)
    means = {name: float(np.mean([reports[name][label]["overall"] for label in WINDOWS]))
             for name in reports}
    return reports, means


def _fit_all(data, pool, mode, test_rows, test_x):
    names = sorted(pool)
    x_oof = np.vstack([np.column_stack([pool[name][label] for name in names])
                       for label in WINDOWS])
    y = np.concatenate([data[label]["rows"].temp.to_numpy(float) for label in WINDOWS])
    x_test = []
    for name in names:
        if name == "base_nom":
            x_test.append(test_x.base_nom.to_numpy(float))
        elif name == "maskaug":
            x_test.append(np.load(ROOT / "preds" / "test_maskaug_E005.npy", allow_pickle=False))
        else:
            x_test.append(np.load(ROOT / "preds" / f"test_{name}.npy", allow_pickle=False))
    x_test = np.column_stack(x_test)
    if mode == "stack_global":
        weights = _fit_simplex(x_oof, y)
        return np.concatenate([np.column_stack([pool[name][label] for name in names]) @ weights
                               for label in WINDOWS]), x_test @ weights

    fallback = _fit_simplex(x_oof, y)
    test_flags = G.regimes(test_x, test_rows)
    weights = {
        regime: _fit_simplex(
            x_oof[np.concatenate([data[label]["flags"][regime] for label in WINDOWS])],
            y[np.concatenate([data[label]["flags"][regime] for label in WINDOWS])],
            fallback,
        )
        for regime in REGIMES
    }
    oof_blocks = []
    test_pred = np.empty(len(x_test), dtype=float)
    for label in WINDOWS:
        block = np.column_stack([pool[name][label] for name in names])
        block_pred = block @ fallback
        for regime in REGIMES:
            local = data[label]["flags"][regime]
            block_pred[local] = block[local] @ weights[regime]
            test_local = test_flags[regime]
            test_pred[test_local] = x_test[test_local] @ weights[regime]
        oof_blocks.append(block_pred)
    return np.concatenate(oof_blocks), test_pred


def _write_submission(test_pred):
    out = ROOT / "subs" / "E013_stack_sepoct.csv"
    with tempfile.TemporaryDirectory(prefix="e013_mix_") as temp:
        missing = Path(temp) / "missing.npy"
        complete = Path(temp) / "complete.npy"
        save_pool_pred(missing, test_pred)
        save_pool_pred(complete, test_pred)
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "mix_sub.py"),
             "--missing", str(missing), "--complete", str(complete), "--out", str(out)],
            cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8"
        )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    print(result.stdout.strip())
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "bugcheck.py"), "--only", "G"],
        cwd=ROOT.parent, capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout.strip())


def main():
    obs = C.load_obs()
    data = _window_data(obs)
    missing_windows = [label for label in WINDOWS if label not in data]
    if missing_windows:
        print("skipped | " + ",".join(missing_windows))
        return 0
    pool, skipped, test_rows, test_x = _candidate_pool(data, obs)
    rows, _ = _candidate_report(data, pool)
    print("표1 | 후보 | 2024_sepoct_전체 | 2024_sepoct_결측 | 2024_sepoct_완전 | S1 | S3 | S4 | sepoct_mean")
    for row in rows[:10]:
        print(" | ".join([
            row["name"], f"{row['2024_sepoct']:.6f}", f"{row['missing']:.6f}",
            f"{row['complete']:.6f}", *(f"{row[label]:.6f}" for label in ("S1", "S3", "S4")),
            f"{row['sepoct_mean']:.6f}",
        ]))
    print("skipped | " + (",".join(skipped) if skipped else "0"))

    if "maskaug" not in pool:
        print("표2 | maskaug | NA")
        print("G | 해당없음")
        return 0

    reports, means = _lowo_report(data, pool)
    baseline = means["maskaug"]
    print("표2 | 구성 | 2024_sepoct | S1 | S3 | S4 | LOWO_mean | Δ_vs_maskaug")
    for name in ("maskaug", "stack_global", "stack_regime"):
        values = [reports[name][label]["overall"] for label in WINDOWS]
        print(" | ".join([name, *(f"{value:.6f}" for value in values),
                           f"{means[name]:.6f}", f"{means[name] - baseline:+.6f}"]))

    winners = [name for name in ("stack_global", "stack_regime") if means[name] < baseline]
    if not winners:
        print("G | 해당없음")
        return 0
    selected = min(winners, key=lambda name: means[name])
    oof_pred, test_pred = _fit_all(data, pool, selected, test_rows, test_x)
    save_pool_pred(ROOT / "preds" / "oof_stack_sepoct_E013.npy", oof_pred)
    save_pool_pred(ROOT / "preds" / "test_stack_sepoct_E013.npy", test_pred)
    print(f"artifacts | 1 | {selected}")
    _write_submission(test_pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
