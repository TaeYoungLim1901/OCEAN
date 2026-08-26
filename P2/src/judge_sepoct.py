"""E013: Sep-Oct aligned OOF judgment and leave-one-window-out stacking."""
import argparse
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
    return re.sub(r"_(?:S[1-4]|2024_sepoct)_E\d+$", "", stem)


def _experiment_number(stem):
    match = re.search(r"_E(\d+)$", stem)
    return int(match.group(1)) if match else -1


def _candidate_pool(data, obs):
    candidates = {}
    sources = {}
    for path in sorted((ROOT / "preds").glob("oof_*.npy")):
        name = _logical_name(path.stem[4:])
        candidates.setdefault(name, {})
        sources.setdefault(name, {})
        try:
            pred = np.load(path, allow_pickle=False)
            if pred.ndim != 1 or not np.isfinite(pred).all():
                continue
            for label in WINDOWS:
                if len(pred) != len(data[label]["rows"]):
                    continue
                current = candidates[name].get(label)
                current_source = sources[name].get(label)
                if (current is None or
                        _experiment_number(path.stem) > _experiment_number(current_source)):
                    candidates[name][label] = pred.astype(float, copy=False)
                    sources[name][label] = path.stem
        except Exception:
            continue

    test_rows = C.test_rows(obs)
    test_x = C.build_features(obs, test_rows)
    candidates["base_nom"] = {
        label: data[label]["x"].base_nom.to_numpy(float) for label in WINDOWS
    }
    full = {name: values for name, values in candidates.items()
            if all(label in values for label in WINDOWS)}
    experiments = {
        name: max(_experiment_number(stem) for stem in sources[name].values())
        for name in full if sources.get(name)
    }
    return full, sorted(set(candidates) - set(full)), test_rows, test_x, experiments


def _test_paths(pool, experiments):
    paths = {}
    excluded = []
    for name in sorted(pool):
        if name == "base_nom":
            continue
        if name == "maskaug":
            path = ROOT / "preds" / "test_maskaug_E005.npy"
        else:
            number = experiments.get(name, -1)
            path = ROOT / "preds" / f"test_{name}_E{number:03d}.npy"
        if path.exists():
            paths[name] = path
        else:
            excluded.append(name)
    return paths, excluded


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


def _fit_all(data, pool, mode, test_rows, test_x, test_paths):
    names = sorted(pool)
    x_oof = np.vstack([np.column_stack([pool[name][label] for name in names])
                       for label in WINDOWS])
    y = np.concatenate([data[label]["rows"].temp.to_numpy(float) for label in WINDOWS])
    x_test = []
    for name in names:
        if name == "base_nom":
            x_test.append(test_x.base_nom.to_numpy(float))
        else:
            x_test.append(np.load(test_paths[name], allow_pickle=False))
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
    out = ROOT / "subs" / "E022_stack_sepoct.csv"
    out_npy = ROOT / "preds" / "test_stack_sepoct_E022.npy"
    if out.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out}")
    if out_npy.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out_npy}")
    with tempfile.TemporaryDirectory(prefix="e022_mix_") as temp:
        missing = Path(temp) / "missing.npy"
        complete = Path(temp) / "complete.npy"
        save_pool_pred(missing, test_pred)
        save_pool_pred(complete, test_pred)
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "mix_sub.py"),
             "--missing", str(missing), "--complete", str(complete), "--out", str(out),
             "--out-npy", str(out_npy)],
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


def main(write_artifacts=False):
    obs = C.load_obs()
    data = _window_data(obs)
    missing_windows = [label for label in WINDOWS if label not in data]
    if missing_windows:
        print("skipped | " + ",".join(missing_windows))
        return 0
    pool, skipped, test_rows, test_x, experiments = _candidate_pool(data, obs)
    test_paths, excluded = _test_paths(pool, experiments)
    rows, _ = _candidate_report(data, pool)
    print("표1 | 후보 | 2024_sepoct_전체 | 2024_sepoct_결측 | 2024_sepoct_완전 | S1 | S3 | S4 | sepoct_mean")
    for row in rows[:10]:
        print(" | ".join([
            row["name"], f"{row['2024_sepoct']:.6f}", f"{row['missing']:.6f}",
            f"{row['complete']:.6f}", *(f"{row[label]:.6f}" for label in ("S1", "S3", "S4")),
            f"{row['sepoct_mean']:.6f}",
        ]))
    print("skipped | " + (",".join(skipped) if skipped else "0"))
    print("excluded | " + (",".join(excluded) if excluded else "0"))

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
    if not write_artifacts:
        return 0
    stack_pool = {name: values for name, values in pool.items() if name not in excluded}
    oof_pred, test_pred = _fit_all(data, stack_pool, selected, test_rows, test_x, test_paths)
    save_pool_pred(ROOT / "preds" / "oof_stack_sepoct_E022.npy", oof_pred)
    print(f"artifacts | 1 | {selected}")
    _write_submission(test_pred)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-artifacts", action="store_true")
    raise SystemExit(main(parser.parse_args().write_artifacts))
