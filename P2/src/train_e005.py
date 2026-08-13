"""E005: 공개층 결측 패턴을 주입한 E002 fixed-depth 모델."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from gate_e004 import HOLDS, holdout, regimes, score
from train import estimator

FEATURES = C.FULL + [
    *(f"public_observed_{layer}" for layer in C.PUBLIC_LAYERS),
    "public_available_count", "nearest_public_layer_distance",
]
MASK_COLUMNS = ("temp", "psal", "depth")
SOURCE_START = pd.Timestamp("2025-09-01", tz="Asia/Seoul")
SOURCE_END = pd.Timestamp("2025-10-31 23:50", tz="Asia/Seoul")
LIMIT_SECONDS = 40 * 60
SEEDS = {
    "2024_julaug": 4201, "2024_sepoct": 4202,
    "2025_mayjun": 4203, "2025_julaug": 4204, "final": 4242,
}
_SOURCE_CACHE = {}


def add_mask_features(x):
    x = x.reset_index(drop=True).copy()
    layers = np.asarray(C.PUBLIC_LAYERS, dtype=float)
    observed = np.column_stack([
        x[f"temp_{layer}"].notna().to_numpy() for layer in C.PUBLIC_LAYERS
    ])
    for i, layer in enumerate(C.PUBLIC_LAYERS):
        x[f"public_observed_{layer}"] = observed[:, i].astype(float)
    x["public_available_count"] = observed.sum(axis=1).astype(float)
    distance = np.abs(x["layer"].to_numpy()[:, None] - layers[None, :])
    distance[~observed] = np.inf
    nearest = distance.min(axis=1)
    nearest[~np.isfinite(nearest)] = len(C.PUBLIC_LAYERS) + 1
    x["nearest_public_layer_distance"] = nearest
    return x


def _runs(flag):
    out = []
    i = 0
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i + 1
        while j < len(flag) and flag[j]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


def _source_pattern(obs, expected):
    key = tuple(expected)
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    source = obs[obs.time.between(SOURCE_START, SOURCE_END)]
    times = pd.DatetimeIndex(source.time.drop_duplicates()).sort_values()
    wides = {column: source.pivot(index="time", columns="layer", values=column).reindex(times)
             for column in MASK_COLUMNS}
    temp = wides["temp"]
    flag = temp.reindex(columns=expected).isna().any(axis=1).to_numpy()
    masks = {}
    for layer in C.PUBLIC_LAYERS:
        for column in MASK_COLUMNS:
            masks[(layer, column)] = wides[column][layer].isna().to_numpy()
    value = times, masks, sorted(_runs(flag), key=lambda item: item[1], reverse=True)
    _SOURCE_CACHE[key] = value
    return value


def _train_time_index(rows):
    counts = rows.groupby("time", sort=True).size()
    times = pd.DatetimeIndex(counts.index).sort_values()
    counts = counts.reindex(times).to_numpy(dtype=int)
    contiguous = np.ones(len(times), dtype=int)
    if len(times) > 1:
        step = pd.Timedelta(minutes=10).value
        for i in range(len(times) - 2, -1, -1):
            if times.asi8[i + 1] - times.asi8[i] == step:
                contiguous[i] = contiguous[i + 1] + 1
    return times, counts, contiguous


def mask_augmentation(obs, rows, ratio, expected, seed):
    """Shift source missing runs onto random contiguous training blocks."""
    source_times, source_masks, source_runs = _source_pattern(obs, expected)
    times, counts, contiguous = _train_time_index(rows)
    target_rows = min(len(rows), max(1, int(round(ratio * len(rows)))))
    if not len(times) or not source_runs:
        raise ValueError("결측 주입용 시간 블록 부족")

    rng = np.random.default_rng(seed)
    occupied = np.zeros(len(times), dtype=bool)
    selected = np.zeros(len(times), dtype=bool)
    placements = []
    remaining = target_rows
    attempts = 0
    run_order = list(rng.permutation(len(source_runs)))
    while remaining > 0 and attempts < 20000:
        run_start, run_length = source_runs[run_order[attempts % len(run_order)]]
        attempts += 1
        max_start = len(times) - 1
        start = int(rng.integers(0, max_start + 1))
        max_length = min(run_length, int(contiguous[start]))
        if max_length <= 0 or occupied[start:start + max_length].any():
            continue
        prefix = np.cumsum(counts[start:start + max_length])
        length = int(np.searchsorted(prefix, remaining, side="left") + 1)
        length = min(max_length, max(1, length))
        if occupied[start:start + length].any():
            continue
        occupied[start:start + length] = True
        selected[start:start + length] = True
        placements.append((start, run_start, length))
        remaining = target_rows - int(counts[selected].sum())

    if remaining > 0:
        raise RuntimeError(f"결측 블록 배치 부족: remaining={remaining}, attempts={attempts}")

    mask_times = {(layer, column): set() for layer in C.PUBLIC_LAYERS for column in MASK_COLUMNS}
    for start, source_start, length in placements:
        dst = times[start:start + length]
        for layer in C.PUBLIC_LAYERS:
            for column in MASK_COLUMNS:
                take = source_masks[(layer, column)][source_start:source_start + length]
                mask_times[(layer, column)].update(dst[take])

    augmented = obs.copy()
    for (layer, column), values in mask_times.items():
        if values:
            use = augmented.layer.eq(layer) & augmented.time.isin(values)
            augmented.loc[use, column] = np.nan
    return augmented, rows.time.isin(times[selected]).to_numpy()


def fit_models(x, rows):
    models = {}
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        model = estimator("lgb_full")
        model.fit(x.loc[use, FEATURES], rows.loc[use, "temp"] - x.loc[use, "base_nom"])
        models[layer] = model
    return models


def predict(models, x, rows):
    out = np.empty(len(rows), dtype=float)
    for layer in C.TARGET_LAYERS:
        use = rows.layer.eq(layer)
        out[use] = x.loc[use, "base_nom"] + models[layer].predict(x.loc[use, FEATURES])
    if not np.isfinite(out).all():
        raise ValueError("비유한 예측")
    return out


def _fit_input(x_clean, x_aug, rows, candidate, augmented_rows):
    if candidate == "maskaug":
        x = x_clean.copy()
        x.loc[augmented_rows, :] = x_aug.loc[augmented_rows, :].to_numpy()
        return x, rows
    return x_aug.loc[augmented_rows].reset_index(drop=True), rows.loc[augmented_rows].reset_index(drop=True)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    obs = C.load_obs()
    cache = C.ROOT / "cache" / "E005"
    cache.mkdir(parents=True, exist_ok=True)
    stats = json.loads((C.ROOT / "cache" / "E004" / "gate_stats.json").read_text(encoding="utf-8"))
    baseline_oof = np.load(C.ROOT / "preds" / "oof_gate_robust_blend_E004.npy", allow_pickle=False)
    baseline_test = np.load(C.ROOT / "preds" / "test_gate_robust_blend_E004.npy", allow_pickle=False)
    runtime = {"limit_seconds": LIMIT_SECONDS, "holds": {}, "total_seconds": None}
    started = time.perf_counter()
    oof_parts = {candidate: [] for candidate in ("maskaug", "missing_only")}
    all_completed = True
    offset = 0

    for label in HOLDS:
        if time.perf_counter() - started >= LIMIT_SECONDS:
            all_completed = False
            break
        hold_started = time.perf_counter()
        work, train, val = holdout(obs, label)
        x_clean = add_mask_features(C.build_features(work, train, tide_correct=True, use_target_depth=True))
        augmented, augmented_rows = mask_augmentation(
            work, train, stats[label]["ratio"],
            [1, 5, 6, 7] if label.startswith("2024") else C.PUBLIC_LAYERS,
            SEEDS[label],
        )
        x_aug = add_mask_features(C.build_features(augmented, train, tide_correct=True, use_target_depth=True))
        x_val = add_mask_features(C.build_features(work, val, tide_correct=True, use_target_depth=True))
        flags = regimes(x_val, val)["missing_public"]
        base = baseline_oof[offset:offset + len(val)]
        if len(base) != len(val):
            raise ValueError(f"E004 OOF 길이 부족: {label}")
        hold_reports = {"e004": score(val, x_val, base)}
        for candidate in ("maskaug", "missing_only"):
            x_fit, rows_fit = _fit_input(x_clean, x_aug, train, candidate, augmented_rows)
            raw = predict(fit_models(x_fit, rows_fit), x_val, val)
            gated = np.where(flags, raw, base)
            oof_parts[candidate].append(gated)
            np.save(cache / f"oof_{label}_{candidate}_E005.npy", gated)
            hold_reports[candidate] = score(val, x_val, gated)
        np.save(cache / f"oof_{label}_e004_E005.npy", base)
        _write_json(cache / f"metrics_{label}.json", hold_reports)
        elapsed = time.perf_counter() - hold_started
        runtime["holds"][label] = {"seconds": elapsed, "completed": True, "n": len(val)}
        _write_json(cache / "runtime.json", runtime)
        offset += len(val)
        if time.perf_counter() - started >= LIMIT_SECONDS and offset < len(baseline_oof):
            all_completed = False
            break

    if len(runtime["holds"]) != len(HOLDS):
        all_completed = False
    if all_completed:
        for candidate, parts in oof_parts.items():
            np.save(C.ROOT / "preds" / f"oof_{candidate}_E005.npy", np.concatenate(parts))

        all_rows = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()].reset_index(drop=True)
        x_clean = add_mask_features(C.build_features(obs, all_rows, tide_correct=True, use_target_depth=True))
        augmented, augmented_rows = mask_augmentation(
            obs, all_rows, stats["test"]["ratio"], C.PUBLIC_LAYERS, SEEDS["final"]
        )
        x_aug = add_mask_features(C.build_features(augmented, all_rows, tide_correct=True, use_target_depth=True))
        test_rows = C.test_rows(obs)
        baseline = pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
        x_test = add_mask_features(C.build_features(
            obs, test_rows, baseline, tide_correct=True, use_target_depth=True
        ))
        test_flags = regimes(x_test, test_rows)["missing_public"]
        if len(baseline_test) != len(test_rows):
            raise ValueError("E004 test 예측 길이 불일치")
        for candidate in ("maskaug", "missing_only"):
            x_fit, rows_fit = _fit_input(x_clean, x_aug, all_rows, candidate, augmented_rows)
            raw = predict(fit_models(x_fit, rows_fit), x_test, test_rows)
            gated = np.where(test_flags, np.clip(raw, -5, 45), baseline_test)
            np.save(C.ROOT / "preds" / f"test_{candidate}_E005.npy", gated)

    runtime["total_seconds"] = time.perf_counter() - started
    runtime["completed_all"] = all_completed
    _write_json(cache / "runtime.json", runtime)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
