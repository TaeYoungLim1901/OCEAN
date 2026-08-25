"""E006: season-aligned holdouts for the E005 maskaug path."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from submission_tools import save_pool_pred
import train_e003 as E3
from gate_e004 import holdout, regimes, score
from train_e003 import fit_eof, fit_layers, predict_eof, predict_layers
from train_e005 import add_mask_features, fit_models, mask_augmentation, predict

SEED = 42
LIMIT_SECONDS = 40 * 60
EXPECTED_2024 = [1, 5, 6, 7]
WINDOWS = (
    ("S1", "2024-09-01", "2024-10-31 23:50", False,
     (("2024-09-01", "2024-10-31 23:50"),)),
    ("S2", "2024-08-16", "2024-10-15 23:50", True,
     (("2024-08-16", "2024-10-15 23:50"),)),
    ("S3", "2024-09-16", "2024-11-15 23:50", True,
     (("2024-09-16", "2024-11-15 23:50"),
      ("2024-08-01", "2024-09-30 23:50"))),
    ("S4", "2024-10-01", "2024-11-30 23:50", True,
     (("2024-10-01", "2024-11-30 23:50"),
      ("2024-07-16", "2024-09-14 23:50"))),
)


def _stamp(value):
    return pd.Timestamp(value, tz="Asia/Seoul")


def _available(obs, start, end):
    start_ts, end_ts = _stamp(start), _stamp(end)
    times = obs.loc[obs.time.between(start_ts, end_ts), "time"].drop_duplicates().sort_values()
    return len(times) == 8784 and times.iloc[0] == start_ts and times.iloc[-1] == end_ts, times


def _choose_windows(obs):
    chosen, rows = [], []
    for label, primary_start, primary_end, transfer, candidates in WINDOWS:
        selected = None
        for start, end in candidates:
            ok, times = _available(obs, start, end)
            if ok:
                selected = {"label": label, "start": start, "end": end,
                            "transfer": transfer, "times": times}
                break
        if selected is None:
            rows.append({"label": label, "start": primary_start, "end": primary_end,
                         "truth_n": 0, "selected": False, "available": False})
        else:
            selected["available"] = True
            selected["selected"] = True
            chosen.append(selected)
            rows.append({"label": label, "start": selected["start"], "end": selected["end"],
                         "truth_n": None, "selected": True, "available": True})
    return chosen, rows


def _season_holdout(obs, window):
    old_holds = dict(E3.HOLDS)
    E3.HOLDS[window["label"]] = (window["start"], window["end"], window["transfer"])
    try:
        return holdout(obs, window["label"])
    finally:
        E3.HOLDS.clear()
        E3.HOLDS.update(old_holds)


def _existing_s1_baseline(val):
    index = pd.read_csv(C.ROOT / "cache" / "E003" / "oof_index.csv", parse_dates=["time"])
    pred = np.load(C.ROOT / "preds" / "oof_gate_robust_blend_E004.npy", allow_pickle=False)
    frame = index[C.KEY].copy()
    frame["pred"] = pred
    joined = val[C.KEY].merge(frame, on=C.KEY, how="left", validate="one_to_one", sort=False)
    if joined.pred.isna().any():
        raise ValueError("S1 E004 OOF 키 누락")
    return joined.pred.to_numpy(dtype=float)


def _fit_e004_baseline(obs, work, train, val, x_train, x_val, flags, blend_weight):
    e002_models = fit_layers(x_train, train, C.FULL)
    enhanced_models = fit_layers(x_train, train, C.ENHANCED)
    e002 = predict_layers(e002_models, x_val, val, C.FULL)
    enhanced = predict_layers(enhanced_models, x_val, val, C.ENHANCED)
    eof = fit_eof(obs, work, train, x_train)
    val_with_base = val.copy()
    val_with_base["_base"] = x_val.base_nom.to_numpy()
    eof_pred = predict_eof(eof, obs, work, val_with_base)
    robust = (1.0 - blend_weight) * enhanced + blend_weight * eof_pred
    return np.where(flags, e002, robust)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _overlap(windows):
    result = []
    for i, left in enumerate(windows):
        a = set(left["times"])
        for right in windows[i + 1:]:
            b = set(right["times"])
            result.append({
                "pair": f"{left['label']}-{right['label']}",
                "intersection": len(a & b),
                "ratio": len(a & b) / min(len(a), len(b)),
            })
    return result


def _reference_holds():
    out = {}
    for label in ("2024_julaug", "2025_julaug", "2025_mayjun"):
        metrics = json.loads((C.ROOT / "cache" / "E005" / f"metrics_{label}.json").read_text(encoding="utf-8"))
        e002 = metrics["e004"]["segments"]["missing_public"]["rmse"]
        maskaug = metrics["maskaug"]["segments"]["missing_public"]["rmse"]
        out[label] = {"e002": float(e002), "maskaug": float(maskaug),
                      "delta": float(maskaug - e002)}
    return out


def main():
    if C.SEED != SEED:
        raise ValueError(f"SEED 불일치: {C.SEED}")
    obs = C.load_obs()
    cache = C.ROOT / "cache" / "E006"
    cache.mkdir(parents=True, exist_ok=True)
    windows, window_rows = _choose_windows(obs)
    runtime_path = cache / "runtime.json"
    prior_runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    runtime = {"limit_seconds": LIMIT_SECONDS, "windows": {}, "total_seconds": None,
               "completed_all": False}
    runtime["windows"].update(prior_runtime.get("windows", {}))
    started = time.perf_counter()
    reports = {}
    completed = []
    trained_any = False
    blend_weight = float(json.loads(
        (C.ROOT / "cache" / "E003" / "metrics.json").read_text(encoding="utf-8")
    )["final_blend_weight"])

    for window in windows:
        if time.perf_counter() - started >= LIMIT_SECONDS:
            break
        cached_metrics = cache / f"metrics_{window['label']}.json"
        cached_pred = C.ROOT / "preds" / f"oof_maskaug_{window['label']}_E006.npy"
        if cached_metrics.exists() and cached_pred.exists():
            cached = json.loads(cached_metrics.read_text(encoding="utf-8"))
            reports[window["label"]] = cached
            completed.append((window, None, None, None, None,
                              np.load(cached_pred, allow_pickle=False),
                              cached["e004"], cached["maskaug"]))
            runtime["windows"].setdefault(window["label"], {})
            runtime["windows"][window["label"]].update(
                {"completed": True, "cached": True, "truth_n": cached["truth_n"],
                 "missing_n": cached["missing_n"]}
            )
            continue
        hold_started = time.perf_counter()
        label = window["label"]
        work, train, val = _season_holdout(obs, window)
        if len(val) < 20000:
            runtime["windows"][label] = {"seconds": 0.0, "completed": False,
                                          "truth_n": len(val), "reason": "truth<20000"}
            _write_json(cache / "runtime.json", runtime)
            continue

        x_clean = add_mask_features(C.build_features(
            work, train, tide_correct=True, use_target_depth=True
        ))
        x_val = add_mask_features(C.build_features(
            work, val, tide_correct=True, use_target_depth=True
        ))
        flags = regimes(x_val, val)["missing_public"]
        ratio = float(flags.mean())
        if label == "S1":
            e004 = _existing_s1_baseline(val)
        else:
            e004 = _fit_e004_baseline(obs, work, train, val, x_clean, x_val, flags, blend_weight)

        augmented, augmented_rows = mask_augmentation(
            work, train, ratio, EXPECTED_2024, SEED
        )
        x_aug = add_mask_features(C.build_features(
            augmented, train, tide_correct=True, use_target_depth=True
        ))
        x_fit = x_clean.copy()
        x_fit.loc[augmented_rows, :] = x_aug.loc[augmented_rows, :].to_numpy()
        raw = predict(fit_models(x_fit, train), x_val, val)
        e006 = np.where(flags, raw, e004)
        if not np.isfinite(e006).all():
            raise ValueError(f"비유한 예측: {label}")
        if not np.array_equal(e006[~flags], e004[~flags]):
            raise ValueError(f"완전 공개 행 변경: {label}")

        e004_report = score(val, x_val, e004)
        e006_report = score(val, x_val, e006)
        reports[label] = {
            "start": window["start"], "end": window["end"], "transfer": window["transfer"],
            "truth_n": len(val), "missing_n": int(flags.sum()), "ratio": ratio,
            "e004": e004_report, "maskaug": e006_report,
            "source": "existing E004 OOF" if label == "S1" else "seasonal E004-style refit",
        }
        save_pool_pred(C.ROOT / "preds" / f"oof_maskaug_{label}_E006.npy", e006)
        save_pool_pred(cache / f"oof_maskaug_{label}_E006.npy", e006)
        _write_json(cache / f"metrics_{label}.json", reports[label])
        elapsed = time.perf_counter() - hold_started
        runtime["windows"][label] = {"seconds": elapsed, "completed": True,
                                      "truth_n": len(val), "missing_n": int(flags.sum())}
        _write_json(cache / "runtime.json", runtime)
        completed.append((window, val, x_val, flags, e004, e006, e004_report, e006_report))
        trained_any = True
        if time.perf_counter() - started >= LIMIT_SECONDS:
            break

    runtime["total_seconds"] = (time.perf_counter() - started if trained_any
                                 else prior_runtime.get("total_seconds", 0.0))
    runtime["completed_all"] = len(completed) == len(windows)
    _write_json(cache / "runtime.json", runtime)

    for row in window_rows:
        if row["label"] in reports:
            row["truth_n"] = reports[row["label"]]["truth_n"]
    overlaps = _overlap([item[0] for item in completed])
    reference = _reference_holds()
    missing_deltas = [
        item[7]["segments"]["missing_public"]["rmse"]
        - item[6]["segments"]["missing_public"]["rmse"]
        for item in completed
    ]
    overall_deltas = [item[7]["overall"] - item[6]["overall"] for item in completed]
    improved = sum(delta < 0.0 for delta in missing_deltas)
    mean_improvement = float(-np.mean(missing_deltas)) if missing_deltas else float("nan")
    max_missing_worsening = float(max(missing_deltas)) if missing_deltas else float("nan")
    max_overall_delta = float(max(overall_deltas)) if overall_deltas else float("nan")
    complete_equal = all(
        item[3] is None or np.array_equal(item[5][~item[3]], item[4][~item[3]])
        for item in completed
    )

    test_rows = C.test_rows(obs)
    baseline = pd.read_csv(C.DATA / "baseline_interp.csv", parse_dates=["time"])
    test_x = add_mask_features(C.build_features(
        obs, test_rows, baseline, tide_correct=True, use_target_depth=True
    ))
    test_flags = regimes(test_x, test_rows)["missing_public"]
    test_base = np.load(C.ROOT / "preds" / "test_gate_robust_blend_E004.npy", allow_pickle=False)
    test_maskaug = np.load(C.ROOT / "preds" / "test_maskaug_E005.npy", allow_pickle=False)
    test_final = np.where(test_flags, test_maskaug, test_base)
    test_complete_equal = np.array_equal(test_final[~test_flags], test_base[~test_flags])
    complete_equal = complete_equal and test_complete_equal

    rules = {
        "R1": improved >= 3 and len(completed) > 2,
        "R2": bool(np.isfinite(mean_improvement) and mean_improvement >= 0.020),
        "R3": bool(np.isfinite(max_missing_worsening) and max_missing_worsening <= 0.010),
        "R4": bool(np.isfinite(max_overall_delta) and max_overall_delta <= 0.000),
        "R5": complete_equal,
    }
    accepted = len(completed) > 2 and all(rules.values())
    payload = {
        "windows": window_rows, "reports": reports, "overlap": overlaps,
        "reference_nonseason": reference, "blend_weight": blend_weight,
        "rules": {key: {"pass": value} for key, value in rules.items()},
        "rule_values": {"R1_improved": improved, "R1_n": len(completed),
                         "R2_mean_improvement": mean_improvement,
                         "R3_max_missing_worsening": max_missing_worsening,
                         "R4_max_overall_delta": max_overall_delta,
                         "R5_complete_equal": complete_equal,
                         "R5_test_complete_equal": test_complete_equal},
        "background": {
            "e005_rejected_by": "4-hold uniform mean",
            "post_hoc_season_alignment": True,
            "e005_missing_public": {
                "2024_sepoct": {"e002": 1.135442, "maskaug": 1.000900, "delta": -0.134542},
                "2025_julaug": {"e002": 1.282815, "maskaug": 1.207117, "delta": -0.075698},
                "2024_julaug": {"e002": 1.215896, "maskaug": 1.282816, "delta": 0.066920},
                "2025_mayjun": {"e002": 1.814061, "maskaug": 1.871968, "delta": 0.057907},
            },
        },
        "accepted": accepted,
    }
    _write_json(cache / "metrics.json", payload)

    print("표1 사용 창: 창 | 시작 | 끝 | truth행 | 채택여부")
    for row in window_rows:
        print(f"{row['label']} | {row['start']} | {row['end']} | {row['truth_n']} | {row['selected']}")
    print("표2 missing_public: 창 | n | e002 | maskaug | Δ")
    for window, _, _, _, _, _, e004, e006 in completed:
        a = e004["segments"]["missing_public"]["rmse"]
        b = e006["segments"]["missing_public"]["rmse"]
        print(f"{window['label']} | {e004['segments']['missing_public']['n']} | {a:.6f} | {b:.6f} | {b-a:+.6f}")
    print("표3 전체 RMSE: 창 | E004조립 | E006조립 | Δ")
    for window, _, _, _, _, _, e004, e006 in completed:
        print(f"{window['label']} | {e004['overall']:.6f} | {e006['overall']:.6f} | {e006['overall']-e004['overall']:+.6f}")
    print("표4 창 겹침 비율: 창쌍 | 겹침비율")
    for row in overlaps:
        print(f"{row['pair']} | {row['ratio']:.6f}")
    print("표5 참고(비계절 hold): hold | e002 | maskaug | Δ")
    for label, row in reference.items():
        print(f"{label} | {row['e002']:.6f} | {row['maskaug']:.6f} | {row['delta']:+.6f}")
    print("표6 규칙 체크: R1 | R2 | R3 | R4 | R5 (숫자와 pass/fail)")
    print(f"R1 | {improved}/{len(completed)} | {'pass' if rules['R1'] else 'fail'}")
    print(f"R2 | {mean_improvement:.6f} | {'pass' if rules['R2'] else 'fail'}")
    print(f"R3 | {max_missing_worsening:+.6f} | {'pass' if rules['R3'] else 'fail'}")
    print(f"R4 | {max_overall_delta:+.6f} | {'pass' if rules['R4'] else 'fail'}")
    print(f"R5 | {int(complete_equal)} | {'pass' if rules['R5'] else 'fail'}")

    if accepted:
        save_pool_pred(C.ROOT / "preds" / "test_gate_maskaug_E006.npy", test_final)
        test_index = pd.read_csv(C.DATA / "test_index.csv")
        test_index[C.KEY].assign(temp=test_final).to_csv(
            C.ROOT / "subs" / "E006_gate.csv", index=False
        )
    print(f"판정: {'채택' if accepted else '기각'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
