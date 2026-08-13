"""E005 게이트: 결측 행만 mask-augmented 후보로 교체."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from gate_e004 import HOLDS, holdout, regimes, score

CANDIDATES = ("maskaug", "missing_only")


def _key_frame(rows):
    out = rows[C.KEY].copy()
    out["time"] = out.time.astype(str)
    return out.reset_index(drop=True)


def _load(path, expected):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or len(values) != expected or not np.isfinite(values).all():
        raise ValueError(f"예측 shape/유한값 오류: {path} {values.shape}")
    return values


def _rules(reports, complete_equal):
    base_mean = float(np.mean([reports[h]["e004"]["overall"] for h in HOLDS]))
    result = {}
    for candidate in CANDIDATES:
        mean = float(np.mean([reports[h][candidate]["overall"] for h in HOLDS]))
        missing_delta = [
            reports[h][candidate]["segments"]["missing_public"]["rmse"]
            - reports[h]["e004"]["segments"]["missing_public"]["rmse"]
            for h in HOLDS
        ]
        segment_delta = []
        violations = []
        for hold in HOLDS:
            base = reports[hold]["e004"]["segments"]
            other = reports[hold][candidate]["segments"]
            for segment, base_seg in base.items():
                if base_seg["n"] >= 200 and other[segment]["n"] >= 200:
                    delta = other[segment]["rmse"] - base_seg["rmse"]
                    segment_delta.append(delta)
                    if delta > 0.030:
                        violations.append((hold, segment, base_seg["n"], base_seg["rmse"],
                                           other[segment]["rmse"], delta))
                if segment == "missing_public" and missing_delta[list(HOLDS).index(hold)] > 0.005:
                    violations.append((hold, segment, base_seg["n"], base_seg["rmse"],
                                       other[segment]["rmse"], missing_delta[list(HOLDS).index(hold)]))
        r1 = base_mean - mean
        improved = sum(delta <= -0.010 for delta in missing_delta)
        max_missing_delta = max(missing_delta)
        r4 = reports["2025_julaug"]["e004"]["overall"] - reports["2025_julaug"][candidate]["overall"]
        r5 = max(segment_delta) if segment_delta else float("inf")
        result[candidate] = {
            "mean": mean, "r1": r1, "missing_delta": missing_delta,
            "improved": improved, "max_missing_delta": max_missing_delta,
            "complete_equal": complete_equal[candidate], "r4": r4, "r5": r5,
            "violations": violations,
            "passed": (r1 >= 0.010 and improved >= 3 and max_missing_delta <= 0.005
                       and complete_equal[candidate] and r4 >= 0.005 and r5 <= 0.030),
        }
    return base_mean, result


def _print_report(reports, rules, runtime):
    print("표1 전체: model | 4hold_mean | Δ_vs_E004 | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug")
    base_mean = float(np.mean([reports[h]["e004"]["overall"] for h in HOLDS]))
    base_values = [reports[h]["e004"]["overall"] for h in HOLDS]
    print("E004 | {:.6f} | +0.000000 | {}".format(
        base_mean, " | ".join(f"{value:.6f}" for value in base_values)))
    for candidate in CANDIDATES:
        values = [reports[h][candidate]["overall"] for h in HOLDS]
        print(f"{candidate} | {rules[candidate]['mean']:.6f} | {rules[candidate]['mean'] - base_mean:+.6f} | "
              + " | ".join(f"{value:.6f}" for value in values))

    print("표2 missing_public 세그먼트: model | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug")
    for model in ("E004", *CANDIDATES):
        values = [reports[h]["e004" if model == "E004" else model]["segments"]["missing_public"]["rmse"]
                  for h in HOLDS]
        print(f"{model} | " + " | ".join(f"{value:.6f}" for value in values))

    print("표3 규칙 체크: cand | R1 | R2 | R3 | R4 | R5")
    for candidate in CANDIDATES:
        rule = rules[candidate]
        print(f"{candidate} | {rule['r1']:+.6f}/{'pass' if rule['r1'] >= .010 else 'fail'} | "
              f"{rule['improved']}/4,maxΔ={rule['max_missing_delta']:+.6f}/"
              f"{'pass' if rule['improved'] >= 3 and rule['max_missing_delta'] <= .005 else 'fail'} | "
              f"diff={0 if rule['complete_equal'] else 1}/{'pass' if rule['complete_equal'] else 'fail'} | "
              f"{rule['r4']:+.6f}/{'pass' if rule['r4'] >= .005 else 'fail'} | "
              f"{rule['r5']:+.6f}/{'pass' if rule['r5'] <= .030 else 'fail'}")

    print("표4 위반 세그먼트: cand | hold | segment | n | E004 | cand | Δ")
    violations = [(candidate, *row) for candidate in CANDIDATES for row in rules[candidate]["violations"]]
    if not violations:
        print("없음")
    else:
        seen = set()
        for candidate, hold, segment, n, base, other, delta in violations:
            key = (candidate, hold, segment)
            if key in seen:
                continue
            seen.add(key)
            print(f"{candidate} | {hold} | {segment} | {n} | {base:.6f} | {other:.6f} | {delta:+.6f}")

    print("표5 runtime: hold | 초 | 완료여부")
    for hold in HOLDS:
        row = runtime.get("holds", {}).get(hold, {})
        print(f"{hold} | {row.get('seconds', float('nan')):.2f} | {row.get('completed', False)}")


def main():
    obs = C.load_obs()
    test_index = pd.read_csv(C.DATA / "test_index.csv")
    runtime_path = C.ROOT / "cache" / "E005" / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected_oof = pd.read_csv(C.ROOT / "cache" / "E003" / "oof_index.csv")
    baseline_oof = _load(C.ROOT / "preds" / "oof_gate_robust_blend_E004.npy", len(expected_oof))
    baseline_test = _load(C.ROOT / "preds" / "test_gate_robust_blend_E004.npy", len(test_index))
    candidates_oof = {candidate: _load(C.ROOT / "preds" / f"oof_{candidate}_E005.npy", len(expected_oof))
                     for candidate in CANDIDATES}
    candidates_test = {candidate: _load(C.ROOT / "preds" / f"test_{candidate}_E005.npy", len(test_index))
                       for candidate in CANDIDATES}

    hold_data = {}
    expected_parts = []
    offset = 0
    for hold in HOLDS:
        work, _, rows = holdout(obs, hold)
        x = C.build_features(work, rows)
        flags = regimes(x, rows)["missing_public"]
        hold_data[hold] = (rows, x, flags, offset)
        expected_parts.append(rows[C.KEY].assign(temp=rows.temp, hold=hold))
        offset += len(rows)
    expected = pd.concat(expected_parts, ignore_index=True)
    aligned = (_key_frame(expected).equals(_key_frame(expected_oof)) and
               expected["hold"].equals(expected_oof["hold"]))
    if not aligned:
        raise ValueError("E003 OOF index와 E005 holdout 순서 불일치")

    reports = {hold: {} for hold in HOLDS}
    complete_equal = {candidate: True for candidate in CANDIDATES}
    for hold in HOLDS:
        rows, x, flags, offset = hold_data[hold]
        base = baseline_oof[offset:offset + len(rows)]
        reports[hold]["e004"] = score(rows, x, base)
        for candidate in CANDIDATES:
            pred = np.where(flags, candidates_oof[candidate][offset:offset + len(rows)], base)
            reports[hold][candidate] = score(rows, x, pred)
            complete_equal[candidate] &= np.array_equal(pred[~flags], base[~flags])

    test_rows = C.test_rows(obs)
    baseline_test_x = C.build_features(obs, test_rows)
    test_flags = regimes(baseline_test_x, test_rows)["missing_public"]
    for candidate in CANDIDATES:
        complete_equal[candidate] &= np.array_equal(
            candidates_test[candidate][~test_flags], baseline_test[~test_flags]
        )
    base_mean, rules = _rules(reports, complete_equal)
    accepted = [candidate for candidate in CANDIDATES if rules[candidate]["passed"]]
    selected = min(accepted, key=lambda candidate: rules[candidate]["mean"]) if accepted else "E004"

    metrics = {
        "means": {"e004": base_mean, **{candidate: rules[candidate]["mean"] for candidate in CANDIDATES}},
        "reports": reports, "selected": selected, "accepted": bool(accepted),
    }
    cache = C.ROOT / "cache" / "E005"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(reports, rules, runtime)

    if accepted:
        pred = candidates_test[selected]
        if not np.isfinite(pred).all() or not ((pred >= -5) & (pred <= 45)).all():
            raise ValueError("채택 test 예측 범위/유한값 오류")
        keys = test_index[C.KEY]
        keys.assign(temp=pred).to_csv(C.ROOT / "subs" / "E005_gate.csv", index=False)
    print(f"판정: {'채택 ' + selected if accepted else '기각'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
