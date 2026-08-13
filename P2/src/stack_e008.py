"""E008: saved-prediction simplex stacking with leave-one-hold-out scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import gate_e004 as G

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
HOLDS = tuple(G.HOLDS)
REGIMES = ("missing_public", "complete_public")
def _key_frame(rows):
    out = rows[C.KEY].copy()
    out["time"] = pd.to_datetime(out.time, utc=True).astype("int64")
    return out.reset_index(drop=True)


def _simplex_projection(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, len(u) + 1) > 0)[0]
    if not len(rho):
        return np.full(len(v), 1.0 / len(v))
    theta = cssv[rho[-1]] / (rho[-1] + 1.0)
    return np.maximum(v - theta, 0.0)


def _fit_simplex(x, y, fallback=None):
    if len(y) == 0:
        return np.full(x.shape[1], 1.0 / x.shape[1]) if fallback is None else fallback.copy()
    try:
        from scipy.optimize import nnls
        w, _ = nnls(x, y)
    except ImportError:
        w = np.full(x.shape[1], 1.0 / x.shape[1])
    w = _simplex_projection(w)
    gram = x.T @ x
    rhs = x.T @ y
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        return w
    step = 1.0 / lipschitz
    # ponytail: fixed projected-gradient cap; use a constrained optimizer only if convergence fails.
    for _ in range(5000):
        new = _simplex_projection(w - step * (gram @ w - rhs))
        if np.max(np.abs(new - w)) < 1e-11:
            return new
        w = new
    return w


def _load_pool(oof_n, test_n):
    oof_files = {p.stem[4:]: p for p in (ROOT / "preds").glob("oof_*.npy")}
    test_files = {p.stem[5:]: p for p in (ROOT / "preds").glob("test_*.npy")}
    pool, audit = {}, []
    for name in sorted(set(oof_files) | set(test_files)):
        if name in {"stack_global_E008", "stack_regime_E008"}:
            audit.append((name, "제외", "현재 실험 산출물"))
            continue
        reasons = []
        opath, tpath = oof_files.get(name), test_files.get(name)
        if opath is None:
            reasons.append("oof 파일 없음")
        if tpath is None:
            reasons.append("test 파일 없음")
        oa = ta = None
        if opath is not None:
            oa = np.load(opath, allow_pickle=False)
            if oa.ndim != 1 or len(oa) != oof_n:
                reasons.append(f"oof 길이={len(oa)} 기대={oof_n}")
            elif not np.isfinite(oa).all():
                reasons.append("oof 비유한값")
        if tpath is not None:
            ta = np.load(tpath, allow_pickle=False)
            if ta.ndim != 1 or len(ta) != test_n:
                reasons.append(f"test 길이={len(ta)} 기대={test_n}")
            elif not np.isfinite(ta).all():
                reasons.append("test 비유한값")
        if not reasons:
            pool[name] = (oa.astype(float, copy=False), ta.astype(float, copy=False))
            audit.append((name, "사용", ""))
        else:
            audit.append((name, "제외", "; ".join(reasons)))
    return pool, audit


def _weight_dict(names, weights):
    return {name: float(value) for name, value in zip(names, weights)}


def _top3(names, weights):
    order = np.argsort(-weights, kind="stable")[:3]
    return ", ".join(f"{names[i]}={weights[i]:.6f}" for i in order)


def _rules(reports, candidate):
    base_mean = float(np.mean([reports[h]["e004"]["overall"] for h in HOLDS]))
    cand_mean = float(np.mean([reports[h][candidate]["overall"] for h in HOLDS]))
    overall_deltas = [reports[h][candidate]["overall"] - reports[h]["e004"]["overall"] for h in HOLDS]
    missing_deltas = [
        reports[h][candidate]["segments"]["missing_public"]["rmse"]
        - reports[h]["e004"]["segments"]["missing_public"]["rmse"]
        for h in HOLDS
    ]
    segment_deltas = []
    for hold in HOLDS:
        base = reports[hold]["e004"]["segments"]
        cand = reports[hold][candidate]["segments"]
        for name, seg in base.items():
            if seg["n"] >= 200 and cand[name]["n"] >= 200:
                segment_deltas.append(cand[name]["rmse"] - seg["rmse"])
    values = {
        "r1": base_mean - cand_mean,
        "r2": max(overall_deltas),
        "r3": max(missing_deltas),
        "r4": reports["2025_julaug"]["e004"]["overall"] - reports["2025_julaug"][candidate]["overall"],
        "r5": max(segment_deltas) if segment_deltas else float("-inf"),
    }
    values["passed"] = (
        values["r1"] >= 0.010 and values["r2"] <= 0.0 and values["r3"] <= 0.005
        and values["r4"] >= 0.005 and values["r5"] <= 0.030
    )
    return values


def _violations(reports, candidate):
    rows = []
    for hold in HOLDS:
        base = reports[hold]["e004"]["segments"]
        cand = reports[hold][candidate]["segments"]
        for name, seg in base.items():
            delta = cand[name]["rmse"] - seg["rmse"]
            if (name == "missing_public" and delta > 0.005) or (
                seg["n"] >= 200 and cand[name]["n"] >= 200 and delta > 0.030
            ):
                rows.append((hold, name, seg["n"], seg["rmse"], cand[name]["rmse"], delta))
    return rows


def _context():
    obs = C.load_obs()
    oof_index = pd.read_csv(ROOT / "cache" / "E003" / "oof_index.csv")
    test_index = pd.read_csv(C.DATA / "test_index.csv")
    hold_data, expected, y_parts, flag_parts, base_parts = {}, [], [], {r: [] for r in REGIMES}, []
    for hold in HOLDS:
        work, _, rows = G.holdout(obs, hold)
        x = C.build_features(work, rows)
        hold_data[hold] = (rows, x)
        expected.append(rows[C.KEY].assign(temp=rows.temp, hold=hold))
        y_parts.append(rows.temp.to_numpy(float))
        base_parts.append(x.base_nom.to_numpy(float))
        for regime in REGIMES:
            flag_parts[regime].append(G.regimes(x, rows)[regime])
    expected = pd.concat(expected, ignore_index=True)
    if not _key_frame(expected).equals(_key_frame(oof_index)) or not expected["hold"].equals(oof_index["hold"]):
        raise ValueError("E003 OOF index와 재생성 holdout 순서 불일치")
    test_rows = C.test_rows(obs)
    if not _key_frame(test_rows).equals(_key_frame(test_index)):
        raise ValueError("test index와 재생성 행 순서 불일치")
    test_x = C.build_features(obs, test_rows)
    test_flags = G.regimes(test_x, test_rows)
    return (
        hold_data,
        oof_index["hold"].to_numpy(),
        np.concatenate(y_parts),
        {r: np.concatenate(flag_parts[r]) for r in REGIMES},
        np.concatenate(base_parts),
        test_rows,
        {r: test_flags[r] for r in REGIMES},
        test_x.base_nom.to_numpy(float),
    )


def _stack(names, x_oof, x_test, y, hold_labels, flags, test_flags):
    oof_global = np.empty(len(y), dtype=float)
    global_weights = {}
    for hold in HOLDS:
        train = hold_labels != hold
        valid = ~train
        w = _fit_simplex(x_oof[train], y[train])
        oof_global[valid] = x_oof[valid] @ w
        global_weights[hold] = {"n": int(train.sum()), "weights": _weight_dict(names, w)}
    final_global = _fit_simplex(x_oof, y)
    test_global = x_test @ final_global

    oof_regime = np.empty(len(y), dtype=float)
    loho_regime = {}
    for hold in HOLDS:
        train = hold_labels != hold
        valid = ~train
        loho_regime[hold] = {}
        fallback = _fit_simplex(x_oof[train], y[train])
        for regime in REGIMES:
            fit = train & flags[regime]
            w = _fit_simplex(x_oof[fit], y[fit], fallback)
            use = valid & flags[regime]
            oof_regime[use] = x_oof[use] @ w
            loho_regime[hold][regime] = {"n": int(fit.sum()), "weights": _weight_dict(names, w)}
    final_regime, test_regime = {}, np.empty(len(x_test), dtype=float)
    for regime in REGIMES:
        w = _fit_simplex(x_oof[flags[regime]], y[flags[regime]], final_global)
        final_regime[regime] = {"n": int(flags[regime].sum()), "weights": _weight_dict(names, w)}
        test_regime[test_flags[regime]] = x_test[test_flags[regime]] @ w
    weights = {
        "loho": {hold: {"global": global_weights[hold], "regime": loho_regime[hold]} for hold in HOLDS},
        "final": {
            "global": {"n": int(len(y)), "weights": _weight_dict(names, final_global)},
            "regime": final_regime,
        },
    }
    return {"stack_global": (oof_global, test_global), "stack_regime": (oof_regime, test_regime)}, weights


def _print_report(pool_audit, reports, rules, weights, names, baseline_label):
    print("표0 풀 구성: 성분 | 사용여부 | 제외사유")
    print(f"base_nom | 사용 | common.build_features 재구성")
    for name, used, reason in pool_audit:
        print(f"{name} | {used} | {reason}")
    print("표1 LOHO 전체: cand | 4hold_mean | Δ_vs_E004 | hold별 RMSE/Δ")
    base_vals = [reports[h]["e004"]["overall"] for h in HOLDS]
    base_mean = float(np.mean(base_vals))
    print(f"{baseline_label} | {base_mean:.6f} | +0.000000 | " + " | ".join(f"{v:.6f}/+0.000000" for v in base_vals))
    for candidate in ("stack_global", "stack_regime"):
        vals = [reports[h][candidate]["overall"] for h in HOLDS]
        print(f"{candidate} | {np.mean(vals):.6f} | {np.mean(vals) - base_mean:+.6f} | " +
              " | ".join(f"{v:.6f}/{v - b:+.6f}" for v, b in zip(vals, base_vals)))
    print("표2 LOHO missing_public: cand | hold | E004 | cand | Δ")
    for candidate in ("stack_global", "stack_regime"):
        for hold in HOLDS:
            b = reports[hold]["e004"]["segments"]["missing_public"]["rmse"]
            v = reports[hold][candidate]["segments"]["missing_public"]["rmse"]
            print(f"{candidate} | {hold} | {b:.6f} | {v:.6f} | {v - b:+.6f}")
    print("표3 가중치: cand | hold | 상위3성분=값")
    for candidate in ("stack_global", "stack_regime"):
        for hold in HOLDS:
            if candidate == "stack_global":
                row = weights["loho"][hold]["global"]
                print(f"{candidate} | {hold} | {_top3(names, np.array(list(row['weights'].values())))}")
            else:
                for regime in REGIMES:
                    row = weights["loho"][hold]["regime"][regime]
                    vals = np.array([row["weights"][name] for name in names])
                    print(f"{candidate} | {hold}/{regime} | {_top3(names, vals)}")
        if candidate == "stack_global":
            row = weights["final"]["global"]
            vals = np.array([row["weights"][name] for name in names])
            print(f"{candidate} | all | {_top3(names, vals)}")
        else:
            for regime in REGIMES:
                row = weights["final"]["regime"][regime]
                vals = np.array([row["weights"][name] for name in names])
                print(f"{candidate} | all/{regime} | {_top3(names, vals)}")
    print("표4 규칙 체크: cand | R1 | R2 | R3 | R4 | R5")
    for candidate in ("stack_global", "stack_regime"):
        r = rules[candidate]
        print(f"{candidate} | {r['r1']:+.6f}/{'pass' if r['r1'] >= .010 else 'fail'} | "
              f"{r['r2']:+.6f}/{'pass' if r['r2'] <= 0 else 'fail'} | "
              f"{r['r3']:+.6f}/{'pass' if r['r3'] <= .005 else 'fail'} | "
              f"{r['r4']:+.6f}/{'pass' if r['r4'] >= .005 else 'fail'} | "
              f"{r['r5']:+.6f}/{'pass' if r['r5'] <= .030 else 'fail'}")
    print("표5 위반 세그먼트: cand | hold | segment | n | E004 | cand | Δ")
    violations = [(candidate, *row) for candidate in ("stack_global", "stack_regime")
                  for row in _violations(reports, candidate)]
    if violations:
        for candidate, hold, name, n, base, cand, delta in violations:
            print(f"{candidate} | {hold} | {name} | {n} | {base:.6f} | {cand:.6f} | {delta:+.6f}")
    else:
        print("없음")


def main():
    hold_data, hold_labels, y, flags, base_oof, test_rows, test_flags, base_test = _context()
    oof_n, test_n = len(y), len(test_rows)
    pool, pool_audit = _load_pool(oof_n, test_n)
    pool["base_nom"] = (base_oof, base_test)
    names = sorted(pool)
    x_oof = np.column_stack([pool[name][0] for name in names])
    x_test = np.column_stack([pool[name][1] for name in names])
    baseline_oof = np.load(ROOT / "preds" / "oof_gate_robust_blend_E004.npy", allow_pickle=False)
    baseline_test = np.load(ROOT / "preds" / "test_gate_robust_blend_E004.npy", allow_pickle=False)
    if len(baseline_oof) != oof_n or len(baseline_test) != test_n:
        raise ValueError("E004 baseline 길이 불일치")
    candidates, weights = _stack(names, x_oof, x_test, y, hold_labels, flags, test_flags)
    np.save(ROOT / "preds" / "oof_stack_global_E008.npy", candidates["stack_global"][0])
    np.save(ROOT / "preds" / "test_stack_global_E008.npy", candidates["stack_global"][1])
    np.save(ROOT / "preds" / "oof_stack_regime_E008.npy", candidates["stack_regime"][0])
    np.save(ROOT / "preds" / "test_stack_regime_E008.npy", candidates["stack_regime"][1])

    reports = {hold: {"e004": None, "stack_global": None, "stack_regime": None} for hold in HOLDS}
    offset = 0
    for hold in HOLDS:
        rows, x = hold_data[hold]
        n = len(rows)
        reports[hold]["e004"] = G.score(rows, x, baseline_oof[offset:offset + n])
        for candidate in candidates:
            reports[hold][candidate] = G.score(rows, x, candidates[candidate][0][offset:offset + n])
        offset += n
    rules = {candidate: _rules(reports, candidate) for candidate in candidates}
    accepted = [candidate for candidate in candidates if rules[candidate]["passed"]]
    selected = min(accepted, key=lambda candidate: np.mean([reports[h][candidate]["overall"] for h in HOLDS])) if accepted else None
    baseline_vals = [reports[h]["e004"]["overall"] for h in HOLDS]
    baseline_expected = [1.285701, 0.495506, 1.303431, 1.271631]
    baseline_ok = all(abs(a - b) <= 5e-6 for a, b in zip(baseline_vals, baseline_expected))
    metrics = {
        "means": {"e004": float(np.mean(baseline_vals)), **{
            candidate: float(np.mean([reports[h][candidate]["overall"] for h in HOLDS]))
            for candidate in candidates
        }},
        "reports": reports,
        "rules": rules,
        "selected": selected,
        "accepted": bool(selected),
        "baseline_expected": baseline_expected,
        "baseline_match": baseline_ok,
    }
    cache = ROOT / "cache" / "E008"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (cache / "weights.json").write_text(json.dumps({"components": names, "weights": weights}, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_label = "E004[OK]" if baseline_ok else "E004[MISMATCH]"
    _print_report(pool_audit, reports, rules, weights, names, baseline_label)
    if selected is not None:
        pred = candidates[selected][1]
        if not np.isfinite(pred).all() or not ((pred >= -5) & (pred <= 45)).all():
            raise ValueError("채택 test 예측 범위/유한값 오류")
        keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
        sub = keys.assign(temp=pred)
        sub.to_csv(ROOT / "subs" / "E008_stack.csv", index=False)
    print(f"판정: {'채택 ' + selected if selected else '기각'}")


if __name__ == "__main__":
    main()
