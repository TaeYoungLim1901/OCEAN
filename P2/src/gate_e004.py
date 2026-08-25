"""E004: E003 예측의 행 단위 missing_public 게이트."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from submission_tools import save_pool_pred

SEED = 42
CANDIDATES = ("eof", "enhanced", "robust_blend")

try:
    from train_e003 import HOLDS, holdout, regimes, score
except ImportError:
    # ponytail: train_e003 imports optional training packages; gate-only fallback keeps the
    # existing definitions from train_e003.py:48-64,130-145,148-159 unchanged.
    HOLDS = {
        "2024_julaug": ("2024-07-02", "2024-08-31 23:50", True),
        "2024_sepoct": ("2024-09-01", "2024-10-31 23:50", False),
        "2025_mayjun": ("2025-05-02", "2025-07-01 23:50", True),
        "2025_julaug": ("2025-07-02", "2025-08-31 23:50", True),
    }

    def bounds(label):
        start, end, transfer = HOLDS[label]
        return pd.Timestamp(start, tz="Asia/Seoul"), pd.Timestamp(end, tz="Asia/Seoul"), transfer

    def pseudo_obs(obs, label):
        start, end, transfer = bounds(label)
        work = obs.copy()
        if transfer:
            source = work[work.time.between(pd.Timestamp("2025-09-01", tz="Asia/Seoul"),
                                            pd.Timestamp("2025-10-31 23:50", tz="Asia/Seoul"))]
            target = work[work.time.between(start, end)]
            for layer in C.PUBLIC_LAYERS:
                src = source[source.layer.eq(layer)].sort_values("time")
                dst = target[target.layer.eq(layer)].sort_values("time")
                if dst.empty:
                    continue
                if len(src) != 8784 or len(dst) != 8784:
                    raise ValueError(f"pseudo mask 길이 {label} layer={layer}: {len(src)}/{len(dst)}")
                for col in ("temp", "psal", "depth"):
                    work.loc[dst.index[src[col].isna().to_numpy()], col] = np.nan
        hidden = work.time.between(start, end) & work.layer.isin(C.TARGET_LAYERS)
        work.loc[hidden, ["temp", "psal"]] = np.nan
        return work

    def holdout(obs, label):
        start, end, _ = bounds(label)
        truth = obs[obs.time.between(start, end) & obs.layer.isin(C.TARGET_LAYERS)
                    & obs.temp.notna()].copy()
        work = pseudo_obs(obs, label)
        public_n = (work[work.layer.isin(C.PUBLIC_LAYERS) & work.temp.notna()]
                    .groupby("time").size())
        truth = truth[truth.time.map(public_n).fillna(0).ge(2)].reset_index(drop=True)
        train = obs[obs.layer.isin(C.TARGET_LAYERS) & obs.temp.notna()
                    & ~obs.time.between(start, end)].reset_index(drop=True)
        if len(truth) < 25000:
            raise ValueError(f"holdout truth 부족 {label}: {len(truth)}")
        return work, train, truth

    def regimes(x, rows):
        expected = [1, 5, 6, 7] if rows.time.dt.year.iloc[0] == 2024 else C.PUBLIC_LAYERS
        availability = x[[f"temp_{layer}" for layer in expected]].notna().sum(axis=1)
        strat = x.strat_1_6.abs()
        return {
            "missing_public": availability.lt(len(expected)).to_numpy(),
            "complete_public": availability.eq(len(expected)).to_numpy(),
            "strat_low": strat.lt(5).to_numpy(),
            "strat_mid": strat.between(5, 10, inclusive="left").to_numpy(),
            "strat_high": strat.ge(10).to_numpy(),
            "transition": x.transition.abs().ge(2.2).to_numpy(),
            "calm": x.transition.abs().lt(2.2).to_numpy(),
        }

    def score(rows, x, pred):
        result = {"overall": C.rmse(rows.temp, pred), "n": len(rows), "segments": {}}
        masks = {f"layer_{layer}": rows.layer.eq(layer).to_numpy() for layer in C.TARGET_LAYERS}
        masks.update(regimes(x, rows))
        for name, mask in masks.items():
            result["segments"][name] = {
                "n": int(mask.sum()),
                "rmse": C.rmse(rows.loc[mask, "temp"], pred[mask]) if mask.any() else None,
            }
        return result


def _key_frame(rows):
    out = rows[C.KEY].copy()
    out["time"] = out.time.astype(str)
    return out.reset_index(drop=True)


def _load_predictions(kind, names, expected):
    out = {}
    for name in names:
        path = C.ROOT / "preds" / f"{kind}_{name}_E003.npy"
        pred = np.load(path, allow_pickle=False)
        if pred.ndim != 1 or len(pred) != expected:
            raise ValueError(f"예측 shape 오류 {path}: {pred.shape}, expected={expected}")
        if not np.isfinite(pred).all():
            raise ValueError(f"예측 비유한값 {path}")
        out[name] = pred
    return out


def _rule_values(reports, cand):
    baseline = np.mean([reports[hold]["e002"]["overall"] for hold in HOLDS])
    candidate = np.mean([reports[hold][cand]["overall"] for hold in HOLDS])
    r1 = baseline - candidate
    r2 = reports["2025_julaug"]["e002"]["overall"] - reports["2025_julaug"][cand]["overall"]
    r3 = max(reports[hold][cand]["segments"]["missing_public"]["rmse"]
             - reports[hold]["e002"]["segments"]["missing_public"]["rmse"] for hold in HOLDS)
    segment_deltas = []
    for hold in HOLDS:
        base = reports[hold]["e002"]["segments"]
        other = reports[hold][cand]["segments"]
        for name, seg in base.items():
            if seg["n"] >= 200 and other[name]["n"] >= 200:
                segment_deltas.append(other[name]["rmse"] - seg["rmse"])
    r4 = max(segment_deltas) if segment_deltas else float("-inf")
    r5 = max(reports[hold][cand]["overall"] - reports[hold]["e002"]["overall"] for hold in HOLDS)
    passed = (r1 >= 0.010 and r2 >= 0.005 and r3 <= 0.002 and r4 <= 0.030 and r5 <= 0.0)
    return {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "passed": passed}


def _violations(reports, cand):
    rows = []
    for hold in HOLDS:
        base = reports[hold]["e002"]["segments"]
        other = reports[hold][cand]["segments"]
        for name, seg in base.items():
            delta = other[name]["rmse"] - seg["rmse"]
            if (name == "missing_public" and delta > 0.002) or \
                    (seg["n"] >= 200 and other[name]["n"] >= 200 and delta > 0.030):
                rows.append((hold, name, seg["n"], seg["rmse"], other[name]["rmse"], delta))
    return rows


def main():
    obs = C.load_obs()
    oof_index = pd.read_csv(C.ROOT / "cache" / "E003" / "oof_index.csv")
    test_index = pd.read_csv(C.DATA / "test_index.csv")

    hold_data = {}
    expected_index = []
    flag_parts = []
    for hold in HOLDS:
        work, _, rows = holdout(obs, hold)
        x = C.build_features(work, rows)
        flags = regimes(x, rows)["missing_public"]
        hold_data[hold] = (rows, x, flags)
        expected_index.append(rows[C.KEY].assign(temp=rows.temp, hold=hold))
        flag_parts.append(flags)
    expected_index = pd.concat(expected_index, ignore_index=True)
    if not _key_frame(expected_index).equals(_key_frame(oof_index)) or \
            not expected_index["hold"].equals(oof_index["hold"]):
        raise ValueError("E003 OOF index와 재생성 holdout 순서 불일치")

    oof = _load_predictions("oof", ("e002", *CANDIDATES), len(oof_index))
    test = _load_predictions("test", ("e002", *CANDIDATES), len(test_index))
    test_rows = C.test_rows(obs)
    test_x = C.build_features(obs, test_rows)
    test_flags = regimes(test_x, test_rows)["missing_public"]
    if len(test_flags) != len(test_index):
        raise ValueError("test flag 행 수 불일치")

    reports = {hold: {"e002": None} for hold in HOLDS}
    gated_oof, gated_test = {}, {}
    offset = 0
    for hold in HOLDS:
        rows, x, flags = hold_data[hold]
        n = len(rows)
        base = oof["e002"][offset:offset + n]
        reports[hold]["e002"] = score(rows, x, base)
        for cand in CANDIDATES:
            gated = np.where(flags, base, oof[cand][offset:offset + n])
            gated_oof.setdefault(cand, []).append(gated)
            reports[hold][cand] = score(rows, x, gated)
        offset += n

    for cand in CANDIDATES:
        gated_oof[cand] = np.concatenate(gated_oof[cand])
        gated_test[cand] = np.where(test_flags, test["e002"], test[cand])
        save_pool_pred(C.ROOT / "preds" / f"oof_gate_{cand}_E004.npy", gated_oof[cand])
        save_pool_pred(C.ROOT / "preds" / f"test_gate_{cand}_E004.npy", gated_test[cand])

    rules = {cand: _rule_values(reports, cand) for cand in CANDIDATES}
    selected = next((cand for cand in CANDIDATES if rules[cand]["passed"]), "E002")
    accepted = selected != "E002"
    means = {"e002": float(np.mean([reports[h]["e002"]["overall"] for h in HOLDS]))}
    means.update({cand: float(np.mean([reports[h][cand]["overall"] for h in HOLDS])) for cand in CANDIDATES})
    metrics = {"means": means, "reports": reports, "selected": selected, "accepted": accepted}
    cache = C.ROOT / "cache" / "E004"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {}
    for hold, flags in zip(HOLDS, flag_parts):
        stats[hold] = {"n": int(len(flags)), "missing_flag_true": int(flags.sum()),
                       "ratio": float(flags.mean())}
    stats["test"] = {"n": int(len(test_flags)), "missing_flag_true": int(test_flags.sum()),
                     "ratio": float(test_flags.mean())}
    (cache / "gate_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_ok = (abs(means["e002"] - 1.161517) <= 5e-6 and
                   abs(reports["2025_julaug"]["e002"]["overall"] - 1.301486) <= 5e-6 and
                   abs(reports["2024_sepoct"]["e002"]["overall"] - 0.517441) <= 5e-6)
    print("표1 후보별 전체: cand | 4hold_mean | Δ_vs_E002 | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug")
    baseline_label = "E002[OK]" if baseline_ok else "E002[MISMATCH]"
    vals = [reports[h]["e002"]["overall"] for h in HOLDS]
    print(f"{baseline_label} | {means['e002']:.6f} | +0.000000 | " + " | ".join(f"{v:.6f}" for v in vals))
    for cand in CANDIDATES:
        vals = [reports[h][cand]["overall"] for h in HOLDS]
        print(f"{cand} | {means[cand]:.6f} | {means[cand] - means['e002']:+.6f} | " +
              " | ".join(f"{v:.6f}" for v in vals))

    print("표2 규칙 체크: cand | R1 | R2 | R3 | R4 | R5")
    for cand in CANDIDATES:
        r = rules[cand]
        print(f"{cand} | {r['r1']:+.6f}/{'pass' if r['r1'] >= .010 else 'fail'} | "
              f"{r['r2']:+.6f}/{'pass' if r['r2'] >= .005 else 'fail'} | "
              f"{r['r3']:+.6f}/{'pass' if r['r3'] <= .002 else 'fail'} | "
              f"{r['r4']:+.6f}/{'pass' if r['r4'] <= .030 else 'fail'} | "
              f"{r['r5']:+.6f}/{'pass' if r['r5'] <= 0 else 'fail'}")

    print("표3 위반 세그먼트: cand | hold | segment | n | e002 | cand | Δ")
    violations = [(cand, *row) for cand in CANDIDATES for row in _violations(reports, cand)]
    if violations:
        for cand, hold, name, n, base, other, delta in violations:
            print(f"{cand} | {hold} | {name} | {n} | {base:.6f} | {other:.6f} | {delta:+.6f}")
    else:
        print("없음")

    print("표4 게이트 비율: hold/test | n | missing_flag_true | 비율")
    for hold, row in stats.items():
        print(f"{hold} | {row['n']} | {row['missing_flag_true']} | {row['ratio']:.6f}")

    if accepted:
        pred = gated_test[selected]
        if not np.isfinite(pred).all() or not ((pred >= -5) & (pred <= 45)).all():
            raise ValueError("채택 test 예측 범위/유한값 오류")
        keys = pd.read_csv(C.DATA / "test_index.csv")[C.KEY]
        keys.assign(temp=pred).to_csv(C.ROOT / "subs" / "E004_gate.csv", index=False)
    print(f"판정: {'채택 ' + selected if accepted else '기각'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
