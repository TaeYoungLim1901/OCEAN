"""공식 지표(통합 RMSE) + 세그먼트 분해.

**기본 인자를 두지 않는다.** 편한 기본값이 생산 경로와 갈라져 조용히 다른 것을 재는 사고가
BARAM 2026 에서 2회 있었다. `by` 는 필수 인자다.

  python metric.py --pred subs/exp001.csv --true cache/answer_local.csv --by lead station
"""
import numpy as np
import pandas as pd
import sys

KEY = ["case_id", "station", "lead_h"]
LEADS = [3, 6, 9, 12, 18, 24]


def rmse(pred, true):
    e = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sqrt(np.mean(e ** 2)))


def join(pred, true):
    """키로 엄격 결합. 누락·중복·초과는 즉시 예외."""
    for name, d in (("pred", pred), ("true", true)):
        if d.duplicated(KEY).any():
            raise ValueError(f"{name} 키 중복 {int(d.duplicated(KEY).sum())}건")
    m = true[KEY + ["hs"]].merge(pred[KEY + ["hs_pred"]], on=KEY, how="outer",
                                 indicator=True, validate="one_to_one")
    if not m["_merge"].eq("both").all():
        raise ValueError(f"키 불일치 {int(m['_merge'].ne('both').sum())}건")
    return m.drop(columns="_merge")


def evaluate(pred, true, by, hs0=None):
    """by: 분해 축 리스트. ['lead'], ['station'], ['hs0bin'] 조합. 빈 리스트 금지."""
    if not by:
        raise ValueError("by 를 명시하라. 전체 점수 하나만 보는 실험은 시작하지 마라 (PLAYBOOK §2.4)")
    m = join(pred, true)
    if hs0 is not None:
        m = m.merge(hs0, on=["case_id", "station"], how="left", validate="many_to_one")
        m["hs0bin"] = pd.cut(m["hs0"], [0, 1.75, 2.25, 3.0, 99],
                             labels=["1.5-1.75", "1.75-2.25", "2.25-3", "3+"])
    total = rmse(m.hs_pred, m.hs)
    out = {"total": total, "n": len(m)}
    for ax in by:
        col = {"lead": "lead_h"}.get(ax, ax)
        if col not in m:
            raise ValueError(f"분해 축 없음: {ax}")
        out[ax] = {str(k): rmse(g.hs_pred, g.hs) for k, g in m.groupby(col, observed=True)}
    return out


def report(res, baseline=None):
    line = f"RMSE {res['total']:.6f}  (n={res['n']:,})"
    if baseline is not None:
        line += f"   Δ {res['total'] - baseline:+.6f} vs 기준 {baseline:.6f}"
    print(line)
    for ax, d in res.items():
        if isinstance(d, dict):
            print(f"  {ax:8} " + "  ".join(f"{k}={v:.4f}" for k, v in d.items()))


def wide_to_long(cases, pred_wide=None):
    """cases(y3..y24 wide) → 채점용 long. pred_wide 는 같은 열이름의 예측."""
    out = []
    for l in LEADS:
        d = cases[["case_id", "station"]].copy()
        d["lead_h"] = l
        d["hs"] = cases[f"y{l}"].to_numpy()
        if pred_wide is not None:
            d["hs_pred"] = np.asarray(pred_wide[f"y{l}"], float)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _selftest():
    c = pd.DataFrame({"case_id": ["A", "B"], "station": ["G-ORS", "I-ORS"],
                      **{f"y{l}": [1.0 + l / 100, 2.0] for l in LEADS}})
    t = wide_to_long(c)
    p = t.rename(columns={"hs": "hs_pred"})
    assert evaluate(p, t, by=["lead"])["total"] == 0.0, "완벽예측이 0이어야 한다"
    p2 = p.copy(); p2["hs_pred"] += 0.5
    assert abs(evaluate(p2, t, by=["lead"])["total"] - 0.5) < 1e-12
    # 행 가법성: 부분집합 MSE 의 가중평균 = 전체 MSE
    m = evaluate(p2, t, by=["station"])
    w = np.mean([v ** 2 for v in m["station"].values()])
    assert abs(np.sqrt(w) - m["total"]) < 1e-12, "가법성 위반"
    try:
        evaluate(p, t, by=[])
    except ValueError:
        pass
    else:
        raise AssertionError("빈 by 를 허용하면 안 된다")
    try:
        evaluate(p.iloc[:5], t, by=["lead"])
    except ValueError:
        pass
    else:
        raise AssertionError("키 불일치를 잡아야 한다")
    print("metric selftest OK")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--pred", required=True)
        ap.add_argument("--true", required=True)
        ap.add_argument("--by", nargs="+", required=True)
        ap.add_argument("--baseline", type=float)
        a = ap.parse_args()
        report(evaluate(pd.read_csv(a.pred), pd.read_csv(a.true), a.by), a.baseline)
