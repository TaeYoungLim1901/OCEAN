"""P3 사전 게이트. 토큰·시간 태우기 전에 돌린다. `판정: OK` 아니면 진행 금지.

  python src/check.py data       # 원본 무결성 · test 스키마 (최초 1회, 전처리 변경 시)
  python src/check.py pipeline   # 생산 경로 == baseline_persistence.csv 완전일치
  python src/check.py cases      # 학습 컨텍스트가 test 컨텍스트와 같은 형태인가
  python src/check.py cv <fold열>  # 폴드 누출 (컨텍스트 48h + 타깃 24h + 완충)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import cases as C
import metric as M

FAILS, WARNS = [], []


def bad(msg): FAILS.append(msg)
def warn(msg): WARNS.append(msg)


def verdict():
    for w in WARNS:
        print(f"  WARN {w}")
    for f in FAILS:
        print(f"  FAIL {f}")
    print("판정: " + ("OK" if not FAILS else "STOP"))
    return 1 if FAILS else 0


def c_data():
    w = pd.read_csv(C.DATA / "train_wave.csv", parse_dates=["time"])
    a = pd.read_csv(C.DATA / "train_atmos.csv", parse_dates=["time"])
    for name, d, dt in (("wave", w, 20), ("atmos", a, 10)):
        for st, g in d.groupby("station"):
            t = g.time.sort_values()
            gaps = t.diff().dt.total_seconds() / 60
            if t.duplicated().any():
                bad(f"{name}/{st} 시각 중복 {int(t.duplicated().sum())}")
            if (gaps.dropna() != dt).any():
                bad(f"{name}/{st} 격자 이탈 {int((gaps.dropna() != dt).sum())}행")
            print(f"  {name}/{st} {len(g):,}행 {t.min().date()}~{t.max().date()}")
    if a[a.station != "G-ORS"].time.min().year == 2025:
        print("  주의: I/S 기상은 2025-01 부터만 (f.atmos_gap_is)")

    t = pd.read_parquet(C.DATA / "test_context.parquet")
    if list(t.columns) != ["case_id", "station", "step_minute"] + C.COLS:
        bad(f"test 열 순서 {list(t.columns)}")
    if t.groupby("case_id").size().nunique() != 1 or t.groupby("case_id").size().iat[0] != C.CTX_ROWS:
        bad("test case 당 행 수가 289 가 아니다")
    st = t.step_minute.to_numpy()
    if set(np.unique(st)) != set(range(-2880, 1, 10)):
        bad("step_minute 격자 이상")
    # 파랑은 20분 격자에만 값이 있어야 한다
    off = t[(t.step_minute % 20 != 0) & t.hs.notna()]
    if len(off):
        bad(f"20분 격자 밖에 hs 값 {len(off)}행")
    on = t[t.step_minute % 20 == 0]
    print(f"  test 20분격자 hs 결측률 {on.hs.isna().mean():.4f}  기상 결측률 {t.wspd.isna().mean():.4f}")
    if t[t.step_minute == 0].hs.isna().any():
        bad("기준시각 hs 결측 (문제지상 불가)")
    if (t[t.step_minute == 0].hs < C.MIN_HS).any():
        bad("기준시각 hs < 1.5 (사례 조건 위반)")
    idx = pd.read_csv(C.DATA / "test_index.csv")
    if len(idx) != t.case_id.nunique() * len(C.LEADS):
        bad(f"인덱스 {len(idx)} != case {t.case_id.nunique()} × 6")
    return verdict()


def c_pipeline():
    """추론 경로로 만든 지속성 예측이 배포 기준값과 완전일치해야 한다."""
    rows = []
    for cid, st, ctx in C.iter_test_contexts():
        hs0 = ctx[-1, 0]
        if not np.isfinite(hs0):
            bad(f"{cid} 기준시각 hs 결측")
            continue
        rows += [(cid, st, l, round(float(hs0), 2)) for l in C.LEADS]
    mine = pd.DataFrame(rows, columns=M.KEY + ["hs_pred"])
    base = pd.read_csv(C.DATA / "baseline_persistence.csv")
    m = base.merge(mine, on=M.KEY, how="outer", suffixes=("_ref", "_mine"), indicator=True)
    if not m["_merge"].eq("both").all():
        bad(f"키 불일치 {int(m['_merge'].ne('both').sum())}건")
        return verdict()
    d = (m.hs_pred_ref - m.hs_pred_mine).abs()
    print(f"  1,200건 최대 절대차 {d.max():.10f}")
    if d.max() > 0:
        bad(f"지속성 재현 불일치 {int((d > 0).sum())}건 — 컨텍스트 정렬 또는 t0 정의가 다르다")
    return verdict()


def c_cases():
    cs = C.build_cases(stride_min=360)
    (cid, st, ctx) = next(C.iter_train_contexts(cs.head(1)))
    (_, _, tctx) = next(C.iter_test_contexts())
    if ctx.shape != tctx.shape:
        bad(f"학습 컨텍스트 {ctx.shape} != test {tctx.shape}")
    if ctx.dtype != tctx.dtype:
        bad(f"dtype {ctx.dtype} != {tctx.dtype}")
    # 파랑 격자 패리티: 두 경로 모두 짝수(20분) 위치에만 값
    for name, x in (("train", ctx), ("test", tctx)):
        odd = np.isfinite(x[::2, 0]).sum() if False else np.isfinite(x[1::2, 0]).sum()
        if odd:
            bad(f"{name} 컨텍스트의 10분 위치에 hs 값 {odd}개")
    if not np.isfinite(ctx[-1, 0]) or ctx[-1, 0] < C.MIN_HS:
        bad("학습 사례의 t0 hs 가 조건을 어긴다")
    # 사례 선정 분포가 공식과 같은가 (그리디 78h 시닝)
    full = C.build_cases(stride_min=60)
    k = C.eval_split(full)
    b = pd.read_csv(C.DATA / "baseline_persistence.csv").query("lead_h == 3").hs_pred
    q = [.1, .25, .5, .75]
    lo, of = full.loc[k, "hs0"].quantile(q).to_numpy(), b.quantile(q).to_numpy()
    print("  t0 hs 분위(로컬/공식) " + " ".join(f"{a:.2f}/{c:.2f}" for a, c in zip(lo, of)))
    if np.abs(lo - of).max() > 0.25:
        warn("사례 선정 분포가 공식과 어긋난다 — eval_split 규칙 재확인")
    print(f"  학습 사례 {len(full):,}건, 78h 평가셋 {k.sum():,}건")
    return verdict()


def c_cv(fold_csv):
    """폴드 배정에 시간 누출이 없는가. case_id,fold,t0,station 열 필요."""
    d = pd.read_csv(fold_csv, parse_dates=["t0"])
    for st, g in d.groupby("station"):
        g = g.sort_values("t0")
        dt = g.t0.diff().dt.total_seconds() / 3600
        cross = g.fold.ne(g.fold.shift())
        near = ((dt < C.SPACING_H) & cross).sum()
        if near:
            bad(f"{st}: 폴드 경계에서 {C.SPACING_H}h 미만 간격 {int(near)}건 — 컨텍스트/타깃 중첩")
    print(f"  폴드별 사례 수 {d.fold.value_counts().sort_index().to_dict()}")
    return verdict()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"data": c_data, "pipeline": c_pipeline, "cases": c_cases}.get(cmd)
    if fn:
        sys.exit(fn())
    if cmd == "cv":
        sys.exit(c_cv(sys.argv[2]))
    sys.exit(__doc__)
