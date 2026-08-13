"""토큰·제출 사고 방지 게이트. 문제 무관.

  python scripts/preflight.py spec P3/specs/T1-harness.md
  python scripts/preflight.py sub P3/subs/exp001.csv --index P3/P3_data/test_index.csv \
      --base P3/P3_data/baseline_persistence.csv
  python scripts/preflight.py selftest

`판정: OK` 가 아니면 codex 를 부르지도, 제출하지도 마라.
"""
import argparse, pathlib, re, sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parent.parent

# 사양 린트 — BARAM 2026 에서 실제로 codex 실행 1회(≈50k 토큰)씩 날린 항목만 남겼다
SPEC_RULES = [
    ("실행 파이썬을 명시했는가", r"python"),
    ("pip install 시 정지 지시가 있는가", r"pip install"),
    ("배경(재조사 금지 사실)을 줬는가", r"배경|이미 검증|재조사"),
    ("기준선 숫자를 줬는가", r"기준선|baseline"),
    ("수락 기준이 있는가", r"수락 기준"),
    ("보고할 숫자를 항목별로 나열했는가", r"보고|완료 보고"),
    ("결론 문장 금지를 넣었는가", r"결론 문장 금지|숫자와 표만"),
]
SPEC_WARN = [
    ("수락 기준이 참고치 '일치' 요구로 보인다 — 절대 하한으로 바꿔라", r"참고치.*일치|일치.*참고치"),
    ("가설이 여러 개로 보인다 — 실험당 변경 1개", r"가설\s*[2-9]|가설.*그리고.*가설"),
]
LEADS = [3, 6, 9, 12, 18, 24]


def rel(p):
    p = pathlib.Path(p)
    return p if p.is_absolute() else ROOT / p


def verdict(fails, warns):
    for w in warns:
        print(f"  WARN {w}")
    for f in fails:
        print(f"  FAIL {f}")
    print("판정: " + ("OK" if not fails else "STOP"))
    return 1 if fails else 0


def cmd_spec(a):
    f = rel(a.path)
    if not f.exists():
        sys.exit(f"사양 없음: {f}")
    s = f.read_text(encoding="utf-8")
    fails = [name for name, pat in SPEC_RULES if not re.search(pat, s, re.I)]
    warns = [name for name, pat in SPEC_WARN if re.search(pat, s, re.I)]
    # 사양이 쓰라고 지시한 디렉터리가 실재하는가 (없어서 실행이 2회 죽었다)
    for d in sorted(set(re.findall(r"[\w./-]*(?:preds|cache|subs|experiments|specs)/", s))):
        if not (rel(d.lstrip("./")).exists() or (ROOT / d).exists()):
            fails.append(f"출력 디렉터리 부재: {d}")
    print(f"사양 {f.name} — {len(s)}자")
    return verdict(fails, warns)


def cmd_sub(a):
    import pandas as pd
    import numpy as np
    sub = pd.read_csv(rel(a.path))
    idx = pd.read_csv(rel(a.index))
    K = ["case_id", "station", "lead_h"]
    fails, warns = [], []
    if list(sub.columns) != K + ["hs_pred"]:
        fails.append(f"열 이름/순서: {list(sub.columns)}")
        return verdict(fails, warns)
    if len(sub) != len(idx):
        fails.append(f"행 수 {len(sub):,} != 인덱스 {len(idx):,}")
    if sub.duplicated(K).any():
        fails.append(f"키 중복 {int(sub.duplicated(K).sum())}건")
    miss = idx.merge(sub, on=K, how="left", indicator=True)
    if not miss["_merge"].eq("both").all():
        fails.append(f"인덱스 키 누락 {int(miss['_merge'].ne('both').sum())}건")
    extra = sub.merge(idx, on=K, how="left", indicator=True)
    if not extra["_merge"].eq("both").all():
        fails.append(f"인덱스에 없는 키 {int(extra['_merge'].ne('both').sum())}건")
    if not sub["lead_h"].isin(LEADS).all():
        fails.append("lead_h 값 오류")
    p = pd.to_numeric(sub["hs_pred"], errors="coerce").to_numpy()
    if not np.isfinite(p).all():
        fails.append(f"비유한/결측 {int((~np.isfinite(p)).sum())}건")
    elif not ((p >= 0.0) & (p <= 30.0)).all():
        fails.append(f"범위 이탈 {int(((p < 0) | (p > 30)).sum())}건")
    elif (p < 0.05).sum():
        warns.append(f"0.05m 미만 {int((p < 0.05).sum())}건 — 사례 조건상 이례적")
    print(f"제출 {rel(a.path).name} — {len(sub):,}행, hs_pred 평균 {np.nanmean(p):.4f} 범위 [{np.nanmin(p):.3f}, {np.nanmax(p):.3f}]")
    if a.base:
        b = pd.read_csv(rel(a.base)).rename(columns={"hs_pred": "base"})
        m = sub.merge(b, on=K, how="inner")
        d = m["hs_pred"] - m["base"]
        print(f"  기준 대비 dist=mean|Δ| {d.abs().mean():.4f}  평균Δ {d.mean():+.4f}")
        print("  리드별 평균Δ " + " ".join(
            f"+{int(l)}h {g:+.3f}" for l, g in d.groupby(m["lead_h"]).mean().items()))
        if d.abs().max() < 1e-12:
            warns.append("기준과 완전히 동일하다 — 제출 슬롯 낭비")
    return verdict(fails, warns)


def cmd_selftest(a):
    import pandas as pd, tempfile, os
    d = pathlib.Path(tempfile.mkdtemp())
    idx = pd.DataFrame({"case_id": ["C1"] * 6, "station": ["G-ORS"] * 6, "lead_h": LEADS})
    idx.to_csv(d / "idx.csv", index=False)
    good = idx.assign(hs_pred=1.5)
    good.to_csv(d / "good.csv", index=False)
    assert cmd_sub(argparse.Namespace(path=d / "good.csv", index=d / "idx.csv", base=None)) == 0
    for name, bad in [
        ("범위", good.assign(hs_pred=[1.5, 1.5, 1.5, 1.5, 1.5, 31.0])),
        ("결측", good.assign(hs_pred=[1.5, 1.5, 1.5, 1.5, 1.5, float("nan")])),
        ("키누락", good.iloc[:5]),
        ("열순서", good[["station", "case_id", "lead_h", "hs_pred"]]),
    ]:
        bad.to_csv(d / "bad.csv", index=False)
        assert cmd_sub(argparse.Namespace(path=d / "bad.csv", index=d / "idx.csv", base=None)) == 1, name
    (d / "s.md").write_text("가설 하나", encoding="utf-8")
    assert cmd_spec(argparse.Namespace(path=d / "s.md")) == 1
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec"); s.add_argument("path"); s.set_defaults(f=cmd_spec)
    s = sub.add_parser("sub"); s.add_argument("path"); s.add_argument("--index", required=True)
    s.add_argument("--base"); s.set_defaults(f=cmd_sub)
    s = sub.add_parser("selftest"); s.set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
