"""사례 생성 — 학습 컨텍스트를 test_context 와 **같은 배열 형태**로 만든다.

생산 경로와 평가 경로가 갈라지면 조용히 다른 것을 잰다(BARAM 2026, 0.0127 오판 2회).
그래서 학습·추론 양쪽 모두 `(289, 10)` float32 컨텍스트 배열 하나만 피처 함수에 넘긴다.

  from cases import COLS, load_grid, build_cases, iter_train_contexts, iter_test_contexts
"""
import numpy as np
import pandas as pd
import pathlib

P3 = pathlib.Path(__file__).resolve().parent.parent
DATA = P3 / "P3_data"
CACHE = P3 / "cache"

COLS = ["hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph"]
WAVE = ["hs", "tp", "hmax", "wvdir"]
STATIONS = ["G-ORS", "I-ORS", "S-ORS"]
LEADS = [3, 6, 9, 12, 18, 24]
CTX_ROWS = 289          # step_minute -2880..0, 10분 간격
STEP = 10               # 분
MIN_HS = 1.5            # f.case_selection
SPACING_H = 78          # 동일 정점 t0 최소 간격 (평가용 분할에만 적용)


def load_grid(station):
    """10분 격자에 파랑(20분)·기상(10분)을 올린다. 반환 (times, arr[N,10])."""
    f = CACHE / f"grid_{station}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
    else:
        w = pd.read_csv(DATA / "train_wave.csv", parse_dates=["time"])
        a = pd.read_csv(DATA / "train_atmos.csv", parse_dates=["time"])
        w, a = w[w.station == station], a[a.station == station]
        t0, t1 = min(w.time.min(), a.time.min()), max(w.time.max(), a.time.max())
        grid = pd.date_range(t0, t1, freq=f"{STEP}min")
        d = pd.DataFrame(index=grid)
        for src, cols in ((w, WAVE), (a, [c for c in COLS if c not in WAVE])):
            s = src.set_index("time")[cols].reindex(grid)
            d[cols] = s
        d.index.name = "time"
        CACHE.mkdir(exist_ok=True)
        d.reset_index().to_parquet(f, index=False)
        d = d.reset_index()
    times = d["time"].to_numpy()
    return times, d[COLS].to_numpy(np.float32)


def build_cases(stride_min=60):
    """유효 t0 와 6리드 타깃을 모은다. 대회 사례 조건(f.case_selection)을 복제한다."""
    rows = []
    for st in STATIONS:
        times, arr = load_grid(st)
        hs = arr[:, 0]
        n = len(times)
        lead_off = [l * 60 // STEP for l in LEADS]
        start = CTX_ROWS - 1                       # 컨텍스트 48h 확보
        for i in range(start, n - max(lead_off), max(1, stride_min // STEP)):
            if not (hs[i] >= MIN_HS):              # NaN 도 여기서 탈락
                continue
            y = hs[[i + o for o in lead_off]]
            if np.isnan(y).any():
                continue
            rows.append((st, times[i], float(hs[i]), i, *y.tolist()))
    c = pd.DataFrame(rows, columns=["station", "t0", "hs0", "idx"] + [f"y{l}" for l in LEADS])
    c.insert(0, "case_id", [f"L{i:05d}" for i in range(len(c))])
    return c


def context_at(arr, i):
    """격자 배열에서 t0 위치 i 의 (289,10) 컨텍스트. test 와 동일한 형태."""
    return arr[i - CTX_ROWS + 1: i + 1]


def iter_train_contexts(cases, stations=None):
    for st, g in cases.groupby("station", sort=False):
        if stations and st not in stations:
            continue
        _, arr = load_grid(st)
        for cid, i in zip(g.case_id.to_numpy(), g.idx.to_numpy()):
            yield cid, st, context_at(arr, i)


def iter_test_contexts():
    t = pd.read_parquet(DATA / "test_context.parquet")
    t = t.sort_values(["case_id", "step_minute"])
    for cid, g in t.groupby("case_id", sort=False):
        assert len(g) == CTX_ROWS, f"{cid} 행 수 {len(g)}"
        yield cid, g.station.iat[0], g[COLS].to_numpy(np.float32)


def eval_split(cases, spacing_h=SPACING_H, seed=42):
    """대회와 같은 78h 간격으로 솎아낸 평가용 부분집합 마스크."""
    keep = np.zeros(len(cases), bool)
    for st, g in cases.groupby("station", sort=False):
        last = None
        for pos, t in zip(g.index.to_numpy(), g.t0.to_numpy()):
            if last is None or (t - last) / np.timedelta64(1, "h") >= spacing_h:
                keep[cases.index.get_loc(pos)] = True
                last = t
    return keep


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    c = build_cases(stride_min=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
    c.to_parquet(CACHE / "cases.parquet", index=False)
    k = eval_split(c)
    print(f"사례 {len(c):,}건  정점별 {c.station.value_counts().to_dict()}")
    print(f"t0 hs 평균 {c.hs0.mean():.3f}  78h 간격 부분집합 {k.sum():,}건")
    print("리드별 지속성 RMSE(78h 부분집합):")
    for l in LEADS:
        e = c.loc[k, f"y{l}"] - c.loc[k, "hs0"]
        print(f"  +{l:2d}h {np.sqrt((e ** 2).mean()):.4f}  평균편차 {e.mean():+.4f}")
    e = np.concatenate([(c.loc[k, f"y{l}"] - c.loc[k, "hs0"]).to_numpy() for l in LEADS])
    print(f"통합 {np.sqrt((e ** 2).mean()):.6f}  (공식 0.769455)")
