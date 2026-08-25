# T14 — 후처리 실험실: 원시 확률로 제출 후보 3개 (학습 없음. 2시간 내)

실행 파이썬: `python`. `n_jobs` ≤ 2. 상단 `sys.stdout.reconfigure(encoding="utf-8")`.

**절대 재학습하지 마라.** 원시 확률이 이미 저장돼 있다. 모델을 다시 적합하면 사양 위반이다.

## 배경 — 확정 사실 (재조사·재측정 금지)

**핵심 정정:** `P1/preds/oof_exp005.npy`·`test_exp005.npy` 는 **uint8 0/1 후처리 결과**다 (고유값 2개).
임계를 다시 훑을 수 없다. 원시 확률은 여기 있다:

| 파일 | dtype | 길이 | 비고 |
|---|---|---|---|
| `P1/cache/exp005_raw_oof.npy` | float64 | 776706 | 평가 대상이 아닌 행은 **NaN**. NaN 을 제외하고 채점한다 |
| `P1/cache/exp005_raw_test.npy` | float64 | 169011 | min 3.86e-09, p50 3.83e-06, max 0.99999990 |

exp005 확정 설정: threshold **0.62**, hole fill **48**, rule weight **0.00**. 이 설정이 exp005 제출본을 만들었다.

### 기준선 숫자 (확정)

| 경로 | F1 |
|---|---|
| forward inner (fold1·2) | **0.697836** |
| public LB (exp005) | **0.717435** |

exp005 제출: 169011행, 양성 5880 (**3.4791%**), train 양성률 **0.041362**,
sha256 `5E5B8840C3A6FC5E6C8A8DFD461E419E9795548492C566E27758EE5BD0185F5A`.
봉인 fold3 는 이미 3회 판독됐다. **이 작업에서 fold3 를 채점하지 마라.**

T10 진단 (확정): inner FN 6038 중 offset 2156(recall 0.4254) + drift 1689(0.5104) = **63.7%**.
offset 세그먼트 17개 중 **full_miss 8**, 구간내 적중률 p50 **0.0083**.
drift 는 구간내 10분위 recall 이 **0.2393 → 0.7361 단조 증가**. flatline 은 이미 해결(적중률 p50 0.9804).
**정밀도 여유가 있다:** 봉인 fold3 기준 precision 0.888 / recall 0.754, FP 704 대 FN 1831.

기각축(재제안 금지): 유형 multiclass blend, stream 별 양성률 앵커링, 랜덤 행 CV, adversarial drift 보정.
`ruptures`·`xgboost`·`catboost`·`torch` 없다.

## 산출물

새 파일 **2개만** 만든다. 기존 파일을 수정하지 마라.

1. `P1/src/postproc_lab.py`
2. 제출 CSV 3개: `P1/subs/exp006b.csv`(덮어씀), `P1/subs/exp008.csv`, `P1/subs/exp009.csv`

추가로 원시 확률을 앙상블 풀에 복사한다: `P1/preds/score_oof_exp005.npy`, `P1/preds/score_test_exp005.npy`
(`cache` 의 원본을 float32 로 복사. 원본은 지우지 마라).

## 후처리 연산 — 적용 순서를 고정한다

세그먼트는 `(station, year, layer)` stream 안의 연속 예측 양성 런이다. **stream 의 관측 gap 을 넘지 않는다.**

1. `base = score >= t_high`
2. **hysteresis 확장** — `t_low` (≤ `t_high`) 로 만든 마스크에서, `base` 와 **연결된** 런만 살린다
   (flood fill). offset 은 계단이 유지되고 drift 는 자라므로 구간 내부가 낮은 임계에서 켜진다
3. **hole fill h** — 길이 ≤ h 인 0-갭을 1 로
4. **dilation k** — 각 런을 양쪽 k 행 확장
5. **min-length L** — 길이 < L 인 1-런 삭제

## 할 일

### A. `exp006b` — 양성률 앵커 프로브 (정보 목적)

`t_high` 를 격자 `0.02:0.005:0.95` 에서 훑고, `t_low = t_high`(hysteresis 없음), hole 48, k=0, L=0 으로
최종 test 양성률이 **0.041362** 에 가장 가까워지는 `t_high` 를 고른다. → `P1/subs/exp006b.csv`.
보고: 선택 `t_high`, 후처리 전/후 양성률, exp005 대비 양성 행 수 차이, inner F1(참고용).

### B. `exp008` — 형태 연산 격자 (inner argmax)

inner(= `exp005_raw_oof.npy` 의 **NaN 이 아닌** 행) 에서 격자를 전부 채점한다:

| 축 | 값 |
|---|---|
| `t_high` | `0.30:0.02:0.90` |
| `t_low` | `t_high` 만 (이 단계는 hysteresis 없음) |
| hole `h` | `0, 12, 24, 48, 96` |
| dilation `k` | `0, 1, 2, 4, 8, 16` |
| min-length `L` | `0, 3, 6, 12` |

inner F1 argmax 를 고르고 그 설정을 test 에 적용 → `P1/subs/exp008.csv`.

### C. `exp009` — hysteresis (inner argmax)

B 와 같되 `t_low` 를 연다:

| 축 | 값 |
|---|---|
| `t_high` | `0.40:0.04:0.90` |
| `t_low` | `0.002, 0.01, 0.05, 0.15, 0.30` 중 `t_high` 이하만 |
| hole `h` | `0, 24, 48` |
| dilation `k` | `0, 2, 8` |
| min-length `L` | `0, 6` |

inner F1 argmax → `P1/subs/exp009.csv`.

### D. 진단

exp008·exp009 각각의 inner 선택 설정에 대해 **T10 과 같은 형식의 유형별 recall 표**
(anomaly_type × TP/FN/recall) 를 출력한다. offset·drift recall 이 exp005 대비 얼마나 움직였는지가 핵심 숫자다.

격자 전체 점수를 `P1/experiments/exp008_009_grid.json` 에 저장한다.

## 제약

- **재학습 금지. LightGBM 을 부르지 마라.** 저장된 확률만 쓴다
- **새 의존성 금지.** `pip install` 필요하면 실행하지 말고 멈추고 보고
- `P1/P1_data/` 읽기 전용. `P1/cache/exp005_raw_*.npy` 도 **읽기 전용**
- **동시에 다른 프로세스가 `P1/src/features.py`·`P1/src/exp007.py`·`P1/specs/`·루트 `RULES.md` 를 쓰고 있다.
  그 파일들을 수정하지 마라**
- 제출 CSV 열·키·정렬은 `P1/subs/exp005.csv` 와 동일해야 한다
- 새 클래스·설정파일·추상화 금지. `postproc_lab.py` 300줄 상한

## 수락 기준

다음을 **실행하고 숫자를 보고**한다:

```
python P1/src/check.py sub P1/subs/exp006b.csv
python P1/src/check.py sub P1/subs/exp008.csv
python P1/src/check.py sub P1/subs/exp009.csv
python P1/src/submission_tools.py audit P1/subs/exp008.csv
python P1/src/submission_tools.py audit P1/subs/exp009.csv
```

세 CSV 모두 `판정: OK`. exp008·exp009 의 sha256 이 exp005 의
`5E5B8840...` 와 **달라야 한다** (같으면 후처리가 아무것도 안 바꾼 것이므로 FAIL 로 보고).

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 보고할 것:
A/B/C 각각의 선택 파라미터와 inner F1, exp005 inner 0.697836 대비 Δ,
각 CSV 의 행수·양성률·sha256, D 의 유형별 recall 표 2개. 산문·권고 금지.
