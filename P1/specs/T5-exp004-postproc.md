# T5 — exp004: 사후처리 재탐색 (재학습 없음)

## 가설

행 단위 이진화가 긴 이상 이벤트를 조각내어 미탐을 만든다. 모델·피처·학습을 그대로 두고
`(weight, threshold, hole)` 만 inner 에서 다시 고르면 F1 이 오른다.

**변경점은 사후처리 1개다. 모델을 재학습하지 마라. 피처를 추가하지 마라.**

## 배경 — 이미 검증된 사실 (재조사 금지)

T4 진단(`P1/cache/T4/metrics.json`)에서 확정:

- sealed outer FN 2508 중 `recoverable_inside` 2410 (96.1%), `near_miss` 98, `hard_miss` **0**.
- FN 의 81% 가 noise(1167) + drift(855). 두 유형 모두 이벤트 길이 중앙값 225·261행.
- 모든 이벤트가 최소 1행은 탐지된다 (이벤트 hit rate: spike 0.96, 나머지 1.00).
- inner 에서 hole 을 2→6 으로 키우면 F1 `0.729777`→`0.736712`, 12 에서 `0.731586`, 36 에서 `0.726385`.
- rolling mean·rolling max·hysteresis·이벤트 확장(±3, ±12)은 inner 에서 전부 hole 단독보다 나빴다. **재시도 금지.**
- 위 T4 수치는 왜곡된 저장 배열(아래 「입력」 절 참조) 위에서 계산되었다. hole 계열 비교는 유효하지만
  **inner F1 절대값은 raw 점수에서 달라질 수 있다.** T4 수치를 하한으로 삼지 마라.
- `exp002.py:42` 의 원래 hole 격자는 `(0, 1, 2)` 뿐이었다. 그래서 hole 6 이 한 번도 평가되지 않았다.
- type-multiclass 축(exp003), adversarial drift 축, 랜덤 행 분할 축은 **닫혔다.**
- train/test drift source AUC `0.850133` 이미 측정됨. 다시 재지 마라.
- 새 의존성 설치 금지. `pip install` 이 필요해지면 **즉시 멈추고 그 사실만 보고한다.**
  현재 환경에 `xgboost`, `catboost`, `torch` 는 없다.

## 기준선 (baseline)

| 지표 | 값 |
|---|---|
| exp002 inner F1 (fold1·2) | `0.729777` |
| exp002 sealed outer F1 (fold3) | `0.766423` |
| outer TP / FP / FN | `4935` / `500` / `2508` |
| 현행 weight / threshold / hole | `0.15` / `0.44` / `2` |

## 입력 — ★ 저장된 npy 를 threshold 재탐색에 쓰지 마라

**`P1/preds/oof_exp002.npy` 와 `test_exp002.npy` 는 원 확률이 아니다.**
`P1/src/exp002.py:48 store_score()` 가 저장 직전에 값을 왜곡한다:

```python
z[pred]  = np.maximum(z[pred], 0.5)
z[~pred] = np.minimum(z[~pred], np.nextafter(0.5, 0))
```

즉 threshold `0.44` 와 hole `2` 가 이미 구워져 있다. 원 점수 `[0.44, 0.5)` 구간이 전부
정확히 `0.5` 로 뭉개졌고 hole 로 메운 행도 `0.5` 로 올라갔다.
**이 배열에 threshold 격자를 돌리면 inner 선택이 무효다.**

대신 **blend 전 raw 점수를 재구성한다:**

- `P1/cache/exp001/` 의 저장 모델과 `P1/src/exp002.py` 의 `model_scores()`·`reproduce()` 경로를 쓴다.
- **재학습 금지.** 저장된 booster 만 쓴다. fold3 는 exp002 가 별도 저장한 봉인 모델을 그대로 쓴다.
- `rule_score` 는 `features.py` 생산 경로에서 나온 열을 그대로 쓴다.
- 재구성한 raw 점수는 `P1/cache/exp004/raw_oof.npy`, `P1/cache/exp004/raw_test.npy` 에 저장한다.
- 재구성 검증: raw 점수에 `weight=0.15, threshold=0.44, hole=2` 를 적용한 결과가
  `exp002.py:reproduce()` 가 통과시키는 것과 동일한 outer TP/FP/FN 을 내야 한다.
  **일치하지 않으면 표를 내지 말고 불일치 숫자만 보고하고 멈춘다.**

그 외 입력:

- `P1/src/exp002.py`, `P1/src/train.py` — fold 정의, `segment_ids`, `fill_short_gaps`(시간 gap 안전), blend
- `P1/cache/exp002_meta.json`, `P1/cache/T4/metrics.json`
- `P1/P1_data/train.csv`, `P1/P1_data/test.csv` (읽기 전용)

raw 점수를 재구성하면 `weight` 도 재탐색 가능하다. 저장 모델로 재구성이 불가능하다고 판단되면
**추정하지 말고 멈추고 그 사실만 보고한다.**

## 절차

### 1단계 — inner 탐색 (fold1·2 만 사용)

격자:

- `threshold` : `0.20` 부터 `0.70` 까지 `0.02` 간격
- `hole` : `1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24`
- `weight` : 재탐색 가능하면 `0.00, 0.10, 0.15, 0.20, 0.30, 0.40`, 아니면 `0.15` 고정

hole 메움은 `station × layer` 안에서만, 시간 gap 을 넘지 않게 적용한다 (exp002 와 동일 함수).

inner F1 argmax 로 조합 1개를 고른다. **동점이면 hole 이 작은 쪽, 그다음 threshold 가 큰 쪽을 고른다**
(보수적 선택). 고른 조합을 `P1/cache/exp004_meta.json` 에 먼저 기록한 뒤 2단계로 간다.

### 2단계 — 봉인 outer 1회 판독

1단계에서 고른 조합 **1개만** fold3 에 적용해 F1·TP·FP·FN 을 낸다.
**outer 를 보고 파라미터를 바꾸지 마라. outer 격자 탐색 금지.** 1개 조합, 1회 판독이다.

### 3단계 — 산출물

- `P1/preds/oof_exp004.npy` — 사후처리 후 train 전 구간 이진 라벨 (0/1, `uint8`)
- `P1/preds/test_exp004.npy` — 사후처리 후 test 이진 라벨 (0/1, `uint8`)
- `P1/subs/exp004.csv` — `P1/subs/exp002.csv` 와 **동일한 열 순서·행 순서·`anomaly_type` 처리 방식**.
  `label` 만 새 사후처리 결과로 바꾼다.
- `P1/cache/exp004_meta.json` — 선택 조합, inner F1, outer F1/TP/FP/FN, exp002 대비 Δ,
  outer 를 유형별(`anomaly_type`)·station 별로 분해한 F1, `submission_sha256`

## 판정 규칙 (결과 보기 전 확정, 변경 금지)

- sealed outer F1 `>= 0.771423` (exp002 `+0.005`) 이면 exp004 를 제출 후보로 승격.
- 그 미만이면 exp002 를 유지하고 exp004 는 앙상블 풀에만 남긴다.
- 어느 쪽이든 예측 파일 3개는 반드시 저장한다.

## 수락 기준

1. `weight=0.15, threshold=0.44, hole=2` 로 현행을 재현하면 outer TP `4935` / FP `500` / FN `2508`,
   F1 `0.766423` 이 **정확히** 나온다. (이것만 일치 요구다.)
2. inner 최적 조합의 inner F1 이 raw 점수 기준 `weight=0.15, threshold=0.44, hole=2` 의 inner F1 **이상**이다
   (같은 raw 배열에서 계산한 상대 하한. T4 의 `0.736712` 를 절대 하한으로 쓰지 마라).
3. 2단계 outer 판독이 **정확히 1개 조합**에 대해서만 수행되었다.
4. `python P1/src/check.py sub P1/subs/exp004.csv` 가 `판정: OK`.
5. `python P1/src/submission_tools.py audit P1/subs/exp004.csv --pred P1/preds/test_exp004.npy` 가 `판정: OK`.
6. `P1/subs/exp004.csv` 의 행 수·`station,year,layer,time` 값이 `P1/P1_data/sample_submission.csv` 와 완전히 같다.
7. `P1/P1_data/`, `P1/preds/oof_exp002.npy`, `P1/preds/test_exp002.npy` 는 변경되지 않았다.

## 제약

- `P1/P1_data/` 읽기 전용. 시간 gap 을 넘는 rolling·diff·이벤트 연결 금지.
- 랜덤 행 분할 금지. fold 경계 purge 87시간 유지.
- `SEED = 42`, `n_jobs <= 2`, 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 새 코드는 `P1/src/exp004.py` 한 파일. 기존 `src/*.py` 는 수정하지 마라
  (hole 메움 함수를 `exp002.py` 에서 import 하는 것은 허용).

## 보고

**결론 문장 금지. 숫자와 표만.** 아래만 낸다.

1. `oof_exp002.npy` 가 blend 이후인지 여부 1줄, weight 탐색 가능 여부 1줄
2. 현행 재현 검증 1줄 (TP/FP/FN/F1)
3. inner 상위 10개 조합 표: `weight | threshold | hole | inner F1`
4. 선택 조합 1줄
5. outer 결과 1줄: `TP | FP | FN | F1 | Δ`
6. outer `anomaly_type` 별 표: `type | 양성 행 | TP | FN | 행 recall | F1`
7. outer station 별 표: `station | F1`
8. test 예측 양성 비율 vs train 양성 비율 1줄
9. 게이트 2개(`check.py sub`, `submission_tools.py audit`) 출력의 `판정:` 줄
10. `submission_sha256`
