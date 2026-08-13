# T8 — exp005: 점수 경로 통일 + 격자 확장 + stream 별 양성률 앵커링

## 가설

두 결함이 있다.

**B1.** `P1/src/exp004.py:33,37-39` 이 inner fold0·1·2 점수를 `P1/cache/exp001/model_fold{k}.txt` 로 낸다.
그 모델들은 `P1/src/features.py:121 train_mask_with_purge()` 의 **대칭** 마스크로 학습됐다.
`held_out=1` 이면 학습 집합에 fold2·fold3 가 들어간다. 즉 inner 격자 선택이 미래를 본 점수 위에서 이뤄졌다.

**B2.** `exp004.py:180` 이 `model_scores(xte)` 를 `folds=None` 으로 불러 booster **4개 평균**을 낸다
(`exp004.py:34-35`). 반면 튜닝은 fold 당 **단일 모델** 점수 위에서 했다. 평균이 확률을 중앙으로 압축한다.

점수 경로 3개(inner fold1·2 / 봉인 fold3 / test)를 **전부 「과거만 학습」 단일 모델**로 통일하면
threshold 가 같은 분포 위에서 동작하고 test 과소예측이 사라진다.

**변경점은 점수 경로 통일 1개 + 그 위에서의 사후처리 재탐색이다. 피처를 추가하지 마라.**

## 배경 — 이미 검증된 사실 (재조사 금지)

- exp004 = raw blend `weight 0.40` / `threshold 0.54` / `hole 24`. 봉인 outer F1 `0.808765`,
  TP `5352` / FP `440` / FN `2091`. inner F1 `0.762702`. 제출 양성률 `0.032039`.
- exp002 = `weight 0.15` / `threshold 0.44` / `hole 2`. 봉인 outer F1 `0.766423`.
- T7 감사(`P1/cache/T7_audit.json`) 결과: 라벨 누수 0, 피처 42개, 키 정렬 불일치 0행,
  hole 메움의 stream·시간 gap 위반 0행, `raw_*.npy` NaN 0 · 정확히 0.5 인 값 0,
  `label` 고유값 `{0,1}`, 원본 파일 해시 전후 동일. **이 항목들은 다시 검사하지 마라.**
- T7 의 C3 `FAIL` 은 사양 오류였다. purge 는 fold 배정 간격이 아니라 학습 마스크에서 걸린다.
  fold block 이 10분 간격으로 인접한 것은 정상이다. **fold 정의를 바꾸지 마라.**
- `raw_oof` min/max `0.000000005` / `0.999998897`, `raw_test` min/max `0.001249847` / `0.974596477`.
- train 양성률: 2024 `0.04107`, 2025 `0.04153`. stream-year 23개에서 min `0.0399` max `0.0546`
  std `0.00296`. **재측정 금지.** `I-ORS × layer 3` 은 train 에만 있고 test 에 없다.
- exp003 type-multiclass 축, 랜덤 행 분할 축, adversarial drift 축은 **닫혔다.**
- train/test drift source AUC `0.850133` 이미 측정됨. 다시 재지 마라.
- T4 에서 rolling mean·rolling max·hysteresis·이벤트 확장(±3, ±12)은 inner 에서 전부
  hole 단독보다 나빴다. **재시도 금지.**
- 새 의존성 설치 금지. `pip install` 이 필요해지면 **즉시 멈추고 그 사실만 보고한다.**
  현재 환경에 `xgboost`, `catboost`, `torch` 는 없다.

## 기준선 (baseline)

| 지표 | 값 |
|---|---|
| exp004 봉인 outer F1 | `0.808765` |
| exp004 outer TP / FP / FN | `5352` / `440` / `2091` |
| exp004 inner F1 (오염된 배열 기준) | `0.762702` |
| exp004 제출 양성률 | `0.032039` |
| train 전체 양성률 | `0.041362` |

## 절차

### 1단계 — 정직한 점수 배열 3개 생성

`P1/src/train.py` 의 학습 코드와 `P1/src/features.py` 의 피처 생성을 그대로 쓰되,
학습 집합만 **엄격히 과거만**으로 바꾼다.

| 배열 | 학습 집합 | 예측 대상 |
|---|---|---|
| `forward_fold1` | fold0, fold1 시작 87시간 전까지 | fold1 행 |
| `forward_fold2` | fold0·1, fold2 시작 87시간 전까지 | fold2 행 |
| `sealed_fold3` | fold0·1·2, fold3 시작 87시간 전까지 | fold3 행 |
| `full_test` | train 전체(fold0~3) | test 전체 |

- purge 는 **한쪽만** 건다 (검증 구간 시작 이전 87시간). 미래 쪽 purge 는 필요 없다.
- climatology 는 각 모델의 **학습 집합만으로** 계산한다. 검증·test 행을 쓰지 마라.
- **early stopping 금지.** `n_estimators` 는 `P1/cache/exp001/reference.json` 또는 `train.py` 에
  고정된 값을 그대로 쓰고, 4개 모델에 **같은 값**을 쓴다. 값을 보고에 1줄로 적는다.
- `sealed_fold3` 는 기존 `P1/models/exp002_outer_honest.txt` 가 이 정의와 일치하면
  **재학습하지 말고 그대로 재사용한다.** 일치 여부를 학습 집합 행 수로 대조해 1줄로 적는다.
- 결과를 `P1/cache/exp005_raw_oof.npy` (train 전 구간, fold0 구간은 `NaN`),
  `P1/cache/exp005_raw_test.npy` 에 저장한다.

학습이 총 20분을 넘으면 **중단하고, 스크립트를 저장한 뒤 그 사실과 경과 시간만 보고한다.**
`n_jobs <= 2` 를 지켜라.

**분포 정합 확인 (필수).** `sealed_fold3` 점수와 `full_test` 점수의 분위수
(1·5·25·50·75·95·99%)를 표로 낸다. exp004 의 `P1/cache/exp004/raw_test.npy` 분위수도 같이 낸다.

### 2단계 — inner 격자 (fold1·2 만 사용)

blend: `score = (1 - weight) * lgbm + weight * rule_score`.
그 뒤 threshold, 그 뒤 hole 메움 (`station × layer` 안에서만, 시간 gap 을 넘지 않게).

**2-A. 전역 threshold 탐색**

- `threshold` : `0.20` ~ `0.80`, `0.02` 간격
- `hole` : `1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48`
- `weight` : `0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60`

격자가 크다. hole 메움은 이진화 결과의 0 구간 길이를 **한 번** 계산해 재사용하면
hole 값 전체를 한 번의 통과로 처리할 수 있다. 이 최적화를 쓰되 결과는 `fill_short_gaps` 와
행 단위로 동일해야 한다 (임의 1개 조합에서 대조해 불일치 행 수를 보고한다).

2-A 의 inner F1 argmax 조합을 `(w*, t*, h*)` 로 고정한다.

**2-B. stream 별 양성률 앵커링**

`weight = w*`, `hole = h*` 로 고정하고 다음 2차원 격자를 inner 에서 탐색한다.

- `target_rate r` : `0.030, 0.034, 0.038, 0.041, 0.045, 0.050, 0.055`
- `shrinkage λ` : `0.00, 0.25, 0.50, 0.75, 1.00`

각 `station × layer` stream `s` 에 대해 그 stream 안 blend 점수의 `(1 - r)` 분위수를 `q_s` 라 할 때
적용 threshold 는 `t_s = (1 - λ) * t* + λ * q_s` 다. `λ = 0` 이면 2-A 와 같다.
stream 별 행 수가 500 미만이면 앵커링하지 말고 `t*` 를 쓴다 (그 stream 수를 보고한다).
분위수는 **각 구간(inner / outer / test) 안에서 그 구간의 점수만으로** 계산한다.

2-B 의 inner F1 argmax 를 최종 조합으로 고정한다.
**동점이면 λ 가 작은 쪽, 그다음 hole 이 작은 쪽, 그다음 threshold 가 큰 쪽을 고른다.**
고른 조합을 `P1/cache/exp005_meta.json` 에 먼저 기록한 뒤 3단계로 간다.

### 3단계 — 봉인 outer 1회 판독

2단계에서 고른 조합 **1개만** fold3 에 적용해 F1·TP·FP·FN 을 낸다.
**outer 를 보고 파라미터를 바꾸지 마라. outer 격자 탐색 금지.**

별도로, `sealed_fold3` 배열 위에서 exp004 설정 `weight 0.40 / threshold 0.54 / hole 24 / λ=0` 이
`0.808765` 를 재현하는지 확인한다 (재현 검증이며 파라미터 선택에 쓰지 않는다).

### 4단계 — 산출물

- `P1/preds/oof_exp005.npy` — 사후처리 후 train 전 구간 이진 라벨 (`uint8`, fold0 구간은 0)
- `P1/preds/test_exp005.npy` — 사후처리 후 test 이진 라벨 (`uint8`)
- `P1/subs/exp005.csv` — `P1/subs/exp004.csv` 와 동일한 열 순서·행 순서·`anomaly_type` 처리
- `P1/cache/exp005_meta.json` — 선택 조합, inner F1, outer F1/TP/FP/FN, exp004 대비 Δ,
  outer 를 `anomaly_type` 별·station 별로 분해한 F1, test 예측 양성률(전체·stream 별), `submission_sha256`

## 판정 규칙 (결과 보기 전 확정, 변경 금지)

- sealed outer F1 `>= 0.813765` (exp004 `+0.005`) 이면 exp005 를 제출 후보로 승격.
- 그 미만이면 exp004 를 유지하고 exp005 는 앙상블 풀에만 남긴다.
- 어느 쪽이든 예측 파일 3개는 반드시 저장한다.

## 수락 기준

1. 1단계 4개 모델의 학습 집합에 **자기 예측 구간 이후 행이 0개**임을 행 수로 증명한다.
2. `sealed_fold3` 위에서 exp004 설정이 outer F1 `0.808765`, TP `5352` / FP `440` / FN `2091` 을
   **정확히** 재현한다. 불일치하면 표를 내지 말고 불일치 숫자만 보고하고 멈춘다.
3. hole 최적화 구현이 `fill_short_gaps` 와 행 단위로 동일하다 (불일치 행 수 `0`).
4. 3단계 outer 판독이 **정확히 1개 조합**에 대해서만 수행되었다.
5. `python P1/src/check.py sub P1/subs/exp005.csv` 가 `판정: OK`.
6. `python P1/src/submission_tools.py audit P1/subs/exp005.csv --pred P1/preds/test_exp005.npy` 가 `판정: OK`.
7. `P1/subs/exp005.csv` 의 행 수·`station,year,layer,time` 값이 `P1/P1_data/sample_submission.csv` 와 완전히 같다.
8. `P1/P1_data/`, 기존 `P1/preds/*.npy`, 기존 `P1/subs/*.csv` 는 변경되지 않았다.

## 제약

- `P1/P1_data/` 읽기 전용. 시간 gap 을 넘는 rolling·diff·이벤트 연결 금지.
- 랜덤 행 분할 금지. fold 정의(`features.py:112 time_folds`)를 바꾸지 마라.
- `SEED = 42`, `n_jobs <= 2`, 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 새 코드는 `P1/src/exp005.py` 한 파일. 기존 `P1/src/*.py` 는 import 만 하고 수정하지 마라.
- 새 모델 파일은 `P1/models/exp005_*.txt` 로만 저장한다. 기존 모델 파일을 덮어쓰지 마라.

## 보고

**결론 문장 금지. 해석·제안·요약 산문 금지. 숫자와 표만.**

1. `n_estimators` 값 1줄, `exp002_outer_honest.txt` 재사용 여부 1줄
2. 1단계 4개 모델 표: `모델 | 학습 행 수 | 학습 max time | 예측 min time | 미래 행 수`
3. 분포 분위수 표: `배열 | p1 | p5 | p25 | p50 | p75 | p95 | p99`
4. hole 최적화 대조 1줄 (불일치 행 수)
5. 2-A 상위 10개 표: `weight | threshold | hole | inner F1`
6. 2-B 상위 10개 표: `r | λ | inner F1`
7. 선택 조합 1줄
8. exp004 설정 재현 검증 1줄 (TP/FP/FN/F1)
9. outer 결과 1줄: `TP | FP | FN | F1 | Δ`
10. outer `anomaly_type` 별 표: `type | 양성 행 | TP | FN | 행 recall | F1`
11. outer station 별 표: `station | F1`
12. test 양성률: 전체 1줄 + stream 별 상·하위 3개
13. 게이트 2개 출력의 `판정:` 줄
14. `submission_sha256`
