# T4 — exp002 FN 해부 (읽기 전용 진단)

## 목적

exp002 sealed outer 는 TP 4935 / FP 500 / FN 2508 이다. precision 0.908, recall 0.663 으로
**실패의 83% 가 미탐(FN)** 이다. 다음 실험 설계를 위해 FN 이 어디에 몰려 있고
그중 얼마가 **사후처리(확률 평활·구멍 메움)만으로 회수 가능**한지 숫자로 확정한다.

이 작업은 **읽기 전용 진단이다. 모델을 새로 학습하지 말고, 기존 저장 예측만 쓴다.**
어떤 파일도 수정하지 말고, 새 파일은 `P1/cache/T4/` 아래에만 만든다.

## 배경 — 이미 검증된 사실 (재조사 금지)

- exp002 = rule score 와 LGBM 확률의 blend(weight 0.15) + threshold 0.44 + hole 2. 봉인 outer(fold3) F1 `0.766423`.
- exp003 의 type-multiclass 축은 sealed outer `0.760045` (Δ `-0.006379`) 로 **이미 기각**했다. 다시 시도하지 마라.
- 랜덤 행 분할, adversarial drift 축도 닫혔다. fold 정의는 strictly-forward nested 이며 경계 purge 는 87시간이다.
- train/test drift source AUC 는 `0.850133` 으로 이미 측정했다. 다시 재지 마라.
- exp002 대비 exp003 잔차 상관은 `0.850584` 로 이미 측정했다.
- 새 의존성 설치 금지. `pip install` 이 필요해지면 **즉시 멈추고 그 사실만 보고한다.**
  현재 환경에 `xgboost`, `catboost`, `torch` 는 없다.

## 기준선 (baseline) 숫자

| 지표 | 값 |
|---|---|
| exp002 봉인 outer F1 | `0.766423` |
| outer TP / FP / FN | `4935` / `500` / `2508` |
| exp002 inner F1 | `0.720133` |
| threshold / weight / hole | `0.44` / `0.15` / `2` |
| noise 유형 F1 | `0.628935` |
| I-ORS station F1 | `0.641750` |

## 수락 기준

1. 스크립트가 exp002 현행 설정을 재현해 봉인 outer TP `4935` / FP `500` / FN `2508`,
   F1 `0.766423` 을 **정확히** 낸다. (이것만 일치 요구다. 나머지는 하한 조건이다.)
2. 표 A~G 가 전부 채워져 있고 빈 칸이 없다.
3. D 의 세 분류 개수 합이 FN 총계 `2508` 과 같다.
4. E·F 의 11개 설정이 모두 계산되어 있다.
5. `P1/cache/T4/metrics.json` 이 생성되어 위 모든 수치를 담는다.
6. `P1/P1_data/` 와 기존 `P1/src/*.py`·`P1/preds/*.npy` 의 mtime 이 변하지 않았다.

## 입력

- `P1/preds/oof_exp002.npy` — exp002 OOF 확률 (train 행 순서)
- `P1/preds/test_exp002.npy` — test 확률 (진단에 필요하면 분포 비교용으로만)
- `P1/P1_data/train.csv` — `label`, `anomaly_type` 포함
- `P1/cache/exp002_meta.json` — `threshold` 0.44, `weight` 0.15, `hole` 2, fold 정의
- `P1/src/exp002.py`, `P1/src/features.py` — fold 경계·구멍 메움 로직 확인용 (읽기만)

`oof_exp002.npy` 의 행 정렬 규칙은 `P1/src/exp002.py` 를 읽어 확정하고, 확정한 정렬 키를
보고 맨 위에 1줄로 적는다. 정렬이 확정되지 않으면 **작업을 멈추고 그 사실만 보고한다.**

## 정의

- **이벤트** = 같은 `station × layer` 안에서 `label == 1` 인 연속 행 묶음.
  시간 gap(10분 간격이 끊긴 곳)을 넘어 이어붙이지 않는다.
- **적용 규칙** = exp002 와 동일: `p >= 0.44` 이면 1, 그 뒤 예측 1 사이의 길이 ≤ 2 인
  0 구간을 1 로 메움(hole=2).
- 아래 A~E 는 **fold3(봉인 outer) 구간**에서 계산한다. F 는 fold1·2(inner) 구간에서 계산한다.
  fold 경계는 `exp002.py` 의 정의를 그대로 쓴다.

## 보고할 표 (전부 숫자만)

**A. anomaly_type 별 FN**

| anomaly_type | 이벤트 수 | 양성 행 수 | TP | FN | 행 recall | 이벤트 hit rate |

`이벤트 hit rate` = 그 이벤트의 행 중 1개 이상을 잡은 이벤트 비율.

**B. station × layer 별 FN 상위 10** (FN 내림차순)

| station | layer | 양성 행 | FN | recall | 지배적 anomaly_type |

**C. 이벤트 길이 구간별 FN**

이벤트 길이 bin: 1, 2-3, 4-6, 7-12, 13-36, 37-144, 145+ 행.

| bin | 이벤트 수 | 양성 행 | FN | 행 recall | 이벤트 hit rate |

**D. 회수 가능성 — 이 실험의 핵심 숫자**

FN 행을 3분류하여 개수와 비율을 낸다.

1. `recoverable_inside` — 같은 이벤트 안에 TP 가 존재하는 FN (이벤트는 잡았는데 행을 놓침)
2. `near_miss` — 이벤트 전체를 놓쳤지만 그 이벤트 행의 최대 확률이 0.20 이상
3. `hard_miss` — 이벤트 전체를 놓쳤고 최대 확률이 0.20 미만

각 분류를 anomaly_type 으로 교차한 표도 낸다.

**E. 사후처리 상한 (봉인 outer, 파라미터 변경 없이 규칙만)**

exp002 확률을 그대로 두고 아래 사후처리만 바꿨을 때의 outer F1 을 표로 낸다.
**이것은 튜닝이 아니라 상한 측정이다. 여기서 고른 값을 제출에 쓰지 않는다.**

| 설정 | TP | FP | FN | F1 | exp002 대비 Δ |
| 현행 (thr 0.44, hole 2) | | | | 0.766423 | 0 |
| hole 6 | | | | | |
| hole 12 | | | | | |
| hole 36 | | | | | |
| 이벤트 확장: 예측 1 구간 양끝을 ±3 행 확장 | | | | | |
| 이벤트 확장 ±12 행 | | | | | |
| 확률 rolling mean 창 7 후 thr 0.44 | | | | | |
| 확률 rolling mean 창 37 후 thr 0.44 | | | | | |
| 확률 rolling max 창 7 후 thr 0.44 | | | | | |
| 이력 임계(hysteresis): 진입 0.44 / 유지 0.15 | | | | | |
| 이력 임계: 진입 0.44 / 유지 0.25 | | | | | |

rolling·확장은 모두 `station × layer` 안에서만, 시간 gap 을 넘지 않게 적용한다.

**F. inner 에서의 동일 표**

E 와 동일한 11개 설정을 fold1·2 inner 구간에서 계산해 같은 형식으로 낸다.
inner 순위와 outer 순위가 일치하는지 확인할 수 있어야 한다.
inner 최상위 설정이 outer 에서도 상위 3 안에 드는지 **예/아니오** 한 단어만 덧붙인다.

**G. noise 전용 관찰**

`anomaly_type == noise` 인 이벤트에 한정해:

| 지표 | 값 |
| 이벤트 수 | |
| 이벤트 길이 중앙값 | |
| 이벤트 내 확률 중앙값 | |
| 이벤트 내 확률 최대값의 중앙값 | |
| 정상 행 확률 중앙값 (같은 station-layer, 이벤트 밖) | |
| 이벤트 내 `std_7 / std_1009` 중앙값 | |
| 정상 행 `std_7 / std_1009` 중앙값 | |

drift 이벤트에 대해서도 같은 표를 낸다 (`std` 대신 `drift_score` 사용).

## 제약

- `P1/P1_data/` 는 읽기 전용이다. 학습·재학습·threshold 재선택 금지.
- 시간 gap 을 넘는 rolling·diff·이벤트 연결 금지.
- `SEED = 42`, `n_jobs <= 2`, 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 진단 스크립트는 `P1/src/diag_t4.py` 한 파일로만 만들고, 수치 결과는
  `P1/cache/T4/metrics.json` 에도 저장한다. 기존 파일은 수정하지 않는다.
- 검증: `python P1/src/diag_t4.py` 를 **직접 실행하고** 위 표의 실제 숫자를 보고한다.
  현행 설정 재현이 outer F1 `0.766423`, TP 4935 / FP 500 / FN 2508 과 일치하지 않으면
  **표를 내지 말고 불일치 숫자만 보고하고 멈춘다.**

## 보고 형식

**결론 문장 금지. 해석·제안·요약 산문 금지. 위 표 A~G 와 숫자만 낸다.**
정렬 키 1줄 + 현행 재현 검증 1줄 + 표 A~G. 그 외 어떤 문장도 쓰지 마라.
