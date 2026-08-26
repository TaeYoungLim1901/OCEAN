# Q3 — 이웃 층 회귀 잔차의 offset·drift 분리력 (읽기 전용 측정)

## 목적

Q2 에서 층간 단면 신호 `x1 = temp − median(같은 기지·시각 다른 층 temp)` 의 offset AUC 는 0.806 으로
기존 `resid_1009`(0.871) 를 못 이겼다. 그러나 `x1` 은 층별 평균·분산 차이를 무시한 **거친** 단면이다.
층 간 수온 관계는 물리적으로 안정하므로, **정상 구간에서 적합한 이웃 층 회귀**로 각 stream 의 수온을
예측하고 그 잔차를 보면 offset·drift 가 훨씬 선명해야 한다. 이 잔차는 climatology 와 달리 **연도 의존이 약해**
2026 test 로 전이될 가능성이 높다. 이 사양은 **측정만** 한다. 학습(GBDT)·제출·기존 파일 수정 금지.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 층 구성: G-ORS 1층(이웃 없음, test 행 9.7%), I-ORS 7층, S-ORS 8층. **test 는 I-ORS 층 3 이 없다** —
  이웃 집합은 test 에 존재하는 층으로만 구성 가능해야 한다.
- Q2 행 단위 AUC 기준선 (train 전체, 유형별): offset — `resid_1009` 0.871 / `clim_resid` 0.855 /
  `x4` 0.806 / `x1` 0.704. drift — `x5` 0.822 / `x4` 0.807 / `resid_1009` 0.806 / `clim_resid` 0.794.
  구현은 `P1/src/q2_cross_layer.py` 에 있다. **재구현하지 말고 재사용해라.**
- 형제 층 동시 주입 겹침 p50 = 0.0. 단면은 상쇄되지 않는다.
- train 2024-01-01~2025-12-09 전 계절, test 2025-12-31~2026-06-30 (1~6월뿐).
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- GBDT 학습·제출·기존 파일 수정 금지. 새 파일은 `P1/src/q3_neighbor.py` 하나.
- 회귀 적합에 **label 을 쓰지 마라.** 적합 표본 선택은 label 없이 robust 하게 한다 (아래).
- 시간 gap 을 넘는 rolling 금지.

## 측정

### 잔차 3종

각 (station, layer) stream 마다 이웃 = 같은 기지의 **다른 층 전부**. G-ORS 는 이웃이 없으니 전부 NaN 이고
그 행수를 보고한다.

- `n1` — **robust 선형 회귀 잔차**: 같은 시각 이웃 층 temp 를 설명변수로 `temp` 를 예측한다.
  적합은 train 전 구간에서 하되 label 을 쓰지 말고 **Huber 손실 또는 IRLS 2회 재가중**으로 이상치를 눌러라
  (`sklearn` 이 있으면 `HuberRegressor`, 없으면 numpy 로 IRLS). 결측 이웃은 그 시각 이웃 평균으로 대체하고
  대체 비율을 보고한다. 잔차 `n1 = temp − 예측`.
- `n2_W` — `n1 − rolling_median(n1, W)`, W ∈ {144, 432}
- `n3` — `rolling_mean(n1, 36) − rolling_mean(n1, 432)` (x4 와 같은 형태, 입력만 n1)
- `n4` — `rolling slope(n1, 144) × 144`

추가로 **연도 전이 점검용**: 회귀 계수를 2024 행만으로 적합한 것과 2025 행만으로 적합한 것 두 벌 만들어,
stream 별 계수 코사인 유사도와 잔차 σ 비를 보고한다.

### 표 Q3-A — 분리력 (Q2 와 같은 방식: 대상 행 = 해당 유형 행 ∪ 정상 행 전체)

| signal | type | n_pos | n_nan_pos | auc | recall_at_fpr_0.03 |

신호: `n1, n2_144, n2_432, n3, n4` × 유형 `offset, drift, noise, flatline`.
비교 기준선으로 `x1, x4, resid_1009, clim_resid` 를 같은 표에 넣어라.

### 표 Q3-B — 기지별 (I-ORS, S-ORS), offset·drift 만

| station | signal | type | auc | recall_at_fpr_0.03 |

### 표 Q3-C — 연도 전이

| station | layer | coef_cos_sim_2024_vs_2025 | sigma_ratio | n_neighbors | imputed_frac |

### 표 Q3-D — 상관

`n3`·`n1` 과 `resid_1009`·`clim_resid`·`x4` 의 Spearman (전체 행, NaN 제외).

## 수락 기준

| 항목 | 기준 |
|---|---|
| 표 | Q3-A~D 4개 전부 출력 |
| 실행 시간 | 15분 이내. 넘으면 stream 을 S-ORS·I-ORS 로 제한하고 표 머리에 적어라 |
| 기준선 | 표 Q3-A 에 `resid_1009` offset AUC 가 0.871 로 재현되어야 한다 (Q2 와 같은 계산인지 확인) |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 산문·코드 붙여넣기 금지.
샌드박스 시간이 부족하면 스크립트를 완성해 저장만 하고 그 사실을 한 줄로 보고해라.

## 실행

```
python P1/src/q3_neighbor.py
```
