# Q2 — 단면(층간·기지간) 잔차의 offset·drift 분리력 측정 (읽기 전용 EDA)

## 목적

현재 최고 제출 public F1 **0.741274**, 1위 **0.865 추정**. 남은 격차의 63.7% 가 offset·drift 미탐이다
(inner FN 6038 중 offset 2156 recall 0.4254 + drift 1689 recall 0.5104).
행 단위 GBDT 가 계단 이후를 「새 정상」으로 보기 때문이다.

가설: **같은 기지의 다른 층은 같은 시각 수온이 강하게 상관**한다. 한 stream 에만 주입된 offset·drift 는
층간 잔차에서 **지속적 편차**로 남는다. 현재 base feature 의 `layer_resid` 는 **순간값**이고
자기 이력 대비 기준선이 없어 이 신호를 못 준다.

이 사양은 **feature 를 만들지 않는다. 측정만 한다.** 학습 금지, 제출 금지, 파일 수정 금지.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- train 776,706 행 / test 169,011 행. train 2024~2025, test 2026. 양성률 train 4.1362%.
- 층 구성: G-ORS 층 1개(단면 불가), I-ORS 7개, S-ORS 8개. test 는 I-ORS 층 3 이 없다.
  G-ORS 는 test 행의 9.7% 뿐이다.
- 유형별 행수(train): noise 9442, drift 8575, offset 7296, flatline 6030, spike 101, 조합형 ~680.
- 기준선 feature 42개는 `P1/cache/exp001/train_features.pkl` 에 있다. 그중 단면 관련은
  `layer_resid`(같은 기지·시각 층 median 대비 순간 잔차), `layer_spread`, `layer_count` 뿐이다.
- 적분 feature 를 39개 한꺼번에 넣으면 붕괴한다(inner 0.5520). 5개만 넣으면 이득(0.702859). **개수가 적어야 한다.**
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- 학습·제출·기존 파일 수정 금지. 새 스크립트는 `P1/src/q2_cross_layer.py` 하나만.
- 시간 gap 을 넘는 rolling 금지 (`P1/src/features.py` 의 gap 분할 방식을 그대로 쓴다).
- 봉인 fold3 를 따로 판독하지 마라. 이 측정은 train 전체 행 기준 통계이며 모델 선택에 쓰지 않는다.

## 측정 1 — 단면이 상쇄되는가

anomaly 구간 = (station, layer) 별 `label==1` 연속 run. run 마다:

- `type` = 그 run 의 최빈 `anomaly_type`
- `sibling_overlap` = 그 run 의 시각들에서 **같은 기지의 다른 층이 동시에 label==1 인 비율**
  (분모: run 의 시각 수 × 그 시각에 존재하는 형제 층 수, 분자: 그중 label==1 인 것)

보고 표 A — 유형별 (offset, drift, noise, flatline, spike, 그 외 묶어서 `other`):

| type | n_runs | median_len | sibling_overlap_p50 | sibling_overlap_p90 | runs_with_zero_overlap |

## 측정 2 — 후보 신호의 분리력

`P1/src/features.py` 의 gap 분할·rolling 유틸을 재사용해 아래를 계산한다. 전부 stream(station-layer) 내부,
gap 을 넘지 않는다. W 는 10분 스텝 개수다.

층간(같은 기지·시각, **자기 자신을 제외한** 다른 층):
- `x1` = temp − median(다른 층 temp)
- `x2_W` = x1 − rolling_median(x1, W), W ∈ {36, 144, 432}
- `x3_W` = x2_W / (rolling_MAD(x1, 432) + 1e-3)
- `x4` = rolling_mean(x1, 36) − rolling_mean(x1, 432)
- `x5` = rolling slope(x1, 144) × 144

기지간(같은 층 번호·시각, 다른 기지). I-ORS·S-ORS 만 대상, 없으면 NaN:
- `y1` = temp − median(다른 기지 같은 층 temp), 그리고 `y2_144`, `y3_144`, `y4` 를 x 와 같은 정의로

기준선 비교군: `offset_score`, `drift_score`, `clim_resid`, `layer_resid`, `resid_1009`.

각 신호 s 에 대해 **|s|** (또는 부호가 의미 있으면 그대로도) 로 다음을 잰다.
평가 대상 행 = (그 유형 행) ∪ (모든 정상 행). NaN 행은 분모에서 빼고 그 수를 보고한다.

보고 표 B — 신호 × 유형 (offset, drift, noise, flatline):

| signal | type | n_pos | n_nan_pos | auc | recall_at_fpr_0.03 |

보고 표 C — 기지별 (I-ORS, S-ORS, G-ORS), offset·drift 만, 상위 신호 5개:

| station | signal | type | auc | recall_at_fpr_0.03 |

## 측정 3 — 결합 상한

표 B 의 offset·drift AUC 상위 **3개 신호만** 골라, 그 3개 + 기준선 5개로 요약 표 D 를 낸다:
각 신호쌍의 Spearman 상관(정상 행 포함 전체, NaN 제외).

| signal_a | signal_b | spearman |

## 수락 기준

- 표 A·B·C·D 4개가 전부 출력된다
- 표 B 는 신호 12개 이상 × 유형 4개를 덮는다
- 실행 시간 20분 이내. 넘으면 stream 을 S-ORS·I-ORS 로 제한하고 그 사실을 표 머리에 적는다

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 산문·코드 붙여넣기 금지. 표 A~D 와 스크립트 경로만.

## 실행

```
python P1/src/q2_cross_layer.py
```

샌드박스 시간이 부족하면 **스크립트를 완성해 저장만 하고 실행하지 마라.** 그 사실을 한 줄로 보고한다.
스크립트는 표 A~D 를 순서대로 print 하고, 각 표 직후에 `elapsed_seconds=` 를 찍는다.
