# T22 — 구간(segment) 단위 offset·drift 검출기. 2단계 모델

## 목적

행 단위 GBDT 는 계단 이후를 「새 정상」으로 본다. inner FN 6038 중 offset 2156(recall 0.4254) +
drift 1689(0.5104) = 63.7% 가 여기서 나온다. offset 구간 17개 중 8개가 full_miss 이고
구간 내부 적중률 p50 은 **0.0083** — 행 확률을 모아봐야 회수되지 않는다. 구간 자체를 증거로 삼는
**2단계 검출기**를 만든다: 변화점으로 후보 구간을 뽑고, 구간 feature 로 분류하고, 채택된 구간의
행 전체를 양성으로 켠다.

이 사양은 **A(후보 생성 상한 측정) → B(구간 분류기) 순서**다. A 의 커버리지가 낮으면 B 는 무의미하니
A 표를 먼저 출력하고 이어서 B 를 돈다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 기준선: 최고 제출 exp010 public F1 **0.741274** (양성률 2.9832%, `t_high 0.775 / hole 48`).
  1위 30.885889점 ≈ F1 0.865. 격차의 대부분이 offset·drift 다.
- 행 모델 정직한 원시 확률이 이미 있다: `P1/cache/exp005_raw_oof.npy` (fold1·2 는 forward 모델,
  fold3 는 봉인 모델), `P1/cache/exp005_raw_test.npy` (전체 train 모델). 재학습하지 마라.
- inner(fold1·2) 고정 후처리 기준선: exp005 inner F1 **0.697836** (`t 0.62`, `hole 48`, dilation 0, min-length 0).
- 유형별 inner recall(exp005): flatline 0.894, noise 0.700, offset 0.425, drift 0.510.
- 층 구성: G-ORS 층 1개(층간 단면 불가), I-ORS 7개, S-ORS 8개. test 는 I-ORS 층 3 이 없다.
- Q2 측정(`P1/experiments/logs/q2_out.log`): 형제 층 동시 주입 겹침 p50 **0.0**(상쇄 없음).
  행 단위 AUC — offset: `resid_1009` 0.871 / `clim_resid` 0.855 / `x4` 0.806,
  drift: `x5` 0.822 / `x4` 0.807 / `resid_1009` 0.806. `x4` 는 `resid_1009` 와 Spearman 0.194 로 **비상관**이다.
  `x1` = temp − median(같은 기지·시각 **다른** 층 temp), `x4` = rolling_mean(x1,36) − rolling_mean(x1,432),
  `x5` = rolling slope(x1,144)×144. 구현은 `P1/src/q2_cross_layer.py` 에 있다 — 재구현하지 말고 재사용해라.
- 적분 feature 39개 동시 투입은 붕괴한다(inner 0.5520). 5개는 이득(0.702859). **feature 는 적게.**
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것 (하드 제약)

- 행 모델을 재학습하지 마라. `exp005_raw_*.npy` 를 그대로 쓴다.
- 봉인 fold3 를 학습·판독·채점하지 마라. 채점은 **inner = fold1·2 뿐**이다.
- 시간 gap 을 넘는 rolling·구간 연결 금지. purge 87시간 규칙 유지.
- 후처리 격자를 inner argmax 로 고르지 마라 (`f.inner_delta_no_transfer`). 후처리는 **exp005 확정값 고정**
  (`t 0.62`, `hole 48`, dilation 0, min-length 0). 이 사양에서 임계 탐색은 없다.
- 외부 데이터 금지. `P1/P1_data/` 읽기 전용. `SEED = 42`, `n_jobs <= 2`.
- 새 추상화 금지. 파일은 `P1/src/t22_segment.py` 하나.
- **전체 실행이 25분을 넘길 것 같으면 코드를 완성해 저장만 하고 멈춰라.** 실행은 사람이 한다.

## A. 후보 구간 생성과 상한 측정

stream = (station, layer) 를 gap 으로 자른 블록. 각 stream 에서 잔차 `r` 를 만든다:
`r = clim_resid` (fold 별 train-only climatology. `P1/src/features.py::fit_reference`·`apply_reference` 재사용),
보조로 `x1` (층간 잔차, 적합 불필요).

후보 경계: W ∈ {36, 144, 432, 1008} 각각에 대해 `d_W(i) = mean(r[i:i+W]) − mean(r[i−W:i])` 를 계산하고
(gap·경계는 NaN), `|d_W|` 를 stream 내 robust σ 로 나눈 값이 **2.0 이상인 국소 최대점**을 경계 후보로 모은다.
경계 후보 + stream 시작·끝으로 구간을 자르고, 인접 구간 병합 없이 그대로 후보 구간으로 쓴다.
길이 6 미만 후보는 버린다.

**표 A1 — 후보 생성 상한** (inner 행만, 유형별: offset/drift/noise/flatline/other):

| type | n_true_runs | covered_iou50 | covered_any | best_iou_p50 | 후보구간수 | 후보가 덮는 행 비율 |

`covered_iou50` = 참 run 중 어떤 후보 구간과 IoU ≥ 0.5 인 것의 수.
`covered_any` = 어떤 후보 구간과 1행이라도 겹치는 참 run 수.

**표 A2 — 상한 F1**: 모든 후보 구간에 대해 「그 구간의 참 양성 비율 ≥ 0.5 이면 켠다」는 오라클을 적용했을 때의
inner F1 / TP / FP / FN. 이것이 B 단계가 도달 가능한 천장이다.

## B. 구간 분류기

후보 구간마다 feature (전부 그 구간과 인접 문맥에서만 계산. 미래 label 사용 금지):

- 길이, 시작 시각의 month, station, layer, depth 중앙값
- 구간 내 `r` 평균·중앙값·표준편차, 구간 밖 좌우 각 432행 대비 **레벨 차이 / stream robust σ**
- 구간 내 `r` 기울기 × 길이, 구간 양끝의 step 크기 (`|d_144|` 값)
- 구간 내 `x1` 평균과 좌우 문맥 대비 차이 (층이 1개면 NaN)
- 구간 내 `temp` diff 의 절대값 중앙값, 정확히 0 인 diff 비율 (flatline 지표)
- 구간 내 psal 결측 비율, psal 잔차 평균
- 구간 내 **행 모델 확률**(`exp005_raw_oof.npy`)의 평균·p90·최대

label = 구간 내 `label==1` 비율 ≥ 0.5.

학습: 구간을 그 시작 시각으로 fold 에 배정한다. fold1·2 각각에 대해 **그 fold 의 forward mask 시간대에
시작하는 구간만** 학습에 쓰고(purge 87h 동일 적용), 해당 fold 구간을 예측한다. LightGBM,
`objective=binary`, `learning_rate 0.05`, `num_leaves 31`, `min_child_samples 20`, `n_estimators 400`,
`random_state 42`, `n_jobs 2`. 구간 수가 적으므로 트리 수를 늘리지 마라.

행 점수 합성: `combined = max(row_prob, seg_prob_broadcast)`. `seg_prob` 는 구간 내 모든 행에 동일 값.

**표 B1 — inner F1 비교** (후처리는 exp005 확정값 고정, 임계 탐색 없음):

| variant | inner_f1 | TP | FP | FN | 양성률 |
|---|---|---|---|---|---|
| row_only (exp005 재현) | | | | | |
| segment_only (seg_prob ≥ 0.5) | | | | | |
| combined (max) | | | | | |

**표 B2 — 유형별 recall** (위 3 variant × offset/drift/noise/flatline/spike).

**표 B3 — 구간 분류기 gain importance 상위 10**.

## 산출물

- `P1/src/t22_segment.py`
- `P1/cache/t22/segments_inner.parquet` (또는 csv) — 후보 구간과 feature·label·예측
- `P1/preds/score_oof_t22.npy` — combined 행 점수 float32 (`submission_tools.save_pool_pred` 사용, fold3 행은 exp005 행 확률 그대로)

## 수락 기준

| 항목 | 기준 |
|---|---|
| 표 A1·A2·B1·B2·B3 | 전부 출력 |
| 누출 검사 | 각 fold 학습 구간의 최대 시각 < 그 fold 검증 시작 − 87h. 위반 수를 0 으로 print |
| combined | row_only 대비 inner F1 과 offset·drift recall 을 같이 보고 (판정은 Opus 가 한다) |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 산문·코드 붙여넣기 금지. 파일 경로 + 표 A1·A2·B1·B2·B3 만.

## 실행

```
python P1/src/t22_segment.py
```
