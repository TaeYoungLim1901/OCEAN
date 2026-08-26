# Q4 — 2차 신호 AUC 스크리닝 (읽기 전용 측정)

## 목적

exp016 이 이웃 회귀 잔차 4개로 public +0.034782 를 냈다. 확정된 사실: **행 단위 AUC 는 public 으로
전이되고 inner F1 은 전이되지 않는다** (`f.neighbor_wins`). 따라서 다음 feature 도 **AUC 로 사전 선별**한다.
이 사양은 후보 신호 8계열의 유형별 AUC 와 **계절 안정성**을 잰다. 학습·제출·기존 파일 수정 금지.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 현재 최고 신호 (train 전체 행 AUC): offset — `n1` 0.903359 / `resid_1009` 0.871459 / `n3` 0.861343.
  drift — `n4` 0.894243 / `n3` 0.860194 / `n2_432` 0.855001. noise — `n2_144` 0.825717 / `n2_432` 0.815020.
  flatline 은 전부 0.63 이하다. 구현은 `P1/src/q3_neighbor.py`·`P1/src/q2_cross_layer.py` — **재사용해라.**
- `n1` = 같은 기지 다른 층 temp 로 robust 회귀한 잔차. `n3` = mean(n1,36) − mean(n1,432).
  `n4` = slope(n1,144)×144. `n2_W` = n1 − rolling_median(n1,W).
- **주입은 temp 에만 한다.** psal·depth 는 참고 변수이고 이상 주입 대상이 아니다 (문제지 1-2).
- 문제지 경고: 층간 상관은 조석 혼합기엔 유효하지만 **수온약층 발달기에는 소실**되어 정상 성층을
  이상으로 오탐할 수 있다. test 는 2026-01~06 (겨울 혼합 → 봄 성층 전이)이다.
- 문제지 권고: **정점·층 단위로 표준화한 지표가 전 구간 공통 임계보다 안정적**이다.
- 층 구성: G-ORS 1층(이웃 없음), I-ORS 7층(test 는 층 3 미운영), S-ORS 8층.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- GBDT 학습·제출·기존 파일 수정 금지. 새 파일은 `P1/src/q4_screen.py` 하나.
- label 을 신호 계산에 쓰지 마라. 시간 gap 을 넘는 rolling 금지.
- 이미 측정된 `n1~n4`·`x1`·`x4`·`resid_1009`·`clim_resid` 를 다시 설계하지 마라 (비교군으로만 포함).

## 후보 신호 8계열

| id | 정의 |
|---|---|
| `s1` | `n1 / (rolling_MAD(n1, 4320) + 1e-3)` — stream 단위 표준화 |
| `s2` | `n3 / (rolling_MAD(n1, 4320) + 1e-3)` |
| `s3` | 이웃 상관 게이트: `rolling_corr(temp, neighbor_median, 432)`. 낮을수록 성층. **신호 자체로도 재라** |
| `s4` | `s1 × clip(s3, 0, 1)` — 성층기엔 이웃 단서를 죽인다 |
| `s5` | CUSUM: stream 내 `cumsum(clip(z, -3, 3))` 를 창 432 로 이동평균 뺀 값. `z = s1` |
| `s6` | 지속성: `run_length(|s1| > 2)` (현재 행이 속한 연속 초과 구간 길이) |
| `s7` | T–S 잔차: stream 별 `temp ~ psal` robust 회귀 잔차. psal 결측 행은 NaN |
| `s8` | `slope(n1, W)×W`, W ∈ {432, 1008} 두 개 (n4 는 W=144) |

## 측정

### 표 Q4-A — 유형별 AUC (train 전체 행, Q3 와 같은 방식)

| signal | type | n_pos | n_nan_pos | auc | recall_at_fpr_0.03 |

신호: `s1..s8` (s8 은 2개) + 비교군 `n1`, `n3`, `n4`, `resid_1009`. 유형: offset, drift, noise, flatline.

### 표 Q4-B — 계절 안정성

같은 AUC 를 **1~6월 행만**과 **7~12월 행만**으로 나눠 다시 잰다. test 는 1~6월이다.

| signal | type | auc_jan_jun | auc_jul_dec | diff |

### 표 Q4-C — 연도 안정성

`auc_2024` 와 `auc_2025` (1~6월 행만, 유형 offset·drift·noise).

| signal | type | auc_2024 | auc_2025 | diff |

### 표 Q4-D — 기존 46 feature 와의 중복

상위 신호 5개 각각과 `n1`,`n3`,`n4`,`resid_1009`,`clim_resid` 의 Spearman.

| signal | vs | spearman |

## 수락 기준

| 항목 | 기준 |
|---|---|
| 표 | Q4-A~D 전부 출력 |
| 기준선 재현 | Q4-A 의 `n1` offset AUC 가 0.903359 ± 0.001 |
| 실행 시간 | 20분 이내. 넘으면 s5·s6 를 빼고 그 사실을 print |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 표가 길면 Q4-A 는 AUC 0.75 이상 행만 출력하고 그 사실을 적어라.
샌드박스 시간이 부족하면 스크립트를 완성해 저장만 하고 한 줄로 보고해라.

## 실행

```
python P1/src/q4_screen.py
```
