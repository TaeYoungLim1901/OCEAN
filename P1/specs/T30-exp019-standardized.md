# T30 — exp019: 정점·층 표준화 이웃 잔차 3개 추가 (46 → 49 feature)

## 목적

Q4 스크리닝에서 **정점·층 단위 표준화가 drift 를 확실히 개선**했다 (train 전체 행 AUC):

| 신호 | offset | drift | recall@FPR 0.03 (drift) | n1 과 Spearman |
|---|---|---|---|---|
| `s1` = n1 / (rolling_MAD(n1, 4320) + 1e-3) | 0.899995 | **0.894843** | **0.426770** | 0.901067 |
| `s2` = n3 / (rolling_MAD(n1, 4320) + 1e-3) | 0.863602 | **0.897333** | 0.372516 | 0.254774 |
| `s8_432` = slope(n1, 432) × 432 | 0.787921 | 0.858509 | 0.263653 | 0.029410 |
| `n1` (현재 사용 중) | 0.903359 | 0.852175 | 0.298751 | — |
| `n4` (현재 사용 중) | 0.681650 | 0.894243 | 0.385842 | — |

계절 안정성도 확인됐다: `s1` 의 1~6월 vs 7~12월 AUC 차이 offset −0.023511 / drift −0.027172 로
현재 쓰는 `n1`(−0.022974 / −0.020775)과 동급이다. test 는 2026-01~06 이다.

기각된 후보(재제안 금지): `s3`·`s4` 성층 게이트, `s5` CUSUM, `s7` T–S 잔차 — 전부 AUC 0.75 미만.
`s6` 지속성은 AUC 0.872914 이지만 recall@FPR0.03 이 **0.0** 이라 임계 근방에서 쓸모없다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 현재 최고 제출 **exp016** public F1 **0.776056** (27.381306점): base 42 + `n1`,`n2_432`,`n3`,`n4` = 46 feature,
  전체 train 단일 모델 700 tree, `t_high 0.737`, `hole 48`, 양성 5070 (2.9998%).
  코드는 `P1/src/exp016_neighbor.py`, 이웃 잔차 구현은 `P1/src/q3_neighbor.py`, s 계열은 `P1/src/q4_screen.py`.
  **전부 재사용해라. 재구현 금지.**
- **AUC 는 public 으로 전이되고 inner F1 은 전이되지 않는다** (`f.neighbor_wins`). inner 를 계산하지 마라.
- feature 개수 절벽 전례: 적분 feature 39개 투입 시 붕괴. 이번 추가는 3개다.
- 후처리는 `hole=48` 하나뿐. 임계는 test 양성률 **0.0300** 앵커. 격자 탐색 금지.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

### 기준선

기준선은 exp016 public F1 **0.741274 대비 +0.034782 인 0.776056** 이다. exp019 는 그것을 이겨야 채택된다.
규칙 베이스라인 전체 F1 은 0.548255 다.

## 하지 말 것

- fold 학습·inner 채점·봉인 fold3 판독 금지. 전체 train 단일 모델만.
- 추가는 **정확히 `s1`, `s2`, `s8_432` 3개**, 최종 49 컬럼 (assert). 다른 feature 발명 금지.
- 새 파일은 `P1/src/exp019_standardized.py` 하나. 기존 파일 수정 금지. `SEED=42`, `n_jobs<=2`.
- **학습을 실행하지 마라.** 코드 완성 + 문법·컬럼 수 검증까지. 실행은 사람이 한다.

## 구현

`P1/src/exp016_neighbor.py` 를 그대로 따르되 feature 3개를 더한다.
train·test 모두 **train 에서 적합한 회귀 계수·MAD 기준**으로 계산한다. test 에만 있는 정보를 쓰지 마라.
rolling MAD 는 stream 내부에서 gap 을 넘지 않는다.

산출: `P1/models/exp019_full.txt`, `P1/cache/exp019_raw_test.npy`, `P1/subs/exp019.csv`,
`P1/preds/test_exp019.npy` (`exp005.store_score` 규약 float32), `P1/cache/exp019_meta.json`.

검증 2개를 직접 실행하고 마지막 3줄 보고:
`python P1/src/submission_tools.py audit P1/subs/exp019.csv --pred P1/preds/test_exp019.npy`,
`python P1/src/check.py sub P1/subs/exp019.csv`.

## 수락 기준

| 항목 | 기준 |
|---|---|
| feature 수 | 49 (assert) |
| post_rate | 0.0300 ± 0.0005 |
| audit / check sub | `판정: OK` |
| 재실행 | CSV sha256 동일 |
| NaN | G-ORS 는 이웃이 없어 s 계열도 NaN 이다. 그 행수를 print |

## 보고 형식

**결론 문장 금지. 숫자와 표만.**

| exp | n_features | train_rows | t_high | raw_rate | post_rate | n_positive | sha10 |

| rank | feature | gain |  ← 상위 15, 추가 3개의 순위가 보이게

## 실행

```
python P1/src/exp019_standardized.py
```
