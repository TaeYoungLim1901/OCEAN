# T24 — exp016: 이웃 층 회귀 잔차 4개를 행 모델에 추가 + 제출

## 목적

Q3 측정에서 **이웃 층 robust 회귀 잔차 계열이 기존 최고 신호를 offset·drift 양쪽에서 이겼다.**
행 단위 AUC:

| 신호 | offset | drift | noise |
|---|---|---|---|
| `n1` 이웃회귀 잔차 | **0.903359** | 0.852175 | 0.787195 |
| `n4` = slope(n1,144)×144 | 0.681650 | **0.894243** | 0.650752 |
| `n3` = mean(n1,36) − mean(n1,432) | 0.861343 | 0.860194 | 0.666679 |
| `n2_432` | 0.745969 | 0.855001 | 0.815020 |
| `resid_1009` (기존 최고) | 0.871459 | 0.806467 | 0.781885 |
| `clim_resid` (기존) | 0.854839 | 0.794493 | 0.686397 |

`n1`–`resid_1009` Spearman 0.389, `n3`–`resid_1009` 0.207 로 **기존 feature 와 대체로 직교**한다.
offset·drift 는 inner FN 의 63.7% 다. 이 4개를 행 모델에 넣고 제출한다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 기준선: exp010 public F1 **0.741274** (양성 5042, 2.9832%, `t_high 0.775 / hole 48`).
  exp012(적분 feature 5개 추가) 0.717975 로 **졌다** → 적분 feature 축 폐쇄.
- **inner 는 판정기가 아니다** (`f.inner_delta_no_transfer`, `f.inner_not_model_judge`).
  임계·후처리·모델 비교 어느 것에도 부호가 전이되지 않았다. **inner 를 계산하지 마라.**
- 후처리는 `hole=48` 하나뿐. 임계는 test 양성률 **0.0300** 앵커. 격자 탐색 금지.
- 이웃 회귀 잔차 구현은 **이미 있다**: `P1/src/q3_neighbor.py` 의 `_neighbor_residuals(train, test)` 와
  `_derive_neighbor_signals`. **재구현하지 말고 import 해서 재사용해라.** train·test 를 같은 계수로 처리한다.
- test 는 I-ORS 층 3 이 없다. 이웃 결측은 q3 의 대체 규칙을 그대로 쓴다 (대체 비율 최대 0.3612).
- 계수 안정성(Q3-C): S-ORS 층별 2024 vs 2025 코사인 유사도가 0.98 ~ −0.01 로 **층에 따라 불안정**하다.
  따라서 `n1` 절대값만 믿지 말고 `n3`·`n4` 같은 상대형을 함께 넣는다.
- 제출 파이프라인은 `P1/src/exp013_shift.py` 에 있다 (전체 train 단일 모델 → 양성률 앵커 → CSV/audit).
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- fold 학습·inner 채점 금지. 전체 train 단일 모델만.
- 새 feature 를 발명하지 마라. 추가는 **정확히 `n1`, `n2_432`, `n3`, `n4` 4개**, 최종 46 컬럼 (assert).
- 봉인 fold3 판독 금지. `P1/P1_data/` 읽기 전용. `SEED=42`, `n_jobs<=2`.
- 새 파일은 `P1/src/exp016_neighbor.py` 하나. 기존 파일 수정 금지.
- **학습을 실행하지 마라.** 코드 완성 + 문법·컬럼 수 검증까지. 실행은 사람이 한다.

## 구현

`P1/src/exp016_neighbor.py`:

1. base 42 = `P1/cache/exp001/train_features.pkl`, test 는 `make_features(test, ref_full)`
   (`P1/cache/exp001/reference.json`).
2. `q3_neighbor` 에서 train·test 이웃 잔차를 만들고 `n1, n2_432, n3, n4` 를 뽑는다.
   **test 잔차는 train 에서 적합한 계수로 계산**해야 한다. train 에만 있는 층으로 test 를 계산하지 마라.
   NaN 은 그대로 둔다 (LightGBM 이 처리한다). G-ORS 는 전부 NaN 이 되며 그 행수를 print 한다.
3. 학습 파라미터는 `P1/src/exp013_shift.py` 와 동일 (700 tree, SEED 42, n_jobs 2).
   전체 train 행으로 1회 학습 → `P1/models/exp016_full.txt`.
4. `P1/cache/exp016_raw_test.npy` 저장 → 양성률 0.0300 앵커 임계 → `P1/subs/exp016.csv`,
   `P1/preds/test_exp016.npy` (`exp005.store_score` 규약 float32), `P1/cache/exp016_meta.json`.
5. 검증 2개를 직접 실행하고 마지막 3줄 보고:
   `python P1/src/submission_tools.py audit P1/subs/exp016.csv --pred P1/preds/test_exp016.npy`,
   `python P1/src/check.py sub P1/subs/exp016.csv`.
6. gain importance 상위 10을 표로 print — 추가한 4개가 몇 위인지 본다.

## 수락 기준

| 항목 | 기준 |
|---|---|
| feature 수 | 46 (assert) |
| post_rate | 0.0300 ± 0.0005 |
| NaN 행수 | G-ORS train/test 행수와 일치하는지 print |
| audit / check sub | `판정: OK` |
| 재실행 | CSV sha256 동일 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 파일 경로 + 아래 표 2개 + 검증 마지막 3줄.

| exp | n_features | train_rows | t_high | raw_rate | post_rate | n_positive | sha10 |

| rank | feature | gain |

## 실행

```
python P1/src/exp016_neighbor.py
```
