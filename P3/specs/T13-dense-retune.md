# 작업 지시: T13 — 조밀 재가중 판정기로 LGBM 재탐색 + 재블렌드

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라**. `P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용.

**`n_jobs` 는 2 로 고정한다.** 다른 세션이 같은 8코어를 쓴다. 이 값을 올리지 마라.

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.thinned_judge_inverts  씨닝 봉인 fold4 는 53사례뿐이고 hs0>3m 0건. damped/lgbm 순위조차 뒤집는다
f.dense_judge_bias       조밀은 저분산·유편향. 씨닝셋 층 비율 재가중으로 편향 제거 (T9, ESS fold4 649.3)
f.t1f_weights            현행 제출 T1F = persistence 0.00 / damped 0.35 / lgbm_t1 0.65
```

핵심: **현행 `lgbm_t1` 하이퍼파라미터는 씨닝 53사례 판정기로 선택됐다.** 판정기를 바꿨으니
파라미터를 다시 골라야 한다. 그것만 한다.

현행 파라미터 (`P3/src/train.py:78-79`):

```
LGBMRegressor(objective="regression_l2", n_estimators=500, learning_rate=0.03,
              num_leaves=31, min_child_samples=30, ...)
```

T8/T9/T11 이 측정한 조밀 **재가중** 점수 (재계산 금지, 대조군 기준):

| 후보 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|
| lgbm_t1 | — | — | — | `0.614009` |
| T1F (W_old) | `0.668083` | `0.630441` | `0.656407` | `0.611959` |

fold 별 ESS = fold1 `571.931657` / fold2 `948.602515` / fold3 `1124.684355` / fold4 `649.308480`

## 기준선 — 반드시 재사용할 자산

```
P3/cache/p3_t8/features.npy        # 조밀 피처 (재계산 금지)
P3/cache/p3_t8/feature_ids.npy
P3/cache/p3_t8/dense_index.csv
P3/cache/p3_t9/dense_weights.csv   # case_id, stratum, w — T9 재가중. 재생성 금지
P3/preds/oof_{persistence,damped,lgbm_t1}_p3_t8.npy
```

**피처를 다시 만들지 마라.** 캐시가 없거나 형상이 안 맞으면 **멈추고 보고하라.**
fold 경계·층 정의·재가중·strict-forward 마스크를 바꾸지 마라. 바꾸는 것은 LGBM 파라미터 하나뿐이다.

재가중 RMSE 정의는 T9·T11 과 동일: `sqrt(Σ w_i (y-p)² / Σ w_i)`, 리드 동일 가중.

## 작업 0 — `P3/src/tune_lgbm_dense.py`

`P3/cache/p3_t8/features.npy` 를 읽어 아래 **6개 후보만** 조밀 4폴드로 적합한다.
**그리드를 늘리거나 줄이지 마라. 다른 파라미터를 건드리지 마라.**

| ID | 변경점 (그 외 현행과 동일) |
|---|---|
| `A_base` | 없음 — 대조군 |
| `B_leaves15` | `num_leaves=15` |
| `C_leaves63` | `num_leaves=63` |
| `D_mcs60` | `min_child_samples=60` |
| `E_slow` | `learning_rate=0.015`, `n_estimators=1000` |
| `F_bag` | `feature_fraction=0.7`, `bagging_fraction=0.7`, `bagging_freq=1`, seed `42/43/44` 3개 평균 |

`F_bag` 만 시드 3개 평균이다. 나머지는 `SEED = 42` 단일.

출력:

- `P3/preds/oof_{ID}_p3_t13.npy` — 조밀 OOF, 6개 전부. **결과와 무관하게 전부 저장한다**
- `P3/preds/test_{ID}_p3_t13.npy` — 전체데이터 적합 후 test 예측, 6개 전부
- `P3/cache/p3_t13/report.json`

각 후보를 조밀 **재가중** fold1/2/3/4 로 채점하고, 조밀 **비가중** fold4 도 함께 낸다.

### 수락 기준 (결과 보기 전 고정)

- `A_base` 의 조밀 재가중 fold4 점수가 T8 `lgbm_t1` 의 `0.614009` 와 `|델타| <= 0.002`.
  벗어나면 하네스 불일치다 — **점수 해석 없이 그 사실만 보고하고 작업 1을 하지 마라**
- 후보별 `best_iteration` 이 `1` 이 아니고 `n_estimators` 상한에 붙지 않았다
- 6개 후보 점수가 서로 소수 6자리까지 같지 않다 (같으면 파라미터 미반영 버그)
- fold 별 ESS 가 T9 값과 일치

## 작업 1 — 재블렌드 (작업 0 대조군 통과 시에만)

작업 0 에서 **조밀 재가중 folds1-3 평균**이 가장 낮은 후보 1개를 `L*` 로 정한다.
(fold4 는 보지 마라. 봉인이다.)

풀 = `{persistence, damped, L*}`. 가중을 `0.05` 격자 심플렉스로 탐색하되
**folds1-3 재가중 점수만으로 고른다.** 고른 가중을 fold4 에 한 번만 적용해 보고한다.

`P3/preds/oof_final_p3_t13.npy` 와 `test_final_p3_t13.npy` 를 저장한다.
**제출 CSV 는 만들지 마라.**

### 수락 기준 (결과 보기 전 고정)

- 가중 성분 합 `1.0 ± 1e-6`, 전부 `>= 0`
- fold4 는 선택에 쓰이지 않았다 (코드로 보장. 어떻게 보장했는지 한 줄로 보고)
- 봉인 fold4 재가중 점수를 현행 T1F `0.611959` 와 나란히 보고

## 금지

- 피처 재생성·fold 경계 변경·재가중 재계산 금지
- 그리드 확장 금지. 표의 6개 외 파라미터 조합을 시도하지 마라
- fold4 를 후보 선택이나 가중 선택에 사용 금지
- 제출 CSV 생성 금지. `P3/subs/` 를 건드리지 마라
- `P3/preds/_backup_pre_maskfix/` 수정·삭제 금지
- `n_jobs` 를 2 초과로 올리지 마라
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- `P1/`·`P2/`·루트 파일 수정 금지. 쓰기는 `P3/` 안에서만
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[대조군]
A_base 조밀 재가중 fold4 = ___ / T8 0.614009 대비 델타 = ___ / 통과 = 예|아니오
fold 별 ESS = fold1 ___ / fold2 ___ / fold3 ___ / fold4 ___
후보별 best_iteration = ___

[작업 0 — 조밀 재가중]
| 후보 | fold1 | fold2 | fold3 | folds1-3 평균 | fold4(봉인) | 조밀 비가중 fold4 |
| A_base | ___ | ___ | ___ | ___ | ___ | ___ |
| B_leaves15 | ... |
| C_leaves63 | ... |
| D_mcs60 | ... |
| E_slow | ... |
| F_bag | ... |
folds1-3 평균 최저 후보 L* = ___

[작업 1 — 재블렌드]
선택 가중 (persistence/damped/L*) = ___ / 합 = ___
fold4 미사용 보장 방법 = ___ (한 줄)
folds1-3 재가중 = ___
봉인 fold4 재가중 = ___ / 현행 T1F 0.611959 대비 델타 = ___

[산출]
저장 파일 = ___
총 실행 시간 = ___
정지 조건 발생: 없음 | <사유>
```
