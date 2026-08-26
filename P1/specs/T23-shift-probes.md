# T23 — 분포 이동 대응 3종. 계절 정합 학습 / 연도 불변 feature / 랭크 앙상블

## 목적

test 는 **2025-12-31 ~ 2026-06-30, 사실상 1~6월 겨울·봄뿐**이다 (12월 378행, 1월 18005, 2월 18001,
3월 20791, 4월 18958, 5월 39466, 6월 53412). train 은 2024-01-01 ~ 2025-12-09 전 계절이며 7~11월이
가장 무겁다 (7월 96925, 8월 95815). 지금까지 모든 모델은 **여름 위주 train 으로 겨울·봄 test 를 예측**했다.
adversarial source AUC 0.850133 의 상위 구분 feature 가 `month`·`psal`·`doy_sin`·`clim_resid`·`layer_count`
인 것과 일치한다. 이 사양은 그 이동을 세 방향으로 친다. **inner 채점은 하지 않는다** — 판정기가 아니다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- public 대장: exp010 2.9832% → **0.741274**(최고) · exp012 3.0501% → 0.717975 · exp005 3.4791% → 0.717435 ·
  exp009 4.1808% → 0.698524 · exp011 2.1602% → 0.657820. 양성률 곡선 봉우리 3.0191%.
- **inner(fold1·2) 는 판정기가 아니다.** 임계에도, 후처리에도, 후처리 고정 모델 비교에도 부호가 전이되지 않았다
  (`f.inner_delta_no_transfer`, `f.inner_not_model_judge`). 이 사양에서 inner 를 계산하지 마라 — 시간 낭비다.
- 적분 feature 축은 폐쇄됐다 (`a.integrated_features`). 새 feature 를 만들지 마라.
- 기존 원시 확률: `P1/cache/exp005_raw_test.npy`(42 feature 전체 train), `P1/cache/exp012_raw_test.npy`(47 feature).
- base 42 feature 는 `P1/cache/exp001/train_features.pkl`, test 는 `make_features(test, ref_full)`.
- 학습 파라미터는 `P1/src/exp012_v1_full.py` 의 것을 그대로 쓴다 (700 tree, SEED 42, n_jobs 2).
- `numpy`·`pandas`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- fold 학습·inner 채점 금지. **전체 train 단일 모델만** 학습한다 (제출용).
- 봉인 fold3 판독 금지. 새 feature 생성 금지. 후처리는 `hole=48` 하나뿐, dilation·min-length·hysteresis 금지.
- 임계는 **test 양성률 0.0300 앵커**로만 정한다. 격자 탐색·inner argmax 금지.
- `P1/P1_data/` 읽기 전용. `SEED = 42`, `n_jobs <= 2`. 새 추상화 금지. 파일은 `P1/src/exp013_shift.py` 하나.
- **학습을 실행하지 마라.** 코드 완성 + 문법·소규모 검증까지만. 실행은 사람이 한다 (아래 §실행).

## 구현 — `P1/src/exp013_shift.py`, 스테이지 3개

공통: 원시 확률을 `P1/cache/{exp}_raw_test.npy` (float64) 로 저장.
임계는 `fill_runs(raw >= t, test_segments, 48)` 후 양성률이 0.0300 에 가장 가까운 `t` (0.001 격자, 동률이면 큰 t).
제출은 `P1/subs/{exp}.csv`, 풀 예측은 `P1/preds/test_{exp}.npy` (`exp005.store_score` 규약 float32),
메타는 `P1/cache/{exp}_meta.json` (t, hole, 양성수, 양성률, feature 목록, train 행수, sha256).
저장 후 두 검증 명령을 직접 실행하고 마지막 3줄을 보고한다:
`python P1/src/submission_tools.py audit P1/subs/{exp}.csv --pred P1/preds/test_{exp}.npy` 와
`python P1/src/check.py sub P1/subs/{exp}.csv`.

### `season` → exp013 — 계절 정합 학습

base 42 feature 그대로. **학습 행을 `month ∈ {12,1,2,3,4,5,6}` 로 제한**한다 (train 약 345,880행).
그 외 전부 exp012 와 동일. climatology reference 는 **전체 train 으로 적합한 기존 `ref_full`
(`P1/cache/exp001/reference.json`)을 그대로** 쓴다 — reference 를 다시 적합하지 마라.

### `invariant` → exp014 — 연도 의존 feature 제거

전체 train 행. feature 에서 **`month`, `doy_sin`, `doy_cos`, `clim_resid`, `layer_count` 5개를 뺀 37개**만 쓴다.
컬럼 수 37 을 assert 한다. 그 외 exp012 와 동일.

### `ensemble` → exp015 — 랭크 평균 (학습 없음)

`exp005_raw_test.npy` 와 `exp012_raw_test.npy` 를 각각 `scipy.stats.rankdata` 로 [0,1] 정규화한 뒤 평균한다.
그 평균을 raw score 로 보고 같은 임계·후처리 규칙을 적용한다.

## 수락 기준

| 항목 | 기준 |
|---|---|
| 기준선 | exp010 public F1 0.741274 (양성 5042, 2.9832%). 이 3종은 그것을 이기려는 시도다 |
| post_rate | 세 exp 전부 0.0300 ± 0.0005 |
| exp014 feature 수 | 37 (assert) |
| exp013 학습 행수 | 340,000 ~ 350,000 |
| audit / check sub | 세 exp 전부 `판정: OK` |
| 재실행 | 같은 스테이지 두 번 실행 시 CSV sha256 동일 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.**

| exp | n_features | train_rows | t_high | raw_rate | post_rate | n_positive | sha256(앞 10자) | audit | check_sub |

각 exp 의 `n_positive` 를 exp010(5042)·exp012(5155) 와 나란히 놓은 차이 행 1개만 추가한다.
산문·코드 붙여넣기 금지.

## 실행

```
python P1/src/exp013_shift.py season
python P1/src/exp013_shift.py invariant
python P1/src/exp013_shift.py ensemble
```
