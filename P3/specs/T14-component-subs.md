# 작업 지시: T14 — 판정기 교정용 단독 성분 제출 CSV 생성

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라**. `P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용.

**학습·모델 적합 금지. 저장된 test 예측을 CSV 로 포장하고 게이트만 통과시킨다.**

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.thinned_judge_inverts  씨닝 53사례 판정기는 damped 를, 조밀 재가중 판정기는 lgbm 을 1등으로 본다
f.t1f_weights            현행 제출 T1F = persistence 0.00 / damped 0.35 / lgbm_t1 0.65
```

8/20 부터 하루 1회 답안 제출이 열린다 (`P3/AGENTS.md:34`). 초반 3회를 **판정기 선택 측정**에
쓴다 (`P3/SUBMISSION_PLAN.md` 1기). 그 3회 중 2개가 아직 CSV 로 없다.

## 기준선 — 사용할 자산

```
P3/preds/test_damped_p3_t1.npy   # 전체데이터 적합 damped 성분 (T1F 가 쓰는 것과 동일 파일)
P3/preds/test_lgbm_p3_t1.npy     # 전체데이터 적합 LGBM 성분 (동일)
P3/src/inference_t1f.py          # CSV 포장 규약 참조용
P3/P3_data/test_index.csv
```

없거나 형상이 안 맞으면 **멈추고 보고하라. 비슷한 다른 파일로 대체하지 마라.**

현행 제출 `P3/subs/submission_p3_t1f.csv` SHA-256 은
`2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` 이며 **바뀌면 안 된다.**

## 작업 — CSV 2개 생성

`P3/src/make_component_subs.py` 를 만든다. 위 두 `.npy` 를 각각 그대로 제출 형식으로 포장한다.
**혼합·보정·클리핑 외 어떤 변형도 하지 마라.** (0~30m 범위 클리핑은 필요 시 허용, 적용 여부를 보고)

```
P3/subs/submission_p3_lgbm_only.csv
P3/subs/submission_p3_damped_only.csv
```

열 순서 `case_id,station,lead_h,hs_pred`, 1,200행, 값 0~30m 유한 (`P3/AGENTS.md:42`).
`inference_t1f.py` 와 동일한 행 순서·포맷 규약을 따른다.

## 수락 기준 (결과 보기 전 고정)

- 두 CSV 모두 1,200행이며 `case_id,station,lead_h` 조합이 `test_index.csv` 와 정확히 일치
- 각 CSV 의 `hs_pred` 가 대응 `.npy` 와 `max |델타| <= 1e-9` (클리핑으로 바뀐 행 수는 별도 보고)
- `submission_p3_t1f.csv` SHA-256 이 `2E498428…3470C1` 로 **불변**
- 두 CSV 각각에 대해 아래 전부 `OK`:

```bash
python P3/src/submission_tools.py audit P3/subs/submission_p3_lgbm_only.csv --pred P3/preds/test_lgbm_p3_t1.npy
python P3/src/submission_tools.py audit P3/subs/submission_p3_damped_only.csv --pred P3/preds/test_damped_p3_t1.npy
python scripts/preflight.py sub P3/subs/submission_p3_lgbm_only.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python scripts/preflight.py sub P3/subs/submission_p3_damped_only.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/context_static_check.py
```

## 금지

- 학습·재적합·피처 생성 금지
- 기존 `P3/subs/*.csv` 3개 수정·삭제 금지
- `P3/preds/_backup_pre_maskfix/` 수정·삭제 금지
- `P3/experiments/log.md` 와 `P3/graph.json` **수정 금지** (다른 작업이 동시에 쓴다)
- `P3/src/tune_lgbm_dense.py`·`P3/cache/p3_t13/`·`P3/preds/*_p3_t13.npy` 건드리지 마라 (동시 작업 중)
- `P1/`·`P2/`·루트 파일 수정 금지
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- **결론 문장 금지. 숫자와 표만.**

## 완료 보고 형식

```
생성 파일 = ___
행 수 = lgbm ___ / damped ___
test_index 키 일치 = 예|아니오
npy 대비 max|델타| = lgbm ___ / damped ___ / 클리핑된 행 수 = lgbm ___ / damped ___
SHA-256 = lgbm ___ / damped ___
submission_p3_t1f.csv SHA-256 = ___ / 2E498428…3470C1 와 동일 = 예|아니오
게이트 = audit(lgbm) ___ / audit(damped) ___ / preflight sub(lgbm) ___ / preflight sub(damped) ___ / context_static_check ___
정지 조건 발생: 없음 | <사유>
```
