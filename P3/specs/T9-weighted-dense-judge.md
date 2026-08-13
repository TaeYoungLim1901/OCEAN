# 작업 지시: T9 — 분포 재가중 조밀 판정기로 제출 가중 재선택

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라**. `P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용.
출력 디렉터리가 없으면 `mkdir` 하라.

**학습 재실행 금지.** 저장된 예측만 재가중·재채점한다. 몇 분 안에 끝나야 한다.

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.thinned_judge_inverts  fold4 씨닝(53)/조밀(1163) 판정 역전. persistence/damped/lgbm_t1
                         씨닝 0.783322/0.770747/0.775590 vs 조밀 0.900254/0.724137/0.664543
f.dense_judge_bias       씨닝 평가셋만 공식 t0 hs 분위수와 일치. 조밀 fold4 는 hs0>3m 포함, 씨닝 fold4 는 0건
f.gated_weight_overfit   게이트(그룹별) 가중은 조밀 판정기로도 V1 대비 최대 0.000983 개선 — 무의미
f.wspd_missing_37pct     씨닝 216사례 중 79건 wspd 전결측. test 200건은 1건만 결측
```

조밀 판정기는 분산이 낮고(5블록 표준편차 0.10~0.13) 편향이 있다. 이 사양은 편향만 제거한다.

**게이트별 가중을 다시 시도하지 마라.** 전역 가중 하나만 고른다 (`f.gated_weight_overfit`).

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F 씨닝 fold4 | `0.754739` | `P3/cache/p3_t2/report.json` |
| T1F 씨닝 뷰A(216) | `0.761126` | `P3/cache/p3_t6/report.json` |
| 조밀 선택 V1 가중 | persistence `0.05` / damped `0.55` / lgbm_t1 `0.40` | `P3/cache/p3_t8/report.json` |
| 조밀 V1 fold4 조밀 / 씨닝 | `0.684661` / `0.747721` | 같은 파일 |
| 조밀 fold4 사례수 | `1163` (예측행 6978) | 같은 파일 |

사용할 저장 자산:

```
P3/preds/oof_dense_{persistence,damped,lgbm_t1}_p3_t8.npy   # folds1-4 조밀 OOF
P3/cache/p3_t8/dense_index.csv                              # case_id, station, fold, hs0
P3/preds/test_{damped,lgbm}_p3_t1.npy                       # test 예측 (전체데이터 적합)
P3/preds/{oof,test}_final_p3_t1f.npy
P3/P3_data/test_index.csv
```

**여기 지정된 것 외에는 아무것도 바꾸지 마라.**

## 작업 1 — T1F 실제 가중 복원

`P3/src/select_t1f.py` 와 `P3/cache/` 산출물에서 T1F 가 쓴 후보 가중 벡터를 읽어 그대로 보고하라.
추정하지 마라. 파일에서 못 찾으면 **멈추고 보고하라.**

이 가중을 조밀 OOF 에 적용한 것을 `T1F_dense` 라 하고 이후 모든 비교의 기준선으로 쓴다.

## 작업 2 — 분포 재가중

조밀 사례에 중요도 가중 `w_i` 를 준다. 목표 분포는 **씨닝 평가셋**(`cases.eval_split`, 272건)이다.
씨닝셋만 공식 t0 hs 분위수와 일치하기 때문이다.

- 층 정의: `station`(3) × `hs0` 4분위(씨닝셋 전체에서 계산한 경계). 총 12층
- `w_i = (씨닝셋 해당 층 비율) / (조밀셋 해당 층 비율)`. 조밀셋 층 비율이 0이면 그 층은 버린다
- **가중 상한 `10.0`** 으로 클립한 뒤 평균 1이 되게 재정규화한다
- 유효표본수 `ESS = (Σw)² / Σw²` 를 fold 별로 보고하라

RMSE 는 사례 가중을 반영한다: `sqrt(Σ w_i (y-p)² / Σ w_i)`, 리드는 동일 가중.

## 작업 3 — 판정기 3종 비교 (측정만)

`{persistence, damped, lgbm_t1, T1F_dense}` 4개를 fold4 에서 세 판정기로 채점한다:
**씨닝(53건) / 조밀 비가중(1163건) / 조밀 재가중**. 순위가 바뀌는 쌍을 전부 나열하라.

## 실험 4 — 재가중 조밀로 전역 가중 재선택

- 후보 풀 `{persistence, damped, lgbm_t1}` 고정. simplex, 합 1, 음수 없음, grid `0.05`
- 선택은 **folds1-3 조밀 재가중**만 쓴다. fold4 는 선택에 절대 쓰지 마라
- 선택된 가중을 `W_new` 라 한다

### 수락 기준 (결과 보기 전 고정)

- **(필수)** fold4 재가중 `ESS >= 300`. 미만이면 재가중이 실패한 것이므로
  숫자만 보고하고 `accepted=false` 로 끝내라
- **(필수)** `W_new` 각 성분이 `[0, 1]` 이고 합이 `1.0 ± 1e-6`
- (채택) `W_new` 가 fold4 **재가중 조밀**에서 `T1F_dense` 대비 `>= 0.005` 개선
- (채택) `W_new` 가 fold4 **씨닝**에서 T1F(`0.754739`) 대비 `+0.002` 를 초과해 악화하지 않음
- (채택) fold4 재가중 조밀에서 `n >= 100` 세그먼트 최대 악화 `<= 0.03` (`T1F_dense` 대비)

## 산출물 — 결과와 무관하게 항상

```
P3/cache/p3_t9/report.json          # 아래 보고 숫자 전부, 층별 가중과 ESS 포함
P3/cache/p3_t9/dense_weights.csv    # case_id, stratum, w
```

채택된 경우에만:

```
P3/preds/test_final_p3_t9.npy       # W_new 를 test_{damped,lgbm}_p3_t1.npy 와 persistence(hs0) 에 적용
P3/subs/submission_p3_t9.csv
python P3/src/submission_tools.py audit P3/subs/submission_p3_t9.csv --pred P3/preds/test_final_p3_t9.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t9.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
```

test 쪽 persistence 후보는 test 컨텍스트의 `hs0` 를 전 리드에 그대로 쓴다.

## 금지

- 학습·모델 재적합·피처 생성 금지
- **그룹별·세그먼트별 가중 금지.** 전역 가중 하나만 고른다
- fold4 를 가중 선택·층 경계 계산에 쓰지 마라. 층 경계는 씨닝 평가셋 전체에서 계산한다
- 가중 상한 `10.0`·grid `0.05`·층 정의를 스윕하지 마라
- 후보를 추가하지 마라 (eqrelax 는 T7 에서 전역 가중 0.00 이었다)
- 기존 T1/T1F/T2/T8 예측과 제출 CSV 재생성 금지
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- 수락 기준을 결과에 맞춰 고치지 마라. 미달이면 미달로 보고하라
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[작업1]
T1F 실제 가중 = ___ (출처 파일·필드)
T1F_dense fold4 조밀 비가중 = ___ / 재가중 = ___ / 씨닝 = ___

[작업2]
층별 가중 최소/최대/클립된 층 수 = ___
fold 별 ESS = fold1 ___ / fold2 ___ / fold3 ___ / fold4 ___
fold4 재가중 ESS >= 300 = 예|아니오

[작업3 — 판정기 3종, fold4]
| 후보 | 씨닝(53) | 조밀 비가중(1163) | 조밀 재가중 |
| persistence | ___ | ___ | ___ |
| damped | ___ | ___ | ___ |
| lgbm_t1 | ___ | ___ | ___ |
| T1F_dense | ___ | ___ | ___ |
순위가 바뀌는 쌍 = ___

[실험4]
W_new (persistence/damped/lgbm_t1) = ___
folds1-3 재가중 선택 RMSE = ___
fold4 재가중 W_new = ___ / T1F_dense 대비 델타 = ___
fold4 씨닝 W_new = ___ / T1F 0.754739 대비 델타 = ___
fold4 재가중 리드별 = +3h ___ / +6h ___ / +9h ___ / +12h ___ / +18h ___ / +24h ___
fold4 재가중 n>=100 세그먼트 최대 악화 = ___ (세그먼트명, n)
accepted = true | false  (미달 항목 명시)

[산출]
저장 파일 = ___
제출 CSV 생성 여부 = 예(경로·audit 판정·SHA-256) | 아니오(사유)
정지 조건 발생: 없음 | <사유>
```
