# 작업 지시: T6 — 풍속 평형 완화 모델 + T1F 약점 분해 + 마스크 누수 수정

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라** (샌드박스 네트워크 차단).
`P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용. 출력 디렉터리가 없으면 `mkdir` 하라.

학습이 있지만 파라미터가 소수라 짧아야 한다. **10분을 넘길 것 같으면 멈추고 보고하라.**

## 배경 — 이미 검증된 사실 (재조사 금지)

```
p.pm_equilibrium         완전발달 해상 Pierson-Moskowitz: Hs_eq ≈ 0.0246·U10². 현재 hs 와 Hs_eq 의 차이가 성장/감쇠 방향을 준다
a.equilibrium_relax      [open] hs 를 Hs_eq(U) 로 완화시키는 1차 미분방정식 피팅
f.t2_pooled_loss         401피처 direct-delta GBDT 5후보 전부 damped 에 패배. a.station_lead_experts 닫힘
f.t1_symmetric_mask_leak train.py 대칭 abs(hours) 마스크로 fold4 학습행 6830 vs strict-forward 6828
f.fold4_tail_empty       fold4(53건) hs0>3m 0건, 2.25-3m 1건. pooled 216건 기준 hs0bin:2 는 n=23
python scripts/g.py --p P3 check "풍속 평형 완화 hs 감쇠"  → 진행 가능
```

평가 뷰 정의는 T5 와 동일하다. **뷰A = folds1-4 pooled OOF 216사례가 주 판정이다.**
fold4 단독(53건)은 세그먼트 분해가 이미 노출돼 부분 소진됐으므로 보조로만 쓴다.

hs0bin 경계 `[0, 1.75, 2.25, 3, 99]`, regime 은 `hs_6h_delta` 의 ±0.10 (`train_t2.py` 정의 그대로).

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F 뷰A(pooled 216) 통합 RMSE | `0.761126` | `P3/cache/p3_t5/report.json` |
| T1F fold4 봉인 | `0.754739` | `P3/cache/p3_t2/report.json` |
| damped 단독 fold평균 | `0.819948` | `P3/cache/p3_t2/report.json` `candidate_rmse` |
| 지속성 로컬 통합(272건) | `0.8329` | `P3/AGENTS.md` |

**여기 지정된 것 외에는 아무것도 바꾸지 마라.** fold 정의(`P3/cache/p3_t2/folds.csv`)와
평가 사례 집합은 고정이다. 새 피처 캐시를 만들지 마라.

## 작업 0 — T1F 약점 분해 (측정만)

`P3/preds/oof_final_p3_t1f.npy` 를 뷰A 216사례에서 세그먼트 분해한다.
비교군은 지속성(`hs_pred = hs0`)과 damped(`P3/preds/oof_damped_p3_t2.npy`).
**모든 세그먼트에 사례 수 `n` 을 붙여라.** 이 표가 T6 이후 실험의 표적을 정한다.

## 작업 1 — 마스크 누수 수정

`P3/src/train.py` 의 대칭 `abs(hours)` 학습 마스크를 strict-forward 로 고친다:
validation fold 시작 시각보다 **78시간 이상 이전** t0 만 학습에 쓴다 (`train_t2.py` 의
`forward_train_mask` 와 동일 의미). 미래 방향 행은 전부 배제한다.

수정 후 fold4 학습행 수가 `6828` 인지 출력하라. `inference.py` 는 건드리지 마라.
**T1/T1F 예측을 재생성하지 마라.** 기존 `P3/preds/*_p3_t1*.npy` 와
`P3/subs/submission_p3_t1f.csv` 는 그대로 둔다. 이 수정은 T6 이후 학습에만 적용된다.

## 실험 1 — 1차 완화 평형 모델 (`eqrelax`)

### 근거

damped persistence 는 hs 를 정점 평균 `μ(station)` 으로 끌어당긴다. 실제 파고는 바람이 만드는
평형값 `Hs_eq(U)` 로 완화한다. 바람이 죽으면 μ 가 아니라 낮은 `Hs_eq` 로, 바람이 세면 μ 위로 간다.
`hs0bin:2` 이상과 falling regime 의 실패는 끌어당기는 목표가 틀린 데서 온다는 가설이다.

### 사양

컨텍스트 289행에서 `hs`, `wspd` 만 쓴다 (`cases.COLS` 순서 고정).

리드 `τ` 별 예측:

```
Hs_eq = a_st * U_eff^2 + b_st              # U_eff = 컨텍스트 말단 wspd 의 로버스트 대표값
pred(τ) = Hs_eq + (hs0 - Hs_eq) * exp(-τ / T_st)
```

- `U_eff` 정의는 **최근 6시간 wspd 중앙값** 하나로 고정한다. 대안을 스윕하지 마라
- 파라미터는 정점(station)별 `a_st`, `b_st`, `T_st` **3개씩, 총 9개뿐**이다
- `a_st` 초기값은 PM 관계 `0.0246`, `b_st` 초기값 0, `T_st` 초기값 12시간
- 적합은 **folds1-3 학습행만** 사용한다. 6리드 통합 제곱오차 최소화 (`scipy.optimize.least_squares`)
- 결측 wspd 는 컨텍스트 내부 값으로만 채운다. 전부 결측이면 그 사례는 damped 로 폴백하고
  폴백 사례 수를 보고하라
- 예측은 `clip(0, 30)`

`P3/src/train_eqrelax.py` **한 파일**로 만든다. 기존 파일에 끼워넣지 마라.
클래스·설정파일·로깅 프레임워크 금지. `print` 와 `np.save` 로 충분하다.

### 수락 기준 (결과 보기 전 고정)

`eqrelax` 는 단독으로 T1F 를 이길 필요가 없다. **혼합 이득**으로 본다.

- (필수) 뷰A 에서 `eqrelax` 단독 통합 RMSE `<= 0.85` — 이보다 나쁘면 구현 오류로 보고 중단
- (필수) 적합된 `T_st` 가 세 정점 모두 `1 <= T_st <= 200` 시간 범위 안 — 경계에 붙으면 보고
- (채택) `{T1F, eqrelax}` 2원 0.05 grid blend 를 **folds1-3 에서만** 선택했을 때,
  뷰A 통합 RMSE 가 T1F(`0.761126`) 대비 `>= 0.003` 개선
- (채택) 뷰A 에서 `n >= 20` 세그먼트 최대 악화 `<= 0.03` (T1F 대비)
- (참고, 판정에 쓰지 않음) fold4 단독 델타, `n < 20` 세그먼트

## 산출물 — 결과와 무관하게 항상

```
P3/preds/oof_eqrelax_p3_t6.npy
P3/preds/test_eqrelax_p3_t6.npy
P3/cache/p3_t6/report.json
```

채택된 경우에만 `P3/preds/{oof,test}_final_p3_t6.npy` 와 제출 CSV 를 만든다:

```bash
python P3/src/submission_tools.py audit P3/subs/submission_p3_t6.csv --pred P3/preds/test_final_p3_t6.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t6.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
```

## 금지

- fold4 를 파라미터 적합·가중 선택에 쓰지 마라
- `U_eff` 정의·초기값·리드 집합·fold 정의를 스윕하지 마라. 하이퍼파라미터 튜닝 금지
- 정점별 3파라미터를 넘겨 파라미터를 늘리지 마라
- 기존 T1/T1F 예측·제출 CSV 재생성 금지
- 새 피처 캐시 생성 금지. 401피처(`features_train.npy`)를 이 실험에 쓰지 마라
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- 수락 기준을 결과에 맞춰 고치지 마라. 미달이면 미달로 보고하라
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[작업0 — T1F 뷰A 세그먼트 분해, n 포함]
T1F 통합 = ___ / 지속성 = ___ / damped = ___
리드별 T1F = +3h ___ / +6h ___ / +9h ___ / +12h ___ / +18h ___ / +24h ___
hs0bin 별 (n 포함) T1F / 지속성 / damped = bin0 ___ / bin1 ___ / bin2 ___ / bin3 ___
regime 별 (n 포함) T1F / 지속성 / damped = falling ___ / stable ___ / rising ___
station 별 (n 포함) T1F = G-ORS ___ / I-ORS ___ / S-ORS ___
T1F 가 지속성에 지는 세그먼트 = ___

[작업1 — 마스크 수정]
수정 후 fold4 학습행 = ___ (기대 6828)
수정 파일 = ___

[실험1 — eqrelax]
적합 파라미터 = G-ORS a ___ b ___ T ___ / I-ORS a ___ b ___ T ___ / S-ORS a ___ b ___ T ___
wspd 전결측 폴백 사례수 = ___
뷰A eqrelax 단독 통합 = ___
뷰A eqrelax 리드별 = +3h ___ / +6h ___ / +9h ___ / +12h ___ / +18h ___ / +24h ___
eqrelax 잔차 상관 vs T1F = ___ / vs damped = ___
folds1-3 선택 가중 (T1F : eqrelax) = ___
뷰A blend 통합 = ___ / T1F 대비 델타 = ___
뷰A n>=20 세그먼트 최대 악화 = ___ (세그먼트명 포함)
fold4 blend = ___ / T1F 대비 델타 = ___  (참고)
accepted = true | false  (미달 항목 명시)

[산출]
저장 파일 = ___
제출 CSV 생성 여부 = 예(경로·audit 판정·SHA-256) | 아니오(사유)
정지 조건 발생: 없음 | <사유>
```
