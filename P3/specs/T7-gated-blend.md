# 작업 지시: T7 — 컨텍스트 게이트 혼합 가중

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라** (샌드박스 네트워크 차단).
`P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용. 출력 디렉터리가 없으면 `mkdir` 하라.

**학습 재실행 금지.** 저장된 예측만 재가중한다. 몇 분 안에 끝나야 한다.
`numpy` import 가 실패하면 `PYTHONPATH` 문제이므로 **멈추고 그 사실만 보고하라.**

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.t1f_bin0_flat      T1F 는 hs0<=1.75m(n=132, 제곱오차 기여 57%)에서 지속성 대비 -0.0132 밖에 못 번다.
                     hs0bin2/3 은 -0.32/-0.59. I-ORS(n=62)는 T1F 0.7946 > damped 단독 0.7834
f.wspd_missing_37pct 평가 216사례 중 79건이 최근 6h wspd 전결측. test 200건은 1건만 결측
f.t2_pooled_loss     401피처 GBDT 5후보 전부 damped 에 패배 (a.station_lead_experts 닫힘)
python scripts/g.py --p P3 check "정점 hs0 게이트 혼합 가중"  → 진행 가능
```

뷰A(folds1-4 pooled OOF 216사례) 세그먼트 실측 — 재계산 불필요:

| 세그먼트 | n | T1F | 지속성 | damped |
|---|---|---|---|---|
| hs0bin:0 (≤1.75) | 132 | 0.7373 | **0.7505** | 0.7769 |
| hs0bin:1 (1.75–2.25) | 49 | 0.8057 | 0.8400 | 0.8312 |
| hs0bin:2 (2.25–3) | 23 | 0.7302 | 1.0537 | 0.7916 |
| hs0bin:3 (>3) | 12 | 0.8802 | 1.4727 | 1.0458 |
| G-ORS | 78 | 0.7277 | 0.8623 | 0.7942 |
| I-ORS | 62 | 0.7946 | 0.8116 | **0.7834** |
| S-ORS | 76 | 0.7667 | 0.9007 | 0.8415 |

`hs0bin:0` 에서 지속성이 damped 를 이기고, I-ORS 에서 damped 단독이 T1F 를 이긴다.
전역 단일 가중이 이 두 구간에서 틀렸다는 것이 가설이다.

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F 뷰A 통합 | `0.761125519778535` | `P3/cache/p3_t6/report.json` `work0.t1f.total` |
| 지속성 뷰A 통합 | `0.8619732221712867` | 같은 파일 |
| damped 뷰A 통합 | `0.8081531461489104` | 같은 파일 |
| T1F fold4 봉인 | `0.754739` | `P3/cache/p3_t2/report.json` |

**여기 지정된 것 외에는 아무것도 바꾸지 마라.** fold 정의·평가 사례 집합 고정.

사용할 저장 예측 (모두 존재. 없으면 멈추고 보고):

```
P3/preds/oof_damped_p3_t1.npy      P3/preds/test_damped_p3_t1.npy
P3/preds/oof_lgbm_p3_t1.npy        P3/preds/test_lgbm_p3_t1.npy
P3/preds/oof_eqrelax_p3_t6.npy     P3/preds/test_eqrelax_p3_t6.npy
P3/preds/oof_final_p3_t1f.npy      P3/preds/test_final_p3_t1f.npy
```

지속성 후보는 별도 파일 없이 `hs_pred(τ) = hs0` 로 그 자리에서 만든다 (전 리드 동일).

## 실험 — 게이트별 혼합 가중

### 근거

혼합 가중이 전역 상수라 지속성이 이기는 저파고 구간과 damped 가 이기는 I-ORS 에서 손해를 본다.
가중을 컨텍스트에서 관측 가능한 그룹별로 나누면 그 손해를 회수한다.

### 사양

후보 풀 4개 고정: `{persistence, damped, lgbm_t1, eqrelax}`. 다른 후보를 넣지 마라.

가중은 각 그룹 안에서 **단체(simplex, 합=1, 음수 없음)** 다. 4개 변형을 비교한다:

| 변형 | 그룹 정의 | 그룹 수 | grid |
|---|---|---|---|
| V1 (대조군) | 없음 (전역 1개) | 1 | 0.05 |
| V2 | station | 3 | 0.25 |
| V3 | hs0 그룹 (`hs0 <= 1.75` / `hs0 > 1.75`) | 2 | 0.25 |
| V4 | station × hs0 그룹 | 6 | 0.25 |

- **가중 선택은 folds1-3 사례만** 쓴다. fold4 는 선택에 절대 쓰지 마라
- 그룹 판정 변수(`station`, `hs0`)는 해당 case 의 컨텍스트에서만 온다. 다른 case 통계 금지
- folds1-3 에서 어떤 그룹의 사례 수가 `< 15` 면 그 그룹은 **전역 가중으로 폴백**하고
  폴백한 그룹 이름과 사례 수를 보고하라
- `P3/src/blend_t7.py` 한 파일. 클래스·설정파일·로깅 프레임워크 금지

### 수락 기준 (결과 보기 전 고정)

- **(대조군 필수)** V1 뷰A 통합이 T1F(`0.761126`) 와 `|델타| <= 0.005`.
  벗어나면 하네스 오류이므로 **점수 해석 없이 그 사실만 보고하고 중단**하라
- (채택) 뷰A 통합이 T1F 대비 `>= 0.003` 개선
- (채택) 뷰A `n >= 20` 세그먼트 최대 악화 `<= 0.03` (T1F 대비)
- (채택) fold4 통합이 T1F(`0.754739`) 대비 악화하지 않음 (`<= 0.754739`)
- **동점·근접 시 파라미터가 적은 변형을 택한다.** 두 변형의 뷰A 차이가 `0.002` 미만이면
  그룹 수가 적은 쪽이 이긴다. V1 < V3 < V2 < V4 순서

## 산출물 — 결과와 무관하게 항상

```
P3/preds/{oof,test}_persistence_p3_t7.npy
P3/cache/p3_t7/report.json      # 변형별·그룹별 가중과 점수 전부
```

채택된 변형이 있을 때만:

```
P3/preds/{oof,test}_final_p3_t7.npy
P3/subs/submission_p3_t7.csv
python P3/src/submission_tools.py audit P3/subs/submission_p3_t7.csv --pred P3/preds/test_final_p3_t7.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t7.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
```

## 금지

- 학습·모델 재적합·피처 생성 금지
- fold4 를 가중 선택에 쓰지 마라
- 후보 풀에 다른 예측을 추가하지 마라. grid 를 사양보다 촘촘하게 만들지 마라
- 그룹 경계(`1.75`)를 스윕하지 마라
- 수락 기준을 결과에 맞춰 고치지 마라. 미달이면 미달로 보고하라
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[대조군]
V1 뷰A 통합 = ___ / T1F 0.761126 대비 델타 = ___ / 대조군 통과 = 예|아니오
V1 전역 가중 (persistence/damped/lgbm_t1/eqrelax) = ___

[변형별 뷰A]
V2 통합 = ___ / V3 통합 = ___ / V4 통합 = ___
T1F 대비 델타 = V2 ___ / V3 ___ / V4 ___

[선택 변형]
선택 = V_ (동점규칙 적용 여부 명시)
그룹별 가중 = ___ (그룹명: persistence/damped/lgbm_t1/eqrelax)
폴백 그룹 (folds1-3 n<15) = ___

[선택 변형 상세]
뷰A 리드별 = +3h ___ / +6h ___ / +9h ___ / +12h ___ / +18h ___ / +24h ___
뷰A 세그먼트 델타 (T1F 대비), n 포함, 악화 큰 순 5개 = ___
뷰A n>=20 최대 악화 = ___ (세그먼트명)
hs0bin:0 (n=132) = ___ (T1F 0.7373 / 지속성 0.7505 대비)
station:I-ORS (n=62) = ___ (T1F 0.7946 / damped 0.7834 대비)
fold4 통합 = ___ / T1F 0.754739 대비 델타 = ___
accepted = true | false  (미달 항목 명시)

[산출]
저장 파일 = ___
제출 CSV 생성 여부 = 예(경로·audit 판정·SHA-256) | 아니오(사유)
정지 조건 발생: 없음 | <사유>
```
