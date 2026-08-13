# 작업 지시: T5 — T2 재판정(세그먼트 최소표본 규칙) + T1 LGBM 풀 확장

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라** (샌드박스 네트워크 차단).
`P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용. 출력 디렉터리가 없으면 `mkdir` 하라.

**학습 재실행 금지.** 저장된 예측만 재조합·재채점한다. 이 사양 전체가 몇 분 안에 끝나야 한다.

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.t2_outer_gain      T2 blend(damped 0.5 + ExtraTrees 0.5) fold4 봉인 0.738364 vs T1F 0.754739,
                     델타 -0.016376. hs0bin:2(n=1) 악화 +0.0721 로 사전등록 cap 0.03 위반해 accepted=false
f.fold4_tail_empty   봉인 fold4(53건)에 hs0>3m 0건, 2.25-3m 1건. 272건 평가셋 전체도 bin2=28 bin3=14.
                     fold4 단독 세그먼트 cap 은 tail 에서 측정 불가
python scripts/g.py --p P3 check "세그먼트 최소표본 재판정 T1 lgbm 풀 확장"  → 진행 가능
```

fold4 hs0bin 사례수 (측정 완료, 재계산 불필요): `bin0=38 / bin1=14 / bin2=1 / bin3=0`.
hs0bin 경계는 `train_t2.py` 의 `pd.cut(cases.hs0, [0, 1.75, 2.25, 3, 99])`.

**이미 디스크에 있는 자산 — 다시 만들지 마라:**

- `P3/preds/oof_{damped,extra,lgbm_pooled,lgbm_lead,lgbm_station}_p3_t2.npy` (folds1-4 OOF)
- `P3/cache/p3_t2/ckpt/test_preds.npz` (전체데이터 재적합 test 예측, 후보별)
- `P3/preds/oof_{damped,lgbm}_p3_t1.npy`, `P3/preds/test_{damped,lgbm}_p3_t1.npy`
- `P3/preds/{oof,test}_final_p3_t1f.npy`
- `P3/cache/p3_t2/{folds.csv,report.json,selection_locked.json}`

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F fold4 봉인 통합 RMSE | `0.754739` | `P3/cache/p3_t2/report.json` `outer_t1f` |
| T1F all OOF 통합 RMSE | `0.750246` | `p3_t1f` 로그 |
| T2 선택 blend fold4 | `0.738364` | `report.json` `outer_selected` |
| T2 folds1-3 selection RMSE | `0.797937` | `report.json` `selection` |
| 지속성 로컬 통합 (272건) | `0.8329` | `P3/AGENTS.md` |

**여기 지정된 것 외에는 아무것도 바꾸지 마라.** 모델·피처·fold 정의·하이퍼파라미터 변경 금지.

## 실험 1 — 세그먼트 최소표본 규칙으로 T2 재판정

### 근거

T2 를 기각시킨 `hs0bin:2` 악화는 사례 1건에서 나왔다. `0.03` 을 해상할 수 없는 표본이다.

### 사양

학습 없이 `P3/preds/oof_*_p3_t2.npy` 와 `ckpt/test_preds.npz` 만 읽는다.
`report.json` 의 선택 가중(`damped 0.5 + extra 0.5`)을 **그대로 고정**한다. 재선택 금지.

세그먼트 채점을 `train_t2.py` 의 `segment_scores` 와 동일한 정의로 다시 하되,
**모든 세그먼트에 사례 수 `n` 을 함께 낸다.** 두 평가 뷰를 각각 낸다:

- **뷰 A (주 판정)** — folds1-4 pooled OOF (216사례). fold4 단독보다 4배 크다
- **뷰 B (보조)** — fold4 단독 (53사례). 이미 세그먼트 분해를 봤으므로 봉인이 부분 소진됐다.
  이 사실을 리포트에 명시하라

비교 대상은 `P3/preds/oof_final_p3_t1f.npy` 이며 case_id 로 정렬을 맞춘다
(T1F 는 272사례, T2 OOF 는 216사례 — 교집합에서 비교하고 교집합 크기를 보고하라).

### 수락 기준 (결과 보기 전 고정)

`n >= 20` 인 세그먼트만 cap 을 적용한다. `n < 20` 세그먼트는 **보고하되 판정에 쓰지 않는다.**

- 뷰 A 통합 RMSE 가 T1F 대비 `>= 0.002` 개선
- 뷰 A 에서 `n >= 20` 세그먼트 최대 악화 `<= 0.03`
- 뷰 B 통합 RMSE 가 T1F 대비 개선 (부호만, 크기 무관)

셋 다 만족하면 실험1 `accepted = true`.

## 실험 2 — blend 풀에 T1 LGBM 추가

### 근거

선택된 두 후보의 잔차 상관이 damped 0.979 / extra 0.979 로 거의 같은 오차를 낸다.
T1 LGBM 은 damped 와 잔차 상관 `0.872939` 로 더 낮아 혼합 이득 여지가 크다.

### 사양

후보 풀 = `{damped, extra, lgbm_pooled, lgbm_lead, lgbm_station, t1_lgbm}` (T2 5개 + T1 LGBM 1개).

**누수 검사 먼저.** `P3/preds/oof_lgbm_p3_t1.npy` 의 fold4 사례 예측이 fold4 시작 이전
데이터로만 학습된 모델에서 나왔는지 `P3/src/train.py` 의 fold 생성·학습 마스크를 읽고 확인하라.
strict-forward 가 아니면 **실험 2 를 중단하고 그 사실만 보고하라.** 점수를 내지 마라.

가중 선택은 **folds1-3 만** 사용한다. 0.1 grid, 최대 2개 후보 pair blend (T2 와 동일 절차).
fold4 는 선택에 쓰지 않는다.

### 수락 기준 (결과 보기 전 고정)

- 누수 검사 통과
- fold4 통합 RMSE 가 T1F(`0.754739`) 대비 `>= 0.002` 개선
- fold4 통합 RMSE 가 실험1 선택(`0.738364`) 대비 `>= 0.002` 개선
- 뷰 A 에서 `n >= 20` 세그먼트 최대 악화 `<= 0.03` (T1F 대비)

## 산출물 — 결과와 무관하게 항상

```
P3/preds/test_{damped,extra,lgbm_pooled,lgbm_lead,lgbm_station}_p3_t2.npy   # ckpt/test_preds.npz 에서 풀어 저장
P3/preds/{oof,test}_final_p3_t5.npy                                        # 최종 채택 조합 (실험1 또는 실험2 중 이긴 쪽)
P3/cache/p3_t5/report.json                                                 # 아래 보고 숫자 전부
```

실험1·2 **둘 다 accepted=false 면** `t5` 최종 파일은 만들지 말고 그 사실만 보고하라.

채택된 경우에만 제출 CSV 를 만든다:

```bash
python P3/src/submission_tools.py audit P3/subs/submission_p3_t5.csv --pred P3/preds/test_final_p3_t5.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t5.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
```

## 금지

- 학습 재실행 금지. 모델 재적합 금지. 피처 재생성 금지
- fold4 를 가중 선택에 쓰지 마라
- `n < 20` 세그먼트로 판정하지 마라. 보고만 하라
- 하이퍼파라미터 튜닝 금지
- 수락 기준을 결과에 맞춰 고치지 마라. 미달이면 미달로 보고하라
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[공통]
T1F ∩ T2 OOF 교집합 사례수 = ___

[실험1 — 재판정, 가중 damped 0.5 + extra 0.5 고정]
뷰A(folds1-4 pooled, n=216) T2 통합 RMSE = ___ / T1F = ___ / 델타 = ___
뷰A 리드별 T2 = +3h ___ / +6h ___ / +9h ___ / +12h ___ / +18h ___ / +24h ___
뷰A 세그먼트 악화 (T2 - T1F), n 포함, 악화 큰 순 5개 = ___
뷰A n>=20 세그먼트 최대 악화 = ___
뷰B(fold4, n=53) T2 = ___ / T1F = ___ / 델타 = ___
실험1 accepted = true | false  (미달 항목 명시)

[실험2 — 풀 확장]
누수 검사 결과 = strict-forward 확인 | 위반(중단)
folds1-3 선택 가중 = ___
fold4 통합 RMSE = ___ / vs T1F 델타 = ___ / vs 실험1 델타 = ___
뷰A 통합 RMSE = ___ / n>=20 세그먼트 최대 악화 = ___
후보별 잔차 상관 (선택 조합 대비) = ___
실험2 accepted = true | false  (미달 항목 명시)

[산출]
저장 파일 = ___
제출 CSV 생성 여부 = 예(경로·audit 판정·SHA-256) | 아니오(사유)
정지 조건 발생: 없음 | <사유>
```
