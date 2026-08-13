# 작업 지시: T8 — 조밀 평가셋으로 판정기 해상도 올리기

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라** (샌드박스 네트워크 차단).
`P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용. 출력 디렉터리가 없으면 `mkdir` 하라.

**학습이 있다. 코드를 만들고 fold 1 하나만 실행해 시간을 재라. 전체 실행은 사람이 한다.**
`--max-folds` 로 쪼갤 수 있어야 한다 (`train_t2.py` 의 체크포인트 방식과 동일).
10분 안에 fold1 이 안 끝나면 멈추고 보고하라.

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.gated_weight_overfit  게이트 혼합 가중은 그룹 수에 비례해 선택셋 개선·봉인 악화.
                        folds1-3 163사례로는 전역 스칼라 하나 이상을 못 고른다
f.fold4_tail_empty      봉인 fold4(53건)에 hs0>3m 0건, 2.25-3m 1건
f.t2_pooled_loss        401피처 GBDT 5후보 전부 damped 에 패배 (a.station_lead_experts 닫힘)
f.t1_symmetric_mask_leak  train.py 대칭 마스크는 T6 에서 strict-forward 로 수정 완료 (fold4 학습행 6828)
python scripts/g.py --p P3 check "조밀 평가셋 판정기 해상도"  → 진행 가능
```

- `cases` 는 8121행이고 평가셋은 78h 그리디 씨닝으로 272건이다 (`cases.eval_split`).
  씨닝은 **test 사례 간격 재현**이 목적이며, 학습은 이미 조밀 사례를 쓴다 (fold4 학습행 6828).
- 즉 판정 해상도만 인위적으로 53사례로 눌려 있다.

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F 씨닝 뷰A(216) | `0.761125519778535` | `P3/cache/p3_t6/report.json` |
| T1F 씨닝 fold4(53) | `0.754739` | `P3/cache/p3_t2/report.json` |
| 지속성 씨닝 뷰A | `0.8619732221712867` | `P3/cache/p3_t6/report.json` |

**여기 지정된 것 외에는 아무것도 바꾸지 마라.** fold 경계(`P3/cache/p3_t2/folds.csv` 의
정점별 시간 컷오프)는 그대로 쓴다. 모델·하이퍼파라미터를 바꾸지 마라.

## 작업 1 — 조밀 OOF 생성

`P3/src/train_dense.py` 한 파일을 만든다.

- fold 정의는 `folds.csv` 의 정점별 컷오프를 그대로 쓴다. **fold 경계를 새로 만들지 마라.**
- 학습 마스크는 `train_t2.py` 의 `forward_train_mask` (validation 시작 78h 이전만).
  T6 에서 고친 `train.py` 와 동일 의미여야 한다
- **검증 대상만 바꾼다**: 씨닝된 사례가 아니라 그 fold 구간의 **모든 사례**에 예측한다
- 후보 3개만: `persistence`(=hs0), `damped`(`train_t2.py` 의 `fit_damped`/`predict_damped` 그대로),
  `lgbm_t1`(`train.py` 의 LightGBM 설정·피처 그대로. 401피처 캐시를 쓰지 마라)
- fold 별 체크포인트 `P3/cache/p3_t8/ckpt/fold{f}.npz` 저장·재개, `--max-folds N`, `--n-jobs`(기본 6)
- 진행 출력 `print(..., flush=True)`, 상단에 `sys.stdout.reconfigure(encoding="utf-8")`

저장:

```
P3/preds/oof_dense_{persistence,damped,lgbm_t1}_p3_t8.npy
P3/cache/p3_t8/dense_index.csv     # case_id, station, fold, hs0 — 예측 행 순서와 정확히 일치
```

씨닝 사례가 조밀셋의 부분집합임을 확인하고 교집합 크기를 보고하라.

## 작업 2 — 판정기 해상도 비교 (측정만)

같은 후보 3개로 **씨닝 뷰**와 **조밀 뷰** 두 판정기를 나란히 채점한다.
목적은 점수 개선이 아니라 **판정기가 얼마나 시끄러운지** 재는 것이다.

- fold4 에서 후보별 RMSE 를 씨닝(53건)·조밀(전체)로 각각 낸다
- 조밀 fold4 를 정점별 시간순 5블록으로 쪼개 블록별 RMSE 를 내고 **블록 간 표준편차**를 보고한다.
  이것이 판정기 잡음의 크기다
- 씨닝 fold4 에서 두 후보 차이가 `0.002` 일 때 그 차이가 블록 표준편차의 몇 배인지 계산하라

## 실험 3 — 조밀 선택으로 T7 게이트 재검정

T7 의 4변형(V1 전역 / V2 station / V3 hs0 / V4 station×hs0)을 **똑같이** 다시 만든다.
후보 풀은 작업1 의 3개(`persistence, damped, lgbm_t1`)로 한다. eqrelax 는 제외한다
(T7 에서 전역 가중 0.00).

- 가중 선택: **folds1-3 의 조밀 사례** (씨닝 아님)
- 판정: **fold4 조밀** 이 주 판정, **fold4 씨닝(53건)** 이 보조
- grid 는 T7 과 동일 (V1 은 0.05, V2/V3/V4 는 0.25)

### 수락 기준 (결과 보기 전 고정)

- **(대조군 필수)** V1 조밀 선택 가중이 T7 의 V1 전역 가중과 각 성분 `±0.15` 이내.
  벗어나면 하네스 차이이므로 그 사실을 보고하라 (중단은 하지 말 것)
- (채택) 어떤 변형이 fold4 **조밀**에서 V1 대비 `>= 0.003` 개선
- (채택) 그 변형이 fold4 **씨닝**에서 T1F(`0.754739`) 대비 악화하지 않음
- (채택) 조밀 fold4 `n >= 100` 세그먼트 최대 악화 `<= 0.03` (V1 대비)
- 동점 규칙은 T7 과 동일: 차이 `0.002` 미만이면 그룹 수 적은 쪽 (V1 < V3 < V2 < V4)

## 실행 — fold 1 만

```bash
python P3/src/train_dense.py --max-folds 1 --n-jobs 6
python P3/src/train_dense.py --max-folds 1 --n-jobs 6      # 재개 확인
```

**fold 2 이상과 작업2·실험3 을 실행하지 마라.** 조밀 OOF 가 없으면 계산할 수 없다.
작업2·실험3 코드는 작성하되 `--stage report` 같은 별도 진입으로 사람이 나중에 돌리게 하라.

## 금지

- fold 경계를 새로 만들거나 바꾸지 마라
- 401피처 캐시(`features_train.npy`)를 이 실험에 쓰지 마라
- fold4 를 가중 선택에 쓰지 마라
- 하이퍼파라미터 튜닝 금지. 후보를 추가하지 마라
- 씨닝 평가셋(`cases.eval_split`)의 정의를 바꾸지 마라. 조밀 뷰는 **추가**이지 대체가 아니다
- 기존 T1/T1F/T2 예측·제출 CSV 재생성 금지
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[작업1 — fold1 만]
조밀 fold1 검증 사례수 = ___ (씨닝 55건 대비 배수 ___)
씨닝 ⊂ 조밀 확인 = 예|아니오 / 교집합 = ___
fold1 최초 계산 벽시계 = ___ s / 단계별 = damped ___ s / lgbm_t1 ___ s
fold1 재개 벽시계 = ___ s / 예측 동일성 np.allclose(atol=0) = True|False
fold1 조밀 후보별 RMSE = persistence ___ / damped ___ / lgbm_t1 ___
fold1 씨닝 후보별 RMSE = persistence ___ / damped ___ / lgbm_t1 ___
fold2-4 예상 벽시계 (학습행수 비율 환산) = ___ s

[작성만, 미실행]
작업2 진입 명령 = ___
실험3 진입 명령 = ___

[산출]
저장 파일 = ___
수정 파일 = ___
정지 조건 발생: 없음 | <사유>
```
