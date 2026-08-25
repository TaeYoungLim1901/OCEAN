# T23 — 대조군 출력 + T1F 로컬 편향 측정

## 목적

두 가지 숫자만 얻는다. **새 모델 금지. 학습 금지. 제출물 생성 금지.**

1. `run_t21.py` 채점 경로의 **대조군** — 기존 3멤버와 T1F 의 조밀 재가중 폴드 점수
2. 조밀 OOF 에서 **T1F 의 평균 편향** `mean(pred - y)`

## 배경 — 이미 확정된 사실이다. 재조사하지 마라

- `P3/src/run_t21.py` 4폴드 실행 완료. arx fold1-4 = 0.575800/0.639721/0.693044/0.672507,
  wsswell = 0.574258/0.632815/0.663154/0.668720. folds1-3 잔차상관 lgbm_t1 대비
  arx 0.832120 / wsswell 0.759305. 블렌드 봉인 fold4 0.633913
- 산출물 `P3/preds/{oof_dense,test}_{arx,wsswell}_p3_t21.npy` 4개는 이미 존재한다.
  **다시 생성하지 마라. 덮어쓰지 마라**
- 다른 스크립트가 낸 기준값: T11 조밀 재가중 fold1-4 W_old = 0.668083/0.630441/0.656407/0.611959,
  T8 조밀 재가중 fold4 lgbm_t1 = 0.614009 / damped 0.663402 / persistence 0.769115.
  **이 값들과 같은 경로인지가 이 작업의 핵심이다**
- T1F 가중 = persistence 0.00 / damped 0.35 / lgbm_t1 0.65 (상수, 고정)
- public 실측: T1F 0.587165, T1F+0.05 0.592805 → public 전역 편향 +0.041550308
- 폴드 정의·`dense_weights.csv`·`forward_train_mask` 를 바꾸지 마라
- `pip install` 금지. 새 의존성 금지

## 작업 1 — `run_t21.py` 보고에 대조군 추가

`report.csv` 와 표준출력 표에 아래 행을 **추가**한다 (기존 arx·wsswell 행은 그대로).

- `persistence`, `damped`, `lgbm_t1` — 저장된 조밀 OOF 를 그대로 쓴다
- `t1f` — 위 3개를 (0.00, 0.35, 0.65) 로 합성한 것

각 행에 fold1~fold4 조밀 재가중 RMSE 를 채운다. 계산 경로는 arx·wsswell 과 **동일한 함수**를 쓴다.

추가로 한 줄씩 찍는다:
```
CONTROL lgbm_t1_fold4=<...> expected=0.614009 delta=<...>
CONTROL t1f_fold4=<...> expected=0.611959 delta=<...>
CONTROL damped_fold4=<...> expected=0.663402 delta=<...>
CONTROL persistence_fold4=<...> expected=0.769115 delta=<...>
```

## 작업 2 — T1F 로컬 편향

조밀 OOF 에서 T1F 합성 예측의 평균 편향 `mean(pred - y)` 를 구한다.
가중은 `dense_weights.csv` 재가중과 비가중 **양쪽 다** 낸다.

```
BIAS overall weighted=<...> unweighted=<...> n=<...>
BIAS fold f=<1..4> weighted=<...> unweighted=<...> n=<...>
BIAS lead h=<3,6,9,12,18,24> weighted=<...> unweighted=<...> n=<...>
BIAS station s=<...> weighted=<...> unweighted=<...> n=<...>
```

추가로 상수보정 이득을 찍는다. `c = -bias_overall_weighted` 를 T1F 에 더했을 때의
조밀 재가중 fold4 RMSE 와 보정 전 대비 델타:
```
SHIFTGAIN c=<...> fold4_before=<...> fold4_after=<...> delta=<...>
```

**표준오차도 찍어라.** 사례 단위 부트스트랩 2000회로 `bias_overall_weighted` 의 sd 를 낸다:
```
BIASSE sd=<...> z_vs_public=<(0.041550308-bias)/sd>
```

## 금지

- 모델 학습 금지. `P3/preds/` 기존 파일 덮어쓰기 금지. 제출 CSV 생성·수정 금지
- 폴드 정의·가중 파일·`forward_train_mask` 수정 금지
- 가중 탐색 금지. 블렌드 재선택 금지 — 작업 1 은 **보고 추가**일 뿐이다
- `bugcheck.py`·`submission_tools.py`·`make_probe_sub.py` 수정 금지
- 추상화 금지. 클래스·config·로깅 프레임워크 금지

## 기준선 숫자 (baseline)

| 항목 | 값 |
|---|---|
| 조밀 fold별 검증 건수 | 988 / 1712 / 2100 / 1163 |
| T1F 가중 | 0.00 / 0.35 / 0.65 |
| public T1F 편향 | +0.041550308 |
| 대조군 기대 fold4 | lgbm_t1 0.614009 / t1f 0.611959 / damped 0.663402 / persistence 0.769115 |
| SEED | 42 |

## 수락 기준 (직접 실행하고 숫자를 보고하라)

```
python P3/src/run_t21.py --folds 4
```

보고: 확장된 표, `CONTROL` 4줄, `BIAS` 전체·폴드·리드·정점, `SHIFTGAIN`, `BIASSE`, exit code.
**결론 문장 금지, 숫자와 표만.**
