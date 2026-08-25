# T21 — 비트리 멤버 2종: ARX/Kalman, 풍파·너울 성분분해

## 목적

조밀 앙상블 풀에 **트리가 아닌** 멤버 2개를 넣는다. 목표는 단독 점수가 아니라
`lgbm_t1` 과의 **낮은 잔차 상관**이다. 축 `a.arx_kalman`·`a.windsea_swell_components`.

**코드를 만들고 짧은 검증만 돌려라. 전체 4폴드 학습 실행은 사람이 한다.**

## 배경 — 이미 검증된 사실이다. 재조사하지 마라

- 조밀 판정기가 옳다는 것은 public LB 로 검증됐다 (`f.lb_picks_dense_judge`:
  LB lgbm 0.591543 < damped 0.642689, 조밀 fold4 와 같은 부호, 씨닝은 반대 부호).
  **씨닝 판정기를 쓰지 마라**
- 조밀 OOF 생성 경로는 `P3/src/train_dense.py` 에 있고 fold별 검증 988/1712/2100/1163 건이다.
  재가중치는 `P3/cache/p3_t9/dense_weights.csv`. **폴드 정의·가중 파일을 바꾸지 마라**
- 시간 분리는 `forward_train_mask` 가 정점별 `t0 < 검증시작 - 78h` 로 강제한다.
  fold0 은 학습행 0 이라 건너뛰고 OOF 는 NaN 이다. **이 구조를 바꾸지 마라**
- `check.py cv` 를 조밀 인덱스에 걸지 마라. 구조적으로 FAIL 이며 누수가 아니다 (`f.dense_cv_gate_na`)
- LGBM 변형 6개는 `lgbm_t1` 과 rho 0.988~0.996 로 전부 중복이다 (`f.pool_no_diversity`).
  트리 계열을 또 만들지 마라
- 피처 함수는 `(289,10) float32` 하나만 받는다. 열 순서 `cases.COLS` 고정
- `torch`·`xgboost`·`catboost` 없다. **`pip install` 하지 말고 필요하면 멈추고 보고하라**
- `P3/src/bugcheck.py` 에 `all`·`wcurve`·`train`·`simplex`·`subset` 이 있다. 건드리지 마라

## 작업 1 — `P3/src/model_arx.py`

결측 허용 선형 상태공간 / ARX 예측기. `numpy`+`scipy` 만.

- 입력은 `(289,10)` 컨텍스트 하나. 출력은 6리드 `hs` 예측 `(6,)`
- 원시 `hs` 시퀀스를 상태로 쓴다. 윈도 요약 통계(mean/std/slope)를 **피처로 만들지 마라** —
  그러면 기존 GBDT 와 같은 정보가 된다. 이 축의 요점은 원시 지연 구조다
- 결측은 관측 갱신을 건너뛰는 방식으로 처리한다. 보간으로 메우지 마라
- 외생 입력은 `wspd`·`gust` 만 허용한다 (있으면 사용, 전결측이면 자동 무시)
- 파라미터는 folds1-3 에서만 적합한다. fold4 를 보지 마라
- 예측은 `[0, 30]` 으로 클립한다

## 작업 2 — `P3/src/model_wsswell.py`

풍파·너울 성분분해 예측기. `numpy` 만.

- `tp`·`wspd`·`wvdir`·`wdir` 로 풍파 성분비 `r in [0,1]` 을 추정하고
  `hs^2 = hs_ws^2 + hs_sw^2` 로 에너지 분해한다
- 풍파 성분은 국지 바람에 빠르게 반응, 너울 성분은 느리게 감쇠하는 서로 다른 시간상수를 준다
- 시간상수와 성분비 파라미터는 folds1-3 에서만 적합. fold4 를 보지 마라
- `wspd` 전결측이면 성분분해를 포기하고 `hs` 단일 감쇠로 폴백한다. 폴백 비율을 보고하라

## 작업 3 — 실행 스크립트 `P3/src/run_t21.py`

`train_dense.py` 의 폴드 정의·인덱스를 **재사용**해 두 모델의 조밀 OOF 와 test 예측을 만든다.

산출물 (경로 고정):
```
P3/preds/oof_dense_arx_p3_t21.npy
P3/preds/oof_dense_wsswell_p3_t21.npy
P3/preds/test_arx_p3_t21.npy
P3/preds/test_wsswell_p3_t21.npy
P3/cache/p3_t21/report.csv
```

`report.csv` 에 담을 것: 멤버별 조밀 재가중 fold1-4 RMSE, folds1-3 에서 `lgbm_t1` 과의
잔차 Pearson 상관, `wsswell` 폴백 비율, 각 멤버 예측의 NaN 수.

## 작업 4 — 블렌드 선택 (별도 함수, 같은 스크립트)

- 풀 = {persistence, damped, lgbm_t1} + 상관 0.90 미만인 신규 멤버
- 가중은 조밀 재가중 **folds1-3 에서만** 0.05 grid simplex 로 고른다
- 봉인 fold4 점수를 계산해 보고만 한다. **선택에 쓰지 마라**
- 상관 0.90 이상인 멤버는 블렌드에서 제외하고 그 사실을 찍는다

## 금지

- 씨닝 판정기 사용 금지. fold4 를 선택에 쓰는 것 금지
- 폴드 정의·`dense_weights.csv`·`forward_train_mask` 수정 금지
- 트리 모델 추가 금지. 하이퍼파라미터 탐색 금지
- 제출 CSV 생성 금지. 기존 `P3/preds/` 파일 덮어쓰기 금지 (새 파일명만)
- `pip install` 금지. 새 의존성 금지
- 추상화 금지. 클래스 계층·config 파일·로깅 프레임워크 만들지 마라

## 기준선 숫자 (baseline)

| 항목 | 값 |
|---|---|
| 조밀 재가중 fold4 T1F | 0.611959 |
| 조밀 재가중 fold4 lgbm_t1 (대조군) | 0.614009 |
| 조밀 재가중 fold4 damped | 0.663402 |
| 조밀 재가중 fold4 persistence | 0.769115 |
| 조밀 fold별 검증 건수 | 988 / 1712 / 2100 / 1163 |
| 신규 멤버 단독 필수 하한 | fold4 재가중 <= 0.85 |
| 상관 임계 | 0.90 |
| SEED | 42 |

## 수락 기준 (직접 실행하고 숫자를 보고하라)

빠른 검증만 돌린다. **fold1 하나만** 실행해 배관이 도는지 확인하라.

```
python P3/src/run_t21.py --folds 1
python P3/src/bugcheck.py all
```

보고: fold1 두 멤버 RMSE, 폴백 비율, NaN 수, 산출물 파일 존재 여부, 두 명령의 exit code.
전체 4폴드 실행은 사람이 한다. **결론 문장 금지, 숫자와 표만.**
