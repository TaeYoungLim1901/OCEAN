# T29 — 판정기 버그: 확률 대신 0/1 라벨을 저장했다

## 목적

`P1/cache/t26/base42_val_raw.npy` 의 **고유값이 2개(0.0, 1.0)** 다. 확률이 아니라 클래스 라벨이다.
`P1/src/t26_judge.py` 가 `LGBMClassifier.predict()` 를 썼기 때문이다 — 그것은 라벨을 돌려준다.
이 때문에 T26·T27 판정 결과가 전부 무의미했다 (`t_high` 전부 1.000, 상위 동점 203,088행,
AUC 0.70대, Spearman 0.200). **재학습 판정 결과를 버리고 확률로 다시 낸다.**

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 올바른 확률 예측 방법은 이미 코드에 있다: `P1/src/exp016_neighbor.py`·`P1/src/exp012_v1_full.py` 는
  `booster.predict(x)` 를 쓴다. `LGBMClassifier` 를 쓸 거면 `predict_proba(x)[:, 1]` 이다.
- 판정기 구간(학습 2024년 279,185행 purge 87h, 검증 2025-01-01~06-30)과 feature 정의는 **바꾸지 마라.**
- 판정 지표는 AUC 우선 (`f.neighbor_wins`).
- public 기준값: neighbor46 0.776056 > base42 0.741274 > integrated47 0.717975 > season42 0.582477.
  판정기가 쓸모 있으려면 이 순서를 재현하고 잔차가 작아야 한다.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

### 기준선

기준선 변형은 `base42` (public 0.741274). 규칙 베이스라인 전체 F1 0.548255.
수정 후 base42 의 val AUC 는 최소 0.85 이상이어야 한다 (train 전체 행 기준 `n1` offset AUC 0.903,
`resid_1009` 0.871 인 신호들을 쓰는 모델이다). 그 미만이면 다른 버그가 남아 있다는 뜻이다.

## 구현

1. `P1/src/t26_judge.py` 에서 val 예측을 **확률**로 바꾼다 (`booster.predict` 또는 `predict_proba[:,1]`).
2. **저장 직전 가드**를 넣는다: `raw` 의 고유값이 1000개 미만이거나 `{0,1}` 부분집합이면
   저장하지 말고 예외를 던진다. 메시지에 고유값 수를 넣어라.
3. 같은 가드를 `P1/src/t27_judge_fix.py` 의 로드 직후에도 넣는다.
4. 그 외 로직 변경 금지. `SEED=42`, `n_jobs<=2`.
5. **학습을 실행하지 마라.** 사람이 아래를 돌린다.

## 수락 기준

| 항목 | 기준 |
|---|---|
| 문법 | `python -m py_compile` 통과 |
| diff | 두 파일 합쳐 20줄 이내 |
| 가드 | 고유값 1000 미만이면 예외 |

## 보고 형식

**결론 문장 금지.** 바뀐 파일 경로와 diff 줄 수만.

## 실행 (사람이 돌린다)

```
python P1/src/t26_judge.py fit base42
python P1/src/t27_judge_fix.py
```
