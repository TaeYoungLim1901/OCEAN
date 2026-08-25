# T16 — exp010: 깨끗한 임계 프로브 (단일 변수. 학습 없음)

실행 파이썬: `python`. `n_jobs` ≤ 2. 상단 `sys.stdout.reconfigure(encoding="utf-8")`.

**재학습 금지.** 저장된 원시 확률만 쓴다.

## 배경 — 확정 사실 (재조사·재측정 금지)

public 관측 2점이 양성률에 대해 단조 감소를 보인다:

| 실험 | 양성률 | public F1 |
|---|---|---|
| exp005 (t 0.62 / hole 48) | 0.034791 | **0.717435** |
| exp009 (hysteresis, t_high 0.88 / t_low 0.010 / hole 48) | 0.041808 | **0.698524** |

train 주입률은 0.041362 다. **양성률을 train 에 맞추는 것이 이득이 아님이 확인됐다.**
안 재본 방향은 **더 낮은 양성률**뿐이고, 그것이 이후 모든 제출의 임계 정책을 정한다.

inner(forward fold1·2) 델타는 public 으로 전이되지 않는다 (exp009 inner +0.004606 vs public -0.018911).
**따라서 이 작업에서 inner 로 파라미터를 고르지 마라.** 양성률 목표만으로 임계를 정한다.

원시 확률: `P1/cache/exp005_raw_test.npy` (float64, 169011, min 3.86e-09, p50 3.83e-06, max 0.99999990),
`P1/cache/exp005_raw_oof.npy` (float64, 776706, 평가 밖 행 NaN).
exp005 제출 sha256 `5E5B8840C3A6FC5E6C8A8DFD461E419E9795548492C566E27758EE5BD0185F5A`.
봉인 fold3 는 이미 3회 판독됐다. **채점하지 마라.**

기각축(재제안 금지): 유형 multiclass blend, stream 별 양성률 앵커링, 랜덤 행 CV, adversarial drift 보정.
`ruptures`·`xgboost`·`catboost`·`torch` 없다.

### 기준선 숫자 (확정)

exp005 inner F1 0.697836, public 0.717435, 양성 5880/169011.

## 할 일

`P1/src/postproc_lab.py` 의 기존 후처리 함수를 **재사용**한다. 새 후처리 로직을 쓰지 마라.

exp005 와 **완전히 동일한 설정**에서 `t_high` **하나만** 바꾼다:

- `t_low = t_high` (hysteresis 없음)
- `hole = 48`
- `dilation = 0`, `min_length = 0`

`t_high` 를 격자 `0.62:0.005:0.99` 에서 훑어, 후처리 **후** test 양성률이 **0.030000** 에 가장 가까워지는 값을 고른다.
그 예측을 `P1/subs/exp010.csv` 로 쓴다. 열·키·정렬은 `P1/subs/exp005.csv` 와 동일해야 한다.

## 제약

- **재학습 금지. LightGBM 을 부르지 마라.**
- **새 의존성 금지.** `pip install` 필요하면 실행하지 말고 멈추고 보고
- `P1/P1_data/`·`P1/cache/exp005_raw_*.npy` 읽기 전용
- **기존 파일을 수정하지 마라.** `P1/subs/exp010.csv` 하나만 새로 만든다.
  `postproc_lab.py` 를 고쳐야 하면 고치지 말고 멈추고 보고한다
- **동시에 다른 프로세스가 `P1/src/exp007.py`·`P1/preds/`·`P1/specs/` 를 쓰고 있다. 그 경로를 건드리지 마라**
- 변수는 `t_high` 하나뿐이다. 다른 축을 열지 마라

## 수락 기준

```
python P1/src/check.py sub P1/subs/exp010.csv
python P1/src/submission_tools.py audit P1/subs/exp010.csv
```

둘 다 `판정: OK`. sha256 이 exp005 의 `5E5B8840...` 와 **달라야 한다**.

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 보고할 것: 선택 `t_high`, 후처리 전/후 양성률,
양성 행 수와 exp005 대비 차이, sha256, 참고용 inner F1 1개. 산문 금지.
