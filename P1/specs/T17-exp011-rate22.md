# T17 — exp011: 양성률 2.2% 임계 프로브 (단일 변수. 학습 없음)

실행 파이썬: `python`. `n_jobs` ≤ 2. 상단 `sys.stdout.reconfigure(encoding="utf-8")`.

**재학습 금지.** 저장된 원시 확률만 쓴다. **T16(exp010)과 동일한 코드 경로를 재사용한다.**

## 배경 — 확정 사실 (재조사·재측정 금지)

public 관측 3점이 양성률에 대해 **가속하는 단조 감소**를 보인다:

| 실험 | 양성률 | public F1 |
|---|---|---|
| exp009 | 0.041808 | 0.698524 |
| exp005 | 0.034791 | 0.717435 |
| **exp010** (t_high 0.775) | 0.029832 | **0.741274** |

기울기: pp 당 +0.0269 → +0.0481. **봉우리는 2.98% 보다 아래에 있다.**
inner 는 임계 판정기가 아니다 — exp010 의 inner 0.682413 은 exp005 의 0.697836 보다 **낮은데** public 은 더 **높다**.
**inner 로 파라미터를 고르지 마라. 양성률 목표만으로 임계를 정한다.**

원시 확률: `P1/cache/exp005_raw_test.npy` (float64, 169011), `P1/cache/exp005_raw_oof.npy` (float64, 776706, 평가 밖 NaN).
exp010 sha256 `59D97691CE3E41408DA76D003CA5FB9DCDE9C0BC0A09ADBA47196E8AC413A769`, t_high 0.775, 양성 5042.
봉인 fold3 는 이미 3회 판독됐다. **채점하지 마라.**
기각축(재제안 금지): 유형 multiclass blend, 양성률 앵커링, 랜덤 행 CV, adversarial drift 보정, hysteresis.
`ruptures`·`xgboost`·`catboost`·`torch` 없다.

### 기준선 숫자 (확정)

exp005 inner 0.697836 / public 0.717435. exp010 inner 0.682413 / public 0.741274.

## 할 일

T16 과 **완전히 같은 절차**에서 목표 양성률만 바꾼다:

- `t_low = t_high` (hysteresis 없음), `hole = 48`, `dilation = 0`, `min_length = 0`
- `t_high` 를 격자 `0.62:0.005:0.999` 에서 훑어, 후처리 **후** test 양성률이 **0.022000** 에 가장 가까워지는 값 선택
- 결과를 `P1/subs/exp011.csv` 로 쓴다. 열·키·정렬은 `P1/subs/exp005.csv` 와 동일

추가로 **참고표 1개**를 출력한다 (제출물 생성 없음): 목표 양성률
`0.026, 0.024, 0.022, 0.020, 0.018, 0.016` 각각에 대한 `t_high`, 후처리 후 양성률, 양성 행 수.
다음 슬롯 계획에 쓴다.

## 제약

- **재학습 금지. LightGBM 을 부르지 마라.**
- **새 의존성 금지.** `pip install` 필요하면 실행하지 말고 멈추고 보고
- `P1/P1_data/`·`P1/cache/exp005_raw_*.npy` 읽기 전용
- **기존 파일을 수정하지 마라.** `P1/subs/exp011.csv` 하나만 새로 만든다
- **동시에 다른 프로세스가 `P1/src/exp007.py`·`P1/preds/`·`P1/specs/` 를 쓰고 있다. 그 경로를 건드리지 마라**
- 변수는 `t_high` 하나뿐이다. 다른 축을 열지 마라

## 수락 기준

```
python P1/src/check.py sub P1/subs/exp011.csv
python P1/src/submission_tools.py audit P1/subs/exp011.csv
```

둘 다 `판정: OK`. sha256 이 exp005·exp010 과 **달라야 한다**.

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 보고할 것: 선택 `t_high`, 후처리 전/후 양성률, 양성 행 수,
exp010 대비 차이, sha256, 참고용 inner F1, 그리고 위 6개 목표 양성률 참고표. 산문 금지.
