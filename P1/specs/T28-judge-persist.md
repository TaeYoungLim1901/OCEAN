# T28 — 판정기 재학습 산출물 저장. T27 이 돌 수 있게 만든다

## 목적

`P1/src/t27_judge_fix.py` 가 `FileNotFoundError: base42: saved validation raw probabilities or exact
variant booster not found` 로 멈춘다. `P1/src/t26_judge.py` 가 요약 json 만 저장하고
**val 원시 확률과 booster 를 저장하지 않았다.** 저장하도록 고치고 5개 변형을 다시 학습한다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 판정기 구간: 학습 2024년 279,185행(purge 87h), 검증 2025-01-01~06-30. 누출 위반 0. 이 정의는 그대로 둔다.
- T26 J1 결과(수정 전, 확률 포화로 오염됨): base42 0.567485 / neighbor46 0.548252 /
  integrated47 0.600142 / invariant37 0.623101 / season42 0.478943. `t_high` 가 전부 1.000 이었다.
- public 기준선: exp016(neighbor46) 0.776056 > exp010(base42) 0.741274 > exp012(integrated47) 0.717975
  > exp013(season42) 0.582477. 규칙 베이스라인 전체 F1 0.548255.
- 판정 지표는 AUC 가 우선이다 (`f.neighbor_wins`: AUC 는 전이되고 F1 은 안 된다).
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- 판정기 구간·feature 정의·학습 파라미터를 바꾸지 마라. 저장만 추가한다.
- `t27_judge_fix.py` 의 판정 로직(AUC, 순위 상위 N행, hole 48)을 바꾸지 마라.
  경로만 맞춰라. 필요하면 `_load_raw` 가 읽는 경로를 아래 규약에 맞춘다.
- **학습을 실행하지 마라.** 코드 수정 + 문법 검증까지. 5회 학습은 사람이 돌린다.
- `SEED=42`, `n_jobs<=2`.

## 구현

1. `P1/src/t26_judge.py` 의 `fit` 스테이지가 추가로 저장하게 한다:
   - `P1/cache/t26/{variant}_val_raw.npy` — val 행 원시 확률 float64, val 행 순서대로
   - `P1/cache/t26/{variant}_val_index.npy` — 그 행들의 train.csv 원본 인덱스 int64
   - `P1/models/t26_{variant}.txt` — booster
   - 기존 json 도 그대로 유지한다
2. 이미 파일이 있으면 재학습하지 않는다.
3. `P1/src/t27_judge_fix.py` 가 위 경로를 읽도록 맞춘다. 없으면 명확한 메시지로 멈춘다.

## 수락 기준

| 항목 | 기준 |
|---|---|
| 문법 | `python -m py_compile` 통과 |
| 저장 경로 | 위 3종 경로가 코드에 그대로 나타난다 |
| 판정 로직 | T27 의 K1·K2·K3 계산은 변경 없음 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 바뀐 파일 경로와 diff 줄 수만 보고한다.

## 실행 (사람이 돌린다)

```
python P1/src/t26_judge.py fit base42
python P1/src/t27_judge_fix.py
```
