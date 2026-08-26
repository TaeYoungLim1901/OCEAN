# T31 — 판정기 변형 추가: 표준화 49 feature + 하이퍼파라미터 3종

## 목적

판정기 J 가 검증을 통과했다 (feature 집합 비교 3개 순서가 public 과 일치, `f.judge_scope`).
이제 **슬롯을 쓰지 않고** 후보를 거른다. 변형 4개를 추가한다. 그중 3개는 **한 번도 검증한 적 없는
하이퍼파라미터**다 — P1 은 처음부터 `n_estimators=700, learning_rate=0.05, num_leaves=63,
min_child_samples=80, scale_pos_weight=2.0` 을 고정해 왔고 아무도 흔들어본 적이 없다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 판정기 정의: 학습 2024년 279,185행(purge 87h), 검증 2025-01-01~06-30, **확률** 예측, 지표 `auc_all`.
  구현은 `P1/src/t26_judge.py`(fit) + `P1/src/t27_judge_fix.py`(report). **정의를 바꾸지 마라.**
- 현재 판정기 값: neighbor46 **0.932374** / season42 0.923480 / base42 0.915129 /
  invariant37 0.906768 / integrated47 0.897338.
- 판정기는 **같은 학습 행에서 feature·하이퍼파라미터만 바꾼 비교에만** 유효하다 (`f.judge_scope`).
  학습 행을 바꾸는 변형을 추가하지 마라.
- `judge_F1` 은 public 과 Spearman −0.8 이다. **F1 로 고르지 마라.**
- `s1`,`s2`,`s8_432` 정의와 구현은 `P1/src/q4_screen.py`, 49 feature 조립은
  `P1/src/exp019_standardized.py` 에 이미 있다. **재구현 금지, import 해서 써라.**
- 저장 전 고유값 1000 미만이면 예외를 던지는 가드가 이미 있다. 유지해라.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

### 기준선

기준선 변형은 `neighbor46`, 판정기 `auc_all` **0.932374** (public 0.776056, 27.381306점).
신규 4개는 이 값을 넘어야 제출 후보가 된다. 규칙 베이스라인 전체 F1 은 0.548255 다.

## 추가 변형 4개

| variant | 정의 |
|---|---|
| `standardized49` | neighbor46 feature + `s1`,`s2`,`s8_432` |
| `hp_trees1400` | **neighbor46 feature 고정**, `n_estimators=1400`, `learning_rate=0.025` |
| `hp_leaves127` | neighbor46 feature 고정, `num_leaves=127`, `min_child_samples=40` |
| `hp_posweight1` | neighbor46 feature 고정, `scale_pos_weight=1.0` |

하이퍼파라미터 변형은 **한 번에 하나씩만** 바꾼다. 위에 적힌 것 외 파라미터는 기존 값 그대로다.

## 하지 말 것

- 학습 행·검증 구간·후처리·purge 를 바꾸지 마라. 새 feature 발명 금지.
- **학습을 실행하지 마라.** 코드 추가 + 문법 검증까지. 4회 학습은 사람이 돌린다.
- `SEED=42`, `n_jobs<=2`. 기존 변형 5개의 정의·저장물을 건드리지 마라.

## 구현

`P1/src/t26_judge.py` 의 variant 표에 위 4개를 추가한다. feature 조립은 기존 함수를 재사용하고,
하이퍼파라미터 변형은 파라미터 dict 만 덮어쓴다. `P1/src/t27_judge_fix.py` 는 새 변형을 자동으로
읽어 K1·K2 표에 포함해야 한다 (K3 는 public 을 아는 4개로만 계산한다 — 변경 없음).

## 수락 기준

| 항목 | 기준 |
|---|---|
| 문법 | `python -m py_compile` 통과 |
| 변형 수 | 9개 (기존 5 + 신규 4) |
| 기존 변형 | 정의·json·npy 불변 |
| K1 | 신규 4개 행이 추가된다 |

## 보고 형식

**결론 문장 금지.** 바뀐 파일 경로와 diff 줄 수, variant 목록만.

## 실행 (사람이 돌린다)

```
python P1/src/t26_judge.py fit standardized49
python P1/src/t26_judge.py fit hp_trees1400
python P1/src/t26_judge.py fit hp_leaves127
python P1/src/t26_judge.py fit hp_posweight1
python P1/src/t27_judge_fix.py
```
