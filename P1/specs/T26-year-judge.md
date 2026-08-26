# T26 — 연도 홀드아웃 판정기 구축과 **검증**. public 점수 예측기

## 목적

현재 P1 에는 작동하는 로컬 판정기가 없다. inner(fold1·2)는 임계·후처리·모델 비교 세 번 모두
public 과 부호가 반대였다. 원인은 inner 가 **station-layer별 시간 블록**이라 실제 train→test 관계를
복제하지 않는 것이다. 실제 관계는 **「과거 연도 전체 → 다음 해 1~6월」**이다.

이 사양은 그 관계를 복제하는 판정기 J 를 만들고, **이미 public 점수를 아는 변형 5개로 검증**한다.
검증 없이 판정기라고 부르지 마라 — 검증 결과가 나쁘면 그것도 숫자로 보고한다.

## 판정기 J 정의

- 학습 구간: `time < 2025-01-01` 에서 검증 시작 87시간 전까지 (purge). 2024년 행만 남는다.
- 검증 구간: `2025-01-01 ~ 2025-06-30` (test 가 2026-01~06 인 것과 같은 계절창).
- climatology reference 는 **학습 구간만으로 적합**한다 (`fit_reference(train, train_mask)`).
- 후처리는 제출과 동일: `hole=48` 만. 임계는 **검증 구간 양성률 0.0300 앵커** (제출과 같은 규칙).
- 점수 = 검증 구간 행에 대한 양성 F1. 유형별 recall 도 같이 낸다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

public 대장 (전부 양성률 3.0% 근방으로 통제됨, exp009·exp011 만 예외):

| 변형 | 설명 | 양성률 | public F1 |
|---|---|---|---|
| exp010 | base 42 feature, 전체 train | 2.9832% | **0.741274** |
| exp016 | base 42 + 이웃회귀 잔차 4개 (46) | 2.9998% | 미제출(오늘 제출 예정) |
| exp012 | base 42 + 적분 step 5개 (47) | 3.0501% | 0.717975 |
| exp014 | 연도의존 5개 제거 (37) | 3.0105% | 미제출 |
| exp013 | base 42, 학습 행 month∈{12,1..6} | 2.9874% | **0.582477** |
| exp005 | exp010 과 같은 모델, 임계만 다름 | 3.4791% | 0.717435 |
| exp009 | 같은 모델 + hysteresis | 4.1808% | 0.698524 |
| exp011 | 같은 모델, 임계 0.995 | 2.1602% | 0.657820 |

- 학습 파라미터·feature 정의는 `P1/src/exp013_shift.py`·`P1/src/exp016_neighbor.py`·
  `P1/src/exp012_v1_full.py` 에 있다. **재구현하지 말고 재사용해라.**
- 이웃 회귀 잔차는 `P1/src/q3_neighbor.py` 재사용.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

### 기준선

판정기 J 의 기준선은 `base42` 다. 나머지 4개는 그것과의 차이로 읽는다.
public 기준선은 exp010 F1 0.741274 이며, 규칙 베이스라인(주최측 사전검증)은 전체 F1 0.548255 다.

## 하지 말 것

- 봉인 fold3 판독 금지. 새 feature 발명 금지. 임계 격자 탐색 금지 (양성률 앵커만).
- 검증 구간(2025-01~06) 행을 학습에 넣지 마라. purge 87시간을 지켜라.
- 새 파일은 `P1/src/t26_judge.py` 하나. 기존 파일 수정 금지. `SEED=42`, `n_jobs<=2`.
- **전체 실행을 codex 가 하지 마라.** 코드 완성 + 문법·컬럼 검증까지. 학습 5회는 사람이 돌린다.
- 스테이지를 나눠라: `python P1/src/t26_judge.py fit <variant>` 로 변형 1개씩 학습·채점하고
  결과를 `P1/cache/t26/{variant}.json` 에 저장. `python P1/src/t26_judge.py report` 로 표를 낸다.
  이미 json 이 있으면 재학습하지 않는다.

## 변형 5개

| variant | 정의 |
|---|---|
| `base42` | base 42 feature, 학습 구간 전체 행 |
| `neighbor46` | base 42 + `n1`,`n2_432`,`n3`,`n4` |
| `integrated47` | base 42 + `temp_step_abs_144`,`max_z_center_144`,`temp_step_144`,`temp_step_abs_48`,`psal_step_abs_144` |
| `invariant37` | base 42 − `month`,`doy_sin`,`doy_cos`,`clim_resid`,`layer_count` |
| `season42` | base 42, 학습 행을 `month∈{12,1,2,3,4,5,6}` 로 제한 |

## 보고 — `report` 스테이지

**표 J1**

| variant | n_features | train_rows | t_high | val_positive_rate | judge_F1 | TP | FP | FN |

**표 J2 — 판정기 검증**: public 을 아는 4개(`base42`↔exp010, `integrated47`↔exp012,
`season42`↔exp013, `neighbor46`↔exp016 은 public 이 나오면 추가)로

| variant | judge_F1 | public_F1 | diff |

그리고 Spearman 순위상관과 최소제곱 직선 `public ≈ a·judge + b` 의 `a`, `b`, 잔차 최대값.

**표 J3 — 유형별 recall** (variant × offset/drift/noise/flatline/spike).

## 수락 기준

| 항목 | 기준 |
|---|---|
| 누출 | 각 variant 학습 행 최대 시각 < 2025-01-01 − 87h. 위반 0 을 print |
| val 양성률 | 전부 0.0300 ± 0.0005 |
| 표 | J1·J2·J3 출력 |
| 재실행 | 같은 variant 재실행 시 judge_F1 동일 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.**

## 실행

```
python P1/src/t26_judge.py fit base42
python P1/src/t26_judge.py fit neighbor46
python P1/src/t26_judge.py fit integrated47
python P1/src/t26_judge.py fit invariant37
python P1/src/t26_judge.py fit season42
python P1/src/t26_judge.py report
```
