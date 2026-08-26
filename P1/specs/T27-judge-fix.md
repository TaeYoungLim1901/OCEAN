# T27 — 판정기 수정: 확률 포화 제거 + AUC 기반 판정

## 목적

T26 판정기가 검증에 실패했다 (Spearman 0.200, 잔차 최대 0.0722). 원인이 둘 있다.

1. **임계 포화.** J1 표에서 5개 변형 전부 `t_high = 1.000` 이고 val 양성률이 목표 0.0300 에 못 미쳤다
   (base42 0.027925, neighbor46 0.024527, invariant37 0.026056, season42 0.020039).
   2024년만으로 700 tree 를 학습해 확률이 1.0 에 붙었고, 임계 격자가 상한에서 잘렸다.
   **변형마다 실효 양성률이 달라서 비교 자체가 오염됐다.**
2. **판정 지표.** 이미 확정된 사실: **행 단위 AUC 는 public 으로 전이되고 F1 은 전이되지 않는다**
   (`f.neighbor_wins`. Q3 AUC offset 0.903·drift 0.894 → exp016 public +0.034782).

수정: 임계를 **순위 기반 상위 N행**으로 뽑고, 판정 지표를 **AUC 우선**으로 바꾼다.
**모델은 이미 저장돼 있으니 재학습하지 마라.** 저장된 val 원시 확률만 다시 쓴다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- public 대장: exp016(neighbor46) **0.776056** > exp010(base42) 0.741274 > exp012(integrated47) 0.717975
  > exp013(season42) 0.582477. invariant37 은 미제출.
- T26 판정기 J1 (수정 전): base42 0.567485 / neighbor46 0.548252 / integrated47 0.600142 /
  invariant37 0.623101 / season42 0.478943. **neighbor46 을 최하위로 놓아 public 순위와 반대다.**
- 판정기 구간: 학습 2024년(279,185행, purge 87h), 검증 2025-01-01~06-30. 누출 위반 0.
- 유형별 recall(J3, base42): offset 0.212468 drift 0.148479 noise 0.854364 flatline 0.983442.
  inner 보다 훨씬 낮다 — 학습 데이터가 279k 로 줄어서다.
- `numpy`·`pandas`·`scipy`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

### 기준선

판정기의 기준선 변형은 `base42` (public 0.741274). 규칙 베이스라인 전체 F1 은 0.548255 다.
판정기가 쓸모 있으려면 K3 의 Spearman 이 1.0 이고 잔차 최대가 0.02 이하여야 한다.

## 하지 말 것

- **재학습 금지.** `P1/models/` 또는 `P1/cache/t26/` 에 저장된 모델·예측을 재사용한다.
  저장된 val 원시 확률이 없으면 저장된 booster 로 예측만 다시 한다.
- 임계 격자 탐색 금지. 상위 N행 규칙만 쓴다.
- 새 파일은 `P1/src/t27_judge_fix.py` 하나. `P1/src/t26_judge.py` 를 수정하지 마라. `SEED=42`.

## 구현

각 변형의 val 원시 확률에 대해:

1. **AUC** — val 전체 행 기준 양성 F1 과 무관한 `roc_auc_score`. 유형별 AUC 도 낸다
   (해당 유형 행 ∪ 정상 행). Q3 와 같은 방식.
2. **순위 기반 예측** — 후처리 전 상위 `k` 행을 1 로 놓는다. `k` 는 val 행수 × `r`,
   `r ∈ {0.020, 0.025, 0.030}`. 동점은 인덱스 오름차순으로 자른다. 그 뒤 `hole=48` 채움.
   포화 때문에 동점이 많으면 동점 행수를 print 한다.
3. 각 `r` 에서 F1 / TP / FP / FN / 실제 후처리 후 양성률.

## 보고

**표 K1 — AUC**

| variant | auc_all | auc_offset | auc_drift | auc_noise | auc_flatline | n_ties_top |

**표 K2 — 순위 기반 F1**

| variant | r | post_rate | judge_F1 | TP | FP | FN |

**표 K3 — 판정기 검증**: public 을 아는 4개(base42 0.741274, neighbor46 0.776056,
integrated47 0.717975, season42 0.582477)로

| 지표 | spearman | a | b | residual_max |
|---|---|---|---|---|
| auc_all | | | | |
| auc_offset | | | | |
| judge_F1 (r=0.030) | | | | |

## 수락 기준

| 항목 | 기준 |
|---|---|
| 재학습 | 0회 (모델 파일 mtime 불변을 print) |
| 표 | K1·K2·K3 출력 |
| 실행 시간 | 10분 이내 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.**

## 실행

```
python P1/src/t27_judge_fix.py
```
