# T18 — exp007 부검: 적분 feature 를 살릴지 축을 닫을지 결정 (학습 최소)

실행 파이썬: `python`. `n_jobs` ≤ 6 (지금 다른 학습 없음). 상단 `sys.stdout.reconfigure(encoding="utf-8")`.

## 배경 — 확정 사실 (재조사·재측정 금지)

exp007 은 exp005 에 적분 feature 39개를 더해 42→81 feature 로 만든 것이다.
**나머지는 전부 동일**: forward fold1·2, 700 tree 고정, early stop 없음, train-only climatology, 한쪽 purge 87h, SEED 42.

| 유형 | exp005 recall | exp007 recall |
|---|---|---|
| offset | 0.425373 | **0.288380** |
| drift | 0.510435 | **0.433043** |
| noise | 0.700364 | **0.429981** |
| spike | 0.686275 | 0.019608 |
| flatline | 0.894231 | 0.894231 |

inner F1: exp005 **0.697836** → exp007 **0.577092** (**-0.120744**).

**임계 축은 닫혔다.** public 양성률 곡선 봉우리는 3.0191%, peak F1 0.741420.
현재 최고 제출은 exp010 (양성률 2.9832%, public **0.741274**). **임계를 다시 탐색하지 마라.**

**inner 델타는 public 으로 전이되지 않는다** (exp009 inner +0.004606 vs public -0.018911).
inner 는 **수준** 비교에는 쓰되, 후처리 파라미터 선택에는 쓰지 마라.

봉인 fold3 는 이미 3회 판독됐다. **채점하지 마라.**
기각축(재제안 금지): 유형 multiclass blend, 양성률 앵커링, 랜덤 행 CV, adversarial drift 보정, hysteresis, 임계 탐색.
`ruptures`·`xgboost`·`catboost`·`torch` 없다.

### 기준선 숫자 (확정)

exp005 inner 0.697836 / public 0.717435. exp010 public 0.741274. exp007 inner 0.577092.
exp007 산출물: `P1/preds/oof_exp007.npy`, `P1/preds/test_exp007.npy`, `P1/experiments/exp007_grid.json`.
exp007 부스터는 `P1/cache/` 아래에 exp005 와 같은 규약으로 저장돼 있다.

## 할 일

### D1 — feature importance 판독 (학습 없음)

exp007 의 저장된 부스터에서 feature importance 를 읽는다 (`gain` 과 `split` 둘 다).
표 1: 적분 feature 39개 각각의 `gain`, `split`, 전체 81개 중 gain 순위.
표 2: 요약 — 적분 feature 39개의 gain 합이 전체의 몇 %인가, split 합이 몇 %인가,
gain 상위 20 안에 적분 feature 가 몇 개인가, gain 이 정확히 0 인 적분 feature 가 몇 개인가.

같은 방식으로 **exp005 부스터**의 기존 42개 importance 도 읽어, exp007 에서 기존 feature 의
gain 점유율이 얼마나 밀렸는지 비교표를 낸다.

### D2 — 희석 가설 검정 (학습 있음. 짧게)

**가설:** feature 를 42→81 로 늘렸는데 트리 700개 고정이라 분할 예산이 희석됐다.

exp005 와 **모든 것을 동일**하게 두고 feature 집합만 바꾼 변형 3개를 forward fold1·2 로 학습해
inner F1 을 보고한다. **후처리는 exp005 확정값(threshold 0.62, hole 48, dilation 0, min-length 0)으로 고정한다.
후처리 격자를 돌리지 마라.**

| 변형 | feature 집합 |
|---|---|
| V1 | 기존 42개 + D1 의 gain 상위 **5개** 적분 feature |
| V2 | 기존 42개 + gain 상위 **12개** 적분 feature |
| V3 | exp007 의 81개 전부, 단 `n_estimators` 를 **1400** 으로 (희석을 트리 예산으로 보상) |

각 변형에 대해 inner F1 과 T10 형식 유형별 recall 표를 낸다.

### D3 — 잔차 상관

exp005 와 exp007 의 **원시 확률** oof (평가 행만) 사이의 Spearman 상관, 그리고 각 유형별 상관.
새 모델은 「기존을 이기는가」가 아니라 「기존과 섞여서 이기는가」로 본다.

## 제약

- **새 의존성 금지.** `pip install` 필요하면 실행하지 말고 멈추고 보고
- **후처리 격자 탐색 금지. 임계 탐색 금지.** 후처리는 exp005 확정값 고정
- **봉인 fold3 채점 금지**
- 원본 데이터(`P1/P1_data`)는 읽기 전용
- **기존 파일을 수정하지 마라.** 새 파일은 `P1/src/exp007_postmortem.py` 하나
- 모델 파라미터(learning_rate·leaves·purge·seed·climatology)는 D2 의 `n_estimators` 외에 바꾸지 마라
- 원시 확률은 **float32 이상으로** `P1/preds/score_oof_{변형}.npy` 에 저장한다. **이진화해서 저장하지 마라**
- V1~V3 학습이 30분을 넘길 것 같으면 학습하지 말고 코드만 만들고 실행 명령을 보고한다

## 수락 기준

```
python P1/src/exp007_postmortem.py d1
python P1/src/exp007_postmortem.py d3
```

가 오류 없이 끝나고 표를 출력한다. D2 는 실행했으면 숫자를, 못 했으면 실행 명령 한 줄을 보고한다.

## 보고 형식

**결론 문장 금지, 숫자와 표만.** D1 표 2개, D2 변형별 inner F1 과 유형별 recall, D3 상관값.
「~로 보인다」·권고·요약 산문 금지.
