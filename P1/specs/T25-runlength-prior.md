# T25 — 주입 지속시간 사전(prior)으로 후처리. 학습 없음

## 목적

문제지가 주입 규격을 **명시**했다. spike 는 10분(1행), 나머지는 최소 2시간(12행) 이상이다:

| 유형 | 지속시간 | 행 수 |
|---|---|---|
| spike | 10분 | 1 |
| noise | 3 ~ 58.8시간 | 18 ~ 353 |
| flatline | 2 ~ 47.2시간 | 12 ~ 283 |
| offset | 8 ~ 86.5시간 | 48 ~ 519 |
| drift | 9 ~ 86.5시간 | 54 ~ 519 |

train 라벨 run 263개를 실측한 결과 **길이 2~11 인 run 이 정확히 0개**다 (길이 1 이 101개, 12 이상이 162개).
즉 우리 예측에 길이 2~11 인 run 이 있으면 그것은 FP 이거나 긴 이상의 파편이다. 이 사전을 후처리에 넣는다.
**학습하지 않는다.** 기존 원시 확률만 쓴다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 최고 제출 exp010: `P1/cache/exp005_raw_test.npy` 에 `t_high 0.775` + `hole 48`, 양성 5042 (2.9832%),
  public F1 **0.741274**. 이 사양의 기준선이다.
- inner OOF 원시 확률: `P1/cache/exp005_raw_oof.npy` (fold1·2 forward, fold3 봉인).
- **inner 는 임계·후처리 선택기가 아니다** (`f.inner_delta_no_transfer`, `f.inner_not_model_judge`).
  이 사양에서 inner 는 **격자 argmax 선택에 쓰지 않고**, 「짧은 run 행이 실제로 FP 인가」라는
  **사실 측정**에만 쓴다. 격자를 만들어 최고점을 고르지 마라.
- T14 에서 min-length L=12 를 적용했더니 spike recall 0.686 → 0.059 로 붕괴했다 (exp008).
  **길이 1 run 은 절대 지우지 마라.** 지우는 대상은 길이 2~11 뿐이다.
- 후처리 순서는 고정: 임계 → hole fill 48 → 이 사양의 run 규칙. 다른 형태 연산 추가 금지.
- `numpy`·`pandas` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**

## 하지 말 것

- 학습 금지, 새 feature 금지, 임계 격자 탐색 금지, 봉인 fold3 판독 금지.
- 새 파일은 `P1/src/t25_runlength.py` 하나. 기존 파일 수정 금지. `SEED=42`.

## 측정 (inner = fold1·2 행만)

`exp005_raw_oof.npy` 에 `t 0.62` + `hole 48` 을 적용한 예측에서 run 을 길이 구간별로 나눈다.

**표 M1 — 예측 run 길이대별 정밀도**

| len_bucket | n_runs | n_rows | TP_rows | precision | 그 버킷을 지웠을 때 inner F1 |
|---|---|---|---|---|---|

버킷: `1`, `2-11`, `12-47`, `48-119`, `120-519`, `520+`.

**표 M2 — 파편 확장 효과**: 길이 2~11 run 각각을 중심에서 좌우로 늘려 길이 12 로 만들었을 때
추가되는 행 중 TP 비율, 그리고 inner F1.

**표 M3 — test 예측의 run 길이 분포** (라벨 없음, 개수만):
exp010 설정(`exp005_raw_test.npy`, t 0.775, hole 48)과 `exp016_raw_test.npy` 가 있으면 그것도.

| source | len_bucket | n_runs | n_rows |

## 제출물 생성

`exp005_raw_test.npy` + `t 0.775` + `hole 48` 을 기준으로 두 변형을 만든다. **임계는 재조정하지 않는다.**

- `exp017` — 길이 2~11 run **삭제**
- `exp018` — 길이 2~11 run 을 **길이 12 로 확장**

각각 `P1/subs/{exp}.csv`, `P1/preds/test_{exp}.npy` (`exp005.store_score` 규약, threshold 0.775 기준),
`P1/cache/{exp}_meta.json` 저장. 저장 후 아래를 직접 실행하고 마지막 3줄 보고:
`python P1/src/submission_tools.py audit P1/subs/{exp}.csv --pred P1/preds/test_{exp}.npy`,
`python P1/src/check.py sub P1/subs/{exp}.csv`.

**표 S1** — exp010 대비 변화:

| exp | n_positive | 양성률 | Δ행 | 영향받은 run 수 | sha10 | audit |

## 수락 기준

| 항목 | 기준 |
|---|---|
| 표 M1·M2·M3·S1 | 전부 출력 |
| 길이 1 run | 두 변형 모두에서 보존 (개수 불변을 print) |
| audit / check sub | 두 exp 다 `판정: OK` |
| 실행 시간 | 5분 이내 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 산문·코드 붙여넣기 금지.

## 실행

```
python P1/src/t25_runlength.py
```
