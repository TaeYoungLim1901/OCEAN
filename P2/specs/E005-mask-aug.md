# E005 — 공개층 결측 전용 mask-augmented 모델

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
루트 `AGENTS.md` §0 의 "코드는 codex 가 쓴다"는 Opus 에게 내린 지시이며 그 codex 가 너다.
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(2026-08-13 E004 1차 시도가 이 재귀로 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
(`xgboost`·`catboost`·`torch` 없음. LightGBM·sklearn·numpy·pandas 만 쓴다.)

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라. §7 표와 판정 1행만 남긴다.

## 1. 배경 (재확인 불필요)

E004 에서 행 단위 `missing_public` 하드 게이트를 채택했다 (`P2/src/gate_e004.py`,
`P2/cache/E004/metrics.json`, 채택 후보 `robust_blend`, 4-hold 평균 `1.089067`).
게이트 구조상 **결측 행에는 여전히 E002 예측이 그대로 들어간다.** 즉 test 행의 `0.297993`
비율은 E001 이후 한 번도 개선되지 않았다.

실측 `missing_public` 세그먼트 RMSE (E002 = 현재 게이트가 쓰는 값):

| hold | n | missing_public | complete_public |
|---|---|---|---|
| 2024_julaug | 7225 | 1.216 | 1.465 |
| 2024_sepoct | 289 | 1.135 | 0.506 |
| 2025_mayjun | 8269 | 1.814 | 1.195 |
| 2025_julaug | 7957 | 1.283 | 1.310 |

가설: 기존 모델은 공개층이 **있는** 상태로만 학습돼서 결측 시 대체 경로를 배우지 못했다.
학습 입력에 test 와 같은 결측 패턴을 주입(mask augmentation)하면 결측 regime 전용 경로가 생긴다.

## 2. 범위 — 허용하는 변경

- 새 파일 2개만: `P2/src/train_e005.py`, `P2/src/gate_e005.py`. 기존 `train_*.py`·`inference_*.py`·`common.py`·`check.py`·`gate_e004.py` 수정 금지.
- 피처 집합·모델 계열은 **E002(`train_e002.py`, fixed-depth)를 그대로** 쓴다. 새 피처 금지,
  하이퍼파라미터 임의 변경 금지. 유일한 변경은 **학습 입력의 결측 주입**과 그에 따른 결측 지시 피처다.
- 허용되는 추가 피처는 **결측 자체를 나타내는 지시 변수뿐**이다: 각 공개층의 관측 유무 플래그,
  사용 가능한 공개층 개수, 가장 가까운 유효 공개층까지의 층 거리. 이들은 공개층 입력만으로 계산된다.
- 외부 데이터·정답 참조 금지. `P2/P2_data/` 읽기 전용. `SEED = 42`. 상대경로.
- `sys.stdout.reconfigure(encoding="utf-8")`.

## 3. 후보 2개

E002 와 동일한 학습 세팅에서 결측 주입 방식만 다르게 한다.

- `maskaug` — 학습 행의 일부에 test 결측 패턴을 무작위 주입한 **단일 모델**.
  주입 비율은 hold 별 실측 결측 비율(`P2/cache/E004/gate_stats.json`, test `0.297993`)에 맞춘다.
  주입은 **연속 시간 블록 단위**로 한다(무작위 행 단위 주입 금지 — test 결측은 연속 구간이다).
- `missing_only` — 결측 주입된 행만으로 학습한 **결측 전용 모델**.

두 후보 모두 결측 행 예측에만 쓰인다. 완전 공개 행은 손대지 않는다.

## 4. 최종 예측 조립

```
pred[i] = cand_missing_model[i]        if missing_public_flag[i]
          E004_gate_robust_blend[i]    otherwise
```

`missing_public_flag` 는 **`gate_e004.py` 가 쓰는 함수를 import 해서 그대로** 쓴다. 재정의 금지.
완전 공개 행은 `P2/preds/test_gate_robust_blend_E004.npy` 값과 **비트 단위로 같아야** 한다.

## 5. 검증 설계

- E003/E004 와 **같은 4개 hold**(`2024_julaug`, `2024_sepoct`, `2025_mayjun`, `2025_julaug`),
  같은 세그먼트 분해, 같은 채점 함수(`by=` 필수).
- 학습은 각 hold 의 학습 구간만 쓴다. hold 구간 정보를 학습에 흘리지 마라.
- **runtime 가드**: hold 하나가 끝날 때마다 `P2/cache/E005/` 에 그 hold 의 예측·지표를 즉시 저장한다.
  전체가 40분을 넘으면 남은 hold 를 건너뛰고, 완료된 hold 수와 그때까지의 표를 보고한다(중단 자체는 실패가 아니다).

## 6. 수락 기준 — 사전등록 판정 규칙 (결과 보기 전 확정. 사후 변경 금지)

기준선은 **E004 게이트**(4-hold 평균 `1.089067`, 2025_julaug `1.271631`)다.
후보가 **아래 5개를 전부** 만족할 때만 채택한다.

1. 4-hold 평균이 E004 대비 **≥ 0.010 개선**
2. `missing_public` 세그먼트가 4개 hold 중 **≥ 3개에서 0.010 이상 개선**되고, 어떤 hold 에서도 **0.005 초과 악화 없음**
3. `complete_public` 행 예측이 E004 와 **비트 단위 동일**(어긋나면 조립 버그이므로 원인까지 보고)
4. test-like hold `2025_julaug` 전체가 E004 대비 **≥ 0.005 개선**
5. `n ≥ 200` 인 모든 세그먼트에서 E004 대비 악화 **≤ 0.030**

둘 다 통과하면 4-hold 평균이 낮은 쪽. 통과 후보가 없으면 **기각**하고 E004 를 유지한다.
규칙을 완화해 억지로 채택하지 마라.

## 7. 산출물과 보고

결과와 무관하게 항상 저장:

- `P2/preds/oof_{maskaug,missing_only}_E005.npy`, `P2/preds/test_{maskaug,missing_only}_E005.npy`
  (결측 행만 다른 **전체 길이** 벡터로 저장한다)
- `P2/cache/E005/metrics.json` — E004 와 같은 스키마(`means`, `reports[hold][model]`, `selected`, `accepted`)
- `P2/cache/E005/runtime.json` — hold 별 학습 초, 완료 여부

채택된 경우에만:

- `P2/subs/E005_gate.csv` (열 `station,layer,time,temp`, -5~45 유한값)
- 아래를 **직접 실행하고 출력 숫자를 보고**:
  ```
  python P2/src/submission_tools.py audit P2/subs/E005_gate.csv --pred P2/preds/test_{선택}_E005.npy
  python P2/src/check.py sub P2/subs/E005_gate.csv
  ```

보고 형식 (이 표만 출력):

```
표1 전체: model | 4hold_mean | Δ_vs_E004 | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug
표2 missing_public 세그먼트: model | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug (E004 기준값 행 포함)
표3 규칙 체크: cand | R1 | R2 | R3 | R4 | R5 (칸마다 실제 숫자와 pass/fail)
표4 위반 세그먼트: cand | hold | segment | n | E004 | cand | Δ (없으면 "없음")
표5 runtime: hold | 초 | 완료여부
판정: 채택 <cand> 또는 판정: 기각
```
