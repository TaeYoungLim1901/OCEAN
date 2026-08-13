# E004 — missing_public 게이트 혼합 (재학습 없음)

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓴다.** 루트 `AGENTS.md` §0 의 "코드는 codex 가 쓴다"는
Opus 에게 내린 지시이며, 그 codex 가 너다. `cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀).

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만 보고한다.** 산문 요약·소감·다음 제안 쓰지 마라.
마지막에 아래 §6 형식의 표와 `판정: 채택 <후보>` 또는 `판정: 기각` 한 줄만 남긴다.

## 1. 배경 (읽고 넘어갈 것, 재확인 불필요)

E003에서 enhanced/eof/robust_blend는 4-hold 평균을 개선했으나 `missing_public` 세그먼트가
악화해 사전 차단선에 걸려 전부 기각됐고 E002만 제출 가능 상태다.

`P2/cache/E003/metrics.json` 실측(세그먼트 RMSE):

| hold | 모델 | missing_public | complete_public |
|---|---|---|---|
| 2025_julaug | e002 | 1.2828 | 1.3096 |
| 2025_julaug | enhanced | 1.3491 | 1.2898 |
| 2025_julaug | eof | 1.3402 | 1.2580 |
| 2024_sepoct | e002 | 1.1354 | 0.5063 |
| 2024_sepoct | enhanced | 1.2134 | 0.4922 |
| 2024_sepoct | eof | 1.1251 | 0.4905 |

즉 결측 regime에서는 E002가, 완전 공개 regime에서는 eof/enhanced가 낫다.
E004는 **행 단위 하드 게이트**로 두 예측을 갈라 쓴다. 새 모델 학습·새 피처 금지.

## 2. 범위 — 이 사양이 허용하는 변경

- 새 파일 `P2/src/gate_e004.py` 하나만 만든다. 기존 `train_*.py`·`inference_*.py`·`common.py`·`check.py` 수정 금지.
- 기존 저장 예측만 사용한다: `P2/preds/oof_{e002,enhanced,eof,robust_blend}_E003.npy`,
  `P2/preds/test_{e002,enhanced,eof,robust_blend}_E003.npy`, 인덱스는 `P2/cache/E003/oof_index.csv`.
- 모델 재학습·하이퍼파라미터 변경·새 피처·외부 데이터 금지.
- `P2/P2_data/`는 읽기 전용.
- **새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.** (`xgboost`·`catboost`·`torch` 없음)
- 콘솔 cp949 대응: 스크립트 상단에 `sys.stdout.reconfigure(encoding="utf-8")`. `SEED = 42`. 상대경로.

## 3. 게이트 정의

`missing_public` 세그먼트를 만드는 **기존 코드의 정의를 그대로 재사용**한다
(`P2/src/` 안에서 `missing_public` 을 grep 해 정의 함수를 찾고, 복붙이 아니라 import 해서 쓴다.
import이 불가능한 구조면 정의를 그대로 옮기되 옮긴 위치와 원본 줄 번호를 보고한다).

게이트 규칙:

```
pred_gated[i] = pred_e002[i]        if missing_public_flag[i]
                pred_cand[i]        otherwise
```

`missing_public_flag` 는 **공개층 입력만으로** 계산 가능해야 한다(정답 미사용). test 세트에도
동일 함수로 계산한다. 정답을 참조해 flag를 만들면 즉시 중단하고 보고하라.

후보 `cand` 3개: `eof`, `enhanced`, `robust_blend`. 각각에 대해 게이트 예측을 만든다.

## 4. 평가

E003과 **동일한 4개 hold**(`2024_julaug`, `2024_sepoct`, `2025_mayjun`, `2025_julaug`)와
동일한 세그먼트 분해(`layer_2/3/4`, `missing_public`, `complete_public`, `strat_low/mid/high`,
`transition`, `calm`)로 채점한다. 채점 함수는 E003이 쓴 것을 그대로 쓴다(`by=` 인자 필수).

기준선 실측값(재계산해서 일치하는지 확인하고, 불일치하면 그 사실을 표에 적어라):
E002 4-hold 평균 `1.161517`, 2025_julaug 전체 `1.301486`, 2024_sepoct 전체 `0.517441`.

## 5. 수락 기준 — 사전등록 판정 규칙 (결과 보기 전 확정. 사후 변경 금지)

후보 하나가 **아래 5개를 전부** 만족할 때만 채택한다.

1. 4-hold 평균 RMSE가 E002 대비 **≥ 0.010 개선**
2. test-like hold `2025_julaug` 전체 RMSE가 E002 대비 **≥ 0.005 개선**
3. 모든 hold의 `missing_public` 세그먼트 악화 **≤ 0.002** (구조상 0이어야 한다. 0이 아니면 게이트 버그이므로 원인까지 보고)
4. `n ≥ 200` 인 모든 세그먼트에서 E002 대비 악화 **≤ 0.030**
5. 어떤 hold에서도 전체 RMSE가 E002보다 악화되지 않음 (악화 허용 0.000)

여러 후보가 통과하면 우선순위: `eof` → `enhanced` → `robust_blend`.
통과 후보가 없으면 **기각**하고 E002를 유지한다. 규칙을 완화해 억지로 채택하지 마라.

## 6. 산출물

결과와 무관하게 항상 저장한다:

- `P2/preds/oof_gate_{cand}_E004.npy`, `P2/preds/test_gate_{cand}_E004.npy` (후보 3개 전부)
- `P2/cache/E004/metrics.json` — E003 metrics.json과 같은 스키마(`means`, `reports[hold][model].overall/n/segments`, `selected`, `accepted`)
- `P2/cache/E004/gate_stats.json` — hold별·test별 `missing_public_flag` True 비율과 행 수

채택된 경우에만 추가로:

- `P2/subs/E004_gate.csv` (열 순서 `station,layer,time,temp`, 값 -5~45 유한)
- 검증 명령을 **직접 실행하고 출력 숫자를 보고**:
  ```
  python P2/src/submission_tools.py audit P2/subs/E004_gate.csv --pred P2/preds/test_gate_{cand}_E004.npy
  python P2/src/check.py sub P2/subs/E004_gate.csv
  ```

## 7. 보고 형식 (이 표만 출력)

```
표1 후보별 전체: cand | 4hold_mean | Δ_vs_E002 | 2024_julaug | 2024_sepoct | 2025_mayjun | 2025_julaug
표2 규칙 체크: cand | R1 | R2 | R3 | R4 | R5   (각 칸에 실제 숫자와 pass/fail)
표3 위반 세그먼트: cand | hold | segment | n | e002 | cand | Δ   (R3·R4 위반 행만. 없으면 "없음")
표4 게이트 비율: hold/test | n | missing_flag_true | 비율
판정: 채택 <cand>  또는  판정: 기각
```
