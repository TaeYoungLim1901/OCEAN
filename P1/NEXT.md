# P1 재개 지점 — 2026-08-26

## 세션 시작 게이트 (2026-08-26 확정)

착수 전 2줄. 실패하면 실험하지 말고 원인부터 고친다. **검사를 완화하지 마라.**

```powershell
python P1/src/bugcheck.py all                    # 판정: OK 여야 한다
python scripts/g.py --p P1 check "<키워드>"       # exit 1 이면 닫힌 축
```

제출 후보를 확정했을 때:

```powershell
python P1/src/submission_tools.py audit <제출csv> --pred <예측npy>   # --pred 는 필수다
python P1/src/check.py final <제출csv>                              # 빈 폴더 바이트 재현 + manifest
```

**모델 제출 직전에만** (세 문제 후보가 전부 정해졌을 때. 1회 확정이라 복구 불가):

```powershell
python scripts/preflight.py final --p1 <P1csv> --p2 <P2csv> --p3 <P3csv>
```

2026-08-26 기준 세 문제 `bugcheck all` 전부 `판정: OK`, 전역 장벽 `판정: READY`.
마감·잔여 제출 횟수는 세션 시작 훅이 출력한다 (`.claude/hooks/ocean_context.py`).
규정 수치를 문서·graph 에 복사하지 마라 — `python scripts/preflight.py rules` 가 잡는다.

## 현재 성적

| 항목 | 값 |
|---|---|
| **최고 제출** | `P1/subs/exp010.csv` — public F1 **0.741274** / **26.456883점** |
| sha256 | `59D97691CE3E41408DA76D003CA5FB9DCDE9C0BC0A09ADBA47196E8AC413A769` |
| 설정 | exp005 파이프라인 + `t_high 0.775` / `hole 48` / dilation 0 / min-length 0 |
| 8/25 순이득 | +0.633606 (25.823277 → 26.456883). **재학습 0회, 임계 재조정만** |
| 1위(ChangCv)와 격차 | F1 기준 **+0.117** (0.858 추정) |

## Public 제출 대장 (`python P1/src/submission_tools.py public list`)

| 실험 | 양성률 | public F1 |
|---|---|---|
| exp011 | 2.1602% | 0.657820 |
| **exp010** | **2.9832%** | **0.741274** |
| exp005 | 3.4791% | 0.717435 |
| exp009 | 4.1808% | 0.698524 |

## 확정된 판정 원칙 — 어기지 마라

1. **`a.threshold_rate` 축 폐쇄.** 양성률-public 곡선 2차 적합 봉우리 **3.0191%**, peak F1 **0.741420**.
   exp010 은 최적에서 **0.000146** 이내다. 임계만 더 만져서 얻을 여지는 0.0002 미만 — 슬롯 쓰지 마라.
   곡선은 **비대칭**이다 (아래로 −0.0835, 위로 −0.0238). 새 모델 임계는 **3.0% 보다 살짝 위**에서 잡는다.
2. **inner 는 후처리·임계 판정기가 아니다** (`f.inner_delta_no_transfer`).
   exp009 는 inner +0.004606 인데 public **−0.018911** 로 부호가 뒤집혔다.
   후처리 격자 argmax 를 inner 로 고르지 마라. 유효 표본이 행이 아니라 **세그먼트**(offset 17·drift 12·noise 34)다.
   단 **후처리를 고정한 채 모델만 비교**하는 용도로는 쓴다 (exp005 inner 0.6978 → public 0.7174, 오프셋 안정).
3. **사전 판정 규칙을 결과 보기 전에 `log.md` 에 적는다.** 8/25 에 두 번 값을 했다 —
   exp009 를 승격선 0.702836 으로 걸렀고(실제 −0.0189), exp011 분기 예측도 맞혔다.
4. 봉인 fold3 는 **이미 3회 판독됐다.** 채점하지 마라.

## 다음 수 — 이 순서

### 1. P1 결함 2건 수정 (최우선. 안 고치면 다음 학습이 오염된다)

- **C1 지뢰**: `P1/src/features.py::train_mask_with_purge` 가 **대칭 purge 마스크**다.
  `bugcheck C1` 이 4 fold 합계 `future_rows=1,140,222` 로 잡았다. exp004 를 오염시킨 그 버그.
  수정: `keep = folds < held_out` + station-layer 별 `time < validation_start − 87h`.
  함수 내부에 `keep & (folds >= held_out)` 가 0 인지 **assert**.
  수락: `python P1/src/bugcheck.py C1` → `future_rows=0 mask_differences=0`
- **C4 이진화**: `P1/preds/oof|test_exp002~005.npy` 가 전부 **uint8 0/1**. 앙상블 재조합이 원천 불가였고
  임계 프로브 1건을 날렸다. 원시 확률은 `P1/cache/exp005_raw_{oof,test}.npy` 에 살아 있다.
  수정: `save_pool_pred(path, values)` 를 `submission_tools.py` 에 추가 — 저장 전 floating dtype·유한·[0,1]·
  고유값 3개 이상 검사 후 float32 저장, 위반 시 **저장하지 않고 예외**. `astype(float32)` 위장 금지.
  수락: `python P1/src/bugcheck.py C4` → `binary_uint8=0`
- 그 뒤 `python P1/src/bugcheck.py all` 의 C3(`config_issues=1`)·C8(`recorded_files=0`) 원인 제거.
  **검사를 완화해서 통과시키지 마라.**

### 2. V1 full-train + 제출 (오늘 슬롯 1번)

`f.v1_wins`: 후처리 exp005 고정(t 0.62 / hole 48) 기준 inner F1 —

| 변형 | feature | 트리 | inner F1 |
|---|---|---|---|
| exp005 | 42 | 700 | 0.697836 |
| **V1** | **47** | 700 | **0.702859** |
| V2 | 54 | 700 | 0.587059 |
| V3 | 81 | 1400 | 0.551970 |

**개수 절벽이다.** 5개는 이득, 12개는 붕괴, 트리 2배로도 복구 불가.
V1 이 고른 5개 — `temp_step_abs_144`, `max_z_center_144`, `temp_step_144`, `temp_step_abs_48`, `psal_step_abs_144`.
**전부 step 검출기 계열**로 T10 의 offset 진단과 일치한다.

**V1 은 inner OOF 만 있고 test 예측이 없다.** 해야 할 것:
전체 train 단일 모델 학습 → 원시 확률 **float 저장** → **양성률 3.0% 로 재임계** → 제출.
`t 0.62` 그대로 내면 양성률이 3.5% 근처가 되어 exp010 보다 나쁘다.
코드는 `P1/src/exp007_postmortem.py` 재사용. 학습은 사람이 실행 (`AGENTS.md` 예외 3).

### 3. 남은 미탐

T10(`P1/experiments/exp006_fn_decomp.json`): inner FN 6038 중 **offset 2156(recall 0.4254) + drift 1689(0.5104) = 63.7%**.
offset 세그먼트 17개 중 full_miss 8, 구간내 적중률 p50 **0.0083**.
drift 는 구간내 10분위 recall **0.2393 → 0.7361 단조 증가**.
flatline(0.894)·spike 는 건드리지 마라. **step feature 를 더 좁게 파는 것이 유일하게 열린 큰 레버다.**

## 닫힌 축 — 재제안 금지

`a.threshold_rate`(양성률 곡선으로 폐쇄) · `a.rate_anchoring`(train 4.1362% 정합이 최악) ·
`a.type_multiclass` · `a.random_rows` · `a.adversarial_drift` · hysteresis(exp009 public −0.0189) ·
min-length L=12(spike recall 0.686→0.059) · 적분 feature 12개 이상(V2·V3 붕괴).

## 구조 반영 — `docs/PORT-PLAN-2026-08-25.md` v3 (sol 적대검토 반영)

**새 문서를 만들지 않는다.** 사고는 `graph.json` 의 `f.incident_*`, 예방은 `g.*` gate 노드,
실행은 `bugcheck`/`check.py` exit code. 게이트를 못 붙이는 개선항목은 계약 문서에 넣지 마라 — 죽는다.

미반영 순서: 1(A `cx.sh` 재귀 가드, root) · 2(`rules_runtime.py` + D 훅, root) ·
**3·4(P1 C1·C4 수정 — 위 참조)** · 5(bugcheck all 정리) · 6(B `--pred` fail-closed) ·
7(`preflight.py rules`) · 8(빈 폴더 최종 재현 + manifest) · **9(세 문제 전역 READY 장벽, root)**.

**순서 9 가 가장 비싼 실패를 막는다** — 한 터미널의 조기 모델 제출이 세 문제 전부의 답안 업로드를 끝낸다.
전역 ready 상태가 없고 1회 확정이며 복구 불가다.

## 규정 (`RULES.md` 통합본, sol 검토 후 14건 수정 완료)

- 답안 제출 제한: **`RULES.md` §0 참조**. 슬롯 이월 없음
- 예선 마감: **`RULES.md` §0 참조** (자정 아님)
- **모델 제출 시 즉시 최종 확정** — 이후 답안 업로드 불가. 맨 마지막에 낸다
- 운영진이 **제출 모델의 답안 재현을 검증**한다. 실패 = 최종 성적 미인정.
  후처리(임계·hole fill)가 모델 산출물인지는 **명시 없음** → 모든 후처리를 추론 코드 안에 결정론적으로 넣어라
- 외부 데이터 사용 가능(평가는 대회 데이터로만). **안 쓴다** — 격차는 climatology 품질이 아니다
- 남은 슬롯은 **`RULES.md` §0와 public ledger로 계산**

## 위임

- 구현: `scripts\cx.cmd P1/specs/T{n}.md` (codex `gpt-5.6-luna` xhigh)
- 조사·웹검색: `scripts\gk.cmd P1/specs/Q{n}.md` (grok-4.6, 읽기 전용)
- 적대검토: `codex exec --model gpt-5.6-sol -c model_reasoning_effort=xhigh --sandbox read-only`
- 전부 `dangerouslyDisableSandbox: true` + `run_in_background: true`, 로그는 `P1/experiments/logs/`
