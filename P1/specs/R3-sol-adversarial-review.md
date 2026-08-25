# R3 — luna 산출물 적대검토 (읽기 전용. sol 전용)

실행 파이썬: `python`. **이 호출은 읽기 전용이다. 파일을 만들거나 고치거나 지우지 마라.**
검사 명령 실행은 허용한다. 코드 수정은 금지다. 고칠 것은 `수정 지시` 로 적기만 한다.

## 배경 — 재조사 금지 사실 (확정)

2026-08-26 에 codex `gpt-5.6-luna` `CX_EFFORT=xhigh` 가 사양 4개를 수행했다.

| 사양 | 산출 |
|---|---|
| `P1/specs/R2-root-guards.md` | `scripts/cx.sh` 재귀 가드, `scripts/rules_runtime.py`, `preflight.py rules`·`final`, `.claude/hooks/ocean_context.py`, `.claude/settings.json` |
| `P1/specs/T19-p1-guards.md` | P1 C1·C3·C4·C8 수정, `save_pool_pred`, `audit --pred` 필수, `check.py final` |
| `P2/specs/E017-p2-guards.md` | P2 C2·C3·C7·C8 수정, 같은 3종 |
| `P3/specs/T24-p3-guards.md` | P3 C4·C8 수정, graph 중복 규칙 제거, 같은 3종 |

원 계획은 `docs/PORT-PLAN-2026-08-25.md` 다 (sol 이 v2→v3 로 적대검토한 그 문서).

**이미 확인된 사고 2건 (다시 조사하지 말고, 같은 계열이 더 있는지만 찾아라):**

1. **luna 가 `scripts/cx.sh` 를 CRLF 로 저장해 스크립트가 깨졌다.** `codex exec \` 줄연속이
   `\<CR>` 이 되어 `line 75: -c: command not found`. 사람이 LF 로 정규화해 복구했다.
2. **P3 `check.py final` 이 임시 디렉터리 비교가 아니라 원본을 덮어썼다.**
   `P3/subs/submission_p3_t1f.csv` 와 `P3/preds/test_final_p3_t1f.npy` 가 float32 로 재작성되어
   제출·채점된 SHA `2E498428...` 가 `B840E7A7...` 로 바뀌었다. 최대 절대차 `1.19e-07`.
   원값은 `t1f_bias + 0.041550308` 또는 `t1f_shift - 0.05` 로 완전 복원 가능하다.
   P1·P2 는 같은 사고가 없었다 (`4C96D5F3...`·`F88AC4BF...` 일치 확인).

## 기준선 (baseline. 재측정 금지)

luna 보고 수치: P1 `C1 1,140,222→0`, `C4 invalid 4→0 binary 6→0`, C8 `0→6 files`.
P2 `C2 FAIL→PASS`, `C7 0.000000→0.517441`, C8 `0→4`. P3 `C4 PermissionError→PASS`, C8 `0→8`.
세 문제 `bugcheck all` exit 0. 재학습 0회. `preflight.py rules` 는 graph 5건으로 STOP.

**이 수치를 믿지 마라. 검사가 완화돼서 PASS 가 된 것인지 확인하는 것이 이 검토의 핵심이다.**

## 검토 항목 — 우선순위 순

### A. 검사 완화 여부 (최우선)

`git diff` 로 `P1/src/bugcheck.py`·`P2/src/bugcheck.py`·`P3/src/bugcheck.py` 변경을 본다.
P2 는 `+2/-2`, P3 도 변경이 보고됐다. 각 hunk에 대해 판정하라:
**검사 대상 축소·임계 완화·예외 삼킴(`try/except: pass`)·조건 반전인가, 아니면 진짜 버그 수정인가.**
`verdict()` 가 여전히 FAIL 시 1 을 반환하는지 확인한다.

### B. `check.py final` 이 원본에 쓰는 경로

세 문제 `P1/src/check.py`·`P2/src/check.py`·`P3/src/check.py` 의 `final` 경로를 읽는다.
- `P1/subs/`·`P1/preds/`·`P1/models/` (P2·P3 동형) 아래에 **쓰기(`to_csv`·`np.save`·`open(...,'w')`)가 남아 있는가.**
- 재현 출력이 `tempfile.mkdtemp()` 밖으로 나가는 경로가 있는가.
- manifest 만 원본 폴더에 쓰는 것은 허용이다. 그 외 쓰기는 전부 결함이다.
- 재현이 기존 `P1/cache/` (P2·P3 동형) 중간 산출물을 읽어 **재현을 위장**하는 경로가 있는가.
  가중치·코드만으로 도는 것이 맞는지 확인하라.

### C. `--pred` 필수화가 실제로 fail-closed 인가

세 문제 `submission_tools.py` 의 `audit` 를 본다. argparse `required=True` 만 있고
**함수 API 가 여전히 `pred=None` 기본값을 받는 경로**가 남아 있으면 결함이다.
다른 모듈이 그 함수를 `pred` 없이 부르는 호출부가 있는지 `grep` 으로 확인하라.

### D. `preflight.py final` 이 실패를 READY 로 통과시키는가

`scripts/preflight.py` 의 `final` 을 읽는다.
- 6개 검사(manifest 존재·csv/pred sha·code/weights sha·`bugcheck all`·`check.py final`·`rules`) 중
  **빠지거나 조건부로 건너뛰는 것**이 있는가.
- 하위 프로세스 exit code 를 실제로 읽는가, 아니면 stdout 문자열만 보는가.
- 예외 발생 시 READY 로 빠지는 경로가 있는가.
- `--p1/--p2/--p3` 가 전부 필수인가.

### E. manifest 스키마 3문제 동일성

`P1/subs/exp002.manifest.json`, `P2/subs/E002_final.manifest.json`,
`P3/subs/submission_p3_t1f.manifest.json` 을 비교한다. 키 집합·sha 대소문자·경로 표기가 같은가.
**P2 manifest 의 `pred_npy` 가 실제 앙상블 pool 배열(`test_fixed_depth_E002.npy`, float64)이 아니라
재현용 신규 float32 배열(`test_E002_final.npy`)을 가리킨다.** 이것이 D 의 비교 종점으로 타당한가.

### F. P1 purge assert 가 대칭 잔존을 실제로 잡는가

`P1/src/features.py::train_mask_with_purge` 를 읽는다.
- `keep = folds < held_out` 와 station-layer 별 `time < validation_start - 87h` 가 **둘 다** 걸리는가.
- `assert (keep & (folds >= held_out)).sum() == 0` 가 `python -O` 로 죽지 않는가
  (`assert` 는 `-O` 에서 사라진다. 이 프로젝트에서 그것이 실제 위험인지 판정하라).
- 호출부가 이 함수를 우회하는 경로가 있는지 `grep` 으로 확인하라.

### G. `save_pool_pred` 우회 경로

세 문제의 예측 저장 폴더에 쓰는 코드를 전부 찾아라 (`np.save`·`np.savez`).
`save_pool_pred` 를 지나지 않는 writer 가 하나라도 남아 있으면 결함이다.
P1 은 `[0,1]` 범위 검사가 있고 P2·P3 는 없어야 한다 (회귀 문제). 반대로 돼 있으면 결함이다.

### H. 사양 범위 이탈

`git status --short` 와 `git diff --stat` 을 본다.
- **`P2/graph.json` 이 +97 줄 변경됐다.** `E017` 사양은 P2 graph 갱신을 지시하지 않았다.
  추가된 노드가 측정 근거 있는 사실인지, 아니면 luna 가 만들어낸 서술인지 판정하라.
- P1·P3 graph 변경도 같은 기준으로 본다.
- 사양이 지시하지 않은 피처·파라미터·분할·모델 변경이 있는가.

### I. CRLF 재발 방지

`scripts/*.sh` 와 `P{n}/src/*.py` 의 줄끝을 확인한다. CRLF 로 저장된 파일이 더 있는가.
`preflight.py selftest` 가 `scripts/*.sh` 의 CRLF 를 잡도록 하는 것이 타당한 방어인지 판정하라.

### J. P3 제출물 복구 방법 검증

`t1f_old = t1f_bias + 0.041550308` 과 `t1f_shift - 0.05` 두 공식이 같은 값을 주는지 직접 계산해 확인하라.
복구된 CSV 를 원 SHA `2E498428...` 와 **바이트 일치**시키려면 어떤 float 서식이 필요한지 판정하라.
`P3/src/` 의 기존 CSV writer 서식과 현재 파일 서식을 비교해서 답하라.

## 제약

- **읽기 전용.** 파일 생성·수정·삭제 금지. `git checkout`·`git stash` 등 작업트리 변경 금지.
- 새 의존성 금지. **`pip install` 이 필요하면 제안하지 말고 그 사실만 적어라.**
- 원본 데이터 폴더(`P1/P1_data` 등)를 읽지 마라.
- 계획을 다시 설계하지 마라. `docs/PORT-PLAN-2026-08-25.md` 의 결정은 확정이다.
- 「있으면 좋다」 수준의 개선 제안을 하지 마라. **구체적 실패를 막는 것만 적는다.**

## 수락 기준

A~J 각 항목에 판정이 있어야 한다. 판정은 `OK` / `결함` / `확인불가` 셋 중 하나다.
`결함` 은 반드시 `파일:행번호` 와 그 결함이 만드는 **구체적 실패 시나리오**를 함께 적는다.
근거 없이 `결함` 이라고 적지 마라. 코드를 직접 읽고 행번호를 대라.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `항목 | 판정 | 파일:행 | 실패 시나리오 1줄`.
표 2 (결함이 있을 때만): `순위 | 수정 지시 1줄 | 예상 크기(줄 수)`. 심각도 순.
그 외 산문 금지. 요약·소감·격려 문장 금지.
