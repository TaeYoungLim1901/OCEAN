# R2 — 루트 안전장치 4종 (재귀 가드 · rules_runtime · preflight rules/final · 세션 훅)

실행 파이썬: `python`. 셸: Git Bash. 리포 루트에서 실행한다.

## 배경 — 재조사 금지 사실 (확정)

- OCEAN 2026 은 문제 3개(P1·P2·P3). 실제 작업은 codex `gpt-5.6-luna` `CX_EFFORT=xhigh` 가 한다.
- 계획 원본은 `docs/PORT-PLAN-2026-08-25.md` 다. **이 사양은 그 계획의 1·2·7·9 단계다.** 계획을 다시 설계하지 마라.
- 검증 완료: 현재 `scripts/` 에는 `cx.cmd` `cx.sh` `g.py` `gk.cmd` `gk.sh` `preflight.py` 만 있다.
  `scripts/rules_runtime.py` 와 `.claude/hooks/` 는 존재하지 않는다. `.claude/` 아래에는 `skills/` 만 있다.
- `preflight.py` 의 현재 서브커맨드는 `spec` `sub` `selftest` 3개다. `verdict(fails, warns)` 헬퍼가 이미 있다.
- `cx.sh` 는 `set -euo pipefail`, `sandbox` `model` `effort` 변수, `-r`/`-s`/`-f` 파싱, `spec` 인자 검사,
  그리고 `python "$root/scripts/preflight.py" spec "$spec"` 호출 순서로 되어 있다.
- 규정 원문은 `RULES.md` 이고 하드 제약 표는 `## 0. 실행에 영향 주는 하드 제약` 절의 마크다운 표다.
  표 열은 `항목 | 규정 | 출처(공고문 절 / 공지) | 위반 시` 4개다.
- 세 문제의 public 제출 대장은 `P1/experiments/public_scores.csv` (P2·P3 동형)이고
  헤더는 `sha256,file,mtime,recorded_at,public,local,note` 다.

## 기준선 (baseline. 재측정 금지)

현재 `python scripts/preflight.py` 는 `{spec,sub,selftest}` 만 받는다. `rules` 와 `final` 은 없다.
`grep OCEAN_CX scripts/cx.sh` 는 0건이다. 세 문제 `bugcheck.py all` 은 전부 `판정: STOP` 이다.

## 작업

### R2-1. `cx.sh` 재귀 가드 (최소 변경)

1. `set -euo pipefail` 직후에 진입부 검사를 넣는다:
   `OCEAN_CX` 환경변수가 `1` 이면 `codex 안에서 codex 를 부를 수 없다` 를 stderr 로 내고 `exit 2`.
2. codex 자식 프로세스에 그 표시를 전달한다. `cx.sh` 가 이미 `shell_environment_policy.set.PYTHONPATH` 를
   `-c` 옵션으로 넘기고 있다. **같은 방식으로** `shell_environment_policy.set.OCEAN_CX='1'` 을 추가한다.
   값은 TOML 이므로 **작은따옴표**를 쓴다.
3. 기존 `-r`/`-s`/`-f` 동작·`preflight spec` 호출 순서·`PYTHONPATH` 주입은 그대로 둔다.

### R2-2. `scripts/rules_runtime.py` — 규정 단일 파서

`RULES.md` `## 0.` 절의 표만 읽는다. 다른 절·다른 파일을 읽지 마라.

- 함수 `load()` 가 dict 를 돌려준다. 최소 키:
  `deadline_pred` (대학부 예선 제출 마감 datetime, 2026-09-07 10:00 KST),
  `deadline_model` (모델 제출 마감 date, 2026-09-07),
  `quota_per_problem_per_day` (int, 3),
  `submit_open` (date, 2026-08-25).
  값은 표 안의 문장에서 정규식으로 뽑는다. **코드에 날짜·횟수를 상수로 적지 마라.**
- 다음 중 하나라도 걸리면 `RuntimeError`: 해당 행이 없음 / 같은 항목 행이 2개 이상 / 정규식이 못 뽑음.
- CLI: `python scripts/rules_runtime.py --plain` 은 키=값 한 줄씩 stdout.
  `python scripts/rules_runtime.py --selftest` 는 위 4키를 뽑고 값 타입을 확인한 뒤 `판정: OK` 출력 exit 0,
  실패면 `판정: STOP` exit 1.
- 표준 라이브러리만 쓴다 (`re`, `datetime`, `pathlib`). `sys.stdout.reconfigure(encoding="utf-8")` 를 상단에 둔다.

### R2-3. `preflight.py rules` — 규정 복사본 금지 게이트

새 서브커맨드 `python scripts/preflight.py rules`.

- `rules_runtime.load()` 를 부른다. 예외 발생 시 그대로 FAIL.
- 아래 경로에서 **하드코딩된 마감일·제출 횟수 복사본**을 찾는다:
  `AGENTS.md`, `CLAUDE.md`, `PLAYBOOK.md`, `OPUS_HANDOFF_*.md`, `P1/AGENTS.md`, `P2/AGENTS.md`, `P3/AGENTS.md`,
  `P1/NEXT.md`, `P2/NEXT.md`, `P3/NEXT.md`, `P1/graph.json`, `P2/graph.json`, `P3/graph.json`.
  탐지 패턴: `9\s*[.월]\s*7`, `2026-09-07`, `09/07`, `하루\s*3\s*회`, `일\s*3\s*회`, `3\s*회/일`.
- 걸린 파일이 있으면 `파일:행번호` 를 FAIL 로 나열한다. `RULES.md` 와 공고문 원문 파일은 검사 대상이 아니다.
- 통과 시 `rules_runtime` 값도 함께 출력한다. 판정은 기존 `verdict(fails, warns)` 를 재사용한다.

### R2-4. `preflight.py final` — 세 문제 전역 READY 장벽

`python scripts/preflight.py final --p1 <csv> --p2 <csv> --p3 <csv>`

세 인자 전부 필수다. 아래를 **전부** 통과할 때만 마지막 줄에 `판정: READY`, exit 0. 하나라도 실패면
`판정: NOT READY` 와 실패 목록, exit 1.

1. 각 `P1/subs/<이름>.manifest.json` (P2·P3 동형) 이 존재한다.
2. manifest 의 `csv_sha256` 이 지금 파일의 sha256 과 일치한다. `pred_npy`·`pred_sha256` 도 같다.
3. manifest 의 `code_files` 와 `weights` 에 적힌 모든 경로가 존재하고 sha256 이 일치한다.
4. `python P{n}/src/bugcheck.py all` 이 세 문제 모두 exit 0.
5. `python P{n}/src/check.py final <csv>` 가 세 문제 모두 exit 0.
6. `python scripts/preflight.py rules` 가 exit 0.

**manifest 스키마 (세 문제 공통. 이 사양이 계약이다)**

```json
{"csv": "P1/subs/x.csv", "csv_sha256": "...", "pred_npy": "P1/preds/test_x.npy",
 "pred_sha256": "...", "code_files": {"P1/src/train.py": "..."},
 "weights": {"P1/cache/x/model.txt": "..."}, "seed": 42, "created": "ISO8601"}
```

sha256 은 대문자 hex 로 통일한다 (`P1/src/submission_tools.py` 의 기존 `audit` 출력과 같은 형식).
4·5 번 하위 프로세스는 `subprocess.run([sys.executable, ...])` 로 부르고 stdout 마지막 3줄만 중계한다.

### R2-5. `.claude/hooks/ocean_context.py` — 세션 시작 컨텍스트

- `rules_runtime.load()` 만 출처로 쓴다. **날짜·횟수를 훅 안에 복사하지 마라.**
- `AGENTS.md §1.1` 의 자원 숫자(`n_jobs` 등)는 **훅에 넣지 마라.**
- 출력 (3~5줄, stdout):
  - 오늘 날짜, 예선 답안 마감까지 남은 일수, 모델 제출 마감까지 남은 일수.
  - 문제별 오늘 사용 제출 횟수: `P1/experiments/public_scores.csv` (P2·P3 동형) 의 `recorded_at` 이 오늘(KST)인 행 수와
    `quota_per_problem_per_day` 를 `P1 1/3  P2 0/3  P3 3/3` 형식으로.
  - 세 문제 중 quota 를 다 쓴 문제가 있으면 그 문제 이름을 한 줄로 경고.
- `--selftest` 는 위 계산을 수행하고 `판정: OK` exit 0 / 실패 시 `판정: STOP` exit 1.
  `--plain` 은 훅 장식 없이 본문만 출력한다.
- `.claude/settings.json` 에 `SessionStart` 훅으로 등록한다. **파일이 있으면 병합하고 덮어쓰지 마라.**
  `.claude/settings.local.json` 은 건드리지 마라. 명령은
  `python .claude/hooks/ocean_context.py --plain` 이며 상대경로로 적는다.

## 제약

- 새 의존성 금지. 표준 라이브러리만 쓴다. **`pip install` 이 필요하면 즉시 멈추고 그 사실을 보고하라.**
- 기존 인터페이스 유지: `preflight.py spec|sub|selftest`, `cx.sh` 의 `-r`/`-f`, `g.py` 전부 그대로.
- `P{n}/src/`·`P{n}/graph.json`·`RULES.md` 를 수정하지 마라. 이 사양은 루트 파일만 만진다.
- 과한 추상화 금지. 클래스·설정파일·로깅 프레임워크를 만들지 마라. `print` 로 충분하다.
- 원본 데이터 폴더(`P1/P1_data` 등)를 읽지 마라.

## 수락 기준 — 아래 명령을 직접 실행하고 출력 숫자를 보고한다

```bash
grep -c "OCEAN_CX" scripts/cx.sh                      # >= 2
OCEAN_CX=1 bash scripts/cx.sh -r P1/specs/R2-root-guards.md ; echo "exit=$?"   # exit=2
python scripts/rules_runtime.py --selftest ; echo "exit=$?"                    # 판정: OK, exit=0
python scripts/rules_runtime.py --plain
python scripts/preflight.py rules ; echo "exit=$?"
python .claude/hooks/ocean_context.py --selftest ; echo "exit=$?"               # exit=0
python .claude/hooks/ocean_context.py --plain
python scripts/preflight.py selftest ; echo "exit=$?"                           # 회귀 없음. exit=0
python scripts/preflight.py spec P1/specs/R2-root-guards.md ; echo "exit=$?"    # 회귀 없음
python scripts/preflight.py final --p1 P1/subs/exp002.csv --p2 P2/subs/E002_final.csv --p3 P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
```

마지막 `final` 은 지금 시점에 **NOT READY(exit 1)가 정상이다.** manifest 와 `check.py final` 이 아직 없다.
실패 목록이 그 사실을 정확히 가리키는지만 확인한다. **NOT READY 를 없애려고 검사를 완화하지 마라.**

`preflight.py rules` 가 FAIL 이면 걸린 문서에서 마감일·횟수 복사본을 지우고 `RULES.md §0 참조` 로 바꾼다.
단 `P1/graph.json`·`P2/graph.json`·`P3/graph.json` 은 **수정하지 말고 FAIL 목록으로 보고만 하라.**

## 완료 보고 — 결론 문장 금지, 숫자와 표만

| 항목 | 명령 | exit | 핵심 출력 1줄 |
로 표 1개. 그 아래에 변경 파일 경로 목록과 각 파일 추가/삭제 줄 수만 적는다. 산문 금지.
