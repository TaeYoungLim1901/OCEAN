# OPS-D2B — 일일 마감 skill 최소화

루트 `AGENTS.md`를 준수한다. 구현은 Codex `gpt-5.6-luna`, xhigh가 한다. `pip install`이 필요하면 즉시 멈추고 보고한다.

## 배경 — 재조사 금지 사실

- OPS-D2 초안 skill은 quick_validate/selftest를 통과했지만 bundled `daily_close.py`가 593 LOC다.
- 독립 리뷰에서 첫 Git 실행을 막는 stage path 정규화 S0, Git 실패 후 기록 rollback 누락 S0, push 실패 뒤 no-op 재실행이 push를 건너뛰는 S0 등 3건이 확인됐다.
- 각 문제의 기존 `submission_tools.py audit/public add/list` selftest는 3/3 exit 0이고, Git/gh가 native 검증·commit·push를 제공한다.
- 기준선은 신규 실행기 593 LOC, skill 48 LOC, UI metadata 4 LOC다. 목표는 신규 실행 코드 0 LOC, skill 본문 100줄 이하다.
- private 원격은 `TaeYoungLim1901/OCEAN`; 2026-08-20 이전 push 0회; 로컬 `.git` 0개다.
- 표준 경로 `P3/specs/`, `P3/experiments/`, `P3/preds/`, `P3/subs/`는 존재한다.

## 수락 기준

- `.claude/skills/ocean-daily-close/scripts/daily_close.py`와 빈 scripts 디렉터리를 삭제한다.
- skill은 `SKILL.md`, `agents/openai.yaml` 2파일만 유지한다.
- SKILL.md 100줄 이하, 신규 Python/PowerShell/Bash 코드 0줄.
- quick_validate exit 0.
- 오늘 `.git`, commit, push, 실제 public score 파일 변경 0건.

## SKILL.md 계약

description은 다음 요청에만 trigger하도록 명확히 한다: 2026-08-20 이후 P1·P2·P3 Public 점수 3개 기록, 일일 마감, private GitHub snapshot/push. 로컬 채점만 요청하면 기존 `score` skill 영역이다.

본문은 coordinator와 delegated Luna 역할을 분리한다.

### Coordinator

1. 사용자에게 날짜, P1/P2/P3 각각의 실제 제출 CSV, 대응 NPY, Public 점수 3개를 받는다. 하나라도 없으면 멈추고 요청한다.
2. 날짜가 2026-08-20 이상이며 서울 현지 오늘 이하인지 확인한다. 세 점수가 나오기 전, 또는 다른 P 세션/학습/Codex가 active이면 실행하지 않는다.
3. 날짜와 3×(CSV, NPY, 점수), 공통 note를 하나의 `P3/specs/OPS-YYYY-MM-DD-daily-close.md` 사양에 쓴다. 기존 파일과 충돌하면 덮어쓰지 않는다.
4. 사양에 아래 Luna 절차·금지·보고 숫자를 그대로 넣고 `python scripts/preflight.py spec <사양>`이 `판정: OK`일 때만 `scripts\cx.cmd -f <사양>` 한 번으로 위임한다. 점수 기록과 Git push를 둘로 쪼개지 않는다.

### Delegated Luna — 한 번의 호출에서 순서 고정

1. 자신이 위임 실행자이면 `cx.cmd`를 재귀 호출하지 않는다. 입력 경로가 각 문제의 subs와 preds 하위인지, 파일 존재, 점수 유한성과 P1 0~1/P2·P3 0 이상을 확인한다.
2. 세 문제 모두 `python P{n}/src/submission_tools.py audit <csv> --pred <npy>`를 먼저 실행한다. 하나라도 실패하면 점수 기록·Git 변경 0건으로 STOP한다.
3. 기존 각 문제 experiments의 `public_scores.csv`에서 동일 SHA를 찾는다. 동일 SHA·동일 점수이면 해당 문제 `public add`를 건너뛴다. 동일 SHA·다른 점수이면 덮어쓰지 말고 STOP한다.
4. 남은 문제만 `python P{n}/src/submission_tools.py public add <csv> <score> --note <note>`로 기록한 뒤 세 문제 `public list`를 출력한다. 한 문제 기록이 실패하면 Git은 건드리지 않고 실패 문제와 이미 기록된 문제 번호만 보고한다. 재실행은 3번 규칙으로 안전하게 이어간다.
5. Git 단계 직전 `gh auth status`와 `gh repo view TaeYoungLim1901/OCEAN --json nameWithOwner,isPrivate`를 검사한다. private/name 불일치면 Git 변경 없이 STOP한다.
6. `.git`이 없을 때만 `git init -b main`과 `git remote add origin https://github.com/TaeYoungLim1901/OCEAN.git`을 한다. 기존 origin의 fetch/push URL이 같은 owner/repo가 아니거나 현재 branch가 main이 아니면 STOP한다.
7. `git add .` 금지. 다음 allowlist만 명시적으로 `git add -A --`한다: `.gitignore`, `.claude`, `docs`, `scripts`, 루트 운영 Markdown, 각 문제의 `AGENTS.md`, `graph.json`, `src`, `specs`, `experiments\log.md`, `experiments\public_scores.csv`.
8. commit 전 staged 이름·크기를 출력한다. data, cache, models, preds, subs, experiments logs, `.env`/pem/key, binary/model/pred 확장자, 50MB 초과, secret 후보, 삭제 파일이 하나라도 있으면 commit/push 0으로 STOP한다. `git diff --cached --check`도 exit 0이어야 한다. 이때 index만 원복하고 worktree는 절대 바꾸지 않는다.
9. staged 변경이 있으면 `scores: YYYY-MM-DD public results`로 commit한다. 없으면 commit 0건이다. `git push -u origin main` 후 local HEAD와 origin/main SHA 일치를 검사한다. 이전 push 실패 뒤 재실행이면 점수 기록을 건너뛰고 기존 commit을 push한다.
10. 최종 보고는 각 audit exit/SHA/Public/최고 대비 delta, 기록 또는 skip, staged 파일 수/bytes, commit SHA, local/remote SHA, init/commit/push 횟수만 표로 낸다. 결론 문장 금지.

### 금지

- 2026-08-20 전 GitHub push, 점수 1~2개만 기록, Public 점수 추측, 제출/예측/모델/데이터 업로드.
- 자동 scheduler/CI/LFS/lock/queue, `git add .`, branch switch/merge/rebase, worktree reset/checkout.
- Public 결과를 자동으로 graph/log 판단에 반영. 숫자를 받은 뒤 사람/Opus가 판정한다.

## 검증

- `$env:PYTHONUTF8='1'; python C:/Users/livde/.codex/skills/.system/skill-creator/scripts/quick_validate.py .claude/skills/ocean-daily-close` exit 0.
- `rg --files .claude/skills/ocean-daily-close` 결과 2파일.
- `rg -n "daily_close.py|git add \\.|scheduler|CI|LFS"`로 금지 잔존을 확인하되 설명상 금지 문구와 실행 지시를 구분해 표로 보고한다.
- 허용 변경은 `.claude/skills/ocean-daily-close/**`뿐이다. `.gitignore` 3개 예외는 그대로 유지한다.

## 보고

결론 문장 금지. 파일 수, LOC, 삭제 LOC, quick_validate exit, 오늘 git init/commit/push, 실제 점수 파일 변경, 허용 밖 변경 수만 표로 보고한다.
