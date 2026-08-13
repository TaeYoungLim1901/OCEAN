# OPS-D2 — Public 점수 일일 마감 skill/tool

루트 `AGENTS.md`를 준수한다. 구현은 Codex `gpt-5.6-luna`, xhigh가 한다. `pip install`이 필요하면 즉시 멈추고 보고한다.

## 배경 — 재조사 금지 사실

- 2026-08-20 제출분까지 P1·P2·P3 후보가 준비되었고, 다음 실험은 Public 점수 확인 뒤 시작한다.
- 각 문제의 `src\submission_tools.py`에 `audit`, `public add/list`, `selftest`가 있고 selftest 3/3 exit 0이다.
- P1 지표는 F1(클수록 좋음, 0~1), P2·P3는 RMSE(작을수록 좋음, 0 이상)다.
- private 원격은 `TaeYoungLim1901/OCEAN`, 오늘 원격 push 0회, 로컬 `.git`은 아직 없다.
- 현재 `.gitignore`는 data, cache, models, preds, subs, logs 및 각 문제 experiments 아래 CSV를 제외한다. 코드·문서 후보 104파일/590828B, 금지 경로 후보 0건이다.
- 표준 경로 `P3/specs/`, `P3/experiments/`, `P3/preds/`, `P3/subs/`는 존재한다.

## 목표

2026-08-20부터 사용자가 P1·P2·P3의 Public 점수와 실제 제출 CSV/대응 NPY를 모두 준 뒤에만, 한 명령으로 감사·기록·private GitHub snapshot/push·숫자 출력을 수행한다. 오늘은 selftest만 하고 git init/commit/push하지 않는다.

## 기준선과 수락 기준

- 기준선은 수동 명령 최소 13개, 실제 Public 장부 0개, 오늘 push 0회다.
- 수락 기준은 아래 자체검증 전부 exit 0, 오늘 git init/commit/push 각 0건, 허용 밖 변경 0개다.

## 최소 구조

skill-creator의 `init_skill.py`로 신규 프로젝트 skill을 초기화한다.

```text
.claude/skills/ocean-daily-close/
  SKILL.md
  agents/openai.yaml
  scripts/daily_close.py
```

- 이름: `ocean-daily-close`
- UI: `OCEAN Daily Close`; 25~64자 짧은 설명; default prompt는 `$ocean-daily-close`를 명시한다.
- README, references, assets, 별도 테스트 파일은 만들지 않는다.
- `.gitignore`에는 각 문제 experiments의 `public_scores.csv`만 다시 포함하는 예외를 추가한다.

## CLI

```powershell
python .claude/skills/ocean-daily-close/scripts/daily_close.py 2026-08-20 `
  --p1 P1/subs/<csv> <public> P1/preds/<npy> `
  --p2 P2/subs/<csv> <public> P2/preds/<npy> `
  --p3 P3/subs/<csv> <public> P3/preds/<npy> `
  --note "<공통 메모>"
python .claude/skills/ocean-daily-close/scripts/daily_close.py selftest
```

stdlib만 사용한다. 저장소 루트 밖 또는 문제 번호와 불일치하는 CSV/NPY를 거부한다.

## 실행 계약

1. 날짜는 ISO 형식이고 2026-08-20 이상, 로컬 오늘 이하만 허용한다. P1/P2/P3 입력을 모두 요구한다.
2. 점수 범위와 유한성을 먼저 검사한다. 파일 존재·문제별 `subs`, `preds` 하위 여부를 검사한다.
3. 세 문제의 기존 `submission_tools.py audit <csv> --pred <npy>`를 모두 먼저 실행한다. 하나라도 실패하면 public score·manifest·git 변경 0건으로 중단한다. audit JSONL의 기존 SHA upsert/중복 방지는 허용한다.
4. `docs/public_scores.csv`를 canonical 날짜 장부로 쓴다. 열은 `day,problem,metric,direction,sha256,file,public,note` 고정이다. `(day,problem)` 3행을 한 묶음으로 원자적 임시파일+`os.replace` 저장한다. 같은 날짜의 3행이 SHA·점수·파일까지 같으면 no-op이다. 하나라도 다르면 기존 기록을 보존하고 중단한다. 부분 날짜(1~2행)가 이미 있으면 중단한다.
5. canonical 저장 성공 전에는 문제별 `public add`를 호출하지 않는다. 이후 3개를 호출하되, 호출 전 각 문제 experiments의 `public_scores.csv` 존재/내용을 메모리에 백업하고 하나라도 실패하면 세 파일을 정확히 복원한 뒤 canonical에 방금 추가한 3행도 원자적으로 되돌린다.
6. P1/P2/P3 `public list`를 출력한다.
7. GitHub 작업 전 `gh auth status`, `gh repo view TaeYoungLim1901/OCEAN --json nameWithOwner,isPrivate`를 검사하고 private/이름 불일치면 중단한다. `.git`이 없으면 `git init -b main`, `origin=https://github.com/TaeYoungLim1901/OCEAN.git`를 설정한다. 기존 origin이 다른 저장소면 중단한다. 현재 branch가 main이 아니면 중단한다.
8. `git add .`는 금지한다. 다음 allowlist만 `git add -A -- <existing paths>`로 stage한다: 루트 운영 md, `.gitignore`, `.claude`, `docs`, `scripts`, 각 문제의 `AGENTS.md`, `graph.json`, `src`, `specs`, `experiments\log.md`, `experiments\public_scores.csv`.
9. stage 전후 금지 경로(data, cache, models, preds, subs, experiments logs 및 binary/model/pred 확장자), 50MB 초과 파일, `.env`/key/pem, 일반적인 secret 토큰 패턴을 검사한다. 발견 시 commit/push 0으로 중단하고 숫자만 보고한다. `git diff --cached --check`도 필수다.
10. staged 변경이 있으면 `scores: YYYY-MM-DD public results`로 commit하고 `git push -u origin main`한다. 변경이 없으면 commit하지 않고 아직 원격에 없는 기존 HEAD만 push한다. push 뒤 local HEAD와 `origin/main` SHA 일치를 검사한다.
11. 동시 실행용 lock/queue, scheduler, CI, LFS, 새 의존성은 만들지 않는다. SKILL.md는 실행 전 다른 P 세션·학습·Codex가 모두 idle인지 확인하고 루트 coordinator 하나만 도구를 실행하도록 지시한다.
12. SKILL.md는 오늘 또는 점수 3개가 없을 때 git 작업을 절대 수행하지 말고, 누락 입력을 사용자에게 요청하도록 한다. 점수 등록과 GitHub push를 별도 Luna 호출로 쪼개지 말고 한 사양/호출로 실행한다.

## 자체검증

`selftest`는 tempfile 안에서 네트워크·실제 repo·실제 ledger를 건드리지 않고 다음을 assert한다.

- 날짜 하한/미래일/NaN·Inf/P1 범위 거부.
- 문제-경로 불일치 거부.
- canonical 3행 신규 저장, 동일 재실행 no-op, 다른 점수·부분 날짜 거부.
- allowlist와 금지 경로 판정.
- rollback helper가 없음/기존 파일을 byte-for-byte 복원.

추가 검증:

- `$env:PYTHONUTF8='1'; python C:/Users/livde/.codex/skills/.system/skill-creator/scripts/quick_validate.py .claude/skills/ocean-daily-close` exit 0.
- `python .../daily_close.py selftest` exit 0.
- `python .../daily_close.py 2026-08-19 ...`는 외부 명령·파일 변경 없이 exit nonzero.
- 오늘 git init/commit/push 0건, 실제 `docs/public_scores.csv` 생성 0건.
- 허용 변경은 `.claude/skills/ocean-daily-close/**`, `.gitignore`뿐이다.

## 보고

결론 문장 금지. 숫자와 표만. 생성 파일, LOC, selftest assert 수, 각 검증 exit, 오늘 git init/commit/push 수, 허용 밖 변경 수를 보고한다.
