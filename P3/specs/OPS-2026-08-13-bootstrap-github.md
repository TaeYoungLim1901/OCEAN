# OPS — 2026-08-13 private GitHub 최초 스냅샷

루트 `AGENTS.md`를 준수한다. 너는 이미 위임받은 Codex `gpt-5.6-luna`, xhigh 실행자다. `cx.cmd`를 재귀 호출하지 않는다. `pip install`이 필요하면 즉시 멈추고 보고한다.

## 배경 — 재조사 금지 사실

- 사용자가 2026-08-20부터의 일일 push 규칙에 대해 지금 1회 push를 명시적으로 요청했다.
- `TaeYoungLim1901/OCEAN`은 private 빈 저장소이며 기본 브랜치가 없다. `gh auth status`는 성공했다.
- 로컬 `.git`은 없고, 활성 OCEAN 학습/Codex 작업은 0개다.
- `.gitignore`는 원본 데이터, cache, models, preds, subs, 실행 로그와 binary를 제외한다.
- 기준선: local git 0, commit 0, remote branch 0, push 0. 직전 임시 index 감사는 후보 104파일/590828B, 금지 경로 0건이었다.
- 표준 경로 `P3/specs/`, `P3/experiments/`, `P3/preds/`, `P3/subs/`는 존재한다.

## 목표와 수락 기준

private `main`에 안전한 최초 코드/문서 스냅샷 1개만 commit/push한다. PR은 만들지 않는다.

- commit message는 caveman-commit 규약의 `chore: bootstrap OCEAN workspace` 정확히 사용한다.
- local HEAD = `origin/main` = GitHub default branch SHA.
- remote tracked 금지 경로·확장자·50MB 초과 파일·secret 후보 0개.
- 데이터·모델·예측·제출 파일과 실제 Public score는 0개 업로드한다.

## 순서

1. `gh auth status`와 `gh repo view TaeYoungLim1901/OCEAN --json nameWithOwner,isPrivate,defaultBranchRef,url`을 다시 확인한다. 이름/private 불일치면 변경 0으로 STOP한다.
2. 현재 파일 후보를 읽기 전용으로 계산한다. `.gitignore` 적용 뒤 금지 경로(data, cache, models, preds, subs, experiments logs, `P*_data`), 금지 확장자(`pkl,npy,npz,parquet,pyc,log,err,out,pid,pem,key`), `.env`, secret 후보, 50MB 초과가 하나라도 있으면 git init 전 STOP한다.
3. `git init -b main`; `git remote add origin https://github.com/TaeYoungLim1901/OCEAN.git`을 한다. 기존 `.git`이나 origin이 생겼다면 main/owner/repo/fetch+push URL을 검증하고 불일치 시 STOP한다.
4. `git add .` 금지. 존재하는 다음 allowlist만 `git add -A --`로 stage한다: `.gitignore`, `.claude`, `docs`, `scripts`, 루트의 `AGENTS.md`, `CLAUDE.md`, `PLAYBOOK.md`, `OPUS_HANDOFF_2026-08-13.md`; 각 P1/P2/P3의 `AGENTS.md`, `graph.json`, `src`, `specs`, `experiments\log.md`, 존재할 때만 `experiments\public_scores.csv`.
5. staged 파일명·총 bytes를 출력하고 2번 금지 검사를 staged blob 기준으로 다시 한다. 삭제 파일이 있으면 STOP한다. `git diff --cached --check` exit 0이어야 한다. 실패 시 index만 원복하고 worktree는 변경하지 않는다.
6. `python scripts/g.py selftest`, 세 문제의 `python P{n}/src/submission_tools.py selftest`, 신규 skill quick_validate를 실행한다. 하나라도 실패하면 commit/push 0으로 STOP한다.
7. `git commit -m "chore: bootstrap OCEAN workspace"`, `git push -u origin main`을 실행한다. 첫 기본 브랜치이므로 별도 branch와 PR을 만들지 않는다.
8. `git status --short --branch`, local HEAD, `git ls-remote origin refs/heads/main`, `gh repo view ... --json defaultBranchRef,isPrivate,url`을 확인한다. `git ls-tree -r --name-only HEAD`에서 금지 경로/확장자 0건을 확인한다.

## 금지

- 제출 CSV, NPY 예측, 모델, cache, 원본 데이터, 실행 로그, credential 업로드.
- `git add .`, LFS, branch switch, merge, rebase, reset, checkout, force push, PR 생성.
- 파일 내용 수정·리팩터. 이 사양 자체 외 기존 worktree는 그대로 snapshot만 한다.
- 결론 문장 금지. 숫자와 표만.

## 완료 보고

후보/스테이지/커밋 파일 수와 bytes, 금지·secret·대형 파일 수, 검증별 exit, branch, commit SHA, remote SHA, private 여부, init/commit/push/PR 횟수를 표로 보고한다.
