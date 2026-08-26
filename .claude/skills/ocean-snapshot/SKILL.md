---
name: ocean-snapshot
description: OCEAN 작업 트리를 개인 private GitHub 저장소 TaeYoungLim1901/OCEAN 에 스냅샷 커밋·푸시한다. "깃허브에 올려", "백업해", "스냅샷", "푸시해", "커밋해", "저장소에 반영" 요청 시 사용. Public 점수 기록은 하지 않는다 — 그것은 ocean-daily-close 다.
---

# OCEAN 스냅샷 푸시

원격은 **이미 존재한다**: `https://github.com/TaeYoungLim1901/OCEAN.git` (private, origin 설정됨).
저장소를 새로 만들지 마라. `gh repo create` 를 부르지 마라.

## 쓰지 않는 경우

- 긴 학습·다른 codex 호출이 도는 중 (작업트리가 움직인다)
- 목적이 Public 점수 기록이면 `ocean-daily-close` 를 쓴다
- 제출물·예측·모델·데이터를 올리려는 경우 — 전부 금지다

## Coordinator (Opus)

1. `git status -sb` 로 브랜치가 `main` 인지, `gh auth status` 로 계정이 `TaeYoungLim1901` 인지 확인한다.
2. 사양 1개를 `P3/specs/OPS-YYYY-MM-DD-snapshot.md` 로 쓴다. **기존 사양을 덮어쓰지 마라.**
   같은 날 2회째면 `-2` 접미를 붙인다. 본문은 아래 템플릿을 그대로 복사하고 날짜·요약만 채운다.
3. `python scripts/preflight.py spec P3/specs/OPS-YYYY-MM-DD-snapshot.md`
4. `scripts\cx.cmd -f P3/specs/OPS-YYYY-MM-DD-snapshot.md` — `-f` 는 push 에 네트워크가 필요해서다.
   Bash 도구에서 `dangerouslyDisableSandbox: true` + `run_in_background: true`, 출력은
   `P3/experiments/logs/codex_snapshot_YYYY-MM-DD.log` 로 리다이렉트한다.
5. 보고표에서 **로컬 HEAD 와 `origin/main` SHA 일치**만 확인한다. 불일치면 멈추고 사용자에게 보고한다.

## 사양 템플릿 — 그대로 복사해 날짜·요약만 채운다

```markdown
# OPS-YYYY-MM-DD — OCEAN 스냅샷 푸시

실행 파이썬: `python`. 리포 루트에서 실행한다.

## 배경 — 재조사 금지 사실 (확정)

- 원격 `origin` = `https://github.com/TaeYoungLim1901/OCEAN.git`, private, 계정 `TaeYoungLim1901`.
  **저장소를 만들지 마라. `gh repo create` 금지.** 브랜치는 `main` 하나다.
- `.gitignore` 가 원본 데이터·cache·models·preds·subs·codex 로그를 이미 제외한다
  (KORS RS-2021-KS211502 재배포 금지). `public_scores.csv` 3개만 예외로 추적한다.
- 이번 스냅샷 요약: <한 줄로 무엇이 바뀌었는지>

## 기준선 (baseline. 재측정 금지)

작업 전 로컬 HEAD 와 `origin/main` SHA 를 먼저 출력하고 시작한다.

## 절차

1. **`cx.cmd` 를 재귀 호출하지 마라.**
2. `gh auth status` 와 `gh repo view TaeYoungLim1901/OCEAN --json nameWithOwner,isPrivate` 를 확인한다.
   정확히 그 private 저장소가 아니면 즉시 멈춘다. origin 의 fetch/push URL 이 다르거나
   브랜치가 `main` 이 아니면 멈춘다.
3. **`git add .` 를 쓰지 마라.** 아래 경로만 `git add -A -- <경로>` 로 명시 스테이징한다:
   `.gitignore`, `.claude`, `docs`, `scripts`, 루트 `*.md` (공고문 원문 포함),
   그리고 각 문제의 최상위 `*.md` (`AGENTS.md`·`NEXT.md`·`*CONTEXT.md`·`*_RESEARCH.md` 등),
   `graph.json`, `src`, `specs`,
   `P1/experiments/log.md`·`P1/experiments/public_scores.csv`·`P1/experiments/*.json`·
   `P1/data_hashes.json` (P2·P3 동형).
4. 커밋 전 스테이징된 파일 이름과 바이트를 전부 출력한다. 다음이 하나라도 있으면 커밋하지 말고 멈춘다:
   데이터·cache·models·preds·subs·codex 실행 로그, `.env`/`.pem`/`.key`,
   `.npy`/`.npz`/`.pkl`/`.parquet` 등 이진·모델·예측 확장자, 비밀값 후보,
   삭제된 파일, 50MB 초과 파일.
   `git diff --cached --check` 를 돌린다. 실패하면 **인덱스만** 되돌리고 작업트리는 건드리지 마라.
5. 스테이징된 변경이 있으면 `snapshot: YYYY-MM-DD <요약>` 로 커밋한다. 없으면 커밋 0회로 보고한다.
6. `git push -u origin main` 후 로컬 HEAD 와 `origin/main` SHA 를 비교한다.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- 브랜치 전환·merge·rebase·`git reset`·`git checkout <경로>` 금지. 강제 푸시 금지.
- CI·스케줄러·LFS·lock 파일을 추가하지 마라.
- 코드·문서 내용을 고치지 마라. 이 작업은 스테이징·커밋·푸시만 한다.
- Public 점수를 기록하지 마라.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

    git rev-parse HEAD
    git rev-parse origin/main
    git status --porcelain | wc -l

로컬 HEAD 와 `origin/main` 이 같아야 한다. 금지 확장자가 커밋에 하나도 없어야 한다:

    git show --stat --name-only HEAD | tail -40

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `항목 | 값` — 스테이징 파일 수, 총 바이트, 커밋 SHA, 로컬 HEAD, `origin/main`, push 횟수.
표 2 (거부가 있을 때만): `거부 파일 | 사유`. 산문 금지.
```

## 금지

저장소 생성, 제출물·예측·모델·데이터 업로드, `git add .`, 강제 푸시, 브랜치 전환,
점수 기록, CI·LFS·스케줄러 추가.
