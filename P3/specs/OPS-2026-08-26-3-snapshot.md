# OPS-2026-08-26-3 — OCEAN 스냅샷 푸시 (3회차)

실행 파이썬: `python`. 리포 루트에서 실행한다.

## 배경 — 재조사 금지 사실 (확정)

- 원격 `origin` = `https://github.com/TaeYoungLim1901/OCEAN.git`, private, 계정 `TaeYoungLim1901`.
  2026-08-26 확인: `gh auth status` 로그인 상태, `repo` 스코프 보유, 저장소 `isPrivate: true`.
  **저장소를 만들지 마라. `gh repo create` 금지.** 브랜치는 `main` 하나다.
- `.gitignore` 가 원본 데이터·cache·models·preds·subs·codex 로그를 이미 제외한다
  (KORS RS-2021-KS211502 재배포 금지). `public_scores.csv` 3개만 예외로 추적한다.
- 1회차 `5665900` (112파일), 2회차 `1cee1ef` (30파일)가 이미 나갔다. 이번이 3회차다.
  `P1/data_hashes.json` 과 `P1/experiments/*.json` 은 이번에 처음 포함된다.
- 이번 스냅샷 요약: 세 `P{n}/NEXT.md` 에 공통 「세션 시작 게이트」 블록 삽입,
  루트 handoff 에 2026-08-26 세션 결과·사고 3건 기록. 실험 JSON·데이터 해시 대장 포함.

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
   `P1/experiments/log.md`·`P1/experiments/public_scores.csv` (P2·P3 동형).
4. 커밋 전 스테이징된 파일 이름과 바이트를 전부 출력한다. 다음이 하나라도 있으면 커밋하지 말고 멈춘다:
   데이터·cache·models·예측 배열·제출 CSV·codex 실행 로그, `.env`/`.pem`/`.key`,
   `.npy`/`.npz`/`.pkl`/`.parquet` 등 이진·모델·예측 확장자, 비밀값 후보,
   삭제된 파일, 50MB 초과 파일.
   `git diff --cached --check` 를 돌린다. 실패하면 **인덱스만** 되돌리고 작업트리는 건드리지 마라.
5. 스테이징된 변경이 있으면 커밋한다. 커밋 메시지 제목은
   `snapshot: 2026-08-26 session close records` 로 하고 본문에 위 요약 3줄을 넣는다.
   변경이 없으면 커밋 0회로 보고한다.
6. `git push -u origin main` 후 로컬 HEAD 와 `origin/main` SHA 를 비교한다.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- 브랜치 전환·merge·rebase·`git reset`·`git checkout <경로>` 금지. 강제 푸시 금지.
- CI·스케줄러·LFS·lock 파일을 추가하지 마라.
- 코드·문서 내용을 고치지 마라. 이 작업은 스테이징·커밋·푸시만 한다.
- Public 점수를 기록하지 마라.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
git rev-parse HEAD
git rev-parse origin/main
git status --porcelain | wc -l
git show --stat --name-only HEAD | tail -40
```

로컬 HEAD 와 `origin/main` 이 같아야 한다. 커밋에 `.npy`·`.pkl`·`.parquet`·`.log` 가 하나도 없어야 한다.
`P1/subs`·`P2/subs`·`P3/subs` 경로가 커밋에 하나도 없어야 한다.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `항목 | 값` — 스테이징 파일 수, 총 바이트, 커밋 SHA, 로컬 HEAD, `origin/main`, push 횟수.
표 2 (거부가 있을 때만): `거부 파일 | 사유`. 산문 금지.
