# Q2 — SUN 대회 운영 도구·안전장치를 OCEAN 구조로 재설계 (계획만. 읽기 전용)

실행 파이썬: `python`. **이 작업에서 코드를 쓰거나 실행하지 않는다. 계획 문서만 stdout 으로 낸다.**

## 배경 — 재조사 금지 사실 (확정)

OCEAN 2026 은 **문제 3개(P1·P2·P3)를 각각 독립 Claude Code 세션**에서 푼다. 하드웨어는 8코어 / RAM 31.5GB 공유.
Opus 토큰이 최대 제약이며 실제 작업은 codex `gpt-5.6-luna` `CX_EFFORT=xhigh` 에 위임한다.

**이미 OCEAN 에 있는 것 (다시 만들지 마라. 대체 제안 금지):**

| 있는 것 | 역할 |
|---|---|
| `AGENTS.md` (루트) + `P{n}/AGENTS.md` | 실행 계약. 소유 경계, 위임 규칙, 토큰 규율 |
| `PLAYBOOK.md`, `docs/DESIGN-*.md` | 절차·근거 문서 |
| `scripts/cx.cmd` / `cx.sh` | codex `gpt-5.6-luna` xhigh 위임 래퍼 |
| `scripts/gk.cmd` / `gk.sh` | grok-4.6 위임 래퍼 (읽기 전용 기본, 웹 검색) |
| `scripts/preflight.py` | `spec` 사양 게이트, `sub` 제출 게이트 |
| `scripts/g.py` + `P{n}/graph.json` | 문제 온톨로지. `ls`/`show`/`check`/`add`/`close`/`edge` |
| `P{n}/src/check.py` | 문제별 데이터·파이프라인·제출 게이트 |
| `P1/src/submission_tools.py` | `audit` / `public add|list` / `selftest` — 제출 sha256·양성률·public 점수 대장 |
| `P1/experiments/log.md`, `P1/preds/` (P2·P3 동형) | 실험 대장, 앙상블 풀 |

### 기준선 숫자 (확정. 재측정 금지)

 forward inner F1 `0.697836`, 봉인 outer fold3 F1 `0.815757`,
public LB F1 `0.717435` (대회 점수 25.823277, 점수 ≈ 36 × F1). 1위 F1 약 `0.858`.
사용 가능 라이브러리는 numpy·pandas·scikit-learn·lightgbm 뿐. xgboost·catboost·torch 없다.

## 읽을 것

1. `D:/hackathon/SUN/` 트리 전체. 특히 루트 문서(`ARENA.md` `RULES.md` `ADDRULES.md` `CHECKLIST.md`
   `CLOSED_AXES.md` `SEARCH.md` `PACKSTEPS.md` `POSTMORTEM.md` `ONBOARDING.md` `HANDOFF_CODEX.md`
   `HANDOVER_2026-08-20.md` `CONNECT.md` `REPOS.md` `PLAYBOOK.md` `GROK_REVIEW.md` `NEXT.md`),
   `tools/*.py`, `.claude/hooks/*.sh`, `experiments` 폴더, `submit` 폴더.
   `.git/`·`__pycache__`·`audio/`·`transcript/`·`server_backup/`·`work/local_data/` 는 건너뛴다.
2. GitHub: `https://github.com/andy6609/sungnam-to-kaist`, `https://github.com/TaeYoungLim1901/SUN`
3. OCEAN 현재 트리: 루트 `AGENTS.md`(자동 로드됨) · `PLAYBOOK.md` · `scripts/` · `P1/AGENTS.md` ·
   `P2/AGENTS.md` · `P3/AGENTS.md` · `P1/src/` 파일 목록.

## 산출 — 계획 문서 1개 (stdout)

### S1. 인벤토리 표
SUN 의 **LLM 안전장치**와 **대회 운영 도구**를 기능 단위로. 열:
`기능 | SUN 파일 | 한 줄 동작 | OCEAN 대응물 | 상태(중복/부분/없음)`.
파일이 아니라 **기능**으로 묶어라. 20행을 넘기지 마라.

### S2. 포팅 대상 — 이득이 실증되는 것만
S1 에서 `없음`·`부분` 인 것 중, **P1·P2·P3 를 3개 터미널로 동시에 운영할 때** 실제 이득이 있는 것만 고른다.
각 항목에 열: `기능 | 막아주는 구체적 실패 | OCEAN 에서 그 실패가 실제로 가능한 근거 | 예상 크기(파일 수·줄 수)`.
"있으면 좋다" 는 탈락시켜라. **버려야 할 이유를 못 찾은 것만 남긴다.**

### S3. 버릴 것 — 명시 목록
SUN 에 있으나 OCEAN 에 옮기지 **않을** 것과 이유 한 줄씩. (SUN 고유 도메인, OCEAN 에 이미 있음,
1회용, 의존성 필요, 문제 3개 구조와 불일치 등.) 이 목록이 S2 보다 길어야 정상이다.

### S4. 재구조화 설계
S2 항목 각각에 대해: `최종 경로 | 인터페이스(정확한 명령 한 줄) | 어느 파일을 수정하는가 | 루트 파일인가`.
제약:
- OCEAN 기존 도구를 **대체하지 말고 얹는다.** `g.py`·`preflight.py`·`cx.sh`·`check.py` 의 기존 인터페이스는 유지.
- **P1·P2·P3 3문제에 전부 동일하게 적용**되어야 한다. 문제별 분기가 필요하면 `P{n}/` 안에 둔다.
- 새 의존성 금지. **`pip install` 이 필요한 설계는 제안하지 말고 그 사실을 명시하고 대안을 낸다.**
- 세션 소유 경계를 지킨다: 루트 파일(`AGENTS.md`·`scripts/`·`.claude/`)을 건드리는 항목은 **따로 표시**한다.
- 과한 추상화 금지. 구현 1개짜리 인터페이스·안 바뀌는 config·로깅 프레임워크를 만들지 마라.

### S5. 반영 순서
단계별로 `순서 | 무엇 | 수락 기준(실행 가능한 검증 명령 한 줄) | 롤백 방법`.
독립적인 단계는 그렇게 표시한다. 각 단계는 codex luna xhigh 사양 1개로 위임 가능한 크기여야 한다.

### S6. 다른 세션 인계
P2·P3 세션이 이 변경 후 작업을 이어가려면 각각 무엇을 읽고 무엇을 실행해야 하는지 3줄씩.

## 제약

- **읽기 전용이다. 파일 생성·수정·삭제 금지. 코드 작성 금지.** 계획만 낸다.
- **새 의존성 금지. `pip install` 이 필요하면 제안하지 말고 명시만 한다.**
- 각 문제의 원본 데이터 폴더(`P1/P1_data` 등)는 읽지 마라.
- SUN 의 문서 문장을 그대로 옮기지 마라. 기능으로 요약한다.

## 수락 기준

S1~S6 전부 존재. S4 의 모든 항목이 S2 에 근거를 갖는다. S3 이 S2 보다 항목 수가 많다.
S5 의 모든 수락 기준이 실행 가능한 한 줄 명령이다.

## 보고 형식

**결론 문장 금지, 표와 목록만.** 서론·재확인·자기소개 금지. 전체 400줄을 넘기지 마라.
