# CLAUDE.md

실행 계약은 [AGENTS.md](AGENTS.md) 다. Claude Code 는 `AGENTS.md` 를 자동으로 읽지 않으므로
여기서 import 한다.

@AGENTS.md

작업 중인 문제 트리에 `P{n}/AGENTS.md`가 있으면 그 트리 안에서 우선한다.

읽는 순서: `AGENTS.md` → (있으면) `P{n}/AGENTS.md` → (그래프가 있으면)
`python scripts/g.py --p P{n} ls axis --status open`. 현재 P1·P2·P3 모두 문제별 계약·그래프·게이트가 있다.

현재 후보·기각축·재개 지점은 [OPUS_HANDOFF_2026-08-13.md](OPUS_HANDOFF_2026-08-13.md)를 먼저 확인한다.

## ★ Opus 는 코드를 직접 쓰지 않는다

실제 작업은 codex `gpt-5.6-luna` `CX_EFFORT=xhigh` 가 한다 (`AGENTS.md` §0).
Opus 는 사양(`P{n}/specs/*.md`)을 쓰고 `scripts\cx.cmd <사양>` 으로 넘긴 뒤
보고된 숫자로 판정한다. 예외는 한 줄 게이트 명령·문서 쓰기·긴 학습 실행 3개뿐.

2026-08-13 현재 Git Bash·Codex CLI·ChatGPT 인증과 Luna xhigh read-only 호출을 검증했다.

## 토큰 규칙

- 문제가 3개다. Opus 토큰이 최대 제약이다. 애매하면 위임한다
- 문서·코드를 통독하지 마라. 그래프가 있는 문제의 지식은 `P{n}/graph.json`, `g.py show <노드>` 로 3줄만
- `PLAYBOOK.md`·`docs/DESIGN-*.md` 는 근거 문서. 필요한 절만 열어라
- 새 실험 전 `g.py check "<키워드>"`. 닫힌 축이면 exit 1
- 사양에 **"결론 문장 금지, 숫자와 표만"** 을 항상 넣어라
