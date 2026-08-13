# OPS-D1 — 루트 CWD 명령 계약 정합화

`pip install`이 필요하면 즉시 멈추고 보고한다. 새 의존성·코드·테스트·산출물은 만들지 않는다.

## 배경 — 재조사 금지 사실

- 루트 계약은 명령 실행 기준 디렉터리를 `D:/hackathon/OCEAN` 저장소 루트로 둔다.
- 현재 `P1/AGENTS.md`, `P2/AGENTS.md`, `P3/AGENTS.md`의 일부 예시는 `python src/...`와 `python ../scripts/...`를 사용해 루트 CWD에서 실패한다.
- 현재 P1·P2·P3 모두 `src/check.py`, `src/submission_tools.py`, `graph.json`이 존재하며 gate selftest는 3/3 통과했다. 루트 `AGENTS.md`의 “P3만 구현” 문구는 사실과 다르다.
- 기준선: 루트에서 `python src/check.py ...`는 파일 없음 exit 2. 목표 문서 예시는 모두 `python P{n}/src/...` 또는 `python scripts/...` 형태다.
- 표준 경로 `P3/specs/`, `P3/experiments/`, `P3/preds/`, `P3/subs/`는 존재한다. 새 파일 출력은 없다.

## 허용 파일

- `AGENTS.md`
- `CLAUDE.md`
- `P1/AGENTS.md`
- `P2/AGENTS.md`
- `P3/AGENTS.md`

그 밖의 파일 수정 금지다. 다른 세션이 생성한 코드·그래프·로그·spec은 건드리지 않는다.

## 작업

1. 세 문제 문서의 실행 가능한 명령 예시를 모두 저장소 루트 CWD 기준으로 정규화한다. P1 명령은 `P1/`, P2는 `P2/`, P3는 `P3/` 접두사를 사용하고 공용 graph 명령은 `python scripts/g.py --p P{n} ...`로 쓴다.
2. 루트 `AGENTS.md`의 “자동화된 graph/check/공용 제출 게이트는 P3만”이라는 과거 상태를 현재 3문제 모두 구현된 상태로 최소 정정한다. 게이트 우회 금지, 소유권, 자원 규칙은 바꾸지 않는다.
3. `CLAUDE.md`가 이미 현재 상태와 일치하면 수정하지 않는다. 중복 설명을 추가하지 않는다.
4. 실험 숫자·규정·graph 의미·코드 동작은 수정하지 않는다.

## 수락 기준

- 허용 문서의 fenced command와 inline command에서 `python src/`, `python ../scripts/`, `python metric.py`, `python submission_tools.py` 잔존 0건.
- P1/P2/P3의 check, metric, submission audit, graph 예시가 루트에서 `--help` 또는 인자 오류 단계까지 파일 찾기 오류 없이 진입한다.
- “P3만 구현”이라는 오래된 상태 문구 0건.
- 허용 밖 변경 0개.

## 보고 형식

결론 문장 금지. 숫자와 표만. 변경 파일·치환 건수·잔존 패턴 수·명령별 exit code를 보고한다.
