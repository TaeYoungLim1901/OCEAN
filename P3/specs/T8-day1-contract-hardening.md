# T8 — 첫날 공통 계약 하드닝

루트 `AGENTS.md`, `P3/AGENTS.md`, `PLAYBOOK.md`를 준수한다. 실제 구현은 Codex `gpt-5.6-luna`, `xhigh`만 한다. 다른 세션이 P1/P2/P3에서 작업 중이므로 아래 허용 파일 밖은 수정하지 않는다.

`pip install`이 필요하면 즉시 멈추고 보고한다. 새 의존성 설치는 금지다.

## 배경 — 재조사 금지 사실

- 현재 `scripts/g.py`는 hard rule hit가 있어도 `return 1 if closed else 0`이라 exit 0이다.
- 현재 `scripts/cx.sh`는 `-s`, `CX_MODEL`, `CX_EFFORT`로 Luna/xhigh 계약을 우회할 수 있다.
- 현재 `.gitignore` 기준 `git add .` 후보는 233파일/388.33MB이며, 모델 319.32MB·예측/제출 64.74MB가 ignore되지 않는다.
- 현재 기준선: hard rule hit exit=0, closed axis hit exit=1, miss exit=0. 목표: 1, 1, 0.
- 표준 산출물 경로 `P3/experiments/`, `P3/preds/`, `P3/subs/`는 존재한다. 이번 작업은 그 안에 새 산출물을 만들지 않는다.

## 목표

첫날 운영 감사에서 확인된 공통 P0 실패모드만 최소 수정한다. 새 의존성·추상화·CI·LFS·실험 코드는 만들지 않는다.

## 허용 파일

- `scripts/g.py`
- `scripts/cx.sh`
- `.gitignore`
- 위 파일의 기존 selftest가 별도 파일을 요구할 때만 최소 테스트 1개

`P1/`, `P2/`, `P3/`의 코드·그래프·로그·산출물과 그 밖의 루트 파일은 수정 금지다.

## 구현

1. `scripts/g.py check`가 일치하는 `hard=true` rule을 하나라도 찾으면 exit 1을 반환하게 한다. closed axis도 기존대로 exit 1이다. 일치 없음만 exit 0이다. 기존 출력 형식은 유지한다.
2. 위 동작을 임시 graph/표준 라이브러리만으로 검증하는 최소 runnable selftest를 남긴다. hard rule hit=1, closed axis hit=1, miss=0을 확인한다.
3. `scripts/cx.sh`에서 `-s`, `CX_MODEL`, 임의 `CX_EFFORT`로 계약을 우회하지 못하게 한다. 모델은 `gpt-5.6-luna`, effort는 `xhigh`로 고정한다. 기존 `-r` 읽기 전용 동작과 기본 구현 동작은 유지한다. 금지 옵션/불일치 환경값은 조용히 무시하지 말고 비영(0이 아닌) 종료한다.
4. `.gitignore`는 기존 규칙을 보존한다. `P3/models/`, `P3/preds/`, `P3/subs/`, `P3/experiments/logs/`와 `P3/experiments/` 바로 아래 CSV·JSONL에 해당하는 패턴을 문제 번호 별표 와일드카드로 최소 추가한다. 또한 `*.py[cod]`, `*.pkl`, `*.npy`, `*.npz`, `*.parquet`, `*.log`, `*.err`, `*.out`, `*.exit`, `*.pid`, `.tmp-*`, `.env`, `.env.*`, `*.pem`, `*.key`를 추가한다. `P3/experiments/log.md`, 코드, `P3/specs/`, graph JSON은 추적 가능해야 하며 같은 규칙을 P1·P2에도 대칭 적용한다.

## 수락 기준

- `python scripts/g.py selftest` 또는 추가한 동등 selftest: exit 0.
- 임시 graph에서 hard rule hit: exit 1; closed axis hit: exit 1; miss: exit 0.
- `bash scripts/cx.sh -s P3/specs/q-smoke.md`: Codex를 호출하지 않고 exit nonzero.
- `CX_MODEL=gpt-5.6-sol bash scripts/cx.sh -r P3/specs/q-smoke.md`: Codex를 호출하지 않고 exit nonzero.
- `CX_EFFORT=high bash scripts/cx.sh -r P3/specs/q-smoke.md`: Codex를 호출하지 않고 exit nonzero.
- `git check-ignore` 또는 동등 검사에서 모델·예측·제출·실행 로그·비밀 후보는 ignored, `P3/graph.json`과 `P3/experiments/log.md`는 not ignored.
- 허용 파일 밖 변경 0개.

## 보고 형식

결론 문장 금지. 숫자와 표만. 변경 파일, 줄 수, 각 검증 명령, exit code, 허용 파일 밖 변경 수를 표로 보고한다.
