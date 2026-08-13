# 작업 지시: Q0 — Codex 위임 경로 smoke test

루트 `AGENTS.md`와 `P3/AGENTS.md`를 준수한다. 시스템 `python`을 사용한다.
`pip install`이 필요하면 멈추고 보고한다. 이 작업은 읽기 전용이며 파일을 수정하지 않는다.

## 배경 — 이미 검증된 사실

기준선: `python scripts/preflight.py spec` 판정 OK. 재조사하지 않는다.

## 수락 기준

- 현재 작업 디렉터리 이름이 `OCEAN`인지 확인한다.
- 루트 `AGENTS.md`와 `P3/AGENTS.md`가 존재하는지 확인한다. 전문은 출력하지 않는다.
- 절대 하한: 확인 3개가 모두 `1`이어야 한다.
- `python -c "import numpy,pandas,scipy,lightgbm,sys;print(sys.executable,numpy.__version__,lightgbm.__version__)"`
  를 실행해 출력 그대로 보고한다. 실패하면 오류 첫 줄과 `python -c "import sys;print(sys.path)"` 결과를 보고한다.
  `pip install` 은 하지 마라.

## 금지

- 파일 수정, 학습, `pip install` 금지
- 결론 문장 금지. 숫자와 표만

## 완료 보고 형식

| cwd_OCEAN | root_agents | p3_agents |
|---:|---:|---:|
| 1 또는 0 | 1 또는 0 | 1 또는 0 |
