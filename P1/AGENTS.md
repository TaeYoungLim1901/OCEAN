# P1 — 종합해양과학기지 관측 수온 자동 품질관리

루트 [AGENTS.md](../AGENTS.md)가 상위 계약이다. P1 고유 계약만 적는다.

## ★ codex 위임 = 토큰 절약의 전부. 컨텍스트 길어져도 잊지 마라

> **codex 에게: 이 절은 Opus(Claude Code) 전용 지시다. 너는 이미 위임받은 실행자다.**
> `cx.cmd`·`codex exec` 를 다시 호출하지 마라. 사양은 네가 직접 구현한다.
> (2026-08-13 P2 E004: 이 절을 자기 지시로 읽고 codex 를 3회 재귀 호출해 timeout, 산출물 0)

코드 작업은 전부 `scripts\cx.cmd <사양>` (codex `gpt-5.6-luna`, `CX_EFFORT=xhigh`). Opus 직접 구현 금지.

**Claude Code Bash 도구의 샌드박스가 codex 인증·네트워크를 막는다.** 2026-08-13 P2 실측:
샌드박스 상태 호출은 출력 없이 `exit 255`, 산출물 0개. 반드시 아래 형태로 부른다.

```
scripts\cx.cmd P1/specs/<사양>.md > P1/experiments/logs/codex_<ID>.log 2>&1
```

- Bash 도구 인자: `dangerouslyDisableSandbox: true`, `run_in_background: true` (호출이 600초를 넘는다).
- 실패 판단은 로그 `tail -20` 과 산출물(`P1/cache/...`) 존재 여부만으로 한다. 로그 전문 금지.

## 과제

10분 간격 `station × layer` 수온 시계열의 합성 spike/noise/flatline/offset/drift를 행별 이진 탐지한다.
train은 2024~2025년, test는 2026년이며 지표는 양성 F1이다.

## 게이트

```bash
python P1/src/check.py data
python P1/src/check.py pipeline P1/subs/exp001.csv P1/preds/oof_lgbm_exp001.npy P1/preds/test_lgbm_exp001.npy
python P1/src/check.py sub P1/subs/exp001.csv
python P1/src/submission_tools.py audit P1/subs/exp002.csv --pred P1/preds/test_exp002.npy
```

`판정: OK`가 아니면 진행하지 않는다.

제출 전 `python P1/src/submission_tools.py audit`, 리더보드 확인 후 `python P1/src/submission_tools.py public add <csv> <F1>`를 기록한다. 숨은 label 없이는 Public F1을 로컬에서 계산할 수 없다.

## 코드 계약

- `P1_data/`는 읽기 전용. 시간 gap을 넘는 diff·rolling·event 연결을 금지한다.
- 랜덤 행 분할 금지. 검증은 station-layer별 시간 block이며 경계에 최소 87시간 purge를 둔다.
- `train.py`와 `inference.py`를 분리하고 둘 다 `features.py`의 동일 생산 경로를 쓴다.
- `evaluate(..., by=[...])`의 `by`는 필수다.
- 상대경로, `SEED = 42`, `n_jobs <= 2`, 콘솔 UTF-8을 지킨다.
