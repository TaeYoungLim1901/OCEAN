# P2 — 소청초 중간층 수온 복원

루트 [AGENTS.md](../AGENTS.md)가 상위 계약이다. P2 고유 게이트는 아래와 같다.

## ★ codex 위임 = 토큰 절약의 전부. 컨텍스트 길어져도 잊지 마라

> **codex 에게: 이 절은 Opus(Claude Code) 전용 지시다. 너는 이미 위임받은 실행자다.**
> `cx.cmd`·`codex exec` 를 다시 호출하지 마라. 사양은 네가 직접 구현한다.
> (2026-08-13 E004: 이 절을 자기 지시로 읽고 codex 를 3회 재귀 호출해 timeout, 산출물 0)

코드 작업은 전부 `scripts\cx.cmd <사양>` (codex `gpt-5.6-luna`, `CX_EFFORT=xhigh`). Opus 직접 구현 금지.

**Claude Code Bash 도구의 샌드박스가 codex 인증·네트워크를 막는다.** 2026-08-13 P2 실측:
샌드박스 상태 호출은 출력 없이 `exit 255`, 산출물 0개. 반드시 아래 형태로 부른다.

```
scripts\cx.cmd P2/specs/<사양>.md > P2/experiments/logs/codex_<ID>.log 2>&1
```

- Bash 도구 인자: `dangerouslyDisableSandbox: true`, `run_in_background: true` (호출이 600초를 넘는다).
- 실패 판단은 로그 `tail -20` 과 산출물(`P2/cache/...`) 존재 여부만으로 한다. 로그 전문 금지.

```bash
python P2/src/check.py data
python P2/src/check.py pipeline
python P2/src/check.py cases
python P2/src/check.py sub P2/subs/E001_lgb_full.csv
python P2/src/check.py reproduce
python P2/src/check.py hashes
python P2/src/submission_tools.py audit P2/subs/E002_final.csv --pred P2/preds/test_fixed_depth_E002.npy
```

- 2024-09-01~10-31의 층 2·3·4를 동시에 가린 연속 홀드아웃만 판정에 쓴다.
- 검증 시 공개층과 가림 방식은 2025 test와 같아야 한다. 무작위 행 분할 금지.
- 연도별 센서 구성이 다르므로 `layer`를 깊이 좌표로 간주하지 않는다.
- 제출 열 순서는 `station,layer,time,temp`, 값은 -5~45°C 유한값이다.
- `P2_data/`는 읽기 전용이며 원자료를 재배포하지 않는다.
- 제출 전 `python P2/src/submission_tools.py audit` 통과가 필수다. `python P2/src/submission_tools.py public add` 및 `python P2/src/submission_tools.py public list`로 하루 1회 결과를 기록한다.
- 숨은 정답 없이는 Public RMSE를 계산할 수 없으므로 리더보드 값을 `python P2/src/submission_tools.py public add`에 직접 입력한다.
