# E009 — maskaug 차분 제출 파일 조립

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
(numpy·pandas 만 쓰면 된다.)

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라.

## 1. 배경 (재조사 금지, 확정 사실)

- `P2/subs/E004_gate.csv` 를 제출해 **public RMSE 0.537099** 를 받았다 (2026-08-25).
  sha256 `9ab720d0a52f870458021451056d884cbab859d21eaf21e8d5e75ac704af34c5`, 26,061행,
  mean 23.048753, range [13.160076, 29.169471]. 로컬 4-hold 평균은 1.089067 이었다.
- `P2/preds/test_maskaug_E005.npy` 는 E005 에서 만든 maskaug 게이트 test 벡터다.
  `P2/preds/test_gate_robust_blend_E004.npy` 와 **공개층 결측 행 7,766개에서만** 다르다
  (test 26,061행의 29.7993%).
- E005 로컬 판정은 기각이었다(4-hold 평균 +0.004738, missing_public 2/4 개선). 그 판정을
  뒤집자는 게 아니다. **public 차분으로 직접 재는 것이 이 작업의 목적이다.** 로컬 재판정 금지.
- 모델·피처·학습·게이트 규칙은 **일절 손대지 않는다.** 기존 npy 를 CSV 로 옮기고 검증만 한다.

## 2. 작업

경로는 저장소 루트 기준. 상대경로 사용, 스크립트 상단에 `sys.stdout.reconfigure(encoding="utf-8")`.

1. `P2/src/build_sub.py` 를 새로 만든다 (기존 파일 수정 금지). 인자 2개: `<npy> <출력 csv>`.
   - `P2/P2_data/test_index.csv` 의 키 컬럼(`P2/src/common.py` 의 `C.KEY`)을 읽는다.
   - npy 를 `allow_pickle=False` 로 읽고 1차원인지, 길이가 test_index 행수와 같은지,
     전부 유한한지, `-5 <= x <= 45` 인지 검사한다. 하나라도 실패하면 예외로 멈춘다.
   - `keys.assign(temp=pred).to_csv(out, index=False)` — `P2/src/gate_e004.py` 기록 방식과 동일.
   - 출력 파일이 이미 있으면 덮어쓰지 말고 예외.
2. 실행하여 `P2/subs/E009_maskaug_diff.csv` 를 만든다
   (입력 `P2/preds/test_maskaug_E005.npy`).
3. 검증 명령을 **직접 실행하고 출력 마지막 줄을 그대로 보고**한다 (cwd = 저장소 루트):
   - `python P2/src/submission_tools.py audit P2/subs/E009_maskaug_diff.csv --pred P2/preds/test_maskaug_E005.npy`
   - `python P2/src/check.py sub P2/subs/E009_maskaug_diff.csv`
4. `P2/subs/E004_gate.csv` 와 비교해 표 1개를 만든다 (임시 스크립트로 계산하고 그 스크립트는 남기지 마라):
   - 두 CSV 의 키 열 순서가 완전히 동일한가 (True/False)
   - `temp` 값이 다른 행 수
   - 다른 행에서 `mean(E009 - E004)`, `max|E009 - E004|`, `std(E009 - E004)`
   - 다른 행 집합이 test 공개층 결측 행 집합과 일치하는가 (True/False, 불일치면 개수 차)

## 3. 기준선 숫자 (이 값과 비교해 보고)

| 항목 | E004_gate.csv |
|---|---|
| 행수 | 26,061 |
| mean | 23.048753 |
| range | [13.160076, 29.169471] |
| public RMSE | 0.537099 |
| 로컬 4-hold 평균 | 1.089067 |

## 4. 수락 기준

- `P2/subs/E009_maskaug_diff.csv` 존재, 26,061행
- `audit` 과 `check.py sub` 모두 `판정: OK`
- E004 와 다른 행 수 = **7,766**
- 키 열 순서 동일 = True
- 위 4개 중 하나라도 어긋나면 CSV 를 지우지 말고 그대로 두고 숫자만 보고한다.
