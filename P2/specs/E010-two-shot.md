# E010/E011 — public 2발 동시 조립 (행 집합 분리)

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
(numpy·pandas 만 쓴다.) **재학습 금지.** 저장된 `P2/preds/test_*.npy` 만 조합한다.

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라.

## 1. 배경 (재조사 금지, 확정 사실)

public 리더보드 실측 2건:

| 제출 | 결측행(7,766) | 완전행(18,295) | public RMSE |
|---|---|---|---|
| E004_gate | `test_e002_E003` (=E002) | `test_gate_robust_blend_E004` 의 완전행 | 0.537099 |
| E009_maskaug_diff | `test_maskaug_E005` | 동일 | **0.525759** |

즉 **결측행에서 maskaug 가 E002 를 이긴다(Δ -0.011340)**. 로컬 4-hold 균등 판정은
정반대(+0.004738)를 말했으므로 **로컬 4-hold 는 판정기가 아니다.** E005·E008 의 로컬 기각을
근거로 후보를 배제하지 마라. 이 사양은 로컬 재판정을 하지 않는다. public 측정만이 목적이다.

`P2/subs/E009_maskaug_diff.csv` = `P2/preds/test_maskaug_E005.npy` 를 그대로 쓴 것이며
sha256 `8dc6a7cd34c8591919dccad7781a990ff3c99dd9d32b1f80b7d169a3462b674f`, 26,061행,
mean 23.046965, range [14.156363, 29.169471]. 이것이 **현재 기준선**이다.

행 집합 정의는 `P2/src/gate_e004.py` 의 `missing_public` 플래그를 그대로 쓴다.
test 결측행 7,766 / 완전행 18,295 (합 26,061).

## 2. 작업 — CSV 2개를 만든다

`P2/src/build_sub.py` 는 이미 있다(단일 npy → CSV). 이번엔 행 집합별 조합이 필요하므로
`P2/src/mix_sub.py` 를 새로 만든다 (기존 파일 수정 금지). 인자:
`--missing <npy> --complete <npy> --out <csv>`.

- `P2/P2_data/test_index.csv` 의 키(`P2/src/common.py` 의 `C.KEY`) 순서를 유지한다.
- `gate_e004.py` 와 **같은 방법으로** `missing_public` 불리언 마스크를 구한다.
  gate_e004 의 기존 함수를 import 해서 쓰고, 복사·재구현하지 마라.
  마스크 True 개수가 7,766 이 아니면 예외로 멈춘다.
- 결과 벡터 = `where(mask, missing_npy, complete_npy)`.
- 유한값·`-5 <= x <= 45` 검사. 출력 파일이 이미 있으면 덮어쓰지 말고 예외.
- `keys.assign(temp=pred).to_csv(out, index=False)` — `gate_e004.py` 기록 방식과 동일.

만들 CSV 2개:

| 파일 | --missing | --complete | 바뀌는 행 |
|---|---|---|---|
| `P2/subs/E010_stackregime_complete.csv` | `P2/preds/test_maskaug_E005.npy` | `P2/preds/test_stack_regime_E008.npy` | 완전행 18,295 |
| `P2/subs/E011_missingonly.csv` | `P2/preds/test_missing_only_E005.npy` | `P2/preds/test_maskaug_E005.npy` | 결측행 7,766 |

두 파일은 E009 대비 **서로 겹치지 않는 행 집합에서만** 달라야 한다. 이것이 수락 기준이다.

## 3. 검증 — 직접 실행하고 마지막 줄을 보고

CSV 각각에 대해 (cwd = 저장소 루트):

- `python P2/src/submission_tools.py audit <csv>`
- `python P2/src/check.py sub <csv>`

## 4. 보고할 표

CSV 2개 각각에 대해 E009(`P2/subs/E009_maskaug_diff.csv`) 와 비교:

| 항목 |
|---|
| sha256, 행수, mean, range |
| 키 열 순서 E009 와 동일 (True/False) |
| `temp` 가 다른 행 수 |
| 다른 행 집합이 의도한 집합(E010=완전행 / E011=결측행)과 정확히 일치 (True/False) |
| 다른 행에서 mean Δ, std Δ, max\|Δ\| (Δ = 새 파일 − E009) |
| audit 마지막 줄, check.py sub 마지막 줄 |

추가로 1행: `E010` 과 `E011` 이 서로 다른 행 집합의 교집합 크기 (0 이어야 한다).

## 5. 수락 기준

- 두 CSV 모두 26,061행, `audit`·`check.py sub` 둘 다 `판정: OK`
- E010: E009 와 다른 행 수 = 18,295, 전부 완전행
- E011: E009 와 다른 행 수 = 7,766, 전부 결측행
- 두 변경 행 집합의 교집합 = 0
- 어긋나면 CSV 를 지우지 말고 그대로 두고 숫자만 보고한다
