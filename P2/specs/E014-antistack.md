# E014 — 완전행 반대방향 외삽 (w = -1) + 결측행 w* 블렌드

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
**재학습 금지.** 저장된 `P2/preds/test_*.npy` 산술 조합과 검증만. 기존 파일 수정 금지.

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라.

## 1. 배경 (재조사 금지, 확정 사실)

public 원장 4점. test 26,061행 = 결측행 7,766 + 완전행 18,295.

| CSV | 결측행 | 완전행 | public |
|---|---|---|---:|
| E004_gate | `test_e002_E003` | `test_gate_robust_blend_E004` | 0.537099 |
| **E009_maskaug_diff** | `test_maskaug_E005` | `test_gate_robust_blend_E004` | **0.525759** |
| E010_stackregime_complete | `test_maskaug_E005` | `test_stack_regime_E008` | 0.528449 |
| E011_missingonly | `test_missing_only_E005` | `test_gate_robust_blend_E004` | 0.530045 |

`MSE(w) = MSE0 + w²·E[d²] − 2w·E[d·e]` 는 항등식이다. 위 4점에서 역산한 값:

| 축 | d | E[d²] | E[d·e] | w* |
|---|---|---:|---:|---:|
| 완전행 | `test_stack_regime_E008 − test_gate_robust_blend_E004` | 0.00046136 | **−0.00118823** | −2.5755 |
| 결측행 | `test_missing_only_E005 − test_maskaug_E005` | 0.00718763 | +0.00133123 | +0.1852 |

완전행 `E[d·e]` 가 음수다 = `stack_regime` 은 완전행에서 일관되게 틀린 방향으로 민다.
반대로 밀면 이득이 난다. `E[d²]` 는 전체 test 로 계산했고 public 은 부분집합이므로
`w* = −2.5755` 외삽은 검증되지 않았다. **먼저 w = −1 로 포물선 3번째 점을 실측 고정한다.**

## 2. 만들 것 — CSV 2개

`P2/src/mix_sub.py` 는 이미 있다(`--missing`·`--complete`·`--out`). 임의의 w 벡터를 만들려면
`P2/src/wmix.py` 를 새로 만든다 (기존 파일 수정 금지). 인자:
`--base <npy> --delta-from <npy> --delta-to <npy> --w <float> --out <npy>`
로 `base + w·(delta_to − delta_from)` 를 저장한다. 유한값·`-5 <= x <= 45` 검사, 출력 존재 시 예외.

### E014 (다음 제출 후보)

- 완전행: `test_gate_robust_blend_E004 + (−1)·(test_stack_regime_E008 − test_gate_robust_blend_E004)`
  = `2·robust_blend − stack_regime`. `wmix.py` 로 `P2/preds/test_antistack_w-1_E014.npy` 저장.
- 결측행: `test_maskaug_E005` 그대로.
- `mix_sub.py` 로 `P2/subs/E014_antistack_w-1.csv` 생성.

### E015 (예비, 제출 안 함)

- 결측행: `test_maskaug_E005 + 0.1852·(test_missing_only_E005 − test_maskaug_E005)`
  → `P2/preds/test_missblend_w0.185_E015.npy`
- 완전행: `test_gate_robust_blend_E004` 그대로.
- `P2/subs/E015_missblend.csv` 생성.

## 3. 검증 — 직접 실행하고 마지막 줄 보고

각 CSV 에 대해:
- `python P2/src/submission_tools.py audit <csv>`
- `python P2/src/check.py sub <csv>`
- `python P2/src/bugcheck.py` 전체 1회 (마지막 2줄만)

## 4. 보고할 표

E009(`P2/subs/E009_maskaug_diff.csv`, public 0.525759) 를 기준선으로 각 CSV 에 대해:

| 항목 |
|---|
| sha256, 행수, mean, min, max |
| 키 열 순서 E009 와 동일 (True/False) |
| 변경 행 수, 변경 집합이 의도한 집합(E014=완전행 / E015=결측행)과 일치 (True/False) |
| 변경 행에서 mean Δ, std Δ, max\|Δ\| |
| `mse_shift = sum(Δ²)/26061` |
| 위 §1 표의 `E[d²]`·`E[d·e]` 로 계산한 **예측 public RMSE** |

## 5. 수락 기준

- 새 파일은 `P2/src/wmix.py` 하나. 기존 파일 무수정
- `P2/subs/E014_antistack_w-1.csv`·`P2/subs/E015_missblend.csv` 둘 다 26,061행,
  `audit`·`check.py sub` 모두 `판정: OK`, `bugcheck` `판정: OK`
- E014 변경 행 수 = 18,295 (전부 완전행), E015 변경 행 수 = 7,766 (전부 결측행)
- E014 예측 public RMSE 가 0.5239 ± 0.0002 안에 들어온다
- **제출하지 않는다**
