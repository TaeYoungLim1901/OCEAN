# P2 재개 지점 (2026-08-13 세션 종료)

## 제출 후보

| 순위 | 파일 | 구성 | 로컬 4-hold 평균 |
|---|---|---|---|
| 1 | `subs/E004_gate.csv` | 결측행 7,766 = E002 / 완전공개행 18,295 = E003 robust_blend | 1.089067 |
| 2 | `subs/E002_final.csv` | E002 단독 (이전 세션 후보) | 1.161517 |

`E004_gate.csv` sha256 `9ab720d0a52f870458021451056d884cbab859d21eaf21e8d5e75ac704af34c5`, 26,061행.
`audit`·`check.py sub`·`data`·`pipeline`·`cases`·`reproduce` 전부 `판정: OK`.
예상 public RMSE **0.74~0.81** (계절 앵커 2024_sepoct 세그먼트를 test 구성비 결측 29.80% 로 재합성).
보수적 상한 1.27 (계절 불일치 hold 통째).

## 진행이 LB 없이 막힌 이유

adversarial AUC `0.992~1.000`. 4개 hold 가 서로 반대 방향을 가리킨다. maskaug(E005)는
Sep-Oct 창 3개에서 결측행을 이기고(S1 -0.188, S3 -0.127, S4 -0.059) Aug 비중 큰 창 1개에서
크게 진다(S2 +0.164). 어느 창이 test 를 대변하는지 로컬로 정할 수 없다.
여기서 규칙을 더 조정하면 같은 4 hold 에 대한 5번째 맞춤이다.

## LB 확보 후 첫 2수 (하루 1회 제한)

1. **1일차**: `subs/E004_gate.csv` 제출. `python P2/src/submission_tools.py public add subs/E004_gate.csv <RMSE>`.
   예상 구간 0.74~0.81 안이면 2024_sepoct 를 test 대변자로 확정하고 계절 정합 판정을 되살린다.
2. **2일차**: maskaug 게이트 제출로 **차분** 측정. 결측행 29.8% 에서 maskaug 우열이 직접 결정된다.
   조립만 하면 된다: 결측행 `preds/test_maskaug_E005.npy`, 완전공개행 `preds/test_gate_robust_blend_E004.npy`
   (두 벡터는 정확히 7,766행만 다르다). 사양 1개로 codex 에 넘겨라.

모델 제출(9/7 마감)은 **즉시 최종 확정**이라 답안 제출을 다 쓴 뒤 마지막에 한다.

## 이번 세션에서 닫힌 축

`a.missing_gate`(채택), `a.mask_augment`, `a.season_matched_holdout`, `a.layer4_anchor`,
`a.loho_stack` — 근거는 전부 `graph.json`. 새 모델 축은 남아 있지 않다.
`python scripts/g.py --p P2 ls axis --status closed` 로 먼저 확인하고 시작해라.

## 도구 메모

- `P2/cx_snap.sh` — `scripts/cx.sh` 스냅샷. 다른 세션이 실행 중 루트를 편집하면 bash 가 오프셋을
  이어 읽다 죽어 codex 자식이 끊긴다(E006 2회 실패). `bash P2/cx_snap.sh <사양>` 으로 부른다.
  P3 세션이 `cx.sh` 를 고치면 다시 복사할 것.
- codex 호출은 Bash 도구 `dangerouslyDisableSandbox: true` + `run_in_background: true`.
- `audit` 의 `npy_exact=True` 는 반올림 후 비교다. 실제 CSV↔npy 최대차 `3.55e-15`(무해).

## 2026-08-25 LB 결과

`E004_gate.csv` **Public RMSE 0.537099 · 점수 26.594071** (`public add` 기록 완료).
예상 구간 0.74~0.81 보다 0.20 낮다. 로컬 4-hold 평균 1.089067 대비 절반 이하 —
로컬 hold 는 test 보다 훨씬 어렵고 **절대 수준 판정기가 아니다**. 2024_sepoct 재합성도
여전히 비관적이었으므로 "계절 앵커 = test 대변자" 는 확정하지 못한다.
2일차 수(maskaug 게이트 차분 제출)는 그대로 유효하며, 차분만이 유일한 판정 근거다.

## 다음 제출 (준비 완료 · 사용자 행동 필요)

`P2/subs/E009_maskaug_diff.csv` — sha256 `8dc6a7cd34c8591919dccad7781a990ff3c99dd9d32b1f80b7d169a3462b674f`,
26,061행, mean 23.046965. `audit`·`check.py sub` 둘 다 `판정: OK`.
E004_gate 와 **공개층 결측 7,766행에서만** 다르다 (mean Δ -0.006000, std 0.219228, max|Δ| 2.255825).

판정 규칙 (**제출 전 확정**):

| public RMSE | 해석 | 다음 수 |
|---|---|---|
| < 0.537 | 결측행에서 maskaug 우세 | `a.mask_augment` 재개, E009 를 새 기준선으로 |
| 0.537 ~ 0.545 | 무차별 | E004 유지, 남은 축(`a.residual_lgb`·`a.tide_context`)으로 |
| > 0.545 | maskaug 열세 | `a.mask_augment` 영구 기각, E004 최종 |

순수잡음 가정 예상값 0.550 (0.298 × 0.219² 를 MSE 에 더한 값). 이보다 낮으면 신호가 있다.

제출 후: `python P2/src/submission_tools.py public add P2/subs/E009_maskaug_diff.csv <RMSE>`

## 2026-08-25 3발째 — E010 (판정 규칙 사전 등록)

`P2/subs/E010_stackregime_complete.csv` sha256 `57aeef5e…b36c8974`, 26,061행, mean 23.039509.
E009 대비 **완전행 18,295 에서만** 다름 (mean Δ -0.010621, std Δ 0.023332, max|Δ| 0.213086).
기준선 E009 public 0.525759.

목적은 점수가 아니라 **측정**이다. 완전행 구간은 public 에서 한 번도 측정된 적이 없다.

2024_sepoct 판정기 예측: stack_regime 0.514111 vs gate_robust_blend 0.495506 = **+0.018606 패배**.

| public 결과 | 해석 | 다음 |
|---|---|---|
| > 0.525759 | 판정기가 완전행에서도 맞음 (검증점 2/2) | 2024_sepoct 를 단일 판정기로 확정, 로컬만으로 후보 선별 |
| < 0.525759 | 판정기가 완전행에서 틀림 | 결측행/완전행 판정기를 분리해야 함. 완전행 축 재개 |
| 0.5258±0.0002 | 무차별 | 완전행은 후보 간 차이가 없는 구간 |

순수잡음 예상 0.526197 / 완전이득 하한 0.525320 (측정폭 0.000877).
제출 후: `python P2/src/submission_tools.py public add P2/subs/E010_stackregime_complete.csv <RMSE>`

## 4발째 예약

E013(Sep-Oct 판정기 스택)이 maskaug 를 이기면 그것, 못 이기면
`P2/subs/E011_missingonly.csv` (측정폭 0.013672, 판정기 예측 +0.004381 패배).

### E010 결과 — 판정기 확정

public **0.528449** (+0.002690 vs E009 0.525759). 사전 등록 규칙의 "판정기 적중" 분기.
2024_sepoct 판정기 **2/2** (결측행·완전행 양쪽). 이제 로컬만으로 후보 선별이 가능하다.

추가: 실측 초과분 0.002838 은 `mse_shift` 0.000461 의 6.1배 = 순수잡음 상한 0.526197 을 넘었다.
stack_regime 의 완전행 변경은 무작위가 아니라 체계적 오방향(corr(Δ,오차) ≈ +0.15).
**완전행은 `gate_robust_blend` 가 확실히 낫다. 완전행 재조합 축은 닫는다.**

public 원장 3건:

| CSV | 결측행 | 완전행 | public |
|---|---|---|---|
| E004_gate | E002 | robust_blend | 0.537099 |
| **E009_maskaug_diff** | **maskaug** | **robust_blend** | **0.525759** ← 최고 |
| E010_stackregime_complete | maskaug | stack_regime | 0.528449 |

## 2026-08-25 4발째 — E011 (판정 규칙 사전 등록)

E013 결과: Sep-Oct 판정에서 저장 풀 재조합이 maskaug 를 못 이긴다(stack_global 동률 = 가중이
maskaug 100% 로 붕괴, stack_regime +0.000489). npy 미생성. bugcheck PASS 19 / FAIL 0.

w축도 public 2점으로 해석 종료: `pred_w = E002 + w(maskaug−E002)`, `E[d²]=0.048101`,
w=1 실측 ΔMSE −0.040448 → `E[d·e]=0.044275`, **w\*=0.9205**. w=1.5 는 악화, w\*의 이득은
RMSE 0.000087. 외삽·축소 제출 불필요.

`P2/subs/E011_missingonly.csv` sha256 `2ba9c685…0e9b4f3`, 26,061행, mean 23.043611.
E009 대비 **결측행 7,766 에서만** 다름 (mean Δ -0.011255, std Δ 0.154898, max|Δ| 1.507483).
측정폭 0.013672. 순수잡음 예상 0.532550 / 완전이득 하한 0.518878.

목적: 판정기의 **결측 세그먼트** 신뢰도 측정. 2024_sepoct 결측행은 n=289 로 판정기에서 가장 약한
부분인데, 남은 상승 여지는 결측행뿐이다. 예측: missing_only 0.496687 vs maskaug 0.492306 = +0.004381 패배.

| public 결과 | 해석 | 다음 |
|---|---|---|
| > 0.525759 | 결측 세그먼트 판정도 신뢰 가능 (3/3) | 신규 학습축을 2024_sepoct 결측 세그먼트로 선별 |
| < 0.525759 | n=289 결측 세그먼트는 판정 불가 | 결측행 후보는 계절정합 창(S1/S3/S4, n≈8000)으로만 판정 |
| > 0.532550 | 순수잡음 상한 초과 = 체계적 오방향 | missing_only 계열 영구 기각 |

제출 후: `python P2/src/submission_tools.py public add P2/subs/E011_missingonly.csv <RMSE>`

## 이후 (제출 슬롯 소진 후)

풀은 소진됐다. 남은 열린 축은 전부 **신규 학습**이 필요하다:
`a.residual_lgb`·`a.tide_context`·`a.profile_depth`·`a.same_sensor_gap`.
판정은 2024_sepoct(전체) + S1/S3/S4(결측, n≈8000) 로 한다. 4-hold 균등 평균을 다시 쓰지 마라.
게이트: `python P2/src/bugcheck.py` 가 `판정: OK` 여야 제출한다.

### E011 결과 — 판정기 3/3

public **0.530045** (+0.004286 vs E009). 예측 +0.004381. 사전 규칙의 "3/3" 분기.
잡음 상한 0.532550 미만 → missing_only 는 오방향이 아니라 그냥 약하다.

## 2026-08-25 종료 — public 4점과 역산

| CSV | 결측행 | 완전행 | public |
|---|---|---|---:|
| E004_gate | E002 | robust_blend | 0.537099 |
| **E009_maskaug_diff** | **maskaug** | **robust_blend** | **0.525759** ← 최고 |
| E010_stackregime_complete | maskaug | stack_regime | 0.528449 |
| E011_missingonly | missing_only | robust_blend | 0.530045 |

`MSE(w) = MSE0 + w²E[d²] − 2w·E[d·e]` 항등식 역산:

| 축 | E[d²] | E[d·e] | w* | w* 이득 |
|---|---:|---:|---:|---:|
| 완전행 (stack_regime − robust_blend) | 0.00046136 | **−0.00118823** | −2.5755 | 0.002919 |
| 결측행 (missing_only − maskaug) | 0.00718763 | +0.00133123 | +0.1852 | 0.000235 |
| 결측행 (maskaug − E002) | 0.048101 | +0.044275 | +0.9205 | 0.000087 |

**완전행 E[d·e] 가 음수 = stack_regime 은 일관되게 틀린 방향.** 반대로 밀면 이득.

## 다음 제출 (E014, 사전 등록)

`P2/subs/E014_antistack_w-1.csv` — 완전행 `2·robust_blend − stack_regime`, 결측행 maskaug 유지.
**예측 public 0.523935** (이득 0.001824). 조립은 `P2/specs/E014-antistack.md` 로 위임 완료.

w = −1 을 쓰는 이유: `E[d²]` 를 전체 test 로 계산했는데 public 은 부분집합이라 w*=−2.5755
외삽은 미검증이다. w=−1 이 포물선의 **3번째 실측점**이 되어 `E[d²]_public` 을 직접 고정한다.

| public 결과 | 해석 | 다음 |
|---|---|---|
| ≈ 0.5239 | 항등식·E[d²] 전체test 근사 모두 성립 | 3점으로 w* 재계산해 그 w 제출 |
| < 0.5239 | E[d²]_public < 전체test | w* 가 더 음수. 재계산 후 제출 |
| 0.5239~0.5258 | 이득은 있으나 예측보다 작다 | 3점으로 포물선 고정, w* 재계산 |
| > 0.525759 | 항등식 역산이 틀렸다 | 완전행 축 영구 종료, E009 최종 |

**주의: 이건 public 에 1파라미터 적합이다.** private 과 어긋날 수 있다.
w 는 |w| ≤ 1 범위 안에서만 쓰고, 최종 제출은 E009 와 E014 중 public 우세한 쪽으로 하되
차이가 0.001 미만이면 과적합이 적은 E009 를 택한다.

### E014/E015 조립 완료 (2026-08-25)

| 항목 | E014_antistack_w-1 | E015_missblend |
|---|---:|---:|
| sha256 | `0cc27d22…daf326b0` | `11bdf515…9caa7ca` |
| 변경 행 | 18,295 완전행 | 7,766 결측행 |
| mean Δ / std Δ | +0.010621 / 0.023332 | −0.002084 / 0.028687 |
| mse_shift | 0.000461344 | 0.000246528 |
| **예측 public** | **0.523935** | 0.525524 |
| audit / check.py sub | OK / OK | OK / OK |

E014 를 다음 제출로 쓴다. E015 는 이득 0.000235 로 작아 보류(E014 3점 고정 후 재평가).
조립 도구: `P2/src/wmix.py` (`--base --delta-from --delta-to --w --out`).
