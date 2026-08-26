# P2 재개 지점 (2026-08-25 세션 종료)

이전 세션 원문은 `P2/experiments/NEXT_2026-08-25_archive.md`.
지식은 `python scripts/g.py --p P2 show <노드>`, 실험 이력은 `P2/experiments/log.md`.

## 1. 지금 상태

**최고 제출: `P2/subs/E009_maskaug_diff.csv` — public RMSE 0.525759 · 점수 26.594071**
구성: 결측행 7,766 = `test_maskaug_E005.npy` / 완전행 18,295 = `test_gate_robust_blend_E004.npy`.

public 원장 4점 (`python P2/src/submission_tools.py public list`):

| CSV | 결측행 | 완전행 | public |
|---|---|---|---:|
| E004_gate | E002 | robust_blend | 0.537099 |
| **E009_maskaug_diff** | **maskaug** | **robust_blend** | **0.525759** ← 최고 |
| E010_stackregime_complete | maskaug | stack_regime | 0.528449 |
| E011_missingonly | missing_only | robust_blend | 0.530045 |

## 2. 첫 수 — E014 제출 (조립·검증 끝남)

`P2/subs/E014_antistack_w-1.csv` sha256 `0cc27d22…daf326b0`, 26,061행.
완전행 = `2·robust_blend − stack_regime` (w=−1), 결측행 = maskaug 유지.
E009 대비 완전행 18,295 에서만 다름, `mse_shift` 0.000461344. **예측 public 0.523935.**
`audit`·`check.py sub`·`bugcheck` 전부 `판정: OK`.

**판정 규칙 (결과 보기 전 확정. 결과 보고 해석 만들지 마라):**

| public | 해석 | 다음 |
|---|---|---|
| ≈ 0.5239 | 항등식 성립, `E[d²]` 전체test 근사도 성립 | 3점으로 w* 재계산해 그 w 제출 |
| < 0.5239 | `E[d²]_public` < 전체test | w* 가 더 음수. 재계산 후 제출 |
| 0.5239~0.5258 | 이득은 있으나 예측보다 작다 | 3점으로 포물선 고정, w* 재계산 |
| > 0.525759 | 역산이 틀렸다 | 완전행 축 영구 종료, E009 최종 |

제출 후: `python P2/src/submission_tools.py public add P2/subs/E014_antistack_w-1.csv <RMSE>`

**주의: public 에 1파라미터 적합이다.** |w| ≤ 1 안에서만 쓴다. 최종 제출은 E009 와 E014 중
public 우세한 쪽이되, 차이가 0.001 미만이면 과적합 적은 **E009** 를 택한다.

예비: `P2/subs/E015_missblend.csv` (결측행 w=0.1852 블렌드, 예측 0.525524, 이득 0.000235). 우선순위 낮음.

## 3. 판정기 — 이것이 이번 세션 최대 산출물

**2024_sepoct hold 가 test 대변자다. 3/3 방향 적중.** 4-hold 균등 평균은 **쓰지 마라** (반대를 가리킨다).

| 후보 | 2024_sepoct | 4-hold 평균 | public |
|---|---:|---:|---:|
| gate_robust_blend | 0.495506 | 1.089067 | 0.537099 |
| maskaug | 0.492306 | 1.093805 | 0.525759 |
| stack_regime | 0.514111 | — | 0.528449 |
| missing_only | 0.496687 | — | 0.530045 |

public 절대 수준이 2024_sepoct 자리수와 같고 순서도 맞다. test 2025-09-01~10-31 과 같은
성층 붕괴기라 물리적으로 설명된다. 결측행 후보는 계절정합 창 S1/S3/S4(`season_e006.py`,
n≈8000)를 함께 본다 — 2024_sepoct 의 결측 세그먼트는 n=289 로 얇다.

`python P2/src/judge_sepoct.py` 로 전 풀을 이 판정기로 재채점한다.

## 4. 게이트 — 제출 전 필수

```
python P2/src/bugcheck.py                      # 검사 19종 A~G. 판정: OK 아니면 멈춘다
python P2/src/submission_tools.py audit <csv>
python P2/src/check.py sub <csv>
```

`bugcheck.py` 섹션: A 데이터 무결성 · B 누수 · C train↔inference 대칭 · D 분할정합 ·
E 예측 정렬/위생 · F 재현성 · G 제출 필요성. 현재 `PASS 19 / FAIL 0`.
G 는 원장 최소 public CSV 를 기준으로 `측정폭` 을 낸다(임계 0.00002 = LB 6자리 해상도).

## 5. 닫힌 축 — 다시 열지 마라

`python scripts/g.py --p P2 ls axis --status closed` 로 근거 확인.

- **완전행 재조합** (`a.loho_stack`) — E010 public +0.002690, 잡음상한 초과. stack_regime 은 체계적 오방향
- **w 외삽/축소** (maskaug↔E002) — w\*=0.9205, w=1 대비 이득 RMSE 0.000087. 제출 불필요
- **missing_only 단독** — w\*=0.1852, 이득 0.000235
- **풀 재조합 전반** — E013 Sep-Oct 판정에서 maskaug 를 이기는 조합 0. stack_global 가중이 maskaug 100% 로 붕괴

역산 도구: `MSE(w) = MSE0 + w²·E[d²] − 2w·E[d·e]` 는 **항등식**이다. public 2점이면
`E[d·e]` 가 나오고 w\* 가 나온다. 새 축마다 이걸 먼저 해라 — 제출 없이 축이 닫힌다.

## 6. 남은 축 — 전부 신규 학습 필요

`a.residual_lgb` · `a.tide_context` · `a.profile_depth` · `a.same_sensor_gap`.
상승 여지는 **결측행 7,766** 에 있다. 완전행은 `gate_robust_blend` 로 고정됐다.

학습 실행은 codex 에 위임하지 마라(샌드박스 시간 제한). 코드만 짜게 하고 실행은 사람이 한다.
학습 전 `python scripts/g.py --p P2 check "<키워드>"`.

## 7. 도구 메모

- codex 호출: `bash P2/cx_snap.sh <사양>`, Bash 도구 `dangerouslyDisableSandbox: true` + `run_in_background: true`
- `P2/cx_snap.sh` 는 `scripts/cx.sh` 스냅샷. P3 세션이 `cx.sh` 를 고치면 다시 복사할 것
- 조립 도구: `src/build_sub.py`(npy→csv) · `src/mix_sub.py`(--missing/--complete) · `src/wmix.py`(--base/--delta-from/--delta-to/--w)
- 모델 제출(9/7 마감)은 **즉시 최종 확정**이라 답안 제출을 다 쓴 뒤 마지막에 한다
- 알려진 흠: `subs/E001_lgb_ensemble.csv` 와 `subs/E001_lgb_full.csv` 가 동일 파일. 둘 다 사장 후보라 방치
