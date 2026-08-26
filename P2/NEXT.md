# P2 재개 지점 (2026-08-26 세션 종료)

## 세션 시작 게이트

```powershell
python P2/src/bugcheck.py all                    # 판정: OK 여야 한다
python scripts/g.py --p P2 check "<키워드>"       # exit 1 이면 닫힌 축
```

제출 후보 확정 시:

```powershell
python P2/src/submission_tools.py audit <csv> --pred <npy>   # --pred 필수
python P2/src/predict_public.py <csv>                        # 예상 RMSE·점수
python P2/src/check.py sub <csv>
```

**재현 게이트** (모델 제출 전 필수. 20~40분, 전체 재학습):

```powershell
python P2/src/check.py final <csv>     # 빈 폴더 바이트 재현 + manifest
```

2026-08-26 E017 로 `match=True pred_bytes_match=True` 확인됨.

이전 세션 원문 `P2/experiments/NEXT_2026-08-26_archive.md`.

## 1. 지금 상태

**최고 제출: `P2/subs/E017_antistack_w-2.5.csv` — public 0.522638 · 점수 26.775526**
완전행 18,295 = `3.5*robust_blend - 2.5*stack_regime`, 결측행 7,766 = `test_maskaug_E005.npy`.

**1위 28.362053 = RMSE 0.3962. 격차 0.126.** 재조합으로 못 메운다.

원장: `python P2/src/submission_tools.py public list`

| CSV | public | 점수 |
|---|---:|---:|
| **E017_antistack_w-2.5** | **0.522638** | **26.775526** |
| E014_antistack_w-1 | 0.523888 | 26.759837 |
| E009_maskaug_diff | 0.525759 | 26.736334 |
| E010_stackregime_complete | 0.528449 | 26.702577 |
| E011_missingonly | 0.530045 | — |
| E004_gate | 0.537099 | 26.594071 |
| E022_stack_sepoct | 0.560599 | 26.299202 |

## 2. 첫 수 — E023 제출 (조립·검증 끝남)

`P2/subs/E023_antistack2_w-0.5.csv` sha256 `24aaf61d…3e45bce7`, 26,061행,
range [13.842197, 29.348499]. `1.5*E017 - 0.5*E022` (전 행).
`audit npy_exact=True`, `check.py sub 판정: OK`.

E022 방향의 `E[d·e]` 가 음수라 반대로 가면 이득이다. `m1-m0 = 0.0411208`,
`A_full = 0.019589252` → `B = -0.0107658`, `w* = -0.5496`.

**예측 public 0.516994 (A_full 기준) ~ 0.516212 (A_pub = A_full×0.945 기준).**
손익분기: `A_pub > 0.0274` 이면 손해. `A_full = 0.0196` 이고 직전 방향 실측비가 0.945 였다.

**판정 규칙 (결과 보기 전 확정. 결과 보고 해석 만들지 마라):**

| public | 해석 | 다음 |
|---|---|---|
| 0.5155~0.5175 | 항등식 성립 | 3점으로 w* 확정, 그 w 제출 |
| < 0.5155 | A_pub 이 예상보다 작다 | w* 더 음수. 재계산 후 제출 |
| 0.5175~0.522638 | 이득 있으나 예측 미달 | 3점 포물선 고정 후 재계산 |
| > 0.522638 | 역산 전제가 깨졌다 | 이 선분 종료, E017 최종 |

기록: `python P2/src/submission_tools.py public add P2/subs/E023_antistack2_w-0.5.csv <RMSE>`

## 3. 점수 공식 — 확정

```
점수 = 33.3335 - 12.54789 x RMSE      (최대 100/3)
```

public 3점에서 역산, 세 점 모두 잔차 ±0.0001. `g.py show f.score_formula`.

## 4. 판정기 — 한계를 알고 써라

`python P2/src/judge_sepoct.py` (읽기 전용. 제출 CSV 는 `--write-artifacts` 로만 만든다)

| 후보 | 2024_sepoct=S1 | 결측(n=289) | 완전 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| maskaug | 0.491145 | 0.947625 | 0.483651 | 0.340521 | 0.194028 |
| A2 (target=temp-B2) | 0.500639 | 0.986708 | 0.492543 | 0.511307 | 0.462888 |
| A0 (E002 계열 대조군) | 0.517441 | 1.135442 | 0.506344 | 0.412991 | 0.214216 |
| A1 (+B2 피처 9개) | 0.524409 | 1.169203 | 0.512698 | 0.421609 | 0.236953 |
| baseB2 (학습 없음) | 0.686777 | 1.407025 | 0.674455 | 0.640976 | 0.306770 |
| baseB1 PCHIP | 0.860074 | 2.689355 | 0.817014 | 0.861862 | 0.469656 |
| base_nom 선형 | 0.970325 | 2.051059 | 0.951427 | 0.862944 | 0.428320 |

- **2024_sepoct = S1 이다.** 같은 창이다. S3·S4 는 test 계절에서 0.5·1개월 밀렸다
- 2024_sepoct 결측 세그먼트는 n=289 = 홀드의 1.1%. **실제 test 는 29.8%.** 대표성 없다
- **`f.judge_fails_freshstack`: 창 OOF 위에서 가중치를 새로 적합한 스택에는 판정기가 반대를
  가리킨다.** E022 가 판정 -0.012688 우세였는데 public +0.037961 패배. E008 스택은 맞혔다.
  스택 후보를 판정기 우세만으로 제출하지 마라
- 4-hold 균등 평균은 쓰지 마라

## 5. 닫힌 축

`python scripts/g.py --p P2 ls axis --status closed`

- `a.antistack_extrap` — 완전행 w 는 -2.5 로 확정. w*=-2.7523 이지만 추가 이득 0.000029
- `a.b2_target_stack` — E022 public 0.560599 패배
- `a.loho_stack` · `a.profile_eof` · `a.regime_blend` · w 외삽 · missing_only 단독 · 풀 재조합 전반

**항등식이 축을 제출 없이 닫는다.** `MSE(w) = MSE0 + w²·E[d²] - 2w·E[d·e]` 는 정확하다.
public 2점이면 `E[d·e]`, 3점이면 과결정. 새 축마다 먼저 해라.
`python P2/src/predict_public.py <csv>` 가 자동으로 한다.

## 6. 남은 부채 — 착수 전 이것부터

1. **`predict_public.py` 평행쌍 탐지 버그.** E023 처럼 원장 2점 위의 선분인데 `exact=False`
   를 낸다. 기준 CSV 자신을 쌍에서 제외하는 것으로 보인다. 손계산은 맞았다
2. **B2 tanh 적합 미캐시.** A1 학습이 A0 의 22.6배(323 → 7,307초), A2 는 13,341초.
   T19 는 대상행 26k 만 계산했고 학습행 140k+ 는 매 창 재계산한다. 체인 전파하려면 필수
3. `subs/E001_lgb_ensemble.csv` 와 `subs/E001_lgb_full.csv` 가 동일 파일. 둘 다 사장

## 7. 남은 축 — 전부 신규 학습

`a.residual_lgb` · `a.tide_context` · `a.same_sensor_gap` · `a.profile_depth`(열림)

- 조석은 `common.py:139` 에 m2 12.4206 · s2 12.0 이 **전 층 공통**으로 들어있다.
  문제지가 말하는 "내부파에 의한 **층간 위상차**" 는 미구현. B2 의 `z0`(수온약층 중심)가
  조석 주기로 오르내리는 게 그 물리적 실체다. `a.tide_context` 와 한 번에 붙일 수 있다
- 격차 분해: test MSE 0.273151 = 0.702·MSE_완전 + 0.298·MSE_결측.
  결측을 완벽히 맞춰도 0.3843 로 1위 0.3962 를 겨우 넘는다. **완전행 70%가 MSE 의 54%다.**
  완전행 *모델링* 은 미개척이다 (닫힌 것은 완전행 *재조합* 이다)

## 8. 도구

- codex: `bash P2/cx_snap.sh <사양>`, Bash 도구 `dangerouslyDisableSandbox: true` + `run_in_background: true`
- codex 샌드박스는 numpy/pandas 를 못 본다. 거기서 나온 `bugcheck FAIL` 은 로컬에서 재확인해라
- 조립: `build_sub.py`(npy→csv) · `mix_sub.py`(--missing/--complete/--out-npy) · `wmix.py`(--base/--delta-from/--delta-to/--w)
- **`wmix.py` 는 마스크가 없다.** 델타를 26,061행 전부에 적용한다. 완전행만 바꾸려면
  `wmix` 로 벡터를 만들고 `mix_sub` 로 합쳐라. 2026-08-26 에 여기서 한 번 틀렸다
- 재현 체인: `python P2/src/run_all.py --out <csv> --w -2.5`
- `train.py` 에 `n_jobs=2` 고정. 세 터미널 동시 학습해도 6스레드다. 조율 불필요
- 모델 제출(9/7)은 즉시 최종 확정. 답안 제출을 다 쓴 뒤 마지막에 한다
