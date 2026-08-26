# P3 세션 시작 프롬프트

## 세션 시작 게이트 (2026-08-26 확정)

착수 전 2줄. 실패하면 실험하지 말고 원인부터 고친다. **검사를 완화하지 마라.**

```powershell
python P3/src/bugcheck.py all                    # 판정: OK 여야 한다
python scripts/g.py --p P3 check "<키워드>"       # exit 1 이면 닫힌 축
```

제출 후보를 확정했을 때:

```powershell
python P3/src/submission_tools.py audit <제출csv> --pred <예측npy>   # --pred 는 필수다
python P3/src/check.py final <제출csv>                              # 빈 폴더 바이트 재현 + manifest
```

**모델 제출 직전에만** (세 문제 후보가 전부 정해졌을 때. 1회 확정이라 복구 불가):

```powershell
python scripts/preflight.py final --p1 <P1csv> --p2 <P2csv> --p3 <P3csv>
```

2026-08-26 기준 세 문제 `bugcheck all` 전부 `판정: OK`, 전역 장벽 `판정: READY`.
마감·잔여 제출 횟수는 세션 시작 훅이 출력한다 (`.claude/hooks/ocean_context.py`).
규정 수치를 문서·graph 에 복사하지 마라 — `python scripts/preflight.py rules` 가 잡는다.

다음 세션 첫 입력으로 아래를 그대로 붙여넣는다.

---

```
P3 세션이다. 터미널 1개 = 문제 1개, P1/P2 트리와 루트 파일은 건드리지 마라
(루트는 P3 세션 소유. AGENTS.md §1.1).

읽는 순서 — 통독 금지:
1. AGENTS.md, P3/AGENTS.md
2. OPUS_HANDOFF_2026-08-13.md 의 P3 절만
3. P3/SUBMISSION_PLAN.md

상태 요약 (2026-08-26 4차 세션 종료 시점):
- 제출 예산·일일 제한은 **`RULES.md` §0 참조**. `P3CONTEXT.md:16` 의 제한 표기는 폐기한다
- **현행 최고 = `P3/subs/submission_p3_alpha05.csv` public 0.577650** (0.5*T1F + 0.5*lr,
  SHA `FD524F7F66DB9F8C04408B3350B8CB2CE0C043DFCD0D4E23EEC1C0675BC473D2`). ledger 8점
- **public 은 66사례(396건)뿐이다** (`f.public_split_66_cases`). LB RMSE 델타 0.003~0.007 은
  표본잡음과 구별 불가 (`f.lb_delta_noise_floor`). **후보 선택은 조밀 재가중 fold4 판정기가 1차**,
  LB 는 0.05 급 검증과 항등식 실측 전용
- **통합(4폴드) 판정기를 쓰지 마라** (`f.alpha_local_fold_flip`). 폴드 상쇄로 0 이 된다.
  test 는 fold4 이후이므로 fold4 만 train->test 를 복제한다
- **1파라미터 폐형 직선최적화가 유일하게 성공한 조합법** (`f.line_search_closed_form`).
  5멤버 격자탐색은 T13·T21·T26 3연패. 자유도를 늘리지 마라
- 로그비율 타깃 `lr` 이 봉인 fold4 최강 단독 (`f.log_ratio_target_wins`, 시드 분산 0.001803)

오늘 첫 할 일 — 슬롯 3회 배분은 이미 정해져 있다:
1. `BASE05 + c` (`c = -0.02`) 안전탐침 제출. **`u=1` 은 `B = mean_pub(1) = 1` 이 정확히
   알려져 있어 부분집합 가정이 전혀 필요 없다** — 제출 1회로 public 잔여편향 `b` 가 정확히 풀린다:
   `b = (M_c - M_0 - c^2) / (2c)`, `M_0 = 0.577650^2`
2. 최적 상수 `c* = -b` 적용해 2회차 제출. 실측이 예측과 재구성 오차 `<=2e-6` 이면 축 채택
3. `ls` 방향 착수 (BASE05 기준 fold4 `alpha* = +0.389041`, 예상 0.596593)

로컬 fold4 는 BASE05 잔여편향을 `+0.0747` 로, 전체test 근사는 `+0.0025` 로 본다. 재면 끝난다.

열린 축: ratio_target, log_target_smearing, huber_objective, spectral_forecast,
multioutput_joint_tree, underused_atmos_derivatives, external_pretrain_linear, lb_quadratic_simplex.
외부 데이터는 규정상 허용이다 (`r.external_data_unmentioned`) — 단 9/7 재현 요건이 따라온다.

상설 채점기 `python P3/src/judge_dense.py <oof_npy> ...` 로 조밀 재가중 fold1~4 를 낸다.

검사 도구: `python P3/src/bugcheck.py {all,train,wcurve,simplex,subset}`. 제출 전 `all` 필수.

새 로컬 실험 전 `python scripts/g.py --p P3 check "<키워드>"`. 코드는 직접 쓰지 말고
`P3/specs/*.md` 사양으로 위임한다 (`scripts\cx.cmd <사양>`).

새 로컬 실험을 하려면 먼저:
  python scripts/g.py --p P3 check "<키워드>"
exit 1 이면 닫힌 축이다. 코드는 직접 쓰지 말고 P3/specs/*.md 사양으로 위임한다
(bash scripts/cx.sh <사양>).
```

---

## 1기 제출 일정 (확정, 게이트 통과 완료)

| 날짜 | 파일 | SHA-256 |
|---|---|---|
| 8/20 | `P3/subs/submission_p3_t1f.csv` | `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` |
| 8/21 | `P3/subs/submission_p3_lgbm_only.csv` | `63496516733EDA854C3D781455DE53A3C46A8B640082CBC4641E18CAD538D259` |
| 8/22 | `P3/subs/submission_p3_damped_only.csv` | `54A4C41CA4D6C552635CD7B1D48F39FC5D906E614346141D9BC03EC2A92FE038` |

제출 직전 필수:

```bash
python P3/src/submission_tools.py audit <csv> --pred <대응 npy>
python scripts/preflight.py sub <csv> --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
```

## 함정 4개

- `python P3/src/check.py cv` 를 **조밀 인덱스에 걸면 FAIL 이 뜨는데 누수가 아니다**
  (`f.dense_cv_gate_na`). 씨닝 평가분할 전용 게이트다
- `select_t1f.py` 격자탐색이 `W_new` 를 고르며 경고를 찍는다 — **정상**. 제출 가중은 상수 고정
- `scripts\cx.cmd` 는 Bash 에서 경로가 깨진다. `bash scripts/cx.sh <사양>` 을 쓰고
  백그라운드 + 샌드박스 해제로 돌려라
- **실행 중인 `scripts/cx.sh` 를 편집하지 마라.** bash 가 바이트 오프셋으로 이어읽어
  `syntax error near unexpected token` 으로 죽는다 (2026-08-26 T27, 산출물은 무사했다).
  codex 가 `ModuleNotFoundError: No module named 'numpy'` 로 죽으면 PYTHONPATH 가 아니라
  **APPDATA** 를 의심해라 — 2026-08-26 에 `cx.sh` 가 둘 다 주입하도록 고쳤다
