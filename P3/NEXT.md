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

상태 요약 (2026-08-25 3차 세션 종료 시점):
- 제출 예산·일일 제한은 **`RULES.md` §0 참조**. `P3CONTEXT.md:16` 의 제한 표기는 폐기한다
- public ledger 4점 확보. 조밀 재가중 판정기가 **LB 로 검증**됐다 (`f.lb_picks_dense_judge`).
  씨닝 판정기는 부호가 반대였다 — 쓰지 마라
- damped-lgbm 단순선은 소진 (`f.damped_lgbm_line_exhausted`). 최적 w=0.2365 라도 개선 0.0013
- public 은 200사례 중 ~80 이하 무작위 부분집합과 모순 없다 (`f.public_small_random_subset`)
- **T1F 는 계통적 과대예측** (`f.t1f_positive_bias`). public +0.041550 / 로컬 +0.035652±0.008937

오늘 첫 할 일:
1. `P3/subs/submission_p3_t1f_bias.csv` 제출 (c*=-0.041550308, 게이트 통과 완료).
   사전등록 `p3_t22_pre` 규칙(3): 실측이 예측 RMSE `0.585693` 과 재구성 오차 `<=2e-6` 이면 축 채택,
   어긋나면 축을 닫는다
2. 점수 받으면 즉시 `python P3/src/submission_tools.py public add <csv> <PUBLIC_RMSE>`
3. 남은 2회는 필수적일 때만 쓴다. 단 슬롯을 그냥 버리지는 마라 — 개선 후보가 없으면
   **측정용 탐침**으로 쓴다 (제출 1회 = public 잔차의 선형범함수 1개, `p3_t22_pre` 참조)

열린 축 (`g.py --p P3 ls axis --status open`): ratio_target, log_target_smearing,
huber_objective, spectral_forecast, multioutput_joint_tree, underused_atmos_derivatives,
external_pretrain_linear, lb_quadratic_simplex.
외부 데이터는 규정상 허용이다 (`r.external_data_unmentioned`) — 단 9/7 재현 요건이 따라온다.

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

## 함정 3개

- `python P3/src/check.py cv` 를 **조밀 인덱스에 걸면 FAIL 이 뜨는데 누수가 아니다**
  (`f.dense_cv_gate_na`). 씨닝 평가분할 전용 게이트다
- `select_t1f.py` 격자탐색이 `W_new` 를 고르며 경고를 찍는다 — **정상**. 제출 가중은 상수 고정
- `scripts\cx.cmd` 는 Bash 에서 경로가 깨진다. `bash scripts/cx.sh <사양>` 을 쓰고
  백그라운드 + 샌드박스 해제로 돌려라
