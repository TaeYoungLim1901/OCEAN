# Opus handoff — 2026-08-13

상위 실행 계약은 `CLAUDE.md` → `AGENTS.md` → 각 `P{n}/AGENTS.md` 순서로 읽는다.
아래 3개만 현재 제출 가능 후보이며, Public 점수는 아직 기록되지 않았다.

| 문제 | 현재 제출 후보 | SHA-256 | 로컬 판정 |
|---|---|---|---|
| P1 | `P1/subs/exp002.csv` | `4C96D5F38BE6893AC7E2829D43BC35F2C8ECBF9FE6D644781829AE506E833DCB` | honest outer F1 `0.766423` |
| P2 | `P2/subs/E002_final.csv` | `F88AC4BF1254C1A8AD52A8505A0CFFADE67ECC23AC637E296A1CAC8B854D2AA9` | 2024 RMSE `0.517441`, 2025 pseudo-gap `1.301486` |
| P3 | `P3/subs/submission_p3_t1f.csv` | `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` | 씨닝 뷰A(216) `0.801413`, 씨닝 fold4 `0.750315`, 조밀 재가중 fold4 `0.611959` |

## 제출 직전과 직후

```powershell
python P1/src/submission_tools.py audit P1/subs/exp002.csv --pred P1/preds/test_exp002.npy
python P2/src/submission_tools.py audit P2/subs/E002_final.csv --pred P2/preds/test_fixed_depth_E002.npy
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f.csv --pred P3/preds/test_final_p3_t1f.npy

# 리더보드 확인 후 실제 점수만 기록한다.
python P1/src/submission_tools.py public add P1/subs/exp002.csv <PUBLIC_F1>
python P2/src/submission_tools.py public add P2/subs/E002_final.csv <PUBLIC_RMSE>
python P3/src/submission_tools.py public add P3/subs/submission_p3_t1f.csv <PUBLIC_RMSE>
```

## P1

- 읽을 것: `P1/COMPETITION_RESEARCH.md`, `P1/DOMAIN_RESEARCH.md`, `P1/graph.json`, `P1/experiments/log.md`는 마지막 3행만 grep.
- exp003 type-multiclass 전략은 sealed outer `0.760045`, exp002 대비 `-0.006379`로 기각했다.
- **`P1/subs/exp003.csv`는 early-stop 누수 발견 전 산출물이므로 제출 금지.** OOF/test family만 앙상블 풀로 보존한다.
- drift AUC `0.850133`; 우선 실패축은 noise, I-ORS L7/L1/L5다.
- 다음 실험은 exp002 오류와 낮은 상관을 보이는 noise/drift·I-ORS specialist만 검토한다. strict-forward inner와 봉인 outer를 유지한다.

## P2

- 읽을 것: `P2/COMPETITION_RESEARCH.md`, `P2/DOMAIN_RESEARCH.md`, `P2/cache/E003/metrics.json`, `P2/experiments/log.md` 마지막 행.
- E003 enhanced/EOF/blend는 4-hold 평균을 개선했지만 2025 test-like `missing_public`가 각각 `+0.066278/+0.057425/+0.050881` 악화해 기각했다.
- E003 후보 OOF/test는 보존했지만 제출 CSV는 만들지 않았다. 현재 E002만 제출 가능하다.
- adversarial AUC가 `0.992764~1.000`으로 매우 높아 모든 로컬 CV는 탐색기다.
- 다음 실험은 공개층 결측 regime 전용 보정 또는 E002와 E003의 결측-gated 혼합만 사전등록 후 검토한다.

## P3

2026-08-13 2차 세션에서 T4~T12를 돌렸다. **제출은 T1F 그대로이며 예측값은 한 번도 바뀌지 않았다.**

- 읽을 것: `P3/graph.json` (`g.py ls fact`), `P3/experiments/log.md` 마지막 5행. 연구문서는 필요한 절만.
- `r.context_only`: 절대시각, 다른 test case 통계, t0 이후 입력은 금지다. `P3/src/context_static_check.py`를 항상 실행한다.

**닫은 축 3개** — `a.station_lead_experts`(401피처 GBDT 5후보 전부 damped 에 패배),
`a.equilibrium_relax`(eqrelax 단독 1.0203, wspd 37% 결측), `a.strict_forward_nested`(구조 검증 완료).

**판정기가 이 문제의 핵심 제약이다** (`f.thinned_judge_inverts`). 78h 씨닝 봉인 fold4 는 53사례뿐이고
`hs0>3m` 을 0건 포함한다. 이 자로는 damped 와 lgbm 의 순위조차 뒤집힌다:

| 후보 | 씨닝(53) | 조밀 비가중(1163) | 조밀 재가중(ESS 649) |
|---|---|---|---|
| persistence | 0.783322 | 0.900254 | 0.769115 |
| damped | 0.770747 | 0.724137 | 0.663402 |
| lgbm_t1 | 0.775590 | 0.664543 | 0.614009 |
| T1F | 0.750315 | 0.665931 | 0.611959 |

새 실험은 **조밀 재가중 판정기**(`P3/src/train_dense.py`, `P3/cache/p3_t9/dense_weights.csv`)로 판정하라.
씨닝 fold4 단독 판정은 금지에 가깝다.

**누수 이력.** `train.py` 대칭 `abs(hours)` 마스크로 fold0 OOF 가 전부 미래 학습이었다. T6에서 수정,
T10에서 fold0 을 건너뛰도록 고쳤다. 그래서 이전 기록 `all OOF 0.750246` / `뷰A 0.761126` 은 무효다.
정직한 값은 위 표와 `f.t1f_honest_scores`.

**제출 가중은 `select_t1f.py` 에 상수로 고정**돼 있다 (`0.00 / 0.35 / 0.65`, persistence/damped/lgbm).
격자탐색은 진단용으로 남아 `W_new (0,0.65,0.35)` 를 고르며 경고를 찍는다 — 정상이다.
T11 사전등록 규칙(조밀 재가중 4폴드 중 3폴드 승리)에서 `W_new` 가 2/4 로 미달했다.

**최근성 축은 닫혔다.** T11 Δ(=W_new−W_old) 는 fold1 `-0.051305` / fold2 `-0.013585` /
fold3 `+0.009943` / fold4 `+0.015275` 다. 즉 **후반 폴드일수록 `W_old` 가 우세**하고 test 기간은
fold4 이후다. 최근성 가중 판정기는 `W_old` 를 강화할 뿐이라 새 정보가 없다. `W_old` 확정.
(2026-08-13 정정: 이 문단은 앞서 부호를 반대로 적었다.)

**T13·T15 로 로컬 개선 경로가 소진됐다.** 두 후보를 다 돌렸고 둘 다 기각이다.

- T13 (하이퍼파라미터 재탐색): 6후보 중 3/4 폴드 승리 0개. **folds1-3 순위와 봉인 fold4 순위가
  Spearman `-0.83` 역상관** (`f.hp_below_noise_floor`). 로컬로 고르면 반대로 고른다
- T15 (풀 재조합): 조밀 풀은 사실상 **3멤버**뿐 — LGBM 변형 6개가 `lgbm_t1` 과 ρ `0.989~0.996`.
  상관 `0.90` 미만은 `damped`(0.842)·`persistence`(0.746) 둘뿐이고 **그 3개가 이미 T1F 구성**이다
  (`f.pool_no_diversity`). 동일가중 블렌드 봉인 fold4 `0.653256` = T1F 대비 `+0.041297` 악화

**남은 유일한 자는 public LB 다.** 계획은 `P3/SUBMISSION_PLAN.md` — 8/20~9/6 **18회**,
1기 3회를 판정기 교정에 쓴다. 1기 제출물 3개는 게이트 통과 상태로 확보돼 있다.

**게이트 주의.** `python P3/src/check.py cv` 를 조밀 인덱스에 걸면 FAIL 이 뜨는데 **누수가 아니다**
(`f.dense_cv_gate_na`). 씨닝 평가분할 전용 게이트다.

**보존 자산.** `P3/preds/_backup_pre_maskfix/` 27파일(수정 전 스냅샷, 수정 금지),
`P3/subs/submission_p3_t1f_r2.csv`(`W_new` 판, 게이트 통과), T2 5후보·eqrelax·조밀 OOF 3후보.

## 종료 상태

- 3개 최종 CSV는 submission audit·NPY 재현·graph selftest를 통과했다.
- 프로젝트 학습 프로세스는 `0`개다.
- 연구 근거와 적용/기각 판단은 각 `P{n}/COMPETITION_RESEARCH.md` 및 `graph.json`에 기록됐다.
- 원본 데이터는 수정하지 않았다.
