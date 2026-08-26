# P1 재개 지점 — 2026-08-26 (2차 세션 종료)

## 세션 시작 게이트

```powershell
python P1/src/bugcheck.py all                    # 판정: OK 여야 한다 (2~3분)
python scripts/g.py --p P1 check "<키워드>"       # exit 1 이면 닫힌 축
```

제출 후보 확정 시: `python P1/src/submission_tools.py audit <csv> --pred <npy>`
(`check.py final` 은 exp002 전용 하드코딩이라 쓰지 마라.)

## 현재 성적

| 항목 | 값 |
|---|---|
| **최고 제출** | `P1/subs/exp016.csv` — public F1 **0.776056** / **27.381306점** |
| sha256 | `B03A5E9C39DC2ADC4BE983166371195779E895E1A4F3EFFE63D924F88552158A` |
| 설정 | base 42 + 이웃 회귀 잔차 `n1`,`n2_432`,`n3`,`n4` = 46 feature / `t_high 0.737` / `hole 48` / 양성 5070 (2.9998%) |
| 8/26 순이득 | +0.034782 (26.456883 → 27.381306). exp010 → exp016 |
| 1위 | 30.885889 (F1 환산 ≈ 0.906). 격차 F1 **0.130** |
| 점수 산식 | `점수 = 26.58 × F1 + 6.754` (관측 6점 오차 0.02 이내. exp013 만 이탈 — 분할 붕괴) |

## 즉시 할 일 — 내일 슬롯 3개

1. **exp019 제출** — `P1/subs/exp019.csv` (49 feature, 양성 5063, sha `221011B2E2…`).
   판정기에서 유일하게 exp016 을 이겼다 (auc_all 0.936068 vs 0.932374, drift 0.913054 vs 0.899629).
   **제출 전 R1 후처리 적용**: 길이 2~11 run 삭제 (`P1/src/t32_postproc.py` 재사용, 길이 1 은 보존).
2. **양성률 프로브** — exp019 원시확률을 **3.6%** 로 재임계. 근거: exp016 역산 정밀도 0.923 / 재현율 0.669
   (test 양성 4.14% 가정) = 과소예측. 봉우리 3.0191% 는 F1 0.74 짜리 구모델 4점으로 적합한 값이다.
3. **T33 결과물** — 아래 착수분 결과에 따라. 없으면 exp016+exp019+exp014 랭크 평균 앙상블.

## 판정기 — 슬롯 없이 거른다

```powershell
python P1/src/t26_judge.py fit <variant>   # 학습 2024년 279,185행 purge 87h, 검증 2025-01~06
python P1/src/t27_judge_fix.py             # K1(auc) K2(rank F1) K3(public 상관)
```

**지표는 `auc_all` 만 본다.** `judge_F1` 은 public 과 Spearman −0.8 이다.
**유효 범위: 같은 학습 행에서 feature·하이퍼파라미터만 바꾼 비교** (`f.judge_scope`).
학습 행을 바꾸는 변형에는 쓰지 마라 (season42 를 2위로 놓았는데 public 은 최하위였다).

현재 값: standardized49 **0.936068** > neighbor46 0.932374 > hp_leaves127 0.929112 >
hp_trees1400 0.927216 > season42 0.923480 > hp_posweight1 0.923437 > base42 0.915129 >
invariant37 0.906768 > integrated47 0.897338.

## 다음 레버 — 우선순위

1. **G-ORS 구멍** (착수 예정 T33). G-ORS 는 층이 1개라 `n1`·`s1`·`s2`·`s8_432` 가 전부 NaN 이다.
   **test 행의 9.7%(16,331행)가 이웃 신호 0으로 판정된다.** 기지간(같은 층 번호, 다른 기지) robust 회귀
   잔차 `g1`·`g3` 로 덮는다. Q3-B 에서 중앙값 기반 거친 버전 `y4` 의 S-ORS offset AUC 가 0.896873 였다.
2. **양성률 재탐색** (위 슬롯 2).
3. **구간 2단계 재설계** — T22 는 오라클 상한 0.555 로 접었지만 원인이 **경계 해상도**임이 밝혀졌다
   (offset IoU p50 0.9746 / drift 0.3611 / noise 0.0708). 경계를 `n1`·`s1` 기반으로 다시 뽑으면 상한이 오른다.
4. 앙상블 — `exp015`(랭크 평균) 코드가 `P1/src/exp013_shift.py ensemble` 에 있다.

## 확정된 판정 원칙 — 어기지 마라

1. **AUC 는 전이되고 F1 은 전이되지 않는다** (`f.neighbor_wins`, `f.judge_scope`).
   새 feature 는 Q 계열 AUC 스크리닝 → 판정기 `auc_all` 순으로 거른다. 슬롯을 스크리닝에 쓰지 마라.
2. **inner(fold1·2)는 어떤 승격 판정에도 쓰지 않는다** (`f.inner_delta_no_transfer`, `f.inner_not_model_judge`).
   임계·후처리·모델 비교 세 번 다 부호가 뒤집혔다.
3. **사전 판정 규칙을 결과 보기 전에 `log.md` 에 적는다.** 8/26 에 exp012·exp013 을 사전 기준으로 정확히 기각했다.
4. 봉인 fold3 는 판독하지 마라.
5. 확률 저장은 `booster.predict` 다. `LGBMClassifier.predict()` 는 0/1 라벨을 준다 (`f.judge_proba_bug`).

## 닫힌 축 — 재제안 금지

`a.threshold_rate`(구모델 기준. **모델이 바뀌면 재개 가능** — 위 슬롯 2) · `a.rate_anchoring` ·
`a.integrated_features`(exp012 public −0.023299) · `a.season_matched`(exp013 −0.158797) ·
`a.hyperparams`(판정기 3종 전부 하락) · `a.type_multiclass` · `a.random_rows` · `a.adversarial_drift` ·
hysteresis(exp009 −0.018911) · min-length L=12(spike recall 붕괴. **길이 2~11 만 지우는 R1 은 유효**).

## 문제지에서 나온 하드 사전

- 주입 지속시간: spike 1행, noise 18~353, flatline 12~283, offset 48~519, drift 54~519.
  train 라벨 run 263개 실측 — **길이 2~11 이 0개, 520+ 도 0개** (`r.duration_prior`).
- 이상은 원자료 품질플래그 good_data 구간에만 주입. 유효 관측의 약 4%.
- 층간 상관은 조석 혼합기엔 유효하나 **수온약층 발달기엔 소실**. test 는 2026-01~06 (겨울→봄 성층 전이).
- **정점·층 단위 표준화 지표가 전 구간 공통 임계보다 안정적**이라고 문제지가 명시 (s1·s2 가 이걸 구현했고 이겼다).
- G-ORS 2026 depth 전부 결측, I-ORS 2026 층 3 미운영, S-ORS 2025년 9~10월 층 2·3·4 는 P2 보호로 제외.
- 규칙 베이스라인 전체 F1 0.548255. 종합점수는 **각 리더보드 분할에서 베이스라인을 재채점** → 분할이 여러 개다.

## 오늘 제출 대장

| 실험 | 양성률 | public F1 | 점수 |
|---|---|---|---|
| **exp016** | 2.9998% | **0.776056** | **27.381306** |
| exp010 | 2.9832% | 0.741274 | 26.456883 |
| exp012 | 3.0501% | 0.717975 | 25.837638 |
| exp005 | 3.4791% | 0.717435 | 25.823277 |
| exp009 | 4.1808% | 0.698524 | 25.320658 |
| exp011 | 2.1602% | 0.657820 | 24.238822 |
| exp013 | 2.9874% | 0.582477 | 11.455269 |

미제출 대기: `exp014`(37f, `BE5183376F`) · `exp019`(49f, `221011B2E2`) · `exp020`(exp016+R1, `5F16F7FCF5`).

## 위임

- 구현: `bash scripts/cx.sh P1/specs/T{n}.md` — **`scripts\cx.cmd` 는 Bash 도구에서 깨진다.**
  `dangerouslyDisableSandbox: true` + `run_in_background: true`, 로그는 `P1/experiments/logs/`.
- 사양은 `python scripts/preflight.py spec <사양>` 이 `판정: OK` 여야 한다.
  자주 걸리는 것: `기준선 숫자`, `pip install 정지 지시`, `배경(재조사 금지)`, 경로는 `P1/cache/` 처럼 접두사 포함.
- 긴 학습은 사람이 돌린다. codex 는 코드만 만든다.
