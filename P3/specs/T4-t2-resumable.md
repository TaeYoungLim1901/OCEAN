# 작업 지시: T4 — T2 학습 체크포인트·재개 구조

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라** (샌드박스 네트워크 차단).
`P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용. 출력 디렉터리가 없으면 `mkdir` 하라.

**이 사양은 전체 학습을 실행하지 않는다.** fold 1 하나만 돌려 구조를 검증하고 시간을 잰다.
전체 학습은 사람이 돌린다 (`AGENTS.md` §0 예외 3).

## 배경 — 이미 검증된 사실 (재조사 금지)

```
a.station_lead_experts   axis    [open]     pooled, lead-specific, station expert direct-delta GBDT를 clean OOF 잔차와 segment tail로 비교
  <- derives_from f.p3_t2_runtime_stop
python scripts/g.py --p P3 check "station_lead_experts 체크포인트 재개"
  → 겹침 없음 — 진행 가능 (exit 0)
```

- 401피처 캐시는 **이미 생성돼 있고 재사용한다**: `P3/cache/p3_t2/features_train.npy`,
  `features_test.npy`, `feature_ids_train.npy`, `feature_ids_test.npy`, `test_cases.csv`.
  **피처를 다시 만들지 마라.**
- strict-forward fold 정의도 이미 있다: `P3/cache/p3_t2/folds.csv`,
  `foundation_report.json`. 4개 fold 전부 `strict: true` 검증 완료. **fold 정의를 바꾸지 마라.**
- 직전 실행(`p3_t2_runtime_stop`)은 fold 루프가 통째로 하나의 프로세스라 codex 샌드박스
  시간 상한을 넘겼고, 중간 산출물이 없어 **0에서 다시 시작해야 했다.** 이 사양이 고치는 것이 이것뿐이다.
- 로컬 평가셋 272건, 지속성 통합 RMSE `0.8329`. 공식 `0.769455` 와 절대비교 금지, 델타만 쓴다.

## 기준선

`P3/src/train_t2.py` (317행) 현재 상태. 이 스크립트의 **모델·피처·fold·후보(CANDIDATES)·
블렌드 선택 규칙·하이퍼파라미터를 하나도 바꾸지 마라.** 바꾸는 것은 실행 제어 구조뿐이다.

비교 대상 T1F 숫자 (`P3/src/select_t1f.py` 산출):

| | 값 |
|---|---|
| T1F all OOF 통합 RMSE | `0.750246` |
| T1F fold4 봉인 통합 RMSE | `0.754739` |
| fold4 persistence | `0.783322` |
| fold4 damped | `0.770684` |
| fold4 LGBM | `0.784077` |

T2 사전등록 판정 규칙(이미 `P3/experiments/log.md` `p3_t2_pre` 에 기록됨, **변경 금지**):
folds1-3 에서 후보/0.1 pair blend 를 고정하고, fold4 가 T1F 보다 `≥0.002` 개선하며
공통 segment 최대 악화가 `≤0.03` 일 때만 T2 산출물을 생성한다.

## 실험 1 — fold 단위 체크포인트·재개

### 근거

fold 루프가 원자적이라 한 번의 시간 초과가 전체 계산을 버린다. fold 경계에서 중간 결과를
디스크에 남기면 같은 스크립트를 다시 실행하는 것만으로 이어서 계산한다.

### 사양

`P3/src/train_t2.py` 를 수정한다. 새 파일을 만들지 마라.

1. **체크포인트 디렉터리** `P3/cache/p3_t2/ckpt/` 를 `mkdir(parents=True, exist_ok=True)` 한다.

2. **fold 체크포인트.** fold `f` 계산이 끝나면 즉시 `ckpt/fold{f}.npz` 에 저장한다:
   - 각 후보(CANDIDATES) 이름별 val 예측 배열
   - 그 fold 의 val 행 인덱스 `k`
   저장은 fold 루프 안, 다음 fold 로 넘어가기 전에 한다.

3. **재개.** fold 루프 진입 시 `ckpt/fold{f}.npz` 가 있으면 **모델을 학습하지 않고** 로드해
   `oof` 에 채운다. 로드한 배열의 shape 이 기대와 다르면 그 파일을 무시하고 다시 계산한다.

4. **최종 전체학습 체크포인트.** 전체 데이터 재학습 후의 test 예측도
   `ckpt/test_preds.npz` 에 후보 이름별로 저장하고, 존재하면 재학습을 건너뛴다.

5. **진행 출력.** 각 단계 시작·종료에 `print(..., flush=True)` 로 한 줄씩 찍는다.
   형식: `[t2] fold=2 stage=lgbm_pooled elapsed=13.4s` — 사람이 긴 실행을 지켜봐야 한다.
   스크립트 상단에 `sys.stdout.reconfigure(encoding="utf-8")` (콘솔이 cp949).

6. **`--max-folds N` 인자** (argparse, 기본값 4). N 개 fold 까지만 계산하고 종료한다.
   체크포인트만 남기고 최종 전체학습·리포트 단계는 건너뛴다 (`N < 4` 일 때).
   이 인자는 긴 실행을 사람이 쪼개 돌리기 위한 것이다.

7. **`--n-jobs` 인자** (기본값 6). 모든 LightGBM·ExtraTrees 의 `n_jobs` 에 전달한다.
   현재 코드가 이 값을 하드코딩하거나 기본값에 맡기고 있으면 인자로 연결하라.
   8코어를 3세션이 공유한다.

8. **결정론.** 재개해서 얻은 결과와 한 번에 끝까지 돌린 결과가 같아야 한다. 시드는 `SEED = 42`
   고정. LightGBM 에 샘플링(`bagging_fraction`·`feature_fraction`)이 켜져 있으면
   fold 별 시드가 재개 여부에 따라 달라지지 않도록 fold 인덱스에서 시드를 유도하라.

### 실행 — fold 1 만

구조 검증과 시간 측정을 위해 아래 3개만 실행하고 숫자를 보고한다.

```bash
python P3/src/train_t2.py --max-folds 1 --n-jobs 6      # (a) 최초 계산
python P3/src/train_t2.py --max-folds 1 --n-jobs 6      # (b) 재개 — 학습 없이 로드
python P3/src/check.py cv P3/cache/p3_t2/folds.csv      # (c) fold 게이트 회귀 확인
```

**fold 2 이상, 전체 학습, 제출 파일 생성을 실행하지 마라.** 시간 초과의 원인이다.

### 수락 기준

- (a) 가 `ckpt/fold1.npz` 를 남기고 정상 종료한다
- (b) 의 벽시계 시간이 (a) 의 `20%` 미만이고, (b) 가 fold1 에 대해 어떤 모델도 학습하지 않는다
  (진행 출력에 `resume` 표시가 보여야 한다)
- (b) 에서 로드한 fold1 후보별 val 예측이 (a) 의 값과 `np.allclose(atol=0)` 로 동일하다
- (c) 가 `판정: OK`
- `git diff --stat` 기준 수정 파일은 `P3/src/train_t2.py` **하나뿐**이다
  (`ckpt/` 산출물 제외). 새 모듈·설정파일·로깅 프레임워크를 만들지 마라

## 금지

- 가설을 섞지 마라. 이 사양은 실행 제어만 바꾼다. 모델·피처·fold·후보·하이퍼파라미터 변경 금지
- 하이퍼파라미터 튜닝 금지
- fold 2 이상 실행 금지, 전체 학습 실행 금지, 제출 CSV 생성 금지
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- 피처 캐시 재생성 금지 (`features_train.npy` 등은 읽기만)
- 추상화 금지. 인터페이스·config 클래스·로깅 프레임워크 만들지 마라. `np.savez` 와 `print` 로 충분
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
(a) 최초 계산 fold1 벽시계 = ___ s
(a) 단계별 벽시계 = lgbm_pooled ___ s / lgbm_lead ___ s / lgbm_station ___ s / extratrees ___ s / damped ___ s
(a) fold1 후보별 val RMSE = <후보명> ___ (전부 한 줄씩)
(b) 재개 벽시계 = ___ s
(b) 재개 예측 동일성 np.allclose(atol=0) = True | False
(c) check.py cv 판정 = OK | STOP
남긴 파일 = ___
수정 파일 = ___
fold2-4 + 최종 전체학습 예상 벽시계 (fold1 실측 × 학습행수 비율로 환산) = ___ s
정지 조건 발생: 없음 | <사유>
```
