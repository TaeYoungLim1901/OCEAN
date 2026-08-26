# T20 — exp012: V1(47 feature) full-train + test 예측 + 양성률 앵커 임계 + 제출

## 목적

`f.v1_wins` 의 V1 변형(기존 42 feature + exp007 gain 상위 5개 적분 feature = 47개, 700 tree)은
inner F1 0.702859 로 exp005(0.697836)를 이겼으나 **test 예측이 없다**.
전체 train 단일 모델을 학습해 test 원시 확률을 만들고, 확정된 임계 정책으로 제출 파일을 만든다.

## 배경 — 이미 검증된 사실. 재조사·재측정 금지

- 기준선: 현재 최고 제출 exp010 public F1 **0.741274**, 양성률 2.9832%, 설정 `t_high 0.775 / hole 48`.
- V1 inner F1 **0.702859** (exp005 0.697836, +0.005023) — 후처리 exp005 확정값 고정 조건. 재측정하지 마라.
- 적분 feature 개수 절벽: 5개 이득(0.702859), 12개 붕괴(0.587059), 39개+1400tree 최악(0.551970).
- 양성률-public 곡선 2차 적합 봉우리 **3.0191%**, peak F1 0.741420. 곡선 비대칭(아래 −0.0835 / 위 −0.0238).
- inner 는 임계·후처리 판정기가 아니다 (exp009 inner +0.004606 인데 public −0.018911).
- 환경: `numpy`·`pandas`·`lightgbm` 설치됨. **`pip install` 이 필요하면 설치하지 말고 멈추고 보고해라.**
  `xgboost`·`catboost`·`torch` 는 없다.

## 하지 말 것 (하드 제약)

- feature 집합·하이퍼파라미터·fold 정의·purge 를 바꾸지 마라. V1 정의 그대로다.
- 후처리는 **hole fill 48 하나뿐**이다. dilation 0, min-length 0, hysteresis 없음 (`exp010` 과 동일 형태).
- 봉인 fold3 를 학습·채점·판독하지 마라.
- inner F1 로 임계를 고르지 마라 (`f.inner_delta_no_transfer`). 임계는 **양성률 목표**로만 정한다.
- 외부 데이터 금지. `P1/P1_data/` 는 읽기 전용. `SEED = 42`, `n_jobs <= 2`.
- **학습 실행은 네가 하지 않는다.** 코드를 완성하고 문법·소규모 경로만 검증한 뒤 멈춘다 (아래 §산출 3).

## 구현

새 파일 `P1/src/exp012_v1_full.py`. 기존 코드를 재사용하고 새 추상화를 만들지 마라.
`P1/src/exp007.py` 의 full-train 경로(134~205행)와 `P1/src/exp007_postmortem.py` 의 V1 선택 로직,
`P1/src/exp005.py` 의 `store_score`·제출 저장 경로를 그대로 가져다 쓴다.

### stage `train` — `python P1/src/exp012_v1_full.py train`

1. feature 컬럼 확정:
   - `base_names` = `P1/cache/exp001/train_features.pkl` 의 컬럼 42개
   - `P1/models/exp007_full.txt` 의 gain importance 내림차순에서 `INTEGRATED_FEATURES` 에 속한 상위 5개
   - **assert** 그 5개가 정확히
     `temp_step_abs_144, max_z_center_144, temp_step_144, temp_step_abs_48, psal_step_abs_144`
     (집합 비교). 다르면 예외를 던지고 멈춘다.
   - 최종 컬럼 수 == 47 을 assert.
2. `full_reference = fit_reference(train)` 로 train/test feature 를 만든다 (`exp007.prepare_features` 재사용).
3. 모델 파라미터는 `exp007_postmortem.model_params(700)` 과 동일.
4. 학습 3회, 각각 학습 직후 경과초를 print:
   - forward fold1 mask, forward fold2 mask → `raw_oof[folds==fold]` 채움. fold3 행은 **NaN 유지**
   - full mask (전체 train) → `raw_test = full_model.predict(x_test)`
   - 모델은 `P1/models/exp012_{forward_fold1,forward_fold2,full}.txt` 에 저장하고, 파일이 있으면 재학습하지 않는다
5. 저장:
   - `P1/cache/exp012_raw_oof.npy` (float64, fold3 NaN 유지. **pool 저장 아님** — `f.t19_quarantine` 참조)
   - `P1/cache/exp012_raw_test.npy` (float64)
6. 누출 감사: `exp005.model_audit` 와 같은 형식으로 forward fold1·fold2·full_test 3행의
   `train_rows / train_max_time / predict_min_time / future_rows` 표를 print. **`future_rows` 는 0 이어야 한다.**

### stage `submit` — `python P1/src/exp012_v1_full.py submit`

학습 없이 `P1/cache/exp012_raw_test.npy` 만 읽는다.

1. 임계 탐색: `t` 를 0.001 간격 격자에서 훑어, `fill_runs(raw_test >= t, test_segments, 48)` 의
   test 양성률이 **0.0302** 에 가장 가까워지는 `t` 를 고른다 (동률이면 큰 `t`).
   근거: `a.threshold_rate` 폐쇄 시 2차 적합 봉우리 3.0191%, 곡선이 비대칭이라 위쪽이 덜 손해다.
2. 표 1행으로 print: `t_high / raw_rate(후처리 전) / post_rate / n_positive`.
   추가로 t ∈ {선택−0.02, 선택−0.01, 선택, 선택+0.01, 선택+0.02} 의 `post_rate` 5행 표.
3. inner F1 도 **참고용으로만** print 한다 (`P1/cache/exp012_raw_oof.npy`, fold1·2, 같은 후처리).
   선택에 쓰지 마라.
4. 제출 저장:
   - `P1/subs/exp012.csv` — `P1/subs/exp010.csv` 와 동일한 컬럼·순서·개행·dtype
   - `P1/preds/test_exp012.npy` — `exp005.store_score` 규약 (`clip(raw − t + 0.5, 0, 1)` float32)
   - `P1/cache/exp012_meta.json` — 선택 `t`, hole, 양성수, 양성률, feature 47개 이름, sha256
5. 검증 명령을 **직접 실행하고 출력을 그대로 보고**한다:
   ```
   python P1/src/submission_tools.py audit P1/subs/exp012.csv --pred P1/preds/test_exp012.npy
   python P1/src/check.py sub P1/subs/exp012.csv
   ```

## 재현 계약

운영진이 제출 모델의 답안 재현을 검증한다. 임계·hole fill 을 포함한 **모든 후처리가
`submit` 스테이지 코드 안에 결정론적 상수로** 들어가야 한다. 선택된 `t` 를 `exp012_meta.json`
에 기록하고, 코드는 meta 가 있으면 그 값을 그대로 재사용해 같은 CSV 를 만든다.

## 수락 기준

| 항목 | 기준 |
|---|---|
| `train` 표 | 3행 모두 `future_rows=0` |
| feature 수 | 47, 상위 5개 assert 통과 |
| `submit` 표 | `post_rate` 가 0.0302 ± 0.0005 |
| audit | `판정: OK` |
| `check.py sub` | `판정: OK` |
| 재실행 | `submit` 두 번 실행 시 CSV sha256 동일 |

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 산문 금지, 코드 붙여넣기 금지.
파일 경로 + 위 표들 + 두 검증 명령의 마지막 3줄만 보고한다.

## 산출

1. `P1/src/exp012_v1_full.py`
2. 문법·임포트·소규모 경로 검증 (예: `python -c "import exp012_v1_full"` 수준, 또는 train 스테이지의
   feature 선택·assert 부분만 도는 짧은 확인). **전체 학습을 실행하지 마라 — 샌드박스 시간 제한이다.**
3. 마지막 줄에 사람이 실행할 명령 2줄을 그대로 출력:
   ```
   python P1/src/exp012_v1_full.py train
   python P1/src/exp012_v1_full.py submit
   ```
