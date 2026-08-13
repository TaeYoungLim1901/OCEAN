# 작업 지시: T10 — 제출 재현성 복구 (fold0 strict-forward 붕괴)

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라**. `P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용.

**이 사양은 점수를 올리는 실험이 아니다. 규정 위반(재현 불가) 제거가 목적이다.**
학습이 있지만 T1 파이프라인은 짧다(조밀판 4폴드 25.7s 실측). 10분 초과 시 멈추고 보고하라.

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.t1_symmetric_mask_leak  train.py 대칭 abs(hours) 마스크 → T6 에서 strict-forward 로 수정
f.t1f_weights             T1F 실제 가중 persistence 0.00 / damped 0.35 / lgbm_t1 0.65.
                          fold4 에서 씨닝·조밀비가중·조밀재가중 3판정기 모두 1위
f.weight_time_unstable    최적 가중은 시간 불안정. 재선택은 정당화되지 않는다
```

- 현재 `python P3/src/train.py` 는 실행 즉시 죽는다:
  `ValueError: Found array with 0 sample(s) (shape=(0, 371)) while a minimum of 2 is required by LGBMRegressor`
- 원인: strict-forward 마스크 하에서 **fold0 은 이전 데이터가 없어 학습행이 0** 이다.
  기존 대칭 마스크에서는 fold0 OOF 가 전부 미래 데이터로 학습된 모델의 예측이었다
- `P3/P3_data/` 규정: 업로드한 예측을 코드가 재현하지 못하면 성적 무효
- 백업: `P3/preds/_backup_pre_maskfix/` 에 수정 전 `*.npy` 전체와 제출 CSV 사본이 있다.
  **이 백업 디렉터리를 수정·삭제하지 마라.**

현재 제출 후보 해시 (변경 여부 판정용):

```
P3/preds/test_final_p3_t1f.npy   sha256 앞 32: defca2414e39183db46b9fc004b65576
P3/subs/submission_p3_t1f.csv    sha256 앞 32: 2e4984283b047950e5d2da219ef6d8b3
```

## 기준선

| | 값 | 출처 |
|---|---|---|
| T1F 씨닝 fold4 | `0.754739` | `P3/cache/p3_t2/report.json` |
| T1F 씨닝 뷰A(216) | `0.761126` | `P3/cache/p3_t6/report.json` |
| T1F 가중 (persistence/damped/lgbm_t1) | `0.00 / 0.35 / 0.65` | `t1f_report.json` |

**여기 지정된 것 외에는 아무것도 바꾸지 마라.** 모델·피처·하이퍼파라미터·fold 경계 변경 금지.

## 작업 1 — fold0 처리

`P3/src/train.py` 를 고친다. 가장 단순한 방법 하나만 쓴다:

- 학습행이 `< 50` 인 fold 는 **학습·예측을 건너뛰고 그 fold 의 OOF 를 `NaN`** 으로 둔다
- 건너뛴 fold 목록과 각 학습행 수를 출력한다
- OOF 채점·가중 선택은 `NaN` 이 아닌 행만 쓴다. `NaN` 을 0 이나 평균으로 채우지 마라
- `select_t1f.py` 도 같은 규칙을 따르는지 확인하고, 안 따르면 동일하게 고친다

fold 경계 자체를 바꾸지 마라. fold0 을 다른 fold 에 합치지 마라.

## 작업 2 — 재생성과 대조

```bash
python P3/src/train.py
python P3/src/select_t1f.py
```

두 스크립트가 정상 종료해야 한다. 실행 후:

- 새로 선택된 T1F 가중을 보고하라 (기존 `0.00 / 0.35 / 0.65` 와 비교)
- `P3/preds/test_final_p3_t1f.npy` 의 새 SHA-256 을 백업본과 비교하라
- 새 OOF 로 뷰A(씨닝 216) 와 씨닝 fold4 통합 RMSE 를 내라

**기존 `P3/subs/submission_p3_t1f.csv` 를 덮어쓰지 마라.** 변경이 필요하면 새 파일
`P3/subs/submission_p3_t1f_r2.csv` 로 만든다.

## 수락 기준 (결과 보기 전 고정)

- `train.py` 와 `select_t1f.py` 가 예외 없이 종료한다
- 건너뛴 fold 는 학습행 `< 50` 인 것뿐이다
- OOF 에 `NaN` 이 남은 fold 는 채점·선택에서 제외됐음을 숫자로 보인다
- `_r2` 를 만든 경우 `audit`·`preflight sub`·`check.py data`·`check.py pipeline` 이 모두 `OK`

아래 세 경우로 갈린다. 해당하는 것만 실행하라.

세 경우로 갈린다. 해당하는 것만 실행하라.

1. **가중이 같고 test 예측 해시도 같다** → 기존 제출 CSV 가 그대로 재현된다.
   새 CSV 를 만들지 말고 그 사실만 보고하라
2. **가중은 같은데 test 예측이 다르다** → `_r2` CSV 를 만들고 audit 한다.
   두 예측의 최대 절대차와 평균 절대차를 보고하라
3. **가중이 다르다** → `_r2` CSV 를 만들고 audit 한다. 새 가중과 기존 가중 각각으로
   씨닝 fold4·뷰A 점수를 **둘 다** 내라. 어느 쪽을 제출할지는 결정하지 마라 (Opus 판단)

`_r2` 를 만든 경우에만:

```bash
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f_r2.csv --pred P3/preds/test_final_p3_t1f_r2.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t1f_r2.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/check.py data
python P3/src/check.py pipeline
```

## 금지

- 백업 디렉터리 `P3/preds/_backup_pre_maskfix/` 수정·삭제 금지
- 기존 `P3/subs/submission_p3_t1f.csv` 덮어쓰기 금지
- fold 경계·모델·피처·하이퍼파라미터 변경 금지. 후보 추가 금지
- `NaN` OOF 를 채워넣지 마라. 대칭 마스크로 되돌리지 마라
- 어느 제출을 쓸지 **결정하지 마라.** 숫자만 내라
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
[작업1]
건너뛴 fold = ___ (fold 번호: 학습행 수)
수정 파일 = ___

[작업2]
train.py 정상 종료 = 예|아니오 / 벽시계 = ___ s
select_t1f.py 정상 종료 = 예|아니오
새 T1F 가중 (persistence/damped/lgbm_t1) = ___ / 기존 (0.00/0.35/0.65) 과 동일 = 예|아니오
test_final_p3_t1f.npy 새 SHA-256 = ___ / 백업과 동일 = 예|아니오
test 예측 최대 절대차 = ___ / 평균 절대차 = ___
새 OOF 씨닝 뷰A(216) = ___ (기존 0.761126)
새 OOF 씨닝 fold4 = ___ (기존 0.754739)
가중이 다른 경우 — 기존 가중으로 재채점: 뷰A ___ / fold4 ___

[판정]
해당 경우 = 1 | 2 | 3
_r2 생성 여부 = 예(경로·audit 판정·SHA-256·preflight sub 판정) | 아니오
게이트 = check.py data ___ / check.py pipeline ___ / context_static_check ___
정지 조건 발생: 없음 | <사유>
```
