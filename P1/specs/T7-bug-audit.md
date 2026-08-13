# T7 — exp004 산출물 버그 감사 (읽기 전용)

## 목적

`P1/subs/exp004.csv` 를 최종 제출 후보로 확정하기 전에, 그 산출 경로 전체에서
**누수·정렬 오류·재현 불가·봉인 위반**이 있는지 숫자로 확정한다.

**읽기 전용 감사다. 모델을 학습하지 마라. 기존 파일을 수정하지 마라.**
새 파일은 `P1/src/audit_t7.py` 와 `P1/cache/T7_audit.json` 두 개만 만든다.

## 배경 — 이미 검증된 사실 (재조사 금지)

- exp004 = raw 점수 blend(`weight 0.40`) + `threshold 0.54` + `hole 24`.
  inner(fold1·2) F1 `0.762702`, 봉인 outer(fold3) F1 `0.808765`, TP `5352` / FP `440` / FN `2091`.
- exp002 = `weight 0.15` / `threshold 0.44` / `hole 2`, 봉인 outer F1 `0.766423`, TP `4935` / FP `500` / FN `2508`.
- `P1/preds/oof_exp002.npy` · `test_exp002.npy` 는 `exp002.py:48 store_score()` 가 값을 왜곡한
  배열이다. threshold 가 이미 구워져 있다. **이 배열로 threshold 를 재탐색하지 마라.**
- exp003 type-multiclass 축, 랜덤 행 분할 축, adversarial drift 축은 **닫혔다.** 다시 열지 마라.
- train/test drift source AUC `0.850133` 이미 측정됨. 다시 재지 마라.
- train 양성률은 stream-year 23개에서 min `0.0399` max `0.0546` std `0.00296` 로 이미 측정됨. 재측정 금지.
- 새 의존성 설치 금지. `pip install` 이 필요해지면 **즉시 멈추고 그 사실만 보고한다.**
  현재 환경에 `xgboost`, `catboost`, `torch` 는 없다.

## 기준선 (baseline)

| 지표 | 값 |
|---|---|
| exp004 봉인 outer F1 | `0.808765` |
| exp004 outer TP / FP / FN | `5352` / `440` / `2091` |
| exp004 inner F1 | `0.762702` |
| exp002 봉인 outer F1 | `0.766423` |
| `P1/subs/exp004.csv` sha256 | `9F47C487EF6128B4E53D7BE17F45AC62B7EDE7DC21DA8CB80A6AB6902C0A1A0C` |
| 제출 양성 비율 | `0.032039` (5415 / 169011) |

## 입력

- `P1/src/exp004.py`, `P1/src/exp002.py`, `P1/src/train.py`, `P1/src/inference.py`, `P1/src/features.py`
- `P1/cache/exp004/raw_oof.npy`, `P1/cache/exp004/raw_test.npy`, `P1/cache/exp004_meta.json`
- `P1/preds/oof_exp004.npy`, `P1/preds/test_exp004.npy`, `P1/subs/exp004.csv`
- `P1/P1_data/train.csv`, `P1/P1_data/test.csv`, `P1/P1_data/sample_submission.csv` (읽기 전용)

## 점검 항목 — 각각 `PASS` / `FAIL` + 숫자

**C1. 재현성.** `P1/src/exp004.py` 를 그대로 다시 실행했을 때 (재학습 없이 캐시 경로로)
`P1/subs/exp004.csv` 의 sha256 이 `9F47C487…` 와 동일한가. 다르면 두 sha 를 모두 적는다.
재실행이 재학습을 유발한다면 재실행하지 말고 `SKIP` 과 그 이유 1줄만 적는다.

**C2. 라벨 누수.** `features.py` 가 만드는 피처 열 이름 전체를 나열하고, `label`·`anomaly_type`
또는 그 파생값이 피처에 포함되는지 확인한다. 포함 열 수를 숫자로 적는다.

**C3. 시간 누수 — fold 경계.** fold1·2·3 각각의 시간 범위(min/max)와 인접 fold 사이
실제 간격(시간)을 표로 낸다. 간격이 87시간 미만인 경계가 있으면 `FAIL`.

**C4. 봉인 위반.** `exp004.py` 안에서 fold3 인덱스가 파라미터 선택(격자 탐색·argmax)에
쓰인 지점이 있는지 코드 경로로 확인한다. 선택 단계에서 fold3 행이 F1 계산에 들어가면 `FAIL`.
inner 격자 탐색에 쓰인 행 수와 fold1·2 행 수가 같은지 숫자로 대조한다.

**C5. 정렬.** `P1/preds/test_exp004.npy` 의 i번째 값과 `P1/subs/exp004.csv` 의 i번째 행이
같은 `(station, year, layer, time)` 을 가리키는지 확인한다. 불일치 행 수를 적는다.
`sample_submission.csv` 와의 키 순서 불일치 행 수도 적는다.

**C6. hole 메움 gap 안전성.** test 예측에 적용된 hole 메움이 (a) `station × layer` 경계를
넘었는지, (b) 10분 간격이 끊긴 시간 gap 을 넘었는지 확인한다.
경계를 넘어 메워진 행 수를 숫자로 적는다. 0 이 아니면 `FAIL`.
같은 점검을 OOF(train) 예측에도 한다.

**C7. hole=24 의 실제 효과.** test 에서 hole 메움 **전** 양성 행 수와 **후** 양성 행 수,
증가분을 적는다. train OOF 에서도 같은 숫자를 적는다.
메움으로 새로 켜진 구간의 길이 분포(중앙값·최대)를 적는다.

**C8. raw 점수 무결성.** `raw_oof.npy`·`raw_test.npy` 의 dtype·shape·min·max·NaN 개수·
정확히 `0.5` 인 값의 개수를 적는다. `0.5` 값이 전체의 1% 를 넘으면 `FAIL`
(`store_score` 왜곡이 섞여 들어온 신호다).

**C9. blend 식.** `exp004.py` 가 쓰는 blend 가 `(1-w)*lgbm + w*rule` 인지 코드로 확인하고,
`rule_score` 의 범위(min/max)를 적는다. `rule_score` 가 [0,1] 밖이면 `FAIL`.

**C10. test 모델 출처.** test 예측에 쓰인 booster 가 어느 fold 에서 학습된 것인지,
그 학습 구간이 test 기간(2026) 이후 데이터를 포함하는지 확인한다. 포함하면 `FAIL`.

**C11. 결측·상수 열.** test 피처 행렬에서 전부 NaN 인 열, 상수 열의 개수와 이름을 적는다.
train 에는 있으나 test 에 없는 `station × layer` stream 목록을 적는다.

**C12. dtype·값 도메인.** `oof_exp004.npy`·`test_exp004.npy` 의 dtype 과 고유값 집합,
`P1/subs/exp004.csv` 의 `label` 열 고유값 집합을 적는다. `{0,1}` 이 아니면 `FAIL`.

**C13. 원본 불변.** `P1/P1_data/` 의 모든 파일과 `P1/preds/oof_exp002.npy`·`test_exp002.npy` 의
sha256 을 적는다. 감사 실행 전후로 동일한지 확인한다.

## 수락 기준

1. C1~C13 이 전부 `PASS` / `FAIL` / `SKIP` 중 하나로 판정되고 근거 숫자가 채워져 있다.
2. `P1/cache/T7_audit.json` 이 생성되어 위 판정과 숫자를 전부 담는다.
3. `python P1/src/audit_t7.py` 를 **직접 실행하고** 실제 출력 숫자를 보고한다.
4. `P1/P1_data/`, `P1/preds/*.npy`, `P1/subs/*.csv`, 기존 `P1/src/*.py` 의 내용이 변하지 않았다
   (C13 의 sha256 으로 증명).

## 제약

- `P1/P1_data/` 읽기 전용. 학습·재학습·threshold 재선택·파라미터 변경 **전부 금지**.
- 시간 gap 을 넘는 rolling·diff·이벤트 연결 금지.
- `SEED = 42`, `n_jobs <= 2`, 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 새 코드는 `P1/src/audit_t7.py` 한 파일. 기존 `P1/src/*.py` 는 import 만 하고 수정하지 마라.
- 버그를 발견해도 **고치지 마라.** 발견 사실과 숫자만 보고한다.

## 보고

**결론 문장 금지. 해석·제안·요약 산문 금지. 숫자와 표만.**

1. C1~C13 판정 표: `항목 | 판정 | 핵심 숫자`
2. C3 fold 경계 표
3. C7 hole 효과 표
4. C8 raw 점수 통계 표
5. C11 결측·상수 열 목록
6. `FAIL` 이 있으면 그 항목의 재현 숫자만 3줄 이내로 추가
