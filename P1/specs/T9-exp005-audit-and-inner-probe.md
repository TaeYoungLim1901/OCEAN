# T9 — exp005 사후 감사 + inner 전용 격자 경계 탐침

## 목적

두 가지를 한 번에 한다.

1. `P1/subs/exp005.csv` 산출 경로의 결함을 숫자로 확정한다.
2. inner(fold1·2) **만** 써서 격자 경계 밖을 탐침한다. 어디까지 넓히면 최적점이
   경계에서 떨어지는지 본다.

**봉인 fold3 를 읽지 마라. fold3 행이 어떤 F1·집계에도 들어가면 안 된다.**
모델을 새로 학습하지 마라. 기존 파일을 수정하지 마라.
새 파일은 `P1/src/audit_t9.py` 와 `P1/cache/T9_audit.json` 두 개만 만든다.

## 배경 — 이미 검증된 사실 (재조사 금지)

- exp005 = 점수 경로 통일(과거만 학습 단일 모델) + blend `weight 0.00` / `threshold 0.62` /
  `hole 48` / `λ 0.00`. inner F1 `0.697836`, 봉인 outer F1 `0.815757`
  (TP `5612` / FP `704` / FN `1831`). test 양성률 `0.034791`. `n_estimators = 700`.
- exp004 = `weight 0.40` / `threshold 0.54` / `hole 24`. 봉인 outer F1 `0.808765`.
  exp005 의 봉인 배열 위에서 이 설정이 `0.808765` 를 정확히 재현했다. **재확인 금지.**
- 1단계 4개 모델의 미래 행 수는 전부 `0` 으로 이미 보고됐다. 학습 행 수는
  `185842` / `380044` / `574188` / `776706`. **이 숫자는 재계산해 대조만 하고 새로 논하지 마라.**
- T7 감사에서 확정: 라벨 누수 0, 피처 42개, hole 메움의 stream·시간 gap 위반 0행,
  키 정렬 불일치 0행, 원본 파일 해시 불변. **이 항목들은 다시 검사하지 마라.**
- purge 는 학습 마스크에서 걸린다. fold block 이 10분 간격으로 인접한 것은 정상이다.
- train 양성률: 전체 `0.041362`, 2024 `0.04107`, 2025 `0.04153`, stream-year 23개
  min `0.0399` max `0.0546` std `0.00296`. **재측정 금지.**
- T4 에서 rolling mean·rolling max·hysteresis·이벤트 확장은 inner 에서 전부 hole 단독보다
  나빴다. exp003 type-multiclass 축, 랜덤 행 분할 축, adversarial drift 축은 닫혔다. **재시도 금지.**
- 새 의존성 설치 금지. `pip install` 이 필요해지면 **즉시 멈추고 그 사실만 보고한다.**
  현재 환경에 `xgboost`, `catboost`, `torch` 는 없다.

## 기준선 (baseline)

| 지표 | 값 |
|---|---|
| exp005 inner F1 | `0.697836` |
| exp005 선택 조합 | `weight 0.00` / `threshold 0.62` / `hole 48` / `r 0.030` / `λ 0.00` |
| exp005 봉인 outer F1 | `0.815757` |
| exp005 제출 sha256 | `5E5B8840C3A6FC5E6C8A8DFD461E419E9795548492C566E27758EE5BD0185F5A` |

## 입력

- `P1/src/exp005.py`, `P1/src/exp004.py`, `P1/src/features.py`, `P1/src/train.py`
- `P1/cache/exp005_raw_oof.npy`, `P1/cache/exp005_raw_test.npy`, `P1/cache/exp005_meta.json`
- `P1/preds/oof_exp005.npy`, `P1/preds/test_exp005.npy`, `P1/subs/exp005.csv`
- `P1/P1_data/train.csv`, `P1/P1_data/test.csv`, `P1/P1_data/sample_submission.csv` (읽기 전용)

## A부 — 감사 항목. 각각 `PASS` / `FAIL` + 숫자

**A1. rule_score 무결성.** exp005 가 쓴 `rule_score` 열의 inner·test 구간별
min·max·평균·표준편차·고유값 개수·NaN 개수를 적는다. 표준편차가 `0` 이거나 고유값이 2개
이하이면 `FAIL`. inner 구간에서 `rule_score` 단독(`weight=1.0`)의 최적 threshold 와 그때의
inner F1 을 적는다. 그 F1 이 `0.10` 미만이면 `FAIL`.

**A2. weight=0 선택의 근거.** inner 에서 `threshold`·`hole` 을 각 weight 별로 재최적화했을 때의
weight 별 최고 inner F1 을 표로 낸다 (`weight` = `0.00` ~ `0.60`, `0.05` 간격).
weight 0 이 최고인지, 곡선이 단조인지 숫자로 보인다.

**A3. lgbm 점수와 rule_score 의 상관.** inner 구간에서 두 점수의 Spearman 상관, 그리고
각각을 이진화(각자 최적 threshold)했을 때 예측의 잔차 상관을 적는다.

**A4. 재현성.** `P1/src/exp005.py` 를 재학습 없이 다시 실행 가능하면 실행해
`P1/subs/exp005.csv` 의 sha256 이 `5E5B8840…` 와 같은지 본다. 재학습을 유발하면
실행하지 말고 `SKIP` 과 이유 1줄만 적는다.

**A5. 정렬·도메인.** `P1/preds/test_exp005.npy` 의 i번째 값과 `P1/subs/exp005.csv` 의 i번째 행이
같은 `(station, year, layer, time)` 인지 확인해 불일치 행 수를 적는다.
`sample_submission.csv` 와의 키 순서 불일치 행 수, 두 npy 의 dtype·고유값 집합,
`label` 열 고유값 집합을 적는다. `{0,1}` 이 아니면 `FAIL`.

**A6. OOF 배열 정의역.** `P1/cache/exp005_raw_oof.npy` 에서 `NaN` 인 행 수와 그 행들이
정확히 fold0 구간인지 확인한다. fold1·2·3 구간에 `NaN` 이 하나라도 있으면 `FAIL`.
`P1/preds/oof_exp005.npy` 의 fold0 구간이 전부 `0` 인지 확인한다.

**A7. climatology 누수.** exp005 가 각 모델의 climatology 를 그 모델의 학습 집합만으로
계산했는지 코드 경로로 확인한다. 검증 구간·test 행이 climatology 계산에 들어가면 `FAIL`.
확인한 함수·행 번호를 적는다.

**A8. hole=48 의 실제 효과.** test 와 inner 각각에서 hole 메움 **전/후** 양성 행 수와 증가분,
새로 켜진 구간 길이의 중앙값·최대를 적는다.

**A9. 원본 불변.** `P1/P1_data/` 전 파일과 `P1/preds/oof_exp002.npy`·`test_exp002.npy`·
`oof_exp004.npy`·`test_exp004.npy` 의 sha256 을 감사 전후로 대조한다.

## B부 — inner 전용 격자 경계 탐침 (봉인 fold3 사용 금지)

`weight` 는 A2 의 최고 weight 로 고정한다. inner(fold1·2) 에서만 계산한다.

**B1. hole 확장.** `threshold` `0.40` ~ `0.80` `0.02` 간격 × `hole`
`24, 32, 48, 64, 80, 96, 128, 160, 200, 256, 320, 400` 의 inner F1 을 낸다.
`hole` 별 최고 inner F1 을 표로 내고, 최고점이 어느 hole 에서 나오는지 적는다.

**B2. threshold 확장.** B1 의 최적 hole 로 고정하고 `threshold` `0.30` ~ `0.96` `0.02` 간격의
inner F1 곡선에서 상위 10개를 적는다. 최고점이 `0.96` 이면 그 사실을 적는다.

**B3. 앵커링 재시도.** B1·B2 의 최적 `(threshold, hole)` 로 고정하고
`target_rate r` = `0.030, 0.034, 0.038, 0.041, 0.045, 0.050, 0.055, 0.060` ×
`λ` = `0.25, 0.50, 0.75, 1.00` 의 inner F1 을 표로 낸다 (λ=0 은 B2 와 같으므로 생략).
stream 별 행 수 500 미만이면 앵커링하지 않는다.

**B4. inner 최적점 요약.** A·B 전체에서 나온 inner 최고 조합 1개와 그 inner F1,
그리고 exp005 선택 조합(`0.00 / 0.62 / 48 / λ 0`)의 inner F1 `0.697836` 대비 Δ 를 적는다.
**이 조합을 fold3 에 적용하지 마라. outer 값을 계산하지 마라.**

## 수락 기준

1. A1~A9 가 전부 `PASS` / `FAIL` / `SKIP` 으로 판정되고 근거 숫자가 채워져 있다.
2. B1~B4 표가 전부 채워져 있다.
3. 스크립트 어디에서도 `folds == 3` 행이 F1·집계에 들어가지 않는다. 이를 코드로 보증하고
   inner 평가에 쓰인 행 수를 숫자로 적는다.
4. `P1/cache/T9_audit.json` 이 생성되어 위 판정과 숫자를 전부 담는다.
5. `python P1/src/audit_t9.py` 를 **직접 실행하고** 실제 출력 숫자를 보고한다.
6. `P1/P1_data/`, `P1/preds/*.npy`, `P1/subs/*.csv`, 기존 `P1/src/*.py` 가 변하지 않았다 (A9 해시).

## 제약

- `P1/P1_data/` 읽기 전용. 학습·재학습·모델 파일 생성 **전부 금지**.
- **봉인 fold3 판독 금지.** outer F1 을 계산하지 마라.
- 시간 gap 을 넘는 rolling·diff·이벤트 연결 금지. hole 메움은 `station × layer` 안에서만.
- `SEED = 42`, `n_jobs <= 2`, 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 새 코드는 `P1/src/audit_t9.py` 한 파일. 기존 `P1/src/*.py` 는 import 만 하고 수정하지 마라.
- 버그를 발견해도 **고치지 마라.** 발견 사실과 숫자만 보고한다.
- 총 실행이 20분을 넘으면 중단하고 경과 시간과 완료한 항목만 보고한다.

## 보고

**결론 문장 금지. 해석·제안·요약 산문 금지. 숫자와 표만.**

1. A1~A9 판정 표: `항목 | 판정 | 핵심 숫자`
2. A1 rule_score 통계 표
3. A2 weight 별 최고 inner F1 표 (13행)
4. A8 hole 효과 표
5. B1 hole 별 최고 inner F1 표 (12행)
6. B2 threshold 상위 10 표
7. B3 `r × λ` 표
8. B4 1줄: 최적 조합 + inner F1 + Δ
9. inner 평가 행 수 1줄, fold3 사용 0 보증 1줄
