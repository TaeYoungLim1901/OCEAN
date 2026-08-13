# T1 — P1 첫 제출

## 배경

train 776,706행, test 169,011행, 이상 약 4%, 기준선 F1 0.548255다. spike/noise/flatline/offset/drift는 유형별 통계가 다르며 station-layer-season 정규화와 시간 gap reset이 필요하다. `lightgbm`, `sklearn`, `scipy`는 이미 설치됐고 새 `pip install`은 금지한다.

## 구현

Python으로 EDA 후 유형별 시계열 feature, LightGBM 시간-block OOF, 규칙/모델/앙상블 threshold와 event 후처리를 비교한다. `train.py`/`inference.py`를 분리하고 OOF/test `.npy`, 모델, 제출 CSV를 P1 안에 저장한다. `SEED=42`, `n_jobs<=2`, 원본 읽기 전용.

## 수락 기준

- `python P1/src/check.py data` → `판정: OK`
- `python P1/src/train.py` 완료, OOF와 모델 저장
- `python P1/src/inference.py` 완료, test 예측과 제출 저장
- `python P1/src/check.py pipeline P1/subs/exp001.csv P1/preds/oof_lgbm_exp001.npy P1/preds/test_lgbm_exp001.npy` → `판정: OK`
- `python P1/src/check.py sub P1/subs/exp001.csv` → `판정: OK`

## 완료 보고

전체/유형/station-layer F1 상위 3개, 기준선 0.548255 대비 delta, 잔차 상관, 제출 경로, 실행 명령, 변경 파일을 숫자와 표만 보고한다. 결론 문장 금지.

