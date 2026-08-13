# E003 — multi-holdout foundation 및 profile ensemble

## 목적

E002의 두 holdout만으로 생기는 선택 편향을 줄이고, 실제 test 공개층 availability와 수심 이동을 수치화한 뒤 검증된 최소 후보만 결합한다. 원본 `P2_data/`는 읽기 전용이고 P2 밖은 수정하지 않는다.

## 배경과 기준선

이미 검증된 E002 기준선은 2024 holdout RMSE 0.517441, 2025 pseudo-gap 1.301486, 두 holdout 평균 0.909464다. 이 사실은 재조사하지 않고 동일 생산 경로의 대조군으로만 재계산한다.

## 구현

- 기존 E002의 2024-09~10, 2025-07-02~08-31 두 연속 gap을 고정한다.
- 같은 2개월 길이 rolling pseudo-gap을 2024-07-02~08-31, 2025-05-02~07-01에 추가하며, 해당 연도 공개층 availability mask를 test 상대시각에서 이식한다.
- test와 각 holdout의 공개층 결측률·actual depth·성층 proxy 분포와 adversarial AUC를 출력한다.
- E002 동형 층별 LightGBM, lag/rolling 강화 층별 LightGBM, target 3층 잔차 PCA(2성분)+LightGBM을 동일 holdout에서 비교한다.
- 전체·층별·공개층 결측·성층 3구간·급변 상위구간 RMSE를 출력한다. OOF/test prediction을 후보별 `.npy`로 보존하고 잔차 상관 및 0.25 간격 blend를 평가한다.
- actual target depth 사용은 허용하되 no-depth sensitivity를 보고한다. M2/S2는 `elapsed_hours()` 경로만 사용한다.
- `SEED=42`, `n_jobs<=2`, 새 의존성 없음. `pip install`이 필요하면 즉시 정지한다. 학습과 추론을 분리한다.

## 결과 전 판정 규칙

- 네 holdout의 행 가중이 아닌 **holdout RMSE 단순평균**으로 판정한다.
- E003 후보/고정 blend는 E002 동형 대비 평균 RMSE가 0.010 이상 낮고, 각 holdout 전체가 E002보다 0.010 초과 악화되지 않아야 한다.
- 표본 200행 이상인 층·결측·성층·급변 segment 어느 것도 E002보다 0.030 초과 악화되면 탈락한다.
- 통과 후보가 여러 개면 평균 RMSE 최저를 선택한다. 0.002 이내 동률이면 단일 enhanced, 전역 blend, regime gate, EOF 순으로 단순한 것을 우선한다.
- 규칙을 통과하지 못하면 E002를 유지하고 실패 축을 graph에 닫는다. 결과와 무관하게 `P2/preds/oof_*_E003.npy`, `P2/preds/test_*_E003.npy`를 보존한다.

## 산출물

- `P2/src/train_e003.py`, `P2/src/inference_e003.py`
- `P2/cache/E003/metrics.json`, `P2/cache/E003/drift.json`, `P2/models/E003.pkl`
- `P2/preds/oof_*_E003.npy`, `P2/preds/test_*_E003.npy`
- 통과 시에만 `P2/subs/E003_final.csv`; 아니면 E002 유지

## 수락 기준과 명령

```powershell
python P2/src/common.py
python P2/src/check.py data
python P2/src/check.py pipeline
python P2/src/check.py cases
python P2/src/train_e003.py
python P2/src/inference_e003.py
python P2/src/submission_tools.py audit P2/subs/E003_final.csv --pred P2/preds/test_E003.npy
```

보고: holdout·층·regime delta, adversarial AUC, 후보 잔차 상관, 선택/유지 이유, 최종 경로·SHA256. 결론 문장 금지.
