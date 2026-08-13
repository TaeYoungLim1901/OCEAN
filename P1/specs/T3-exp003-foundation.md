# T3 — exp003 foundation + analog-supported specialist

## 배경

이미 검증된 사실: train 776,706행, test 169,011행, 이상 약 4%, 공식 baseline F1 0.548255, exp002 honest fold3 outer F1 0.766423이다. 재조사하지 않는다. `lightgbm`, `sklearn`, `scipy`는 설치됐으며 새 `pip install`이 필요하면 즉시 정지하고 사용자에게 묻는다.

## 목표와 사전 판정

HAICon·Liverpool 해법에서 반복된 다중 시간척도 잔차, 유형별 분류, threshold/후처리와 VSB의 adversarial drift 진단을 P1에만 적용한다. 결과와 무관하게 시도한 각 모델의 `P1/preds/oof_*_exp003.npy`, `P1/preds/test_*_exp003.npy`와 `P1/experiments/` 진단을 보존한다.

후보 선택은 fold1·fold2만 사용한다. 각 validation은 같은 stream의 더 이른 fold만 학습하고 87시간 purge하며, climatology도 해당 train subset만 쓴다. fold0는 warm-up이라 점수를 내지 않는다. binary LightGBM과 anomaly-type multiclass LightGBM의 score, 그리고 exp002와의 제한된 blend weight `{0,.25,.5,.75,1}` × threshold `.20:.02:.70` × hole `{0,1,2}`를 inner에서만 고정한다. fold3는 그 뒤 한 번만 평가하고 결과를 본 뒤 변경하지 않는다.

exp003 제출은 (a) sealed outer F1이 exp002 honest outer 0.766423보다 높거나, (b) inner F1이 개선되고 exp002 대비 오류 상관이 낮은 blend가 outer에서 0.002 이상 악화되지 않을 때만 만든다. 아니면 exp002를 유지하고 축을 기각한다.

## 구현·진단

- `SEED=42`, LightGBM `n_jobs=2`, 새 의존성 없음. raw data는 읽기 전용.
- train/test source classifier는 source별 전반부 train, 후반부 validation으로 AUC·중요 feature만 기록한다. target 학습/threshold에는 사용하지 않는다.
- fold3 전체/유형/station-layer 및 event hit/row recall을 저장한다.
- pool에는 exp002와 모든 exp003 score의 Pearson/Spearman, 이진 오류 상관과 blend F1을 저장한다.
- 저장 모델에서 OOF/test를 재생성하고 제출을 만들었다면 `submission_tools.py audit --pred`를 통과시킨다.

## 수락 기준

```bash
python scripts/preflight.py spec P1/specs/T3-exp003-foundation.md
python P1/src/check.py data
python P1/src/check.py pipeline
python P1/src/check.py cases
python P1/src/foundation.py
python P1/src/foundation_inference.py
python P1/src/foundation.py reproduce
```

## 완료 보고

forward inner 전체 F1, sealed outer F1·TP·FP·FN과 exp002 대비 delta, 유형/station-layer 하위 3개, event hit/row recall, adversarial AUC·중요 feature 5개, pool Pearson/Spearman·오류 상관·blend F1, 후보 경로·SHA256, 재현/audit 결과를 숫자와 표만 보고한다. 결론 문장 금지.
