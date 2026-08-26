# P1 유사 대회 고득점 해법 조사

기준일: 2026-08-13. 공식 대회 페이지·주최자 논문과 실제 입상자 원문만 근거로 사용했다. 아래의 외부 성적은 P1 실험 결과가 아니다.

## 비교표

| 대회 | 지표·데이터 유사성 | 검증된 순위·점수 | 실제 해법 | P1 전이 |
|---|---|---|---|---|
| [HAICon 2020](https://dacon.io/en/competitions/official/235624/overview/description) | 연속 산업 센서의 희귀 이상 구간. P1의 행 F1과 달리 이벤트를 반영한 eTaPR | 1위 private 0.9379, 재현 0.9378; baseline private 0.8416·156위. [주최자 CSET 논문 표 B1](https://doi.org/10.1145/3474718.3474719) | 정상 시계열 예측→실측과의 잔차→threshold가 공통 골격. 주최자 분석상 상위 7팀은 baseline 계열을 개선했고, 모델 자체보다 전처리·잔차 후처리와 여러 시간창 ensemble의 영향이 컸다. 1위 ORIDORI의 후속 공개 비교에는 EWM, low-pass, 양방향 stacked LSTM+attention, 시점별 전체 센서 평균 예측오차 threshold가 명시된다. [ORIDORI 비교 논문](https://public.thinkonweb.com/journals/jkiisc/digital-library/manuscript/file/25879/32-04%2008%20%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%89%E1%85%A1%E1%86%BC%E1%84%89%E1%85%AE.pdf) | **예.** P1은 정상 예측보다 합성 유형 라벨이 더 강한 감독신호를 제공하므로, 다중창 robust residual·유형별 점수·threshold/짧은 hole 후처리를 우선한다. eTaPR용 넓은 구간 확장은 행 F1에는 FP를 늘리므로 그대로 전이하지 않는다. |
| [University of Liverpool — Ion Switching](https://www.kaggle.com/competitions/liverpool-ion-switching) | 합성 잡음이 섞인 1차원 신호를 매 시점 다중분류, macro-F1. P1도 합성 센서 이상을 행 단위 판정 | 공개된 [5위 해법 원문](https://www.kaggle.com/competitions/liverpool-ion-switching/discussion/154220); 정확한 private 점수는 원문에서 확인되지 않아 기재하지 않음 | WaveNet 확률에 signal·잔차를 넣은 XGBoost 보정, lead/lag, sliding-window 예측, 시간 반전, LGBM/MLP stacking. XGB는 public은 올랐지만 private 차이는 거의 없었다고 보고했다. [57위 단일모델 원문](https://www.kaggle.com/competitions/liverpool-ion-switching/discussion/153813)은 cleaned signal+GRU/WaveNet+k-means+focal loss, public 0.943을 재현 가능한 보조 근거로 제공한다. | **부분 전이.** 유형 multiclass→binary와 raw/residual GBDT, 서로 다른 시간척도 ensemble은 직접 전이한다. 시간 반전·대형 신경망·stacking은 새 의존성과 과적합 비용 때문에 이번 축에서 제외한다. 이 대회는 합성 생성 누수로 우승권이 왜곡된 사례이므로 test label 역공학은 전이 금지다. |
| [VSB Power Line Fault Detection](https://www.kaggle.com/competitions/vsb-power-line-fault-detection) | 실제 전력선의 희귀 부분방전/고주파 spike를 신호별 이진 판정. 지표는 MCC라 P1 행 F1과 다름 | [1위 공식 노트북](https://www.kaggle.com/code/mark4h/vsb-1st-place-solution), private MCC 0.719는 독립 논문 비교표로 확인. [9위 실제 해법](https://www.kaggle.com/competitions/vsb-power-line-fault-detection/writeups/ngyope-9th-place-solution-overview)은 private 0.688 | 1위는 신호 처리 특징+LightGBM+수동 threshold. 9위는 raw adversarial AUC >0.96을 확인하고 high-pass+DWT 후 <0.78, drift 기여 feature 제거 후 <0.70으로 낮춘 뒤 다양한 RNN 8개 hard vote를 사용했다. | **진단만 전이.** train/test adversarial validation과 drift feature 중요도, 다양성/상관 확인은 채택한다. DWT·고주파 필터는 P1의 느린 ocean signal과 noise 주입을 같이 지울 수 있어 실험하지 않는다. |

## P1에 고정한 전략

1. 평가 관계를 맞추기 위해 stream별 시간 순서를 지킨다. fold1·fold2로 후보를 고르되 각 검증시점보다 과거인 fold만 학습하고 87시간을 purge한다. fold3는 파라미터를 잠근 뒤 한 번만 보는 outer다.
2. HAICon/Liverpool 근거에 따라 기존 binary LightGBM과 별도로 anomaly type multiclass LightGBM을 만들고, `1-P(normal)`을 binary score로 사용한다. 각 모델의 OOF/test를 모두 보존하고 inner에서만 threshold·아주 짧은 hole fill·보수적 blend를 고른다.
3. VSB 근거에 따라 train/test source classifier를 시간 전반→후반으로 검증해 AUC와 중요 feature를 기록한다. 이는 drift 진단이며 target model 선택 점수로 쓰지 않는다.
4. 최종 승격은 잠긴 fold3 outer F1이 exp002의 0.766423을 이기거나, inner에서 개선되고 exp002와 오류 상관이 낮은 보수적 blend가 outer에서 악화되지 않을 때만 허용한다.

## 전이하지 않는 축

- 랜덤 행 CV: P1 연속 주입 이벤트와 centered window가 분할 양쪽에 걸려 누수다.
- pseudo-label/test label 역공학: 규정·검증 관계 근거가 없고 Liverpool 생성 누수 사례를 반복한다.
- 이벤트 전체 팽창: HAICon eTaPR에는 맞지만 P1 행 F1에서는 경계 FP가 직접 감점된다.
- 대형 RNN/WaveNet/DWT: 현재 설치 자산 밖이며, 같은 목적을 기존 다중창 특징+LightGBM으로 더 직접 검증할 수 있다.
