# P2 유사 대회·해법 조사 (2026-08-13)

외부 근거와 P2 관찰을 분리한다. 아래 순위·점수는 링크된 공식 대회 페이지, 수상 논문 또는 실제 해법 저장소에서만 가져왔다. P2의 RMSE와 타 대회 점수는 척도가 달라 절대값을 비교하지 않는다.

| 사례 | 과제·평가 유사성 | 검증된 성적 | 실제 전략 | P2 전이 판단 |
|---|---|---:|---|---|
| [KDD Cup 2018 공식 결과](https://www.kdd.org/kdd2018/kdd-cup) · [AccuAir 공식 수상 논문 요약](https://www.kdd.org/kdd2019/accepted-papers/view/accuair-winning-solution-to-air-quality-prediction-for-kdd-cup-2018) | 관측소별 대기질을 기상·인접 관측소의 시공간 문맥으로 복원/예측. P2처럼 환경 센서, 급변 regime, 시계열 상관이 핵심이나 미래 48시간 예측이라는 차이가 있다. | AccuAir: 3개 트랙 중 2개 1위, 1개 2위 | LightGBM·Gated-DNN·Seq2Seq를 결합하고, 급변과 관측소 간 시공간 상관에 별도 gate를 둠 | **채택:** LightGBM 잔차 + 성층/급변 regime 진단 + 서로 다른 오차의 앙상블. **불채택:** P2 표본 규모와 1개 station에 비해 무거운 DNN/Seq2Seq. |
| [Ocean Data Challenge 2021 공식 저장소·leaderboard](https://github.com/ocean-data-challenges/2021a_SSH_mapping_OSE) | 부분 위성고도계 관측으로 연속 SSH 지도를 재구성하고 독립 Cryosat-2를 숨겨 평가. P2의 “공개 관측으로 가린 환경장을 재구성” 구조와 가장 직접적이다. | 기준 OI 0.85, DUACS 0.88, 최고 공개 방법군 0.90(4DVarQG·ConvLSTM·NeurOST; 저장소 표의 skill score) | OI, 동역학 regularization, 4DVar, ConvLSTM 등 서로 다른 재구성기를 독립 관측으로 비교; RMSE뿐 아니라 공간/주파수 해상도도 진단 | **채택:** 관측 mask를 실제와 같게 이식한 pseudo-gap, 독립 기간·센서 검증, 서로 다른 재구성기 OOF 보존. **불채택:** P2는 1차원 수심 프로파일이라 4DVar/ConvLSTM 공간망은 과함. |
| [Kaggle LANL 26위 실제 해법 저장소](https://github.com/filipmu/Kaggle-LANL-Earthquake-Prediction) | 긴 센서 시계열에서 test와 같은 길이의 창을 만들고 MAE로 평가. P2와 타깃은 다르지만 연속 시계열의 split leakage와 문맥 피처 문제가 같다. | 26/4,540; 저장소에는 최종 점수 미기재 | test와 같은 150k 창, 실험 단위 group 분할, LightGBM, STFT/Hilbert/matched-filter 등 451 피처, grouped/non-grouped 제출을 헤지 | **채택:** 연속 gap 길이·mask 복제, 시간/기간 group 단위 검증, lag/rolling 피처. **불채택:** 수백 신호 피처와 대규모 모델 sweep은 P2 물리 구조에 비해 과적합 위험. |
| [Kaggle ASHRAE 상위 5개 해법 재현·비교](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis) | 다수 환경 센서와 날씨의 결측·이상치·시간 이동을 다루는 회귀. 공간 수심축은 없지만 센서별 분포와 결측 regime이 비슷하다. | 1위 1.231, 2위 1.232, 3위 1.234, 4위 1.235, 5위 1.237(private score) | 1위는 weather 결측 보간 후 LightGBM/CatBoost/MLP의 가중 generalized mean; 2위는 meter별 다른 blend; 4위는 weather lag; 5위는 building별 LightGBM | **채택:** 결측/층별 세그먼트, lag 피처, OOF 상관을 본 소규모 robust blend. **불채택:** unavailable CatBoost/MLP와 site별 대형 ensemble. |
| [부분 수심 온도에서 전 수심 복원 연구](https://www.mdpi.com/2072-4292/14/23/5984) · [수온·염분 부분 프로파일 conditional MFPCA](https://arxiv.org/abs/2608.05376) | 소수 고정 수심/잘린 수온·염분 프로파일에서 전체 수직장을 복원한다. 대회는 아니지만 P2의 수심축과 직접 일치한다. | 후자는 지리 공변량 포함 시 모의 truncated profile 온도 30%, 염분 33% 정확도 개선(논문 보고) | 완전 프로파일에서 EOF/PCA basis를 학습하고 관측된 수심으로 coefficient를 조건부 추정해 미관측 깊이를 복원 | **조건부 채택:** target 3층 잔차의 저차원 PCA/EOF 후보를 만들되 P2 다중 holdout에서만 채택. **위험:** 3개 target만으로는 저랭크 가정이 과도하고 성층 전이에서 무너질 수 있음. |

## P2에 적용할 최소 원칙

1. 실제 test의 2개월 연속 target mask와 공개층 availability를 복제한 2024/2025 holdout을 고정한다.
2. 층별 LightGBM 수심보간 잔차를 기준으로 두고, lag/rolling 강화 모델과 저랭크 target-profile 잔차 모델만 비교한다.
3. 전체 RMSE 외 층·공개층 결측·성층·급변 regime의 최악 악화를 사전 차단한다.
4. 모든 후보의 OOF/test를 남기고 잔차 상관과 고정 weight grid로만 blend한다.
5. 외부 대회의 순위나 복잡도가 아니라 P2의 고정 다중 holdout 개선만 전이 근거로 인정한다.

## P2 전이 결과

E003에서 lag/rolling 강화, 2성분 EOF, robust blend는 4-hold 평균을 E002보다 각각 0.068386, 0.059417, 0.066248 낮췄다. 그러나 실제 test 결측 mask를 이식한 2025-07~08 `missing_public` RMSE가 각각 0.066278, 0.057425, 0.050881 악화해 사전 차단선을 넘었다. 따라서 외부 해법에서 **다중 pseudo-gap·독립 segment 검증·OOF 보존**만 foundation으로 채택하고, EOF/lag/regime blend 제출 전이는 기각했다. 이는 실험 관찰이며 위 표의 외부 성적과 별개다.
