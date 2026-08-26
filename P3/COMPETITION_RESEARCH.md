# P3 유사대회 상위권 해법 조사

조사일: 2026-08-13. 공식 대회 페이지, 주최자 결과 논문·저장소, 실제 참가자 해법만 사용했다.
P3와 동일한 공개 유의파고 경진대회의 검증 가능한 상위권 해법은 찾지 못했다. 따라서 아래는
`다정점/그룹 × 다중 horizon × 미래 기간 이동` 구조가 직접 겹치는 순서로 정리한다.

| 대회·해법 | P3 유사성 | 검증 순위·점수 | 상위권 전략 | P3 전이 |
|---|---|---|---|---|
| GEFCom2014 Wind, kPower | 10개 풍력단지, 매일 24h rolling forecast, 풍속·방향과 인접 lead 사용. 타깃은 파고가 아닌 풍력이고 확률예측이라는 차이 | 공식 결과표 Wind **1위, 56.6%**(naive 대비 가중 개선율) | 2-layer GBM, 정점과 인접 lead의 풍속·방향, CV | **예**: 원형 방향, lead별/direct GBDT, 정점 상호작용, forward CV. 외부 정점/미래 NWP는 P3 `r.context_only` 때문에 **불가** | 
| M5 Forecasting Accuracy, YJ STU | 42,840개 grouped series의 미래 28일 예측, public 이후 더 미래 private 기간. 물리 도메인은 다름 | 공식 결과논문 **1위, WRMSSE 0.520**, ES benchmark 대비 22.4% 개선 | 서로 다른 train grouping과 recursive/non-recursive LightGBM 6개 단순 평균; 최근 4개 28일 창에서 평균뿐 아니라 변동성도 검증 | **예**: station/pooled·direct 모델 다양화, 여러 forward 창, 잔차가 다른 OOF만 혼합. 달력·가격·타계열 통계는 P3에 **불가** |
| M5 Forecasting Accuracy, monsaraida | 위와 동일 | 공식 결과논문 **4위, WRMSSE 0.536** | store별 LightGBM, forecast week별 non-recursive 모델(총 40개), 최근 5개 28일 창 CV | **예**: station expert와 lead-specific direct 모델 비교. horizon별 모델 수 증가는 sealed outer에서만 채택 |
| Kaggle Hill of Towie Wind Turbine Power Prediction, Ravi Ramakrishnan | 풍력 관측 정형 회귀와 기간 일반화. 다중 horizon·파고는 아님 | Kaggle 실제 해법 글이 public/private **모두 4위**를 명시; 정확한 최종 점수는 글에 미공개 | 5-fold OOF의 AutoGluon/FLAML/XGBoost/LightGBM/CatBoost를 Ridge blend | **부분 예**: 깨끗한 OOF와 이질 모델 혼합. 무작위 5-fold, AutoML, 미설치 모델은 전이하지 않음 |
| GEFCom2012 Wind, Leustagos | 7개 풍력단지의 시간당 48h point forecast로 P3 horizon 구조와 매우 가까움 | 저자 논문이 public/private **모두 1위**를 확인; 접근 가능한 초록에는 수치 점수 미기재 | 시간·기상 feature의 GBDT와 선형회귀 결합 | **부분 예**: GBDT+단순 shrinkage의 잔차 상보성. 절대시간 feature는 P3에서 **불가** |

## 근거

- [GEFCom2014 공식 결과 논문](https://robjhyndman.com/papers/gefcom2014.pdf): 문제 구조·평가와 Table 3 순위/점수, Table 9 상위 5팀 방법.
- [M5 공식 결과 논문](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf): Table 3 순위/WRMSSE와 상위 5팀 방법·rolling CV.
- [M5 주최자 저장소](https://github.com/Mcompetitions/M5-methods): top-50 점수, 우승 코드, 제출물의 공식 보존소.
- [Kaggle M5 공식 대회](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview/abstract): grouped series, 28일 미래, public/private 기간 정의.
- [Kaggle Hill of Towie 4위 해법](https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction/writeups/rank4-approach-automl-pipelines-and-single-ml-mo): 참가자 본인의 public/private 4위와 OOF Ridge blend 설명.
- [GEFCom2012 1위 해법 논문](https://www.sciencedirect.com/science/article/pii/S0169207013000836): 7개 단지 48h, GBDT+선형회귀, public/private 1위.
- [BARAM 2026 DACON 공식 페이지](https://dacon.io/competitions/official/236727/overview/description): 2026-08-13 현재 진행 중이라 최종 상위권 해법이 없어 전략 근거에서는 제외했다.

## P3 적용 원칙

1. 최근 미래 창을 완전히 봉인한 strict-forward nested CV가 먼저다.
2. pooled, lead-specific, station expert의 inductive bias 차이를 OOF 잔차 상관과 segment tail로 비교한다.
3. 파고 level/trend와 여러 window의 풍속 memory, 방향의 sin/cos·풍향-파향 정렬만 case 내부에서 만든다.
4. 단순 평균/비음수 grid blend도 selection fold에서만 고르고 latest outer에는 한 번만 적용한다.
5. 다른 test case 통계, 절대시각, future NWP는 출처에서 유효했더라도 P3 하드 규정 때문에 사용하지 않는다.
