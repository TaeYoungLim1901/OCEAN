# P3 도메인 조사 메모

기준일: 2026-08-13. 이 문서는 외부 근거와 실험 전 가설을 기록한다. 확정 규정·사실·폐쇄축은
`graph.json`이 기준이다.

## 문제 구조

- 각 익명 `case_id`의 과거 48시간(289행)만으로 +3/6/9/12/18/24시간 유의파고를 예측한다.
- 평가 표본은 기준시각 `hs >= 1.5 m`인 상위꼬리 조건부 표본이고 지표는 1,200행 통합 RMSE다.
- 추론 시 다른 사례, 절대시각, 미래 기상장을 사용할 수 없다(`r.context_only`, `r.no_abs_time`).
- 파랑은 20분, 기상은 10분 간격이다. 파랑 열의 중간 10분 결측은 구조적 결측이다.

## 도메인 근거

- 파랑은 국지 바람이 만드는 풍파와 원격 폭풍에서 전파된 너울의 합이다. 파랑모델은 바람 입력,
  비선형 상호작용, 백파, 저면마찰, 수심·해류에 따른 전파를 함께 다룬다
  ([NOAA WAVEWATCH III](https://polar.ncep.noaa.gov/waves/wavewatch/index.shtml),
  [ECMWF wave forecasting](https://www.ecmwf.int/en/about/media-centre/focus/2020/fact-sheet-ocean-wave-forecasting)).
- 과거 관측만으로 6~24시간 유의파고를 예측하는 LSTM 연구는 가능성을 보였지만, 지역·입력·검증이
  다르므로 이 문제에서의 우위 근거는 아니다
  ([Minuzzi & Farina](https://arxiv.org/abs/2201.00356),
  [reference code](https://github.com/felipeminuzzi/lstm-ocean)).
- G/I/S 기지는 위치·수심·취송거리와 태풍 노출이 다르므로 정점 효과를 분리해야 한다
  ([KIOST KORS 기지 정보](https://kors.kiost.ac.kr/demo/about/?tab=stations)).

## 우선 가설

1. **정점×리드별 감쇠 지속성**: `hs0`를 정점·리드별 평균으로 수축한다. 사례 선정이 상위꼬리라
   긴 리드에서 평균회귀가 강해질 가능성이 크며, 로컬 사실 `f.persistence_bias`도 이를 지지한다.
2. **상태·추세 직접예측**: 최근 1/3/6/12/24/48시간의 `hs`, `hs²`, EMA, 기울기, 범위와
   풍속·돌풍의 누적/감쇠 기억으로 `future_hs - hs0`를 리드별 예측한다.
3. **풍파/너울 soft gate**: `wspd`, `tp`, `wvdir-wdir`의 원형 피처로 상태를 나눈다. 대회 `tp`가
   첨두주기가 아니므로 공식 분리식을 hard rule로 쓰지 않는다
   ([NOAA windsea/swell 설명](https://www.ndbc.noaa.gov/faq/windsea.shtml)).
4. **기압 경향**: `caph`의 3/6/12/24/48시간 경향과 곡률을 저기압 접근·이탈의 보조 신호로 쓴다.
5. **최소 결합**: 감쇠 지속성과 direct GBDT의 OOF 잔차 상관을 본 뒤 혼합한다. 새 시퀀스 모델은
   이 둘이 남기는 분명한 실패모드가 있을 때만 검토한다.

## 물리식 사용 한계

- Pierson–Moskowitz 평형파고는 충분한 취송거리·지속시간의 완전발달 해상 근사다. 풍속 센서 높이도
  확인되지 않았으므로 `U²` 상호작용 피처로만 쓰고 정답식이나 hard cap으로 쓰지 않는다.
- `hs/(1.56 tp²)`와 `hmax/hs`도 깊은물·주기 정의·표본 파수 가정에 민감하다. QC 보조피처로만 쓴다
  ([NOAA wave measurements](https://www.ndbc.noaa.gov/faq/measdes.shtml)).
- 리드가 길다고 파고가 단조 감소하지는 않는다. 폭풍 접근 사례는 증가할 수 있어 단조 제약은 금물이다.

## 검증 위험

- 무작위 행/창 분할은 겹치는 48시간 문맥과 24시간 타깃을 누출한다. 정점별 78시간 purge와 미래형
  분할을 유지한다.
- train은 2025-06까지, test는 이후 1년이므로 로컬 CV는 판정기보다 탐색기다. 통합값 외에
  `lead × station × hs0 구간`을 반드시 본다.
- Public은 66사례뿐이다. 단독 모델 선택 근거로 쓰지 않는다.
- 구조적 파랑 결측을 10분 보간해 가짜 관측으로 만들지 말고 원 20분 관측 기반 피처와 관측 age를 쓴다.
- log target은 원공간 RMSE의 조건부 평균과 다르다. 사용하면 역변환 편향 보정이 필요하다.

## 최소 다음 순서

`station×lead damped persistence` → 같은 분할의 direct-delta LightGBM → OOF 혼합. PM ODE,
GRU/Chronos, 외부 사전학습은 앞선 두 모델의 세그먼트 실패와 잔차 다양성이 확인될 때만 연다.
