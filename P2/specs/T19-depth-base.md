# T19 — 깊이 좌표 기반 프로파일 base 를 측정한다 (학습 없음)

## 왜 (읽고 넘어가라, 재조사 금지)

현재 모든 모델은 `base_nom` = 층1↔층5 **선형보간** 의 잔차를 학습한다.
문제지 `P2/P2CONTEXT.md:48` 이 직접 지적한다:

> 공개된 층 1(4.19 m)과 층 5(19.59 m) 사이 약 15 m 구간을 복원해야 합니다.
> 두 층을 잇는 수심 선형보간은 곡률을 표현하지 못하므로, 수온약층의 형태를 설명할 수 있는
> 구조가 필요합니다.

곡률을 표현하는 base 로 바꾸면 잔차가 작아지고 그 위 학습이 쉬워진다.
**이 사양은 base 만 측정한다. 학습을 돌리지 마라.** 결과가 좋으면 다음 사양에서 사람이 학습한다.

`P2/P2_data/observations.csv` 열: `station,year,layer,time,temp,psal,depth,nominal_depth`.
`depth` 가 실측 수심이고 `nominal_depth` 와 다르다 (2024 행 예: depth 4.09 / nominal 4.18).
연도별 센서 수심이 다르다 (`f.depth_shift`). **`depth` 를 쓰고, `nominal_depth` 와 비교 보고한다.**

## 측정 대상 base 3종

가림 층 2·3·4 의 수온을 **동시각 공개층만으로** 추정한다. 학습 금지, 시간축 정보 사용 금지 —
같은 시각의 (depth, temp) 점들만 쓴다.

| ID | 정의 |
|---|---|
| `B0` | 현재 기준선. 층1↔층5 선형보간 (기존 `base_nom` 과 같아야 한다 — 재현으로 검증) |
| `B1` | 공개층 **전부**(1·5·6·7·8)의 실측 depth 위에서 PCHIP 단조 3차 보간 (`scipy.interpolate.PchipInterpolator`) |
| `B2` | 4모수 수온약층 tanh 적합. `T(z) = Td + (Ts-Td)/2 * (1 - tanh((z-z0)/h))`, `scipy.optimize.least_squares` 로 시각별 적합 |

`B2` 초기값·경계: `Ts` = 최상단 공개층 온도, `Td` = 최하단 공개층 온도,
`z0` 은 공개층 중 인접 수직기울기가 최대인 구간의 중점, `h` 초기값 3.0 · 경계 [0.5, 30].
적합이 실패하거나 결과가 -5~45 범위를 벗어나면 그 시각은 `B1` 값으로 대체하고 **대체 건수를 보고한다.**

공개층이 2개뿐인 시각(`P2/DOMAIN_RESEARCH.md:14`: 3개 42시각, 2개 2시각)은
`B1`·`B2` 가 정의되지 않을 수 있다. 그런 시각은 `B0` 으로 대체하고 건수를 보고한다.

## 평가 창

`P2/src/season_e006.py` 의 `WINDOWS` 정의(S1·S3·S4)와 `2024_sepoct` 홀드를 그대로 쓴다.
창 정의를 새로 만들지 마라. 채점은 `P2/src/judge_sepoct.py` 가 쓰는 `G.score` 와 같은 함수를 쓴다.

## 보고할 표

### 표1 — 창별 전체 RMSE

행 = `B0`·`B1`·`B2`, 열 = `2024_sepoct` · `S1` · `S3` · `S4`.
비교용으로 `maskaug` 행을 같이 넣어라 (2024_sepoct 0.491145 · S1 0.491145 · S3 0.340521 · S4 0.194028 가 나와야 한다).

### 표2 — 2024_sepoct 세그먼트 분해

행 = `B0`·`B1`·`B2`, 열 = `전체` · `결측`(n=289) · `완전` · `층2` · `층3` · `층4`.

### 표3 — depth vs nominal_depth

`B1` 을 `depth` 로 계산한 값과 `nominal_depth` 로 계산한 값의 창별 RMSE 차이.

### 표4 — 잔차 크기

각 base 의 `RMSE(base)` 와, 그 base 잔차의 표준편차·첨도. 학습 난이도 비교용이다.

## 산출물 경로

| 경로 | 내용 |
|---|---|
| `P2/preds/oof_baseB1_E019.npy`, `P2/preds/oof_baseB2_E019.npy` | 4개 창 각각. `season_e006.py` 명명 규약을 따라 창별 파일로 저장한다 (`oof_baseB1_S1_E019.npy` 형태) |
| `P2/preds/test_baseB1_E019.npy`, `P2/preds/test_baseB2_E019.npy` | test 26,061행 |
| `P2/cache/T19/` | 중간 산출물 |
| `P2/src/depth_base_e019.py` | 신규 파일 1개. 이것만 만든다 |

**중요:** OOF 배열은 `judge_sepoct.py` 의 `_candidate_pool` 이 읽을 수 있어야 한다.
그 함수는 4개 창 **전부** 에 대해 행 수가 맞는 배열을 요구하고, 하나라도 없으면 후보를 skip 한다.
저장 후 `python P2/src/judge_sepoct.py` 를 실행해 `baseB1`·`baseB2` 가 `skipped` 목록에
**없는지** 확인하고 그 출력을 보고에 넣어라.

## 기준선 숫자

| 항목 | 현재값 |
|---|---|
| `base_nom` 2024_sepoct RMSE | 0.970325 |
| `base_nom` S1 / S3 / S4 | 0.970325 / 0.862944 / 0.428320 |
| `maskaug` 2024_sepoct 전체 / 결측 / 완전 | 0.491145 / 0.947625 / 0.483651 |
| 주최측 선형보간 전체 평가 RMSE | 1.290264 |
| test 행 수 | 26,061 (결측 7,766 / 완전 18,295) |
| 현재 우리 public | 0.522638 |

## 수락 기준

- `B0` 재계산값이 기존 `base_nom` 과 창별 RMSE 6자리까지 일치한다. 불일치면 **멈추고 보고해라**
- 표1~표4 전부 출력
- `judge_sepoct.py` 출력에서 `baseB1`·`baseB2` 가 skipped 에 없다
- 신규 파일은 `P2/src/depth_base_e019.py` 하나뿐. 기존 파일은 수정하지 마라
- 학습 금지. LightGBM 을 import 하지 마라
- `P2/P2_data/` 읽기 전용. `P2/subs/` 건드리지 마라
- 새 의존성 금지. `pip install` 이 필요하면 설치하지 말고 즉시 멈추고 보고해라
- 추상화 금지. 클래스 계층·config·플러그인 만들지 마라. 함수 몇 개면 된다

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 표1~표4, 대체 건수, judge_sepoct skipped 줄, 신규 파일 줄 수.
"개선되었다"·"유망하다" 같은 문장을 쓰지 마라. 산문 금지.
