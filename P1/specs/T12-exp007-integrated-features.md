# T12 — offset·drift 미탐 교정: 다중창 적분 feature 추가 (exp007)

실행 파이썬: `python`. `n_jobs` ≤ 2. 콘솔 cp949 → 스크립트 상단에 `sys.stdout.reconfigure(encoding="utf-8")`.

**학습 실행은 하지 마라.** 코드를 만들고 **한 번만** 아주 짧은 smoke(스트림 3개, 트리 20개)로 동작을 확인한 뒤 멈춘다.
전체 학습은 사람이 돌린다. 학습 명령을 마지막에 한 줄로 적어라.

## 배경 — 확정 사실 (재조사·재측정 금지)

exp005 가 현재 제출본이다. public F1 **0.717435**. 설정: weight 0.00 / threshold 0.62 / hole 48 / λ 0.00,
forward fold1·2 학습, 700 tree 고정, early stop 없음, train-only climatology, 한쪽 purge 87h.

### 기준선 숫자 (확정)

| 경로 | F1 |
|---|---|
| forward inner (fold1·2) | **0.697836** |
| 봉인 outer fold3 | 0.815757 |
| public LB | **0.717435** |

T10(`P1/experiments/exp006_fn_decomp.json`) 이 확정한 실패모드:

| 유형 | inner 행수 | inner recall | FN |
|---|---|---|---|
| **offset** | 3752 | **0.4254** | 2156 |
| **drift** | 3450 | **0.5104** | 1689 |
| noise | 5777 | 0.7004 | 1731 |
| flatline | 2288 | 0.8942 | 242 |
| spike | 51 | 0.6863 | 16 |

offset+drift = inner FN 6038 의 **63.7%**.
세그먼트 형태: offset 17구간 중 **full_miss 8**, 구간내 적중률 p50 **0.0083**.
drift 는 구간내 10분위 recall 이 **0.2393 → 0.7361 단조 증가**.
flatline 은 full_miss 0, 적중률 p50 0.9804 — **이미 해결됐다. 건드리지 마라.**

**원인(확정):** 행 단위 LightGBM 이 계단 이후 구간을 「새 정상」으로 보고, 느린 기울기를 적분하지 못한다.
단기 lag·차분 feature 는 누적량을 담지 못한다.

**기각된 축 (다시 제안하지 마라):** 유형 multiclass blend, stream 별 양성률 앵커링,
랜덤 행 CV, adversarial drift 를 보정기로 사용. 규칙 점수(`rule_score`) 가중치는 확장 격자에서 0 이 최적이었다.
`ruptures`·`xgboost`·`catboost`·`torch` 없다. numpy·pandas·scikit-learn·lightgbm 만 있다.

## 할 일

### T12-0. 임계 보정 프로브 제출본 (먼저. 독립)

`P1/preds/test_exp005.npy` 를 재임계한다. **재학습 금지.**
threshold 를 격자 `0.20:0.005:0.80` 에서 훑어, 후처리(hole fill 48) 적용 **후** 최종 test 양성률이
train 양성률 **0.041362** 에 가장 가까워지는 값을 고른다.
결과를 `P1/subs/exp006b.csv` 로 쓴다. 형식·열은 `P1/subs/exp005.csv` 와 동일해야 한다.
보고: 선택 threshold, 후처리 전/후 양성률, exp005 대비 양성 행 수 차이.
검증 명령: `python P1/src/check.py sub P1/subs/exp006b.csv`.

### T12-1. 적분 feature 추가

`P1/src/features.py` 에 아래를 추가한다. **기존 feature 를 지우거나 바꾸지 마라.**
전부 `(station, year, layer)` stream 안에서만 계산하고, **stream 의 관측 gap 을 넘어가지 않는다**
(gap 판정은 기존 코드가 쓰는 방식을 재사용한다).
기준 신호는 기존 train-only climatology 잔차 `r_t` (`temp`·`psal` 각각). σ 는 **학습집합에서만** 추정한 stream 별 잔차 표준편차.

1. **다중창 z-score** — 창 `W ∈ {12, 48, 144, 288}` (10분 간격 기준 2h/8h/24h/48h).
   중심창과 후행창 둘 다: `z_W = mean_W(r) / (sigma / sqrt(W))`.
   계단·드리프트는 W 가 커질수록 z 가 커지고 백색잡음은 커지지 않는다. 이것이 GBDT 에 없던 적분값이다.
2. **robust step 점수** — 창 `W ∈ {48, 144}`: `(median(직후 W) - median(직전 W)) / sigma`. 부호 유지 + 절댓값 둘 다.
3. **rolling slope 점수** — 창 `W ∈ {144, 288}`: 최소제곱 기울기 × W / sigma (창 전체에서의 총 이동량, σ 단위).
4. **z 의 지속성** — `|z_288|` 이 임계 2.0 을 넘은 연속 길이 (run-length), stream 내에서.

각 항목을 `temp`·`psal` 양쪽에, 그리고 두 변수의 z 를 합친 `max` 1개. 열 이름은 기존 규약을 따른다.
NaN 은 기존 코드가 쓰는 방식대로 처리한다. **누수 금지: 미래 행을 쓰는 중심창은 허용되지만
(offline 행 판정이라 `r.offline_centered` 로 허용됨) label 이나 다른 stream 의 통계는 절대 쓰지 마라.**

### T12-2. `P1/src/exp007.py`

exp005 의 학습·평가 경로를 **그대로 복제**하고 feature 집합만 바꾼다.
forward fold1·2 학습(과거만), 한쪽 purge 87h, train-only climatology, `n_estimators=700` 고정, early stop 없음, `SEED = 42`.
**봉인 fold3 를 읽거나 채점하지 마라.** inner(fold1·2) 만 채점한다.

후처리 격자를 확장한다:

| 축 | 값 |
|---|---|
| threshold | `0.30:0.02:0.90` |
| hole fill h | `0, 12, 24, 48, 96` |
| **dilation k** (예측 1 을 양쪽 k 행 확장) | `0, 2, 6, 12, 24` |
| **min-length L** (길이 < L 인 1-런 삭제) | `0, 3, 6, 12` |

적용 순서는 **hole fill → dilation → min-length** 로 고정한다. 순서를 바꾸지 마라.
격자 전체에서 inner F1 을 계산하고 argmax 를 고른다.

산출: `P1/preds/oof_exp007.npy`, `P1/preds/test_exp007.npy`,
`P1/experiments/exp007_grid.json` (격자 전체 점수), 그리고 T10 과 같은 형식의 **유형별 recall 표**.

## 제약

- **새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 멈추고 보고한다.**
- `P1/P1_data/` 는 읽기 전용.
- 사양 밖의 변경 금지: 모델 파라미터·분할·climatology·purge·seed 를 바꾸지 마라. feature 추가와 후처리 격자만이 변경점이다.
- 동시에 다른 프로세스가 루트 `RULES.md` 와 `P1/specs/` 를 쓰고 있다. **그 경로를 건드리지 마라.**
- 과한 추상화 금지. 새 클래스·설정파일·로깅 프레임워크를 만들지 마라.

## 수락 기준

1. `python P1/src/check.py sub P1/subs/exp006b.csv` 가 `판정: OK`.
2. `python P1/src/features.py` 관련 기존 자체 검사가 있으면 통과. 없으면 만들지 마라.
3. smoke 실행(스트림 3개·트리 20개)이 오류 없이 끝나고 새 feature 열 이름과 개수를 출력한다.
4. 전체 학습 명령을 **한 줄로** 보고한다 (사람이 실행한다).

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 보고할 것: T12-0 의 선택 threshold·양성률·행수차,
추가된 feature 열 이름과 개수, smoke 출력, 전체 학습 명령 1줄. 산문 금지.
