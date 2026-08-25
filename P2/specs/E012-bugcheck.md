# E012 — P2 모델 파이프라인 버그 체커

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
(numpy·pandas 만.) **재학습 금지. 모델·피처·게이트 규칙·기존 파일 수정 금지.**
버그를 발견해도 **고치지 마라.** 발견만 보고한다. 수정은 다음 사양에서 지시한다.

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라.

## 1. 배경 (재조사 금지, 확정 사실)

public 실측 2건: `E004_gate` 0.537099, `E009_maskaug_diff` 0.525759 (결측행 7,766에서만 다름).
로컬 4-hold 균등 판정은 **정반대**를 예측했다(maskaug +0.004738 악화 예측). 로컬 판정기가
왜 틀렸는지가 미해결이며, 그 원인이 분포 차이인지 **파이프라인 버그**인지 구분되지 않았다.

test 26,061행 = 결측행 7,766 + 완전행 18,295. 마스크는 `P2/src/gate_e004.py` 의
`regimes(...)["missing_public"]`. 예측 풀은 `P2/preds/{oof,test}_*.npy`.
제출물은 `P2/subs/*.csv`. 학습·추론 코드는 `P2/src/train*.py`·`inference*.py`·`common.py`.

## 2. 만들 것 — `P2/src/bugcheck.py` 하나

기존 파일 수정 금지. `check.py`(data·pipeline·cases·sub)·`submission_tools.py`(audit) 와
**중복되는 검사는 넣지 마라.** 여기서는 그 둘이 안 하는 것만 한다.

사용법: `python P2/src/bugcheck.py [--only <섹션문자>]`. 인자 없으면 A~G 전부.
출력은 섹션별로 `A1 PASS ...` / `A1 FAIL ...` 1행씩 + 숫자. 마지막에 `PASS n / FAIL n`.
**FAIL 이 하나라도 있으면 exit 1**, 전부 통과면 마지막 줄 `판정: OK` 와 exit 0.
검사 하나가 예외로 죽어도 나머지는 계속 돌고 그 검사는 `ERROR` 로 집계한다(ERROR 도 exit 1).

### A. 데이터 무결성

- A1: 원본 `P2/P2_data/` 로딩 후 관측 키(`common.py` 의 `C.KEY`) 중복 행 수 = 0
- A2: 시간 컬럼이 단조 정렬 가능하고 파싱 실패(NaT) 0
- A3: target 층·공개층 temp 의 물리 범위 이탈 개수(예: -5 미만 / 45 초과)와 결측률 표
- A4: `test_index.csv` 의 키가 전부 관측 프레임 안에 존재하고 순서가 유일하게 정해지는가

### B. 누수 (leakage) — 이 섹션이 최우선

- B1: **시간 누수.** 학습에 쓰이는 행의 시각 최대값 < test 기간 시작 시각. 위반 행 수를 찍는다.
- B2: **hold 누수.** `gate_e004.py` 의 `HOLDS`·`holdout()` 을 그대로 import 해서 각 hold 의
  (학습 인덱스 ∩ 평가 인덱스) 크기를 찍는다. 전부 0 이어야 한다.
- B3: **타깃 누수 피처.** `common.py` 의 `build_features` 로 만든 피처 행렬에서, 각 피처와
  타깃의 |Pearson r| 를 계산해 **0.999 이상인 피처**를 전부 나열한다(없어야 정상).
  타깃 자신·타깃의 결정론적 변환이 피처에 섞였는지 보는 검사다.
- B4: **미래 참조.** 피처 이름에 lead/forward/next 류가 있거나 `shift(-` 가 소스에 있으면
  해당 위치를 `파일:행` 으로 나열한다(존재 자체가 즉시 FAIL 은 아니고 나열 후 FAIL 로 표시).
- B5: **OOF 자기예측.** 각 `P2/preds/oof_*.npy` 의 전체 RMSE 와, 같은 벡터를 hold 별로 쪼갠
  RMSE 를 표로 낸다. 전체 RMSE 가 hold 별 RMSE 최소값보다 **0.30 이상 낮으면** FAIL 로 표시한다.

### C. train ↔ inference 대칭

- C1: `train*.py` 와 `inference*.py` 가 쓰는 피처 컬럼 목록을 각각 추출(가능하면 실제 호출,
  안 되면 소스에서 상수 목록 추출)해 **집합 차이**를 찍는다. 차이가 있으면 FAIL.
- C2: 컬럼 **순서**까지 동일한가 (LightGBM 은 순서 의존). 첫 불일치 인덱스를 찍는다.
- C3: 결측 대치·스케일링 규칙이 train 과 inference 에서 같은 함수를 통과하는가.
  서로 다른 코드 경로로 대치하는 곳이 있으면 `파일:행` 을 나열한다.

### D. 분할·판정기 정합

- D1: `HOLDS` 각각의 기간·행수·타깃층 분포·`missing_public` 비율을 표로 낸다.
  같은 표의 마지막 행에 **test** 의 같은 값을 넣는다.
- D2: 각 hold 의 `missing_public` 비율과 test 의 0.297993 의 차이를 찍는다.
- D3: **결측 원인 대조.** test 결측행의 결측 길이 분포(연속 결측 구간 길이의 중앙값·최대값)와
  각 hold 결측행의 같은 분포를 표로 낸다. test 는 연속 블록, hold 는 산발이라면 여기서 드러난다.
  (이 섹션은 진단이므로 FAIL 판정 없이 `INFO` 로 찍는다.)

### E. 예측 벡터 정렬·위생

- E1: `P2/preds/test_*.npy` 전부 길이 26,061·1차원·유한.
- E2: 정렬 대리 지표 — 기준선 `base_nom` 과의 `rmse_real = RMSE(pred, base_nom)`,
  시드 42 로 셔플한 `rmse_shuffled`. `ratio = rmse_real / rmse_shuffled > 0.9` 면 FAIL 이고
  파일명을 찍는다(정렬이 어긋난 예측은 값·범위 검사를 전부 통과하며 조용히 망가진다).
- E3: `P2/preds/oof_*.npy` 전부 길이가 학습 행수로 서로 동일.
- E4: 각 test 예측의 상수 구간 최대 길이·유일값 개수(모델이 뭉개졌는지).

### F. 재현성

- F1: `SEED = 42` 가 `train*.py` 전부에 있는가. 없는 파일을 나열한다.
- F2: `P2/subs/E009_maskaug_diff.csv` 를 `P2/src/build_sub.py` 로 임시 경로에 재생성해
  sha256 이 `8dc6a7cd34c8591919dccad7781a990ff3c99dd9d32b1f80b7d169a3462b674f` 와 같은가.
  임시 파일은 지운다.

### G. 제출 필요성 게이트

`submission_tools.py` 원장에서 public 최소값 CSV 를 기준선(`base_public`)으로 잡는다.
아직 public 점수가 없는 `P2/subs/*.csv` 각각에 대해:

- `m` = temp 가 기준선과 다른 행 수, `d` = 그 차이 벡터
- `mse_shift = sum(d**2) / 26061`
- `rmse_pure_noise = sqrt(base_public**2 + mse_shift)`
- `rmse_if_perfect = sqrt(max(base_public**2 - mse_shift, 0.0))`
- `측정폭 = rmse_pure_noise - rmse_if_perfect`

**측정폭 < 0.002 → `제출불필요`, 아니면 `제출가치있음`.** 이 임계는 사양이 고정한 값이며
바꾸지 마라. 이 섹션은 FAIL 을 내지 않고 표만 낸다.

## 3. 실행하고 보고

`python P2/src/bugcheck.py` 를 실행하고 **출력 전문**을 그대로 보고한다.
섹션이 무거우면 D3·E2 는 캐시 없이도 60초 안에 끝나도록 짜라.

## 4. 수락 기준

- 새 파일은 `P2/src/bugcheck.py` 하나뿐, 기존 파일 무수정
- A~G 전부 구현, 각 검사가 PASS/FAIL/INFO/ERROR 중 하나를 1행으로 출력
- FAIL 또는 ERROR 가 있으면 exit 1, 없으면 `판정: OK` 와 exit 0
- 발견한 버그를 **고치지 않았다**
