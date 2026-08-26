# T25 — 구조적 상수보정 측정 + t1f_bias 재현 러너 등록

**결론 문장 금지. 숫자와 표만 보고한다.** 산문 요약·권고·다음 단계 제안을 쓰지 마라.

작업 2개다. 서로 독립이며 파일이 겹치지 않는다. 둘 다 수행한다.

---

## A. `check.py final` 에 t1f_bias 러너 등록

`P3/src/check.py:163` 의 runner 매핑에 `submission_p3_t1f` 하나만 있다.
`submission_p3_t1f_bias` 가 없어 `python P3/src/check.py final P3/subs/submission_p3_t1f_bias.csv`
가 `ValueError: unsupported submission runner` 로 STOP 한다.

- `P3/src/inference_t1f_bias.py` 를 만든다. `inference_t1f.py` 와 동일 경로로 T1F 를 생성한 뒤
  전 행에 상수 `C_BIAS = -0.041550308` 를 더하고 `[0, 30]` 클립을 적용해
  `P3/preds/test_final_p3_t1f_bias.npy` 와 `P3/subs/submission_p3_t1f_bias.csv` 를 쓴다.
  상수는 모듈 최상단 상수 1개로 둔다. 설정 파일·CLI 플래그를 만들지 마라.
- runner 매핑에 `"submission_p3_t1f_bias": ("inference_t1f_bias.py", "test_final_p3_t1f_bias.npy")`
  를 추가한다. 매핑 구조 자체는 바꾸지 마라 (dict 한 줄 추가).
- **기존 산출물은 비트 단위로 보존돼야 한다.** 실행 후:
  - `P3/subs/submission_p3_t1f_bias.csv` SHA256 == `605BA2BB6FA70CC5811F1ACAEA3F200543C929E37BDF632E7FDAFEF389E3BA1C`
  - `P3/subs/submission_p3_t1f.csv` SHA256 == `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` (불변)
  - `P3/preds/test_final_p3_t1f.npy` 불변
  달라지면 코드를 고치지 말고 **차이 수치를 보고하고 멈춘다.**

수락 기준 (직접 실행하고 출력 숫자를 보고. 판정이 OK 가 아니면 멈추고 숫자만 보고):

```
python P3/src/check.py final P3/subs/submission_p3_t1f_bias.csv
python P3/src/check.py final P3/subs/submission_p3_t1f.csv
python P3/src/bugcheck.py all
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f_bias.csv --pred P3/preds/test_final_p3_t1f_bias.npy
```

보고 표 A: 명령 | 판정 | 핵심 숫자(해시·max delta)

---

## B. 구조적 상수보정 — 로컬 조밀 재가중 측정

### 배경 · 기준선(baseline) 숫자 — 재조사 금지, 그대로 쓴다

- 현행 T1F 봉인 fold4 조밀 재가중 RMSE = `0.611959`
- 전역 상수보정 `c = -0.035651675` (folds 전체에서 잰 값) 적용 시 fold4 = `0.607162`, 델타 `-0.004798`
- 정점별 가중편향 G-ORS `+0.073441` / I-ORS `+0.048349` / S-ORS `-0.013242`
- 리드별 가중편향 +3h `.032784` / +6h `.040107` / +9h `.036561` / +12h `.025923` / +18h `.042359` / +24h `.036176`
- 폴드별 편향 부호가 갈린다: f1 `+0.389` / f2 `-0.065` / f3 `-0.157` / f4 `+0.100`

### 무엇을 재는가

편향이 **구조(정점·리드)** 를 가지는지, 그 구조가 **폴드를 넘어 전이되는지**.
public 리드별 탐침(제출 6회 소모)에 예산을 쓸 가치가 있는지를 이 측정이 결정한다.

### 방법

`run_t21.py --folds 4` 가 쓰는 조밀 재가중 채점 경로를 **그대로 재사용**한다.
새 학습 없음. 새 모델 없음. 저장된 T1F 조밀 OOF 만 쓴다.

보정 변형 4개. 각 변형의 상수는 **folds1-3 조밀 재가중 잔차 평균으로만** 적합한다.
fold4 는 적합에 일절 쓰지 않는다.

| 변형 | 자유도 | 적합 단위 |
|---|---|---|
| G | 1 | 전역 |
| L | 6 | 리드별 |
| S | 3 | 정점별 |
| SL | 18 | 정점 x 리드 |

각 변형에 대해 봉인 fold4 조밀 재가중 RMSE 를 계산한다.
보정 후 예측은 `[0, 30]` 클립을 적용하고 클립된 행 수를 보고한다.

### 수락 기준 — 결과 보기 전 고정 규칙

- **(대조군)** 보정 없음(c=0) 재현값이 `0.611959` 와 델타 `0.000000` 이어야 한다.
  또한 전역 변형 G 의 적합 상수가 `-0.035651675` 와 `|델타| <= 0.002` 여야 한다
  (G 는 folds1-3 만 쓰므로 정확히 같지 않아도 된다). 어긋나면 **채점경로 불일치이므로 중단**하고 숫자만 보고한다.
- **(구조 존재)** L·S·SL 중 어느 하나라도 fold4 에서 G 변형보다 `0.003` 이상 개선하면
  「구조 존재」로 판정한다. 전부 미달이면 「구조 없음」이며 리드별 public 탐침은 예산 낭비다.
- **(전이 안정성)** folds1-3 에서 적합한 상수 벡터와, 참고로 fold4 에서 적합한 상수 벡터를
  나란히 보고한다. fold4 적합 상수는 **판정에 쓰지 않는다** — 전이 실패 여부를 눈으로 보기 위한 진단이다.
- **(제출 채택)** 어떤 변형도 이 실험만으로 제출 후보가 되지 않는다. 이 실험은 측정이다.

### 보고 표 B (이 3개 표만)

1. `변형 | 자유도 | fold4 RMSE | vs 0.611959 델타 | vs G 델타 | 클립행수`
2. `단위 | folds1-3 적합 c | fold4 적합 c | 차이` — S(3행) 와 L(6행). SL 은 `|차이|` 상위 5행만
3. `대조군 | 값 | 기대값 | 델타`

---

## 산출물

- `P3/src/inference_t1f_bias.py` (신규)
- `P3/src/check.py` runner 매핑 1줄 추가
- `P3/preds/oof_dense_t1f_corr_{g,l,s,sl}_p3_t25.npy` 4개 (결과와 무관하게 저장)
- 표 A·B 를 stdout 에 출력

## 환경

새 의존성 금지. `pip install` 이 필요해지면 **설치하지 말고 즉시 정지**하고 무엇이 없는지만 보고한다.

## 금지

- 새 모델 학습, 하이퍼파라미터 변경, 새 피처
- fold4 를 상수 적합에 사용
- 제출 CSV 신규 생성 (A 의 재현 검증 산출물 제외)
- 추상화·설정 클래스·로깅 프레임워크. `print` 로 충분
- `cx.cmd`·`codex exec` 재귀 호출
- 결론·권고 문장
