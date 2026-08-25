# T16 — bugcheck.py: 제출 자산 교차검증 + LB 가중곡선 해석해

## 목적

제출 전에 「이 CSV 가 우리가 믿는 그것인가」를 기계로 확인한다. 기존 게이트는 각 파일을
**개별로** 본다. 파일 **사이의 관계**(가중 항등식·키 정렬·ledger 해시)는 아무도 안 본다.
그 관계가 깨져 있으면 LB 곡선 해석 전체가 무효다.

## 배경 — 이미 검증된 사실이다. 재조사하지 마라

- 게이트 `check.py data/pipeline/cases`, `metric.py selftest`, `context_static_check.py`,
  `submission_tools.py audit`, `preflight.py sub` 는 전부 구현·통과 상태다. 고치지 마라
- `P3/subs/` 에 CSV 5개 존재: `submission_p3_t1.csv`, `submission_p3_t1f.csv`,
  `submission_p3_t1f_r2.csv`, `submission_p3_lgbm_only.csv`, `submission_p3_damped_only.csv`
- T1F 제출 가중은 상수 (persistence 0.00 / damped 0.35 / lgbm 0.65) 로 `select_t1f.py` 에 고정돼 있다.
  `_r2` 는 (0.00 / 0.65 / 0.35). persistence 성분은 양쪽 다 0 이다
- 단독 CSV 2개는 T14 에서 `test_lgbm_p3_t1.npy`·`test_damped_p3_t1.npy` 를 포장만 한 것이다
  (npy 대비 max|Δ| 4.44e-16, 클리핑 0행)
- 새 의존성은 없다. **`pip install` 이 필요하다고 판단되면 실행하지 말고 멈추고 보고하라**

## 기준선 숫자 (baseline)

| 자산 | 값 |
|---|---|
| public RMSE T1F | 0.587165 |
| public RMSE lgbm_only | 0.591543 |
| 공식 persistence B | 0.769455 |
| `submission_p3_t1f.csv` SHA256 앞 8 | `2E498428` |
| `submission_p3_lgbm_only.csv` SHA256 앞 8 | `63496516` |
| `submission_p3_damped_only.csv` SHA256 앞 8 | `54A4C41C` |
| `submission_p3_t1f_r2.csv` SHA256 앞 8 | `DB4CEF48` |
| 전 CSV 행수 | 1200 |

C2 가 PASS 라면 max|Δ| 는 부동소수 오차 수준(1e-9 이하)이어야 한다.

## 산출물

`P3/src/bugcheck.py` 하나. 새 의존성 금지 (numpy/pandas 만). 표준출력 UTF-8 재설정.
기존 게이트 스크립트를 **재구현하지 마라.** subprocess 로 호출하고 exit code 만 모은다.

## 서브커맨드 2개

### 1. `python P3/src/bugcheck.py all`

아래를 순서대로 실행하고 **한 줄 = 한 검사**로 `PASS/FAIL` + 숫자를 찍는다.

기존 게이트 위임 (exit code 만):
- `P3/src/check.py data`, `check.py pipeline`, `check.py cases`
- `P3/src/metric.py selftest`
- `P3/src/context_static_check.py` (인자는 현행 사용법 그대로)
- `P3/subs/*.csv` 각각에 `submission_tools.py audit`

새 교차검사 (여기가 본체):
- **C1 키 정렬**: `P3/subs/*.csv` 전부와 `P3/P3_data/test_index.csv` 의
  `(case_id, station, lead_h)` 튜플 순서가 완전 동일한가. 불일치 행 수 보고
- **C2 가중 항등식**: `submission_p3_t1f.csv` 가
  `0.35*damped_only + 0.65*lgbm_only` 와 일치하는가.
  `submission_p3_t1f_r2.csv` 는 `0.65*damped_only + 0.35*lgbm_only` 인가.
  각각 max|Δ|, mean|Δ| 를 찍는다. 임계 `1e-6` 초과면 FAIL
- **C3 암시 가중 역산**: t1f 와 r2 각각에 대해 최소제곱으로
  `hs_t1f ≈ a*hs_damped + b*hs_lgbm` 의 (a, b) 를 풀어 실제 값을 찍는다.
  C2 가 FAIL 일 때 무엇이 진짜 가중인지 알기 위한 것이다. 잔차 RMS 도 찍는다
- **C4 CSV↔npy**: `P3/subs/` 각 CSV 와 대응 `P3/preds/test_*.npy` 의 max|Δ|.
  대응이 불명확하면 그 줄은 `SKIP` 으로 찍고 이유를 6단어 이내로
- **C5 ledger 해시**: `submission_tools.py public list` 에 기록된 sha256 이
  디스크 현재 파일 해시와 일치하는가
- **C6 값 위생**: 각 CSV 의 NaN/inf 개수, 0 미만·30 초과 개수, 중복 키 개수

하나라도 FAIL 이면 exit 1.

### 2. `python P3/src/bugcheck.py wcurve`

public ledger 에서 damped–lgbm 단순선 위 점들을 읽어 2차식을 푼다.

- 각 제출의 w(=damped 가중)는 하드코딩 매핑: lgbm_only 0.0 / t1f 0.35 / t1f_r2 0.65 / damped_only 1.0
- ledger 에 점이 3개 미만이면 `INSUFFICIENT n=<k>` 만 찍고 exit 0
- 3점이면 `MSE(w) = A w^2 + B w + C` 를 정확히 풀고, 4점 이상이면 최소제곱 + 잔차 RMS 보고
- 출력: A, B, C, 정점 `w* = -B/(2A)`, `RMSE(w*) = sqrt(MSE(w*))`,
  현행 T1F 대비 예상 개선폭, 그리고 A<=0 이면 `NON-CONVEX` 경고
- **주의**: public RMSE 는 소수 6자리 반올림값이다. w* 의 반올림 민감도를
  각 입력 RMSE 를 ±5e-7 섭동시킨 8개 조합으로 재계산해 **w\* 범위**를 같이 찍는다

## 금지

- 새 예측 생성 금지. 제출 CSV 수정 금지. `P3/preds/` 쓰기 금지
- 추상화 금지. 함수 몇 개와 `if __name__ == "__main__"` 이면 충분하다
- 로깅 프레임워크·config 파일·클래스 계층 만들지 마라

## 수락 기준 (직접 실행하고 숫자를 보고하라)

```
python P3/src/bugcheck.py all
python P3/src/bugcheck.py wcurve
```

**결론 문장 금지, 숫자와 표만.** 각 검사의 PASS/FAIL 과 측정값, 두 명령의 exit code 만 보고한다.
