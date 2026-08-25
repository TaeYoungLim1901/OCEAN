# T22 — public 편향 탐침 제출물 생성

## 목적

제출 1회로 public 부분집합에서 현행 T1F 의 **전역 편향** `mean(pred - y)` 를 정확히 측정한다.
CSV 2개를 만들기만 한다. **모델 없음. 학습 없음. 점수 계산 없음.**

## 배경 — 이미 검증된 사실이다. 재조사하지 마라

- 현행 제출 `P3/subs/submission_p3_t1f.csv`, SHA `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1`,
  public RMSE `0.587165`, 1200행. **이 파일을 수정하지 마라**
- 제출 형식은 `case_id,station,lead_h,hs_pred` 열 순서 고정, 값은 0~30m 유한, 키 1200건 정확일치
- `P3/src/bugcheck.py` 와 `P3/src/submission_tools.py` 는 건드리지 마라
- T21 이 `P3/src/model_arx.py`·`model_wsswell.py`·`run_t21.py` 를 동시에 만들고 있다.
  **그 세 파일과 `P3/preds/` 를 건드리지 마라**
- 수학: `MSE(T1F + d) = MSE(T1F) + 2*d*mean(e) + d^2`, `e = pred - y`.
  `d=0.05` 고정. 재유도하지 마라
- `pip install` 금지. 새 의존성 금지

## 작업 1 — `P3/src/make_probe_sub.py`

`P3/subs/submission_p3_t1f.csv` 를 읽어 아래 2개를 만든다. 행 순서·키를 그대로 유지한다.

```
P3/subs/submission_p3_t1f_shift.csv     # hs_pred + 0.05, 그 뒤 [0,30] 클립
P3/subs/submission_p3_t1f_bias.csv      # hs_pred + C, 그 뒤 [0,30] 클립
```

- `C` 는 명령행 인자 `--c <float>`. 인자가 없으면 `bias.csv` 를 만들지 않고 그 사실을 찍는다
- 클립으로 실제 변경된 행 수를 반드시 찍는다. **0 이 아니면 경고를 찍어라** —
  클립이 일어나면 위 등식이 깨진다
- 부동소수 출력 형식은 기존 제출 CSV 와 동일하게 맞춘다

## 작업 2 — 검증

`shift.csv` 에 대해 아래를 직접 실행하고 숫자를 보고한다.

```
python P3/src/make_probe_sub.py
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f_shift.csv
python scripts/preflight.py sub P3/subs/submission_p3_t1f_shift.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/bugcheck.py all
```

보고 항목: `shift.csv` SHA256, 행수, T1F 대비 `max|Δ|`·`mean Δ` (0.05 여야 한다),
클립된 행 수, 네 명령의 exit code.

## 금지

- `submission_p3_t1f.csv` 및 기존 제출물 수정 금지. `P3/preds/` 쓰기 금지
- 모델 학습·평가·점수 계산 금지. `y` 는 없다
- `bugcheck.py`·`submission_tools.py`·T21 산출물 수정 금지
- 추상화 금지. 클래스·config·로깅 프레임워크 금지

## 기준선 숫자 (baseline)

| 항목 | 값 |
|---|---|
| T1F public RMSE | 0.587165 |
| T1F hs 평균 | 1.617184 |
| T1F hs 범위 | [0.782838, 4.035338] |
| shift 폭 `d` | 0.05 |
| 예상 클립 행 수 | 0 |
| 행수 | 1200 |

## 수락 기준

위 4개 명령의 숫자와 exit code. **결론 문장 금지, 숫자와 표만.**
