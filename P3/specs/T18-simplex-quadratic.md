# T18 — bugcheck.py `simplex`: 3성분 단순형 public MSE 2차형 해석해

## 목적

persistence·damped·lgbm 3성분 단순 가중 혼합의 public MSE 를 **자유가중 2개의 2차형**으로
정확히 복원하고 전역 최적 가중을 해석적으로 구한다. 격자탐색·로컬 폴드 선택을 쓰지 않는다.

## 배경 — 이미 검증된 사실이다. 재조사하지 마라

- `P3/src/bugcheck.py` 는 T16 에서 생성됐고 T17 이 서브커맨드를 추가하는 중이다.
  **파일을 새로 만들지 말고 서브커맨드 `simplex` 만 추가**한다. 기존 출력 형식을 바꾸지 마라
- 제출 CSV 사이의 가중 항등식은 이미 검증됐다 (T16 C2 max|Δ|=1.33e-15,
  C3 역산 (0.35,0.65)/(0.65,0.35), rms 4.3e-16). 재검증 불필요
- 예측은 가중에 대해 정확히 선형이므로 MSE 는 가중의 정확한 2차식이다. 근사가 아니다
- 새 의존성 없다. **`pip install` 이 필요하다고 판단되면 실행하지 말고 멈추고 보고하라**

## 수학 (그대로 구현하라. 재유도하지 마라)

`P` = persistence 예측, `D` = damped, `L` = lgbm, `y` = 미지 정답.
가중 `wp, wd`, `wl = 1 - wp - wd`. 오차 `e = wp(P-L) + wd(D-L) + (L-y)`.

```
MSE(wp,wd) = a*wp^2 + b*wd^2 + 2c*wp*wd + 2d*wp + 2e_*wd + f
a  = mean((P-L)^2)          # 정답 불필요
b  = mean((D-L)^2)          # 정답 불필요
c  = mean((P-L)*(D-L))      # 정답 불필요
f  = MSE(lgbm)     = 0.591543^2
e_ = (MSE(damped)      - b - f) / 2      # MSE(damped)      = 0.642689^2
d  = (MSE(persistence) - a - f) / 2      # MSE(persistence) = 0.769455^2
```

전역 최소: `[[a, c], [c, b]] @ [wp, wd] = [-d, -e_]`.
`a*b - c^2 <= 0` 이면 `NON-CONVEX` 를 찍고 최적 계산을 멈춘다.

## 입력 파일

- `P` : `P3/P3_data/baseline_persistence.csv`
- `D` : `P3/subs/submission_p3_damped_only.csv`
- `L` : `P3/subs/submission_p3_lgbm_only.csv`
- 세 파일을 `P3/P3_data/test_index.csv` 의 `(case_id, station, lead_h)` 순서로 정렬해 맞춘다.
  키 불일치가 1행이라도 있으면 즉시 FAIL 로 멈춘다

## 출력 (한 줄 = 한 항목, 숫자만)

1. `COEF a=… b=… c=… d=… e=… f=…`
2. `V1 b_hat=… A_fit=0.119800 rel_err=…` — `b` 와 p3_lb1 3점 적합 `A=0.119799973975` 의
   상대오차. `<= 0.05` 면 PASS
3. `V2 pred_rmse_at_t1f=… actual=0.587165 delta=…` — 2차형이 `(wp,wd)=(0,0.35)` 에서 예측하는
   RMSE 와 실측의 차. `|delta| <= 0.002` 면 PASS. **T1F 는 적합에 쓰지 마라. 홀드아웃이다**
4. `OPT wp=… wd=… wl=… pred_rmse=… improvement_vs_t1f=…` (개선 = 0.587165 − pred_rmse)
5. `PROJ …` — 최적이 단순형(`wp,wd,wl >= 0`) 밖이면 세 변과 세 꼭짓점에 사영한 후보 중
   최소를 골라 같은 형식으로 다시 찍는다. 안이면 `PROJ inside`
6. `SENS wp_range=[…] wd_range=[…]` — 입력 RMSE 3개를 각각 ±5e-7 섭동시킨 8조합의 최적 범위
7. `GRID` — 참고용으로 `wp,wd ∈ {0,0.05,…}` 격자에서 예측 RMSE 상위 5개 조합. **선택에 쓰지 마라**

`V1` 또는 `V2` 가 FAIL 이면 exit 1.

## 금지

- CSV 생성 금지. 제출 파일 수정 금지. `P3/preds/`·`P3/subs/` 쓰기 금지
- 로컬 OOF·폴드·`dense_weights.csv` 사용 금지. 이 서브커맨드는 public 3점만 쓴다
- T1F 점을 적합에 넣지 마라. 넣으면 V2 가 무의미해진다
- 추상화 금지. 클래스·config·로깅 프레임워크 만들지 마라

## 기준선 숫자 (baseline)

| 항목 | 값 |
|---|---|
| public RMSE lgbm_only | 0.591543 |
| public RMSE T1F (홀드아웃) | 0.587165 |
| public RMSE damped_only | 0.642689 |
| 공식 persistence B | 0.769455 |
| p3_lb1 3점 적합 A | 0.119799973975 |
| 전 CSV 행수 | 1200 |

## 수락 기준 (직접 실행하고 숫자를 보고하라)

```
python P3/src/bugcheck.py simplex
python P3/src/bugcheck.py all
```

**결론 문장 금지, 숫자와 표만.** 위 7개 출력 항목과 두 명령의 exit code 만 보고한다.

---

## 부수 작업 — `train` C1 임계 수정

현행 C1 은 「비트 단위 불변」을 요구해 `t2` 경로의 `4.44e-16` (값 ~2 에서 1 ULP,
누적 순서 의존)에 FAIL 을 낸다. `train` 이 영구 exit 1 이면 게이트로 쓸 수 없다.

- C1 판정 임계를 `1e-12` 로 바꾼다. 초과면 FAIL, 이하이면 PASS
- 단 `0 < max|Δ| <= 1e-12` 인 경로는 같은 줄에 `ULP` 표시와 경로명·변경 사례 수를 남긴다
- 경로별로 분리해 찍는다 (현행처럼 `t1_max`·`t2_max` 유지)
- 다른 검사의 임계·출력은 건드리지 마라
