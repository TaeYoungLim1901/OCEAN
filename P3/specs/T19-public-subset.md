# T19 — bugcheck.py `subset`: public 부분집합 식별

## 목적

public 점수가 전체 1200행이 아님이 확정됐다. **정답 없이 계산되는 통계 하나**를 체로 써서
public 이 어떤 부분집합인지 좁힌다. 제출을 낳지 않는 순수 진단이다.

## 배경 — 이미 검증된 사실이다. 재조사하지 마라

- `P3/src/bugcheck.py` 에 `all`·`wcurve`·`train`·`simplex` 가 있다.
  **서브커맨드 `subset` 만 추가**한다. 기존 출력·임계를 바꾸지 마라
- 체로 쓸 통계: `b = mean((D-L)^2)`, `D`=damped 예측, `L`=lgbm 예측. 정답이 필요 없다
- 전체 1200행에서 `b_full = 0.157493425176` (T18 실측)
- public LB 3점 적합이 주는 public 부분집합 값 `b_pub = 0.119799973975` (p3_lb1)
- 1200행 = 200 `case_id` x 6 `lead_h` (3/6/9/12/18/24). 정점은 3개
- 예측 선형성·가중 항등식은 T16 에서 검증됐다 (max|Δ| 1.33e-15). 재검증 불필요
- 새 의존성 없다. **`pip install` 이 필요하다고 판단되면 실행하지 말고 멈추고 보고하라**

## 입력

- `D` : `P3/subs/submission_p3_damped_only.csv`
- `L` : `P3/subs/submission_p3_lgbm_only.csv`
- 키·정렬 : `P3/P3_data/test_index.csv` 의 `(case_id, station, lead_h)`
- t0 hs 분위 구간이 필요하면 `P3/P3_data/` 의 test 컨텍스트에서 각 `case_id` 의
  마지막 행 `hs` 를 쓴다. 다른 사례 통계를 섞지 마라 (`r.context_only` 는 추론 제약이나
  여기서도 같은 정의를 쓴다)

## 작업 1 — 무작위 사례분할 가설 검정

`case_id` 단위로 무작위 부분집합을 뽑는다 (해당 사례의 6리드 전부 포함).
`k = 20, 40, 60, 80, 100, 120, 140, 160, 180` 각각에 대해 2000회 반복.

출력 한 줄씩:
```
RAND k=<k> mean=<...> sd=<...> z=<(0.119799973975-mean)/sd> p_two_sided=<...>
```
마지막에 `RAND verdict=REJECT_RANDOM` (모든 k 에서 |z|>=3) 또는
`RAND verdict=CANNOT_REJECT k_ok=<|z|<=2 인 k 목록>` 한 줄.

## 작업 2 — 구조 후보 체

각 후보 부분집합에서 `b = mean((D-L)^2)` 를 계산하고
`rel_err = |b/0.119799973975 - 1|` 을 구한다.

후보 목록:
- 단일 리드 6개
- 리드 부분집합 전체 63개 (`2^6-1`)
- 정점 단독 3개, 정점 쌍 3개
- 정점 x 리드부분집합 (정점 3 x 63)
- 각 `case_id` 의 t0 hs 4분위별 1개·2개·3개 조합

출력: `rel_err <= 0.01` 인 후보만 한 줄씩
```
CAND kind=<lead|station|station_lead|hs0q> spec=<...> n_rows=<...> b=<...> rel_err=<...>
```
마지막에 `CAND total=<맞은 후보 수>`.

## 작업 3 — 참고 구조표 (판정에 쓰지 마라)

```
BYLEAD lead=<h> n=<...> b=<...>
BYSTATION station=<...> n=<...> b=<...>
```

## 금지

- 제출 CSV 생성·수정 금지. `P3/preds/`·`P3/subs/` 쓰기 금지. 학습 금지
- 로컬 OOF·폴드·`dense_weights.csv` 사용 금지
- 정답이 필요한 통계를 만들지 마라. `y` 는 없다
- 추상화 금지. 클래스·config·로깅 프레임워크 만들지 마라

## 기준선 숫자 (baseline)

| 항목 | 값 |
|---|---|
| `b_full` (1200행) | 0.157493425176 |
| `b_pub` (LB 적합) | 0.119799973975 |
| 비 `b_pub/b_full` | 0.760666 |
| 전 CSV 행수 | 1200 |
| 사례 수 / 리드 수 / 정점 수 | 200 / 6 / 3 |
| SEED | 42 |

## 수락 기준 (직접 실행하고 숫자를 보고하라)

```
python P3/src/bugcheck.py subset
python P3/src/bugcheck.py all
```

**결론 문장 금지, 숫자와 표만.** `RAND verdict` 1줄, `CAND` 맞은 후보 전부와 `total`,
`BYLEAD` 6줄, `BYSTATION` 3줄, 두 명령의 exit code. 그 외 산문 금지.
