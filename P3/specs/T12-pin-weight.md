# 작업 지시: T12 — 제출 가중 고정으로 원본 제출 재현

루트 `AGENTS.md` 와 `P3/AGENTS.md` 준수. python 은 시스템 `python` 을 쓴다.
`pip install` 이 필요하면 **멈추고 보고하라**. `P3/P3_data/` 와 `P3/src/metric.py` 는 읽기 전용.

**점수 실험이 아니다. 예측값을 바꾸지 않고 재현성만 복구한다.**

## 배경 — 이미 검증된 사실 (재조사 금지)

```
f.t1f_weights           T1F 제출 가중 W_old = persistence 0.00 / damped 0.35 / lgbm_t1 0.65
f.weight_time_unstable  최적 가중은 시간 불안정
f.thinned_judge_inverts 씨닝 53건 판정기는 damped/lgbm 순위조차 못 가린다
```

T11 결과 (측정 완료, 재계산 금지). 조밀 재가중 fold 별 `W_old` / `W_new`:

| fold | W_old | W_new |
|---|---|---|
| 1 | 0.668083 | 0.616779 |
| 2 | 0.630441 | 0.616857 |
| 3 | 0.656407 | 0.666350 |
| 4 | 0.611959 | 0.627234 |

사전등록 규칙(`p3_t11_pre`)은 `W_new` 가 4폴드 중 3폴드 이상 이길 때만 교체였다. 결과 2/4 → **`W_old` 유지 확정.**

이미 확인된 사실:

- `P3/preds/test_damped_p3_t1.npy` 와 `test_lgbm_p3_t1.npy` 는 마스크 수정 전후 SHA-256 이 동일하다
  (전체데이터 적합에는 fold 마스크가 적용되지 않는다)
- 따라서 `W_old` 를 그 둘에 적용하면 원본 `test_final_p3_t1f.npy` 가 그대로 나와야 한다

원본 해시 (판정 기준):

```
P3/preds/test_final_p3_t1f.npy   SHA-256 앞 32: defca2414e39183db46b9fc004b65576
P3/subs/submission_p3_t1f.csv    SHA-256: 2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1
```

## 기준선

현재 `P3/src/select_t1f.py` 는 OOF 격자탐색으로 가중을 고르며, 누수 제거 후 `W_new`(damped 0.65)를
고른다. **모델·피처·fold·하이퍼파라미터는 바꾸지 마라.** 바꾸는 것은 가중 결정 방식 하나뿐이다.

## 작업 — 가중 고정

`P3/src/select_t1f.py` 를 고친다.

- 제출 가중을 **상수로 고정**한다: `persistence 0.00 / damped 0.35 / lgbm_t1 0.65`
- 상수 바로 위에 근거를 2~3줄 주석으로 적는다: 씨닝 53사례 판정기가 damped/lgbm 순위를 역전시키며
  (`f.thinned_judge_inverts`), 조밀 재가중 판정기 4폴드 대조에서 `W_new` 가 2/4 로 사전등록 임계
  3/4 에 미달했다는 것 (`P3/cache/p3_t11/report.json`)
- 격자탐색 코드를 **삭제하지 말고** 진단용으로 남긴다. 탐색 결과 가중과 고정 가중을 둘 다 출력하고,
  둘이 다르면 경고 한 줄을 찍는다. 출력만 하고 예외를 던지지 마라
- 설정 파일·클래스·플래그를 만들지 마라. 상수 하나와 주석이면 된다

수정 후 실행:

```bash
python P3/src/select_t1f.py
```

`P3/preds/test_final_p3_t1f.npy` 와 `P3/subs/submission_p3_t1f.csv` 를 이 실행이 다시 만든다.
**덮어쓰기 전에** 기존 두 파일의 SHA-256 을 기록하고, 실행 후 값과 대조하라.

## 수락 기준 (결과 보기 전 고정)

- `select_t1f.py` 가 예외 없이 종료한다
- 재생성된 `test_final_p3_t1f.npy` SHA-256 이 `defca2414e39183db46b9fc004b65576…` 와 동일
- 재생성된 `submission_p3_t1f.csv` SHA-256 이 `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` 와 동일
- 해시가 다르면 **원본을 복구하고**(`P3/preds/_backup_pre_maskfix/` 에 사본 있음) 차이 크기만 보고하라
- 아래 게이트 전부 `OK`:

```bash
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f.csv --pred P3/preds/test_final_p3_t1f.npy
python P3/src/context_static_check.py
python scripts/preflight.py sub P3/subs/submission_p3_t1f.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/check.py data
python P3/src/check.py pipeline
python scripts/g.py --p P3 selftest
```

## 금지

- 백업 디렉터리 `P3/preds/_backup_pre_maskfix/` 수정·삭제 금지
- 모델·피처·fold 경계·하이퍼파라미터 변경 금지
- 격자탐색 코드 삭제 금지. 가중을 재탐색하거나 다른 값으로 고정하지 마라
- `P3/subs/submission_p3_t1f_r2.csv` 삭제 금지 (앙상블 자산으로 보존)
- 다른 test `case_id` 의 행을 피처·정규화·예측에 사용 금지 (하드 제약, 위반 = 실격)
- 절대시각·달력 파생 금지 (`a.calendar_features` 닫힘)
- **결론 문장 금지. 숫자와 표만.** 코드 전문 붙여넣기 금지

## 완료 보고 형식

```
수정 파일 = ___
select_t1f.py 정상 종료 = 예|아니오
격자탐색이 고른 가중 = ___ / 고정 가중 = 0.00 / 0.35 / 0.65 / 경고 출력 여부 = ___
test_final_p3_t1f.npy SHA-256 = ___ / 원본과 동일 = 예|아니오
submission_p3_t1f.csv SHA-256 = ___ / 원본과 동일 = 예|아니오
해시 불일치 시 최대 절대차 / 평균 절대차 = ___ / ___ / 원본 복구 여부 = ___
게이트 = audit ___ / context_static_check ___ / preflight sub ___ / check data ___ / check pipeline ___ / g.py selftest ___
정지 조건 발생: 없음 | <사유>
```
