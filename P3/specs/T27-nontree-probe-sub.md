# T27 — T21 비트리 블렌드 제출 CSV 생성 (판정기 교정 탐침)

**결론 문장 금지. 숫자와 표만 보고한다.** 산문 요약·권고·다음 단계 제안을 쓰지 마라.

## 배경 · 기준선(baseline) 숫자 — 재조사 금지, 그대로 쓴다

조밀 재가중 판정기는 damped–lgbm 계열에서 public LB 로 검증됐다 (`f.lb_picks_dense_judge`).
**비트리 계열에서는 검증된 적이 없다.** T21 이 만든 arx·wsswell 은 lgbm_t1 과의 잔차 상관이
0.832 / 0.759 로 상관 벽(`f.pool_no_diversity`, 다른 전원 ρ>0.98)을 처음 뚫은 자산이다.

T21 블렌드 가중 (folds1-3 에서만 선택, 봉인 fold4 미사용):

```
persistence 0.00 / damped 0.05 / lgbm_t1 0.25 / arx 0.15 / wsswell 0.55
```

| 조밀 재가중 봉인 fold4 | RMSE |
|---|---|
| 현행 T1F | 0.611959 |
| T21 블렌드 | 0.633913 (+0.021954) |

public ledger 기준선: `submission_p3_t1f.csv` = **0.587165** (상수보정 없음, 비교 대상).

이 제출은 **점수를 올리려는 제출이 아니라 판정기 교정 측정**이다.
상수 편향보정을 섞지 마라 — 교란된다.

## 작업

`P3/src/make_component_subs.py` 와 같은 포장 경로로, 저장된 test 예측을 위 가중으로
선형결합해 제출 CSV 1개를 만든다.

- 입력: `P3/preds/test_*_p3_t21.npy` (arx·wsswell) 및 persistence·damped·lgbm_t1 의
  대응 test 예측. 정확한 파일명은 `P3/preds/` 에서 확인한다.
- 가중은 위 5개 상수 고정. **격자탐색·재적합 금지.**
- `[0, 30]` 클립. 클립된 행 수를 보고한다.
- 산출: `P3/preds/test_blend_t21_p3_t27.npy`, `P3/subs/submission_p3_blend_t21.csv`
- 열 순서 `case_id,station,lead_h,hs_pred`, `P3/P3_data/test_index.csv` 키 정렬 일치

## 수락 기준 (직접 실행하고 출력 숫자를 보고. 판정이 OK 가 아니면 멈추고 숫자만 보고)

```
python P3/src/submission_tools.py audit P3/subs/submission_p3_blend_t21.csv --pred P3/preds/test_blend_t21_p3_t27.npy
python scripts/preflight.py sub P3/subs/submission_p3_blend_t21.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/bugcheck.py all
```

추가 대조군: 기존 제출 CSV 4개의 SHA256 이 **불변**이어야 한다
(`submission_p3_t1f.csv` = `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1`,
`submission_p3_t1f_bias.csv` = `605BA2BB6FA70CC5811F1ACAEA3F200543C929E37BDF632E7FDAFEF389E3BA1C`).
달라지면 고치지 말고 숫자를 보고하고 멈춘다.

## 보고 표 2개만

1. `멤버 | 가중 | test npy 파일명 | 예측 평균`
2. `검증 명령 | 판정 | 핵심 숫자(SHA256·max delta·클립행수)`

## 환경

실행 파이썬은 `python`. 새 의존성 금지. `pip install` 이 필요해지면 **설치하지 말고 즉시 정지**하고
무엇이 없는지만 보고한다.

## 금지

- 학습, 가중 재적합, 격자탐색, 새 피처
- 상수 편향보정 적용
- 기존 제출 CSV·npy 수정
- `train_target_transform.py` 등 T26 산출물 건드리기
- 추상화·설정 클래스. `print` 로 충분
- `cx.cmd`·`codex exec` 재귀 호출
- 결론·권고 문장
