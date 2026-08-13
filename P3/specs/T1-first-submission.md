# 작업 지시: T1 — P3 첫 제출

루트 `AGENTS.md`, `PLAYBOOK.md`, `P3/AGENTS.md`, `P3/P3CONTEXT.md`,
`P3/DOMAIN_RESEARCH.md`를 준수한다. `P3/P3_data/`와 기존 `P3/src/metric.py`는 읽기 전용이다.
새 의존성을 설치하지 않고 `numpy`, `pandas`, `scikit-learn`, `lightgbm`만 사용한다.
`pip install`이 필요하면 멈추고 보고한다.

## 배경 — 이미 검증된 사실, 재조사 금지

- `a.damped_persistence`, `a.lead_shrinkage`, `a.gbdt_direct`, `a.pooled_station`은 open이다.
- `r.context_only`: 피처·정규화·예측은 한 `case_id`의 289행만 사용한다.
- 달력·절대시각·test 전체 통계·다른 test case 결합은 금지한다.

## 구현

- 실험 ID `p3_t1` 전용 자산 경로는 `P3/cache/p3_t1/`, `P3/models/p3_t1/`이다.
- `P3/src/features.py`: `(289, 10) float32` 한 배열에서만 수준/추세/풍속 memory/원형방향/
  missingness 피처를 추출한다. 구조적 10분 파랑 결측을 보간해 새 관측으로 만들지 않는다.
- `P3/src/train.py`: `SEED=42`, 정점별 시간순 5개 block OOF, 각 validation case 전후 78시간의
  학습 후보 purge. station×lead damped persistence와 direct-delta LightGBM을 같은 OOF에서 비교한다.
  LightGBM은 1개 multi-horizon 모델, `n_jobs=2` 이하, 튜닝 금지다.
- `P3/src/inference.py`: 저장 모델로 test를 재현하고 제출 CSV를 만든다. 추론은 case 단위 반복이다.
- 결과와 무관하게 `P3/preds/oof_damped_p3_t1.npy`, `test_damped_p3_t1.npy`,
  `oof_lgbm_p3_t1.npy`, `test_lgbm_p3_t1.npy`를 저장한다.
- 최종 예측도 `P3/preds/oof_final_p3_t1.npy`, `test_final_p3_t1.npy`로 저장한다.

## 수락 기준 — 사전 판정 규칙

- 지속성 0.832899를 먼저 재현한다.
- damped persistence는 지속성보다 통합 RMSE가 낮아야 유지한다.
- LightGBM 혼합 가중치는 damped 기준 `0.00..1.00`을 0.05 간격으로만 비교한다.
- LightGBM을 최종에 넣는 조건: 최적 혼합이 damped보다 RMSE `>=0.003` 개선하고,
  lead, station, hs0-bin 세그먼트의 최대 악화가 `0.05m` 이하이다. 아니면 damped만 제출한다.
- 결과 확인 후 규칙·피처·파라미터를 바꾸지 않는다.

## 검증과 보고

- `python P3/src/check.py cv P3/cache/p3_t1/folds.csv`
- 통합/lead/station/hs0-bin, 지속성 대비 delta, 두 모델 잔차 상관, 혼합 가중치를 출력한다.
- inference 재실행이 저장 test final 및 제출과 완전일치해야 한다.
- `python scripts/preflight.py sub <csv> --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv`
- 결론 문장 금지. 숫자와 표, 경로만 보고한다.
