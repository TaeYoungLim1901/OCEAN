# 작업 지시: T3 — strict-forward foundation 및 P3 T2 후보

루트 `AGENTS.md`, `PLAYBOOK.md`, `P3/AGENTS.md`, `P3/P3CONTEXT.md`,
`P3/COMPETITION_RESEARCH.md`를 준수한다. 실행은 `python`이며 새 의존성이나 `pip install`이 필요하면
멈추고 보고한다. 원본 `P3/P3_data/`는 읽기 전용이고 모든 출력은 `P3/` 아래에 둔다.

## 배경과 이미 확인된 사실

- `r.context_only`: 피처·정규화·예측은 해당 case의 289행만 사용한다.
- `a.calendar_features`, `a.transductive_stats`는 닫혀 있다. 절대시각, 다른 test case 통계, t0 이후 입력은 금지한다.
- T1F sealed latest fold RMSE는 0.754739이고 persistence 0.783322, damped 0.770684, LGBM 0.784077이다.
- 유사대회는 station/group별·horizon별 GBDT, rolling-origin CV, 이질 모델 OOF 혼합을 지지한다.
- 로컬 CV는 train/test 기간 이동을 완전히 복제하지 못하므로 판정기가 아니라 탐색기다.

## 구현 범위

실험 ID는 `p3_t2`다. `P3/cache/p3_t2/`, `P3/models/p3_t2/`, `P3/preds/`를 사용한다.

1. 78h 평가셋을 정점별 시간순 5 block으로 만든다. fold 1~3은 selection, 최신 fold 4는 outer다.
   각 fold 학습에는 동일 정점에서 `t0 < min(validation t0)-78h`인 사례만 허용한다. fold 4는 선택·튜닝에 쓰지 않는다.
2. 기존 case-local 피처에 여러 window의 파고 에너지·풍속 memory·풍향/파향 원형 정렬을 추가한다.
3. 동일 feature/target으로 다음 후보를 folds 1~4와 full fit에서 모두 만든다.
   - station×lead damped persistence
   - pooled direct-delta LightGBM
   - lead-specific direct-delta LightGBM 6개
   - station expert pooled LightGBM 3개
   - pooled direct-delta ExtraTrees 1개 (`n_jobs=2`)
4. 각 후보 OOF/test를 `P3/preds/oof_*_p3_t2.npy`, `test_*_p3_t2.npy`에 결과와 무관하게 저장한다.
5. selection fold에서만 후보 RMSE, 잔차 상관, 비음수 0.1 grid pair blend를 비교한다. 동률은 단순한 후보와 낮은 최대 segment RMSE 순이다.
6. station/lead/hs0-bin/regime segment를 출력한다. regime은 case 내부 hs 최근 6h 변화가 -0.10m 미만/그 사이/+0.10m 초과로 고정한다.
7. adversarial validation과 feature drift는 진단 전용이며 어떤 예측·정규화·가중치에도 넣지 않는다.
8. `P3/src/context_static_check.py`는 feature 함수가 컨텍스트 인자 하나만 받고 데이터/절대시각을 읽지 않는지, inference가 test-wide 집계를 하지 않는지 정적으로 검사한다.
9. 학습과 추론은 분리한다. 저장 model로 test npy와 CSV를 재현하고 `submission_tools.py audit --pred`를 통과시킨다.

## 수락 기준 — 결과 전 판정 규칙

- folds 1~3에서 RMSE 최소 후보 또는 2-model blend를 하나 고정한 뒤 fold 4를 한 번만 연다.
- T2는 고정 후보의 fold 4 RMSE가 T1F 0.754739보다 최소 0.002 낮고, T1F 대비 lead/station/hs0-bin/regime
  공통 segment의 최대 악화가 0.03m 이하일 때만 만든다.
- 위 조건을 못 채우면 T1F를 유지한다. outer 결과를 본 뒤 feature, 모델, hyperparameter, blend weight를 바꾸지 않는다.
- 모든 후보의 OOF/test는 앙상블 풀에 남긴다. 이 실험은 미래 기간 shift와 persistence의 regime 오류를 고치고,
  결과와 무관하게 `P3/preds/{oof,test}_*_p3_t2.npy`를 남긴다.

## 검증과 보고

- 사전: `python scripts/preflight.py spec P3/specs/T3-foundation-t2.md`
- 학습 전: `python P3/src/check.py data`, `pipeline`, `cases`, `python P3/src/metric.py selftest`
- CV 후: `python P3/src/check.py cv P3/cache/p3_t2/folds.csv`
- 정적 검사: `python P3/src/context_static_check.py`
- 채택 시: `python P3/src/inference_t2.py`와 `python P3/src/submission_tools.py audit <csv> --pred <npy>`
- 제출 전: `python scripts/preflight.py sub <csv> --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv`
- 통합, lead/station/hs0-bin/regime 상위 악화 3개, outer delta, 잔차 상관, drift AUC, 경로·SHA256만 보고한다. 결론 문장 금지.
