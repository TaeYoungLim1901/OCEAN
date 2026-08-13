# 작업 지시: T1F — outer fold 혼합 검증

루트 `AGENTS.md`, `PLAYBOOK.md`, `P3/AGENTS.md`를 준수한다. 기존 T1 모델·OOF·test·제출은
읽기 전용으로 보존한다. 새 학습과 `pip install`은 금지하며 필요하면 멈추고 보고한다.

## 배경 — 이미 검증된 사실, 재조사 금지

- T1의 purged 5-fold OOF와 damped/LightGBM test 예측이 저장돼 있다.
- 기존 all-OOF 선택은 같은 272사례에서 가중치를 골라 낙관 위험이 있다.
- `a.damped_persistence`, `a.gbdt_direct`, `a.ensemble` 범위만 사용한다.

## 수락 기준과 결과 전 고정 규칙

1. fold 0~3만 selection set으로 사용한다. fold 4는 선택·튜닝·tie-break에 절대 사용하지 않는다.
2. `pred(w)=(1-w)*damped+w*lgbm`, `w=0.00,0.05,...,1.00` 21개를 selection RMSE로 비교한다.
3. selection RMSE 최소 weight를 고른다. 정확한 동률이면 더 작은 `w`를 고른다.
4. 선택 weight를 변경 없이 fold 4에 한 번 적용한다.
5. fold 4 RMSE가 persistence, damped, LightGBM 각각보다 모두 엄격히 낮을 때만 T1F를 채택한다.
   하나라도 못 이기면 기존 T1을 유지하고 T1F 제출을 만들지 않는다.
6. 채택 시 저장 bundle의 damped/LightGBM 모델에서 test 성분을 다시 계산해 기존 component `.npy`와
   최대 절대차 0을 확인한 뒤 `test_final_p3_t1f.npy`, `submission_p3_t1f.csv`를 생성한다.
7. 추론을 재실행해 component→final npy→CSV hash가 완전일치해야 한다.

## 검증

- selection/all-OOF/outer fold의 persistence·damped·LightGBM·선택 blend RMSE를 기록한다.
- `python scripts/preflight.py sub P3/subs/submission_p3_t1f.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv`
- 결론 문장 금지. 숫자와 표, 경로만 보고한다.

