# 작업 지시: T2 — 제출 감사와 Public 점수 원장

루트/P3 `AGENTS.md`를 준수하고 시스템 `python`을 쓴다. P3만 수정하며 학습·추론·제출값 변경은 금지한다.
새 `pip install`이 필요하면 멈춘다.

## 배경 — 이미 검증된 사실, 재조사 금지

- canonical key는 `P3_data/test_index.csv`, 기준은 `baseline_persistence.csv`다.
- 최종 후보는 `P3/subs/submission_p3_t1f.csv`, 대응 예측은 `P3/preds/test_final_p3_t1f.npy`다.

## 수락 기준

- `submission_tools.py audit`: 열/행/순서/키/값/case-lead 완전성/기준 거리/hash와 optional npy mapping을 검사한다.
- 성공 audit는 hash 중복 없이 JSONL 기록, 실패는 exit 1이다.
- `public add/list`: audit 후 RMSE 점수를 hash 기준 CSV upsert하고 오름차순/delta를 표시한다.
- `selftest`: 임시 루트에서 성공·실패·upsert를 확인하며 실제 원장을 바꾸지 않는다.
- 실제 T1F audit와 기존 preflight가 모두 통과해야 한다. 결론 문장 금지, 숫자와 경로만 보고한다.
