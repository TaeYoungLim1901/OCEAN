# T29 — lr 타깃변환 제출 CSV + 재현 러너

**결론 문장 금지. 숫자와 표만 보고한다.** 권고·다음 단계·요약 산문을 쓰지 마라.

**`P3/experiments/log.md` 와 `graph.json` 을 수정하지 마라.** 기록은 Opus 가 한다.

## 배경 · 기준선(baseline) 숫자 — 재조사 금지, 그대로 쓴다

`train_target_transform.py` 4폴드 실행이 끝났고 T28 채점이 끝났다. 재학습·재채점 금지.

| 조밀 재가중 봉인 fold4 | RMSE |
|---|---|
| 현행 T1F | 0.611959 |
| `lr` 단독 | 0.598926 |
| `ls` 단독 | 0.603982 |
| `lgbm_t1` | 0.614009 |

public ledger 최고 = `submission_p3_t1f_bias.csv` 0.585693.
`submission_p3_t1f.csv` = 0.587165 (상수보정 없음).

## 작업 2개

### A. 제출 CSV 2개

저장된 `P3/preds/test_lr_p3_t26.npy` · `P3/preds/test_ls_p3_t26.npy` 를 그대로 포장한다.
**재학습·재가중·상수 편향보정 금지.** 순수 단독 예측이어야 LB 델타가 해석 가능하다.

- `P3/subs/submission_p3_lr.csv`
- `P3/subs/submission_p3_ls.csv`
- 열 순서 `case_id,station,lead_h,hs_pred`, `P3/P3_data/test_index.csv` 키 정렬 일치
- `[0, 30]` 클립, 클립행수 보고

### B. 재현 러너 등록 (9/7 모델 제출 요건)

`P3/src/check.py` 의 runner 매핑에 `submission_p3_lr` 을 추가한다. `submission_p3_t1f_bias`
때와 같은 형태다 (dict 한 줄 + 추론 스크립트 1개). `train_target_transform.py` 가 학습부터
`test_lr_p3_t26.npy` 생성까지 하므로 추론 스크립트는 그것을 호출하고 CSV 를 쓰면 된다.
빈 폴더 재현이므로 필요한 코드 파일이 복사 목록에 들어가야 한다.

`ls` 러너는 만들지 마라. 제출 후보 확정 전이다.

## 수락 기준 (직접 실행하고 출력 숫자를 보고. 판정이 OK 가 아니면 멈추고 숫자만 보고)

```
python P3/src/submission_tools.py audit P3/subs/submission_p3_lr.csv --pred P3/preds/test_lr_p3_t26.npy
python P3/src/submission_tools.py audit P3/subs/submission_p3_ls.csv --pred P3/preds/test_ls_p3_t26.npy
python scripts/preflight.py sub P3/subs/submission_p3_lr.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python scripts/preflight.py sub P3/subs/submission_p3_ls.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/check.py final P3/subs/submission_p3_lr.csv
python P3/src/bugcheck.py all
```

`check.py final` 이 학습을 다시 돌리므로 시간이 걸린다. `--n-jobs` 는 2 를 넘기지 마라.

추가 대조군 — 기존 제출 CSV SHA256 **불변**:
`submission_p3_t1f.csv` = `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1`,
`submission_p3_t1f_bias.csv` = `605BA2BB6FA70CC5811F1ACAEA3F200543C929E37BDF632E7FDAFEF389E3BA1C`,
`submission_p3_blend_t21.csv` = `77C4F71C19FBCD8357C3389D2C91DB94A7EDBDA430FB664362BC9B0CADA43BAB`.
달라지면 고치지 말고 숫자를 보고하고 멈춘다.

## 보고 표 2개만

1. `CSV | 행수 | hs 평균 | 클립행수 | SHA256 | npy max delta`
2. `검증 명령 | 판정 | 핵심 숫자`

## 환경

실행 파이썬은 `python`. 새 의존성 금지. `pip install` 이 필요해지면 **설치하지 말고 즉시 정지**하고
무엇이 없는지만 보고한다.

## 금지

- 재학습(단 `check.py final` 의 재현 실행은 예외), 하이퍼파라미터·피처·폴드 변경
- 가중 적합, 격자탐색, 상수 편향보정 적용
- 기존 제출 CSV·npy 수정
- `log.md`·`graph.json` 수정
- 추상화·설정 클래스. `print` 로 충분
- `cx.cmd`·`codex exec` 재귀 호출
- 결론·권고 문장
