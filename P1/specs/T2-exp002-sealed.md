# T2 — exp002 sealed outer

## 배경

exp001 모델 학습은 완료됐으나 후보 선택에 4개 fold를 모두 썼다. 새 학습과 `pip install` 없이 저장 모델을 재사용하여 folds 0~2에서만 후보를 고르고 fold3를 봉인 outer로 평가한다. baseline은 공식 F1 0.548255다.

## 구현

Python으로 stream마다 절대 유일한 gap segment ID를 만든다. centered raw-context는 `P1CONTEXT`가 전체 test 시계열의 offline 행 판정을 주고 인과 금지를 명시하지 않아 허용하되 label/test 집계는 사용하지 않는다. exp002 모델·OOF·test·submission을 별도 저장하고 저장 모델에서 exact 재생성한다. `SEED=42`, `n_jobs<=2`.

## 수락 기준

- inner(fold 0~2)에서만 weight/threshold/hole 선택 후 fold3 outer 1회 계산
- `python P1/src/check.py reproduce exp002` → `판정: OK`
- `python P1/src/check.py sub P1/subs/exp002.csv` → `판정: OK`

## 완료 보고

outer F1, 선택 파라미터, exp001 참고치 대비, 유형/station 약점, 파일 해시를 숫자와 표만 보고한다. 결론 문장 금지.
