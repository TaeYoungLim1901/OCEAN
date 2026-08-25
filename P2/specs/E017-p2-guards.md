# E017 — P2 결함 제거 + 제출 재현 장벽 (C2·C3·C7·C8 · audit --pred · check.py final)

실행 파이썬: `python`. 리포 루트에서 실행한다. **P2 트리만 수정한다.**

## 배경 — 재조사 금지 사실 (확정)

- 계획 원본은 `docs/PORT-PLAN-2026-08-25.md` 의 5b·6b·8b 단계다. 계획을 다시 설계하지 마라.
- 진단기 `P2/src/bugcheck.py` 는 이미 있고 정상 동작한다.
  **bugcheck 의 검사를 완화·삭제·주석처리하지 마라. 원인을 고쳐라.** 검사 완화는 즉시 실패로 본다.
- P2 판정기는 `2024_sepoct` hold 다. 4-hold 균등 평균을 판정에 쓰지 마라.
- adversarial AUC 가 `0.992764~1.000` 이라 모든 로컬 CV 는 판정기가 아니라 탐색기다.
- pool NPY 는 연속값만 담는다. `astype(float32)` 로 이진·상수 배열을 위장하지 마라.
- 사용 가능 라이브러리는 numpy·pandas·scikit-learn·lightgbm 뿐이다.

## 기준선 (baseline. 재측정 금지)

`python P2/src/bugcheck.py all` 현재 출력:

```
C1 PASS train_rows=140,776 hold_rows=26,273 overlap=0 boundary_errors=0
C2 FAIL full_observation_fit=1 global_time_features=1
C3 FAIL config_issues=1 paired_arrays=53 p50_median=23.7355
C4 PASS   C5 PASS   C9 PASS
C7 FAIL models=3 trees_min=500 expected=500 nonzero_models=3 pred_std=3.5784 score=0.000000
C8 FAIL data_files=4 recorded=0
판정: STOP
```

현재 제출 후보는 `P2/subs/E002_final.csv` (2024 RMSE `0.517441`).
**이 사양은 점수를 올리는 작업이 아니다. 결함 제거만 한다.**

## 작업

### E017-1. C2 — 전 관측 fit · 전역 시간 피처 누수 제거

`full_observation_fit=1` 과 `global_time_features=1` 이 무엇을 세는지 `bugcheck.py` 에서 확인한다.
- 스케일러·인코더·결측 통계 등 **모든 fit 은 train fold 안에서만** 한다. hold/test 를 포함해 fit 하지 마라.
- 전역 시계열 통계로 만든 시간 피처는 fold-local 로 다시 계산하거나 제거한다.
- 수정 후 `E002` 계열 예측값이 바뀌면 **바뀐 사실과 새 2024_sepoct RMSE 를 보고만 하고**
  기존 `P2/subs/E002_final.csv` 를 덮어쓰지 마라. 새 예측은 새 이름으로 저장한다.

### E017-2. C3 — `config_issues=1` 제거

bugcheck C3 가 무엇을 config issue 로 세는지 코드에서 확인하고 원인을 고친다.
`p50_median` 수치는 이 사양의 판정 대상이 아니다.

### E017-3. C7 — `score=0.000000` 원인 제거

모델 3개가 trees·nonzero 는 정상인데 재현 score 가 0 이다. 채점 경로의 결함이다.
키 정렬·인덱스 정합·타깃 컬럼·역변환 중 어디가 끊겼는지 찾아 고친다.
**점수 임계를 낮추거나 검사를 지우는 방식은 금지다.**

### E017-4. C8 — 데이터 파일 해시 대장

`data_files=4 recorded=0` 을 채운다. 원본 데이터는 **읽기 전용**이며 sha256 계산만 한다.
대장 생성 명령을 `P2/src/check.py` 또는 `submission_tools.py` 서브커맨드로 노출하고
`P2/AGENTS.md` 에 명령 한 줄을 고정한다.

### E017-5. `save_pool_pred` + `audit --pred` 필수화 (fail-closed)

- `P2/src/submission_tools.py` 에 `save_pool_pred(path, values)` 를 만든다.
  저장 전 검사: floating dtype · 전부 유한 · 고유값 3개 이상. 위반 시 **저장하지 않고 예외**.
  통과 시 float32 저장. `P2/preds/` 에 쓰는 모든 writer 가 이 함수를 지나가게 한다.
  (P2 는 회귀 문제이므로 `[0,1]` 범위 검사는 넣지 마라.)
- `audit` 의 `--pred` 를 **argparse 와 함수 API 양쪽** 모두 필수로 만든다.
  argparse 는 `required=True`, 함수는 기본값 없는 인자. `None` 이면 예외.

### E017-6. `check.py final` — 빈 폴더 재현 + manifest

`python P2/src/check.py final <제출csv>` 를 만든다.

1. 빈 임시 디렉터리(`tempfile.mkdtemp`)에서 **제출 코드 + 학습 가중치만으로** 추론을 재실행한다.
   학습을 다시 하지 않는다. 기존 `P2/cache/` 중간 산출물을 읽으면 실패로 본다.
2. 재현 CSV 의 sha256 이 인자 CSV 의 sha256 과 일치하는지 본다.
3. 통과 시 manifest 를 `P2/subs/<이름>.manifest.json` 으로 쓴다. **스키마는 아래가 계약이다** (P1·P3 공통):

```json
{"csv": "P2/subs/x.csv", "csv_sha256": "...", "pred_npy": "P2/preds/test_x.npy",
 "pred_sha256": "...", "code_files": {"P2/src/train.py": "..."},
 "weights": {"P2/cache/x/model.txt": "..."}, "seed": 42, "created": "ISO8601"}
```

sha256 은 대문자 hex. 마지막 줄 `판정: OK` / `판정: STOP`, exit 0 / 1.
`check.py` 의 기존 `sub` 서브커맨드는 그대로 둔다.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- **P2 트리 밖(`scripts/`·루트 `AGENTS.md`·`RULES.md`·`P1/`·`P3/`)을 수정하지 마라.**
- 원본 데이터는 읽기 전용. 해시 계산 외 쓰기 금지.
- **긴 재학습을 실행하지 마라.** 필요하면 스크립트만 만들고 실행은 사람에게 넘긴다.
- 과한 추상화 금지. 구현 1개짜리 인터페이스·설정파일·로깅 프레임워크를 만들지 마라.
- 점수를 올리려고 피처·파라미터·분할·모델을 바꾸지 마라. 사양 밖 변경 금지.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
python P2/src/bugcheck.py all ; echo "exit=$?"          # C1~C9 전부 PASS, 판정: OK, exit=0
python P2/src/submission_tools.py selftest ; echo "exit=$?"
python P2/src/submission_tools.py audit P2/subs/E002_final.csv ; echo "exit=$?"   # --pred 없음 → exit 2
python P2/src/check.py final P2/subs/E002_final.csv ; echo "exit=$?"
python scripts/g.py --p P2 selftest ; echo "exit=$?"
```

C7 은 `score` 가 0 이 아니어야 한다. `check.py final` 이 실패하면 **검사를 완화하지 말고** 원인을 보고하라.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `검사 | before | after`. 표 2: `파일 | +줄 | -줄`.
C2 수정으로 예측값이 바뀌었으면 `배열 | 이전 2024_sepoct RMSE | 이후 RMSE` 표를 추가한다. 산문 금지.
