# T24 — P3 결함 제거 + 제출 재현 장벽 (C4·C8 · audit --pred · check.py final)

실행 파이썬: `python`. 리포 루트에서 실행한다. **P3 트리만 수정한다.**

## 배경 — 재조사 금지 사실 (확정)

- 계획 원본은 `docs/PORT-PLAN-2026-08-25.md` 의 5c·6c·8c 단계다. 계획을 다시 설계하지 마라.
- 진단기 `P3/src/bugcheck.py` 는 이미 있다. C4 가 `PermissionError` 로 **검사 자체를 못 돌고 있다.**
  **bugcheck 의 검사를 완화·삭제·주석처리하지 마라. 원인을 고쳐라.** 검사 완화는 즉시 실패로 본다.
- 문제지가 명시한 `r.context_only` 는 하드 제약이다. 위반 = 실격.
- pool NPY 는 연속값만 담는다. `astype(float32)` 로 이진·상수 배열을 위장하지 마라.
- 사용 가능 라이브러리는 numpy·pandas·scikit-learn·lightgbm 뿐이다.
- 계획서가 지적한 그래프 결함: **P3 graph 에 `r.sub_daily` 와 `r.sub_daily3` 가 동시에 살아 있는데도
  `g.py selftest` 가 통과한다.** 중복 규칙 노드가 남으면 안 된다.

## 기준선 (baseline. 재측정 금지)

`python P3/src/bugcheck.py all` 현재 출력:

```
C1 PASS folds=4 train_boundary_violations=0 size_mismatches=0 min_gap_h=79
C2 PASS   C3 PASS   C5 PASS   C9 PASS
C4 FAIL runtime=1 error=PermissionError
C7 PASS models=1 trees_min=500 pred_std=0.40388 final=0.791336 baseline=0.861973
C8 FAIL data_files=8 recorded=0
판정: STOP
```

현재 제출 후보는 `P3/subs/submission_p3_t1f.csv`. public ledger 4점 중 최저 RMSE 는
`t1f 0.587165` 이고 다음 후보는 `P3/subs/submission_p3_t1f_bias.csv` 다.
**이 사양은 점수를 올리는 작업이 아니다. 결함 제거만 한다.**

## 작업

### T24-1. C4 — `PermissionError` 원인 제거

C4 가 어떤 경로를 열다가 `PermissionError` 를 내는지 찾는다. 흔한 원인은 디렉터리를 파일처럼 여는 것,
Windows 에서 열려 있는 파일에 쓰기, `mkdtemp` 정리 실패다.
- 예외를 `try/except` 로 삼키지 마라. 경로 처리를 고쳐서 검사가 끝까지 돌게 한다.
- 검사가 돌고 나서 나오는 실제 FAIL 수치(예: `invalid`·`binary_uint8`·`key_mismatch`)도 0 이 되게 고친다.

### T24-2. C8 — 데이터 파일 해시 대장

`data_files=8 recorded=0` 을 채운다. 원본 데이터는 **읽기 전용**이며 sha256 계산만 한다.
대장 생성 명령을 `P3/src/check.py` 또는 `submission_tools.py` 서브커맨드로 노출하고
`P3/AGENTS.md` 에 명령 한 줄을 고정한다.

### T24-3. 그래프 중복 규칙 노드 정리

`P3/graph.json` 에서 `r.sub_daily` 와 `r.sub_daily3` 중 **하나만 남긴다.**
남길 쪽은 `RULES.md` 하드 제약 표(문제당 하루 3회)와 일치하는 것이다.
`scripts/g.py` 인터페이스로만 갱신하고 JSON 을 손으로 편집하지 마라.
그리고 `g.py selftest` 가 **같은 사안을 가리키는 규칙 노드 중복을 앞으로 잡아내도록** 만든다.
`scripts/g.py` 는 루트 파일이므로 **수정하지 말고, 필요한 검사 내용을 보고서에 제안으로만 적어라.**

### T24-4. `save_pool_pred` + `audit --pred` 필수화 (fail-closed)

- `P3/src/submission_tools.py` 에 `save_pool_pred(path, values)` 를 만든다.
  저장 전 검사: floating dtype · 전부 유한 · 고유값 3개 이상. 위반 시 **저장하지 않고 예외**.
  통과 시 float32 저장. `P3/preds/` 에 쓰는 모든 writer 가 이 함수를 지나가게 한다.
  (P3 는 회귀 문제이므로 `[0,1]` 범위 검사는 넣지 마라.)
- `audit` 의 `--pred` 를 **argparse 와 함수 API 양쪽** 모두 필수로 만든다.
  argparse 는 `required=True`, 함수는 기본값 없는 인자. `None` 이면 예외.

### T24-5. `check.py final` — 빈 폴더 재현 + manifest

`python P3/src/check.py final <제출csv>` 를 만든다.

1. 빈 임시 디렉터리(`tempfile.mkdtemp`)에서 **제출 코드 + 학습 가중치만으로** 추론을 재실행한다.
   학습을 다시 하지 않는다. 기존 `P3/cache/` 중간 산출물을 읽으면 실패로 본다.
2. 재현 CSV 의 sha256 이 인자 CSV 의 sha256 과 일치하는지 본다.
3. 통과 시 manifest 를 `P3/subs/<이름>.manifest.json` 으로 쓴다. **스키마는 아래가 계약이다** (P1·P2 공통):

```json
{"csv": "P3/subs/x.csv", "csv_sha256": "...", "pred_npy": "P3/preds/test_x.npy",
 "pred_sha256": "...", "code_files": {"P3/src/train.py": "..."},
 "weights": {"P3/cache/x/model.txt": "..."}, "seed": 42, "created": "ISO8601"}
```

sha256 은 대문자 hex. 마지막 줄 `판정: OK` / `판정: STOP`, exit 0 / 1.
`check.py` 의 기존 서브커맨드(`cv` 등)는 그대로 둔다.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- **P3 트리 밖(`scripts/`·루트 `AGENTS.md`·`RULES.md`·`P1/`·`P2/`)을 수정하지 마라.**
- 원본 데이터는 읽기 전용. 해시 계산 외 쓰기 금지.
- **긴 재학습을 실행하지 마라.** 필요하면 스크립트만 만들고 실행은 사람에게 넘긴다.
- 과한 추상화 금지. 구현 1개짜리 인터페이스·설정파일·로깅 프레임워크를 만들지 마라.
- 점수를 올리려고 피처·파라미터·분할·모델을 바꾸지 마라. 사양 밖 변경 금지.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
python P3/src/bugcheck.py all ; echo "exit=$?"          # C1~C9 전부 PASS, 판정: OK, exit=0
python P3/src/submission_tools.py selftest ; echo "exit=$?"
python P3/src/submission_tools.py audit P3/subs/submission_p3_t1f.csv ; echo "exit=$?"   # --pred 없음 → exit 2
python P3/src/check.py final P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
python scripts/g.py --p P3 selftest ; echo "exit=$?"
python scripts/g.py --p P3 show r.sub_daily3
```

C4 는 `PermissionError` 없이 수치를 내고 FAIL 이 0 이어야 한다.
`check.py final` 이 실패하면 **검사를 완화하지 말고** 원인을 그대로 보고하라.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `검사 | before | after`. 표 2: `파일 | +줄 | -줄`.
`g.py selftest` 개선 제안은 표 아래 3줄 이내 목록으로. 산문 금지.
