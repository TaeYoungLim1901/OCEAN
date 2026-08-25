# T19 — P1 결함 제거 + 제출 재현 장벽 (C1·C3·C4·C8 · audit --pred · check.py final)

실행 파이썬: `python`. 리포 루트에서 실행한다. **P1 트리만 수정한다.**

## 배경 — 재조사 금지 사실 (확정)

- 계획 원본은 `docs/PORT-PLAN-2026-08-25.md` 의 3·4·5a·6a·8a 단계다. 계획을 다시 설계하지 마라.
- 진단기 `P1/src/bugcheck.py` 는 이미 있고 정상 동작한다. `verdict(fails)` 가 exit code 를 낸다.
  **bugcheck 의 검사를 완화·삭제·주석처리하지 마라. 원인을 고쳐라.** 검사 완화는 즉시 실패로 본다.
- purge 규칙은 확정이다: `keep = folds < held_out` **그리고** station-layer 별 `time < validation_start − 87h`.
  대칭 purge(미래 방향까지 남기는 것)가 지금의 결함이다.
- pool NPY 는 **연속 score 만** 담는다. CSV `label` 열만 0/1 을 허용한다.
  `astype(float32)` 로 이진 배열을 위장하는 것도 금지다.
- 사용 가능 라이브러리는 numpy·pandas·scikit-learn·lightgbm 뿐이다.

## 기준선 (baseline. 재측정 금지)

`python P1/src/bugcheck.py all` 현재 출력:

```
C1 FAIL future_rows=1,140,222 mask_differences=1,140,222
C2 PASS full_fit_match=0 fold_reference_differences=634
C3 FAIL config_issues=1 p50_delta=-8.69631e-07 p90_delta=0.00154956 p99_delta=0.00302335 max_delta=1.0036e-06
C4 FAIL arrays=21 invalid=4 binary_uint8=6 key_mismatch=0
C5 PASS   C7 PASS   C9 PASS
C8 FAIL recorded_files=0 checked=0
판정: STOP
```

기준 성적: forward inner F1 `0.697836`, 봉인 outer fold3 F1 `0.815757`, public LB F1 `0.717435`,
최고 public 은 exp010 F1 `0.741274`. **이 사양은 점수를 올리는 작업이 아니다. 결함 제거만 한다.**

## 작업

### T19-1. C1 — forward-only purge 수정

`P1/src/features.py::train_mask_with_purge` 를 고친다.

- 마스크는 `keep = (folds < held_out)` 을 먼저 적용하고, 그 위에 station-layer 별
  `time < validation_start - 87h` 만 남긴다.
- 함수 안에 `assert` 를 넣는다: `(keep & (folds >= held_out)).sum() == 0`.
- 호출부가 여러 개면 전부 이 함수를 지나가게 한다. 호출부마다 가드를 복사하지 마라.
- **오염 cache 격리**: 기존 `P1/cache/` 의 exp001·exp004 계열 산출물을 수정판 학습이 읽지 않게 한다.
  새 학습은 새 experiment ID 경로를 쓴다. 구 cache 를 참조하면 시작 전 exit 1 이 되게 한다.

### T19-2. C3 — `config_issues=1` 제거

bugcheck C3 가 무엇을 config issue 로 세는지 코드에서 확인하고 그 원인을 고친다.
`p50/p90/p99/max_delta` 수치는 이 사양의 판정 대상이 아니다.

### T19-3. C4 — 연속 score writer + 이진 산출물 격리

- `P1/src/submission_tools.py` 에 `save_pool_pred(path, values)` 를 만든다.
  저장 전 검사: floating dtype · 전부 유한 · `[0,1]` 범위 · 고유값 3개 이상.
  **위반 시 저장하지 않고 예외를 던진다.** 통과 시 float32 로 저장한다.
- `P1/preds/` 에 쓰는 **모든** writer 가 이 함수를 지나가게 바꾼다.
- 현재 `invalid=4 binary_uint8=6` 인 기존 배열 처리:
  - 연속 raw score 에서 재생성 가능한 것만 재생성한다. **재학습은 하지 마라.**
  - 재생성 불가한 것은 sha256 을 기록한 뒤 `P1/preds/quarantine/` 으로 옮긴다.
    격리된 파일은 앙상블·제출 경로에서 읽히지 않아야 한다.
  - 격리 사실을 `P1/graph.json` 에 `f.` 노드 1개로 남긴다 (파일명·sha256·사유).
    `scripts/g.py` 인터페이스로만 갱신하고 JSON 을 손으로 편집하지 마라.

### T19-4. C8 — 데이터 파일 해시 대장

bugcheck C8 이 읽는 대장을 채운다. 원본 데이터(`P1/P1_data/`)는 **읽기 전용**이며 sha256 계산만 한다.
대장 생성 명령을 `P1/src/check.py` 또는 `submission_tools.py` 의 서브커맨드로 노출하고
`P1/AGENTS.md` 에 그 명령 한 줄을 고정한다.

### T19-5. `audit --pred` 필수화 (fail-closed)

`P1/src/submission_tools.py` 의 `audit` 에서 `--pred` 를 **argparse 와 함수 API 양쪽** 모두 필수로 만든다.
argparse 는 `required=True`, 함수는 기본값 없는 위치 인자로 바꾼다. `None` 을 받으면 예외를 던진다.

### T19-6. `check.py final` — 빈 폴더 재현 + manifest

`python P1/src/check.py final <제출csv>` 를 만든다.

1. 빈 임시 디렉터리를 만든다 (`tempfile.mkdtemp`).
2. **제출 코드 + 학습 가중치만으로** 추론을 재실행한다. 학습을 다시 하지 않는다.
   기존 `P1/cache/` 의 중간 산출물을 읽으면 실패로 본다.
3. 재현한 CSV 의 sha256 이 인자로 받은 CSV 의 sha256 과 일치하는지 본다.
4. 통과 시 manifest 를 `P1/subs/<이름>.manifest.json` 으로 쓴다. **스키마는 아래가 계약이다** (P2·P3 공통):

```json
{"csv": "P1/subs/x.csv", "csv_sha256": "...", "pred_npy": "P1/preds/test_x.npy",
 "pred_sha256": "...", "code_files": {"P1/src/train.py": "..."},
 "weights": {"P1/cache/x/model.txt": "..."}, "seed": 42, "created": "ISO8601"}
```

sha256 은 대문자 hex. 마지막 줄은 `판정: OK` / `판정: STOP`, exit code 0 / 1.
`check.py` 의 기존 서브커맨드(`sub`·`pipeline`·`reproduce`)는 그대로 둔다.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- **P1 트리 밖(`scripts/`·`AGENTS.md`·`RULES.md`·`P2/`·`P3/`)을 수정하지 마라.**
- 원본 데이터 `P1/P1_data/` 는 읽기 전용. 해시 계산 외 쓰기 금지.
- **긴 재학습을 실행하지 마라.** 재학습이 필요한 항목은 스크립트만 만들고 실행은 사람에게 넘긴다.
- 과한 추상화 금지. 구현 1개짜리 인터페이스·설정파일·로깅 프레임워크를 만들지 마라.
- 점수를 올리려고 피처·파라미터·분할·모델을 바꾸지 마라. 사양 밖 변경 금지.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
python P1/src/bugcheck.py all ; echo "exit=$?"          # C1~C9 전부 PASS, 판정: OK, exit=0
python P1/src/submission_tools.py selftest ; echo "exit=$?"
python P1/src/submission_tools.py audit P1/subs/exp002.csv ; echo "exit=$?"   # --pred 없음 → exit 2
python P1/src/check.py final P1/subs/exp002.csv ; echo "exit=$?"
python scripts/g.py --p P1 selftest ; echo "exit=$?"
```

C1 은 `future_rows=0 mask_differences=0`, C4 는 `invalid=0 binary_uint8=0` 이어야 한다.
`check.py final` 이 실패하면 **검사를 완화하지 말고** 실패 원인을 그대로 보고하라.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `검사 | before | after`. 표 2: `파일 | +줄 | -줄`. 격리 파일이 있으면 `파일 | sha256 앞 8자 | 사유` 표.
산문 금지.
