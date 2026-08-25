# R4 — sol 적대검토 결함 6건 수정 + 규정 복사본 제거

실행 파이썬: `python`. 리포 루트에서 실행한다.

## 배경 — 재조사 금지 사실 (확정)

2026-08-26 luna 산출물을 `gpt-5.6-sol` 이 읽기 전용으로 적대검토했다 (`P1/experiments/logs/sol_R3.log`).
판정: A·B·D·E·G·I·J 결함, C·F·H 는 OK. **C·F·H 를 다시 건드리지 마라.**

**이미 사람이 처리한 것 (다시 하지 마라):**

- `scripts/cx.sh` 를 LF 로 정규화했다. `bash -n` 통과, 재귀 가드 `exit 2` 확인.
- **P3 제출물을 복구했다.** `P3/preds/_backup_pre_maskfix/` 의 golden 을 원위치에 복사했다.
  현재 `P3/subs/submission_p3_t1f.csv` = `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1`
  (41,055바이트), `P3/preds/test_final_p3_t1f.npy` = float64 (200,6).
  **이 두 파일을 다시 쓰지 마라.** 현재 `P3/subs/submission_p3_t1f.manifest.json` 은 손상본 해시를 담고 있어 갱신이 필요하다.

**현재 게이트 상태 (기준선. 재측정 금지):**

```
P1 bugcheck all  판정: OK exit=0   (C1 0, C4 invalid 0 binary 0, C8 6/6)
P2 bugcheck all  판정: OK exit=0   (C2 0/0, C7 0.517441, C8 4/4)
P3 bugcheck all  판정: OK exit=0   (C4 unmapped=2, C8 8/8)
g.py selftest ×3 exit=0 / submission_tools selftest ×3 exit=0
preflight.py rules  판정: STOP — P1\graph.json:234,240,264 / P3\graph.json:34,585
```

제출물 기준 해시: P1 `4C96D5F3…E833DCB`, P2 `F88AC4BF…B854D2AA9`, P3 `2E498428…3470C1`.
**이 세 파일의 해시가 작업 후에도 그대로여야 한다.**

## 작업 — sol 지시 순위대로

### R4-1. P3 `check.py final` — 임시 출력만, CSV·NPY 양쪽 비교

`P3/src/check.py:189-203`. 현재 생성 NPY 를 원본과 비교하지 않고 원본 NPY 해시로 manifest 를 만든다.
- 재현 산출물은 **전부 `tempfile.mkdtemp()` 안에만** 쓴다.
- 재현 CSV 와 재현 NPY 를 **각각** golden(`P3/subs/…csv`, `P3/preds/…npy`)과 바이트 비교한다.
  NPY 는 dtype 까지 비교한다 (float32 로 재작성되어 사고가 났다).
- 둘 다 일치할 때만 manifest 를 갱신한다. manifest 외에는 원본 폴더에 아무것도 쓰지 마라.

### R4-2. P2 `check.py final` — 격리 실행

`P2/src/check.py:113-125,141`. 현재 격리 없이 원본 예측 폴더에 저장한다.
- 코드·가중치를 임시 트리로 **복사**한 뒤 그 안에서 `subprocess` 로 추론을 돌린다.
- 원본 예측 폴더에 쓰는 writer 를 제거한다.
- manifest 의 `pred_npy` 를 재현용 float32 mirror 가 아니라 **실제 pool 배열
  `P2/preds/test_fixed_depth_E002.npy`** 로 바꾼다. `pred_sha256` 도 그 파일 기준으로 다시 계산한다.

### R4-3. `preflight.py final` — 사후 재검사 + 빈 manifest 차단

`scripts/preflight.py:182-220,238-249`.
- `check.py final` 실행 **후에** manifest 를 다시 읽어 csv/pred/code/weights 해시를 재검사한다.
- `code_files` 또는 `weights` 가 비어 있으면 FAIL 이다.
- 하위 프로세스는 exit code 로 판정한다. stdout 문자열 매칭으로 판정하지 마라.

### R4-4. bugcheck 검사 강화 (완화 금지)

- `P3/src/bugcheck.py` C4: `unmapped` 를 **실패에 포함**한다. 현재 `unmapped=2` 라 C4 가 FAIL 로 바뀔 것이다.
  그 2건이 무엇인지 찾아 원인을 고쳐서 `unmapped=0` 으로 만들어라. **검사에서 빼지 마라.**
- 세 문제 C8: 기록된 키 집합과 실제 파일 집합의 불일치를 **실패로 센다.**
  기록 누락·기록에만 있고 파일 없음 둘 다 실패다.

### R4-5. `save_pool_pred` 우회 writer 제거

- `P1/src/train.py:89` 의 중간 `np.save` 를 없앤다.
- `P3/src/blend_t7.py:180,287`, `P3/src/train.py:171-173`, `P3/src/train_dense.py:148` 을 포함한
  P3 의 pool writer 13개를 전부 `save_pool_pred` 로 통일한다.
- 불완전 OOF(전 구간이 채워지지 않는 배열)는 예측 폴더가 아니라 `P3/cache/` 로 옮긴다.
- **`P1/preds/`·`P2/preds/`·`P3/preds/` 의 기존 파일 내용을 다시 쓰지 마라.** 코드 경로만 바꾼다.

### R4-6. `preflight.py selftest` — 줄끝 검사

`scripts/*.sh` 에 `\r\n` 또는 단독 `\r` 이 있으면 FAIL. luna 가 `cx.sh` 를 CRLF 로 저장해 실제로 깨뜨렸다.

### R4-7. 규정 복사본 제거

`preflight.py rules` 가 잡은 5건을 없앤다: `P1/graph.json:234,240,264`, `P3/graph.json:34,585`.
마감일·제출 횟수 수치를 지우고 `RULES.md §0 참조` 로 바꾼다.
**`scripts/g.py` 인터페이스로만 갱신하고 JSON 을 손으로 편집하지 마라.**

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- **bugcheck 검사를 완화·삭제·주석처리하지 마라.** 원인을 고쳐라.
- **제출 CSV 3개와 그 짝 NPY 를 다시 쓰지 마라.** 해시가 바뀌면 즉시 실패로 본다.
- **모든 파일을 LF 줄끝으로 저장하라.** CRLF 로 저장하면 셸 스크립트가 깨진다.
- **긴 재학습을 실행하지 마라.** 필요하면 스크립트만 만들고 실행은 사람에게 넘긴다.
- 사양 밖의 피처·파라미터·분할·모델 변경 금지. 과한 추상화 금지.
- 원본 데이터 폴더는 읽기 전용. 해시 계산 외 쓰기 금지.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
python -c "import hashlib,pathlib;[print(p,hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest().upper()[:10]) for p in ['P1/subs/exp002.csv','P2/subs/E002_final.csv','P3/subs/submission_p3_t1f.csv']]"
python P1/src/bugcheck.py all ; echo "exit=$?"
python P2/src/bugcheck.py all ; echo "exit=$?"
python P3/src/bugcheck.py all ; echo "exit=$?"
python scripts/preflight.py selftest ; echo "exit=$?"
python scripts/preflight.py rules ; echo "exit=$?"
python P1/src/check.py final P1/subs/exp002.csv ; echo "exit=$?"
python P2/src/check.py final P2/subs/E002_final.csv ; echo "exit=$?"
python P3/src/check.py final P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
python scripts/preflight.py final --p1 P1/subs/exp002.csv --p2 P2/subs/E002_final.csv --p3 P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
```

제출 해시 3개는 `4C96D5F38B`·`F88AC4BF12`·`2E4984283B` 그대로여야 한다.
`bugcheck all` 3개와 `preflight rules`·`selftest` 는 exit 0 이어야 한다.
`preflight final` 이 `NOT READY` 면 **검사를 완화하지 말고** 실패 목록을 그대로 보고하라.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `R4 항목 | 명령 | exit | 핵심 출력 1줄`.
표 2: `파일 | +줄 | -줄`.
표 3: `제출 CSV | 작업 전 SHA10 | 작업 후 SHA10`.
산문 금지.
