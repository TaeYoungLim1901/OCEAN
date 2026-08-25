# R5 — 제출 재현 바이트 일치 (float32 강제 캐스팅 제거)

실행 파이썬: `python`. 리포 루트에서 실행한다.

## 배경 — 재조사 금지 사실 (확정)

`python scripts/preflight.py final --p1 … --p2 … --p3 …` 이 `판정: NOT READY` 다. 원인이 특정됐다.

- P1 `check.py final` **통과**.
- P2: `csv match=True` 인데 `pred_bytes_match=False`.
- P3: `ValueError: reproduced CSV sha256 mismatch`.

golden 산출물 dtype 실측:

| 파일 | dtype | shape |
|---|---|---|
| `P1/preds/test_exp002.npy` | float32 | (169011,) |
| `P2/preds/test_fixed_depth_E002.npy` | **float64** | (26061,) |
| `P3/preds/test_final_p3_t1f.npy` | **float64** | (200, 6) |

**원인은 사양 T19-3/E017-5/T24-4 가 지시한 `save_pool_pred` 의 `astype(np.float32)` 강제 캐스팅이다**
(`P1/src/submission_tools.py:38` 및 P2·P3 동형). P2·P3 의 golden 은 float64 이므로
재현 경로가 float32 로 내려가면 바이트가 절대 일치하지 않는다. 최대 오차 `1.19e-07`.

이미 확인된 사고 (다시 조사하지 마라): 같은 float32 왕복으로 2026-08-26 에
`P3/subs/submission_p3_t1f.csv` 와 `P3/preds/test_final_p3_t1f.npy` 가 손상됐고,
`P3/preds/_backup_pre_maskfix/` 의 golden 으로 복구했다. golden CSV 는 41,055바이트,
서식은 `str(float)` + CRLF 줄끝, SHA `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1` 다.

## 기준선 (baseline. 재측정 금지)

세 문제 `bugcheck all` 전부 `판정: OK` exit 0. `preflight rules`·`selftest` exit 0.
제출 해시: P1 `4C96D5F38B…`, P2 `F88AC4BF12…`, P3 `2E4984283B…`.
**작업 후에도 이 세 해시가 그대로여야 한다.**

## 작업

### R5-1. `save_pool_pred` — 정밀도 보존

세 문제 `submission_tools.py` 의 `save_pool_pred` 를 고친다.

- **`astype(np.float32)` 강제 캐스팅을 제거한다.** 입력 dtype 이 float32 면 float32 로,
  float64 면 float64 로 저장한다.
- 기존 검사는 **전부 유지**한다: floating dtype 여부 · 전부 유한 · 고유값 3개 이상 ·
  P1 만 `[0,1]` 범위. 위반 시 저장하지 않고 예외.
- floating 이 아닌 dtype(정수·bool)은 저장 거부다. **캐스팅으로 통과시키지 마라.**
- 기존에 저장된 배열을 다시 쓰지 마라. 코드 경로만 바꾼다.

### R5-2. P2 재현 NPY 바이트 일치

`P2/src/check.py final` 의 재현 경로가 golden `P2/preds/test_fixed_depth_E002.npy` 와
**dtype·shape·바이트까지** 일치하는 배열을 만들게 한다. 추론 중간에 float32 로 내려가는 지점을 찾아 없앤다.

### R5-3. P3 재현 CSV·NPY 바이트 일치

`P3/src/inference_t1f.py` 와 `P3/src/check.py final` 을 고친다.

- 재현 NPY 가 golden 과 같이 **float64 (200,6)** 이어야 한다.
- 재현 CSV 서식을 golden 과 일치시킨다: 값은 `str(float)` (반올림·`%.6f` 서식 금지),
  줄끝은 **CRLF**, 열은 `case_id,station,lead_h,hs_pred`, 총 41,055바이트.
- 재현물은 `tempfile.mkdtemp()` 안에만 쓴다. **golden 두 파일을 다시 쓰지 마라.**

### R5-4. manifest 갱신

세 문제 manifest 의 `pred_npy`·`pred_sha256` 이 실제 golden 배열을 가리키게 한다.
재현용 미러 배열을 종점으로 삼지 마라.

## 제약

- 새 의존성 금지. **`pip install` 이 필요하면 즉시 멈추고 보고하라.**
- **bugcheck 검사와 `check.py final` 비교 조건을 완화하지 마라.** 허용 오차(tolerance) 비교로 바꾸는 것은 실패다.
  바이트 일치를 그대로 요구한 채 생산 쪽을 고쳐라.
- **제출 CSV 3개와 짝 golden NPY 를 다시 쓰지 마라.** 해시가 바뀌면 즉시 실패로 본다.
- **모든 파일을 LF 줄끝으로 저장하라** (소스 파일 한정. P3 재현 CSV 의 CRLF 는 golden 요건이라 예외다).
- **긴 재학습을 실행하지 마라.** 필요하면 스크립트만 만들고 실행은 사람에게 넘긴다.
- 사양 밖의 피처·파라미터·분할·모델 변경 금지. 과한 추상화 금지.

## 수락 기준 — 직접 실행하고 숫자를 보고한다

```bash
python -c "import hashlib,pathlib;[print(p,hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest().upper()[:10]) for p in ['P1/subs/exp002.csv','P2/subs/E002_final.csv','P3/subs/submission_p3_t1f.csv']]"
python -c "import numpy as np;[print(p,np.load(p).dtype,np.load(p).shape) for p in ['P1/preds/test_exp002.npy','P2/preds/test_fixed_depth_E002.npy','P3/preds/test_final_p3_t1f.npy']]"
python P1/src/check.py final P1/subs/exp002.csv ; echo "exit=$?"
python P2/src/check.py final P2/subs/E002_final.csv ; echo "exit=$?"
python P3/src/check.py final P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
python P1/src/bugcheck.py all ; echo "exit=$?"
python P2/src/bugcheck.py all ; echo "exit=$?"
python P3/src/bugcheck.py all ; echo "exit=$?"
python scripts/preflight.py final --p1 P1/subs/exp002.csv --p2 P2/subs/E002_final.csv --p3 P3/subs/submission_p3_t1f.csv ; echo "exit=$?"
```

`check.py final` 3개가 전부 exit 0 이고 `preflight final` 이 `판정: READY` exit 0 이어야 한다.
제출 해시 3개와 golden NPY dtype(float32 / float64 / float64)이 위 표와 같아야 한다.
lightgbm 이 import 되지 않으면 그 사실을 보고하고 나머지 명령 결과만 낸다.

## 완료 보고 — 결론 문장 금지, 숫자와 표만

표 1: `R5 항목 | 명령 | exit | 핵심 출력 1줄`.
표 2: `파일 | +줄 | -줄`.
표 3: `산출물 | 작업 전 SHA10·dtype | 작업 후 SHA10·dtype`.
산문 금지.
