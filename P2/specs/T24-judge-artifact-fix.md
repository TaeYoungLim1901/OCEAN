# T24 — judge_sepoct 산출물 경로 버그를 고치고 A2 포함 스택 후보를 만든다

## 버그 (재조사 금지, 원인 확정됨)

`python P2/src/judge_sepoct.py --write-artifacts` 실행 결과:

```
FileNotFoundError: [Errno 2] No such file or directory:
  'D:\hackathon\OCEAN\P2\preds\test_A0.npy'
```

T23 에서 `_logical_name` 이 OOF 파일명에서 실험번호 접미사를 벗기도록 고쳐졌다.
그런데 산출물 생성 경로는 논리명 `A0` 를 그대로 `test_{논리명}.npy` 에 끼워 넣는다.
실제 파일은 `P2/preds/test_A0_E020.npy` 다.

같은 문제가 `baseB2` 에도 있었고, T19 가 `test_baseB2.npy` 를 **추가로** 써서 우회했다.
T23 이 그 우회 파일을 지우면서 버그가 드러났다.

## 할 일

### 1. test 배열 조회를 논리명 기반으로 고친다

각 후보의 OOF 파일이 어느 실험(`_E<숫자>`)에서 왔는지는 `_candidate_pool` 이 이미 안다.
같은 실험번호의 `test_{논리명}_E<숫자>.npy` 를 찾도록 고친다.
후보별로 실험번호가 여럿이면 **가장 큰 번호**를 쓴다.
해당 test 파일이 없는 후보는 스택 구성에서 조용히 제외하고, **제외된 후보 목록을 출력한다.**

`P2/preds/` 에 우회용 별칭 파일을 새로 만들지 마라. 그게 T19 가 한 실수다.

### 2. 산출물 파일명

`--write-artifacts` 로 만드는 제출 CSV 이름을 `P2/subs/E022_stack_sepoct.csv` 로 바꾼다.
기존 `P2/subs/E013_stack_sepoct.csv` 는 A2 이전에 만들어진 구본이다. **삭제하지 말고 그대로 둬라.**
출력 파일이 이미 있으면 덮어쓰지 말고 `FileExistsError` 로 멈춘다.

### 3. 검증 명령을 직접 실행하고 숫자를 보고

```
python P2/src/judge_sepoct.py
python P2/src/judge_sepoct.py --write-artifacts
python P2/src/submission_tools.py audit P2/subs/E022_stack_sepoct.csv --pred <해당 npy>
python P2/src/check.py sub P2/subs/E022_stack_sepoct.csv
python P2/src/predict_public.py P2/subs/E022_stack_sepoct.csv
python P2/src/bugcheck.py all
```

`--write-artifacts` 는 제출 CSV 와 짝이 되는 예측 npy 도 `P2/preds/` 에 저장해야 한다
(`mix_sub.py --out-npy` 와 같은 방식). 그래야 `audit --pred` 가 가능하다.

## 산출물 경로

| 경로 | 내용 |
|---|---|
| `P2/src/judge_sepoct.py` | 수정. 신규 파일 금지 |
| `P2/subs/E022_stack_sepoct.csv` | 신규 제출 후보 |
| `P2/preds/test_stack_sepoct_E022.npy` | 위 CSV 와 바이트 일치하는 예측 |

`P2/subs/` 의 다른 CSV 를 건드리지 마라. E017 은 이미 제출됐다.

## 기준선 숫자

| 후보 | 2024_sepoct | 결측 | 완전 | S1 | S3 | S4 |
|---|---:|---:|---:|---:|---:|---:|
| maskaug | 0.491145 | 0.947625 | 0.483651 | 0.491145 | 0.340521 | 0.194028 |
| A0 | 0.517441 | 1.135442 | 0.506344 | 0.517441 | 0.412991 | 0.214216 |
| A1 | 0.524409 | 1.169203 | 0.512698 | 0.524409 | 0.421609 | 0.236953 |
| A2 | 0.500639 | 0.986708 | 0.492543 | 0.500639 | 0.511307 | 0.462888 |
| baseB2 | 0.686777 | 1.407025 | 0.674455 | 0.686777 | 0.640976 | 0.306770 |

표2 현재값: `stack_regime` 2024_sepoct 0.478457 · S3 0.346312 · S4 0.192956 · Δ_vs_maskaug −0.005164.
**수정 후에도 이 값들이 6자리까지 같아야 한다. 다르면 멈추고 보고해라.**

- 현재 원장 최소 public: `E017_antistack_w-2.5.csv` 0.522638 (점수 26.775526)
- 점수 = 33.3335 − 12.54789 × RMSE
- test 행 수 26,061 (층2 8,713 · 층3 8,712 · 층4 8,636)

## 수락 기준

- `judge_sepoct.py` 가 인자 없이도, `--write-artifacts` 로도 예외 없이 끝난다
- 표2 값이 위 기준선과 6자리 일치
- `audit` `npy_exact=True`, `check.py sub` `판정: OK`, `bugcheck all` `판정: OK`
- `predict_public.py` 출력 전문을 보고에 넣어라
- 수정 파일은 `judge_sepoct.py` 하나. 신규 소스 파일 금지
- `P2/P2_data/` 읽기 전용
- 새 의존성 금지. `pip install` 이 필요하면 설치하지 말고 즉시 멈추고 보고해라
- 학습 금지. LightGBM 을 import 하지 마라

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 표1·표2 전문, 제외된 후보 목록, audit·check·bugcheck 결과,
`predict_public.py` 출력 전문, 수정 줄 수. 산문 금지.
