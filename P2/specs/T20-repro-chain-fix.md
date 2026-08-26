# T20 — run_all.py 체인의 끊긴 연결을 고친다

## 실패 증거 (재조사 금지)

`python P2/src/check.py final P2/subs/E017_antistack_w-2.5.csv` 실행 결과:

```
FAIL final run_all failed rc=1:
FileNotFoundError: [Errno 2] No such file or directory:
  '...\p2_final_p7z88j14\P2\cache\T18\run_0deew1xi\P2\preds\test_e002_E003.npy'
판정: STOP
```

원인: `P2/src/run_all.py` 의 `main()` 이 `train_e002.main` · `train_e003.main` 만 호출하고
`inference_e002.main` · `inference_e003.main` 을 호출하지 않는다.

- `P2/src/train_e002.py:101` → `oof_{name}_{label}_E002.npy` 만 저장
- `P2/src/inference_e002.py:33` → `test_{selected}_E002.npy` 저장 — **체인에서 호출되지 않음**
- `P2/src/train_e003.py:303` → `oof_{name}_E003.npy` 만 저장
- `P2/src/inference_e003.py:43` → `test_{name}_E003.npy` 저장 — **체인에서 호출되지 않음**

`gate_e004.py:189-190` · `train_e005.py:253` · `stack_e008.py:291-294` 는
oof 와 test 를 모두 저장하므로 끊긴 곳이 아니다.

## 할 일

### 1. 정적 의존성 표를 먼저 만들어 보고한다

`train_e002.py` · `inference_e002.py` · `train_e003.py` · `inference_e003.py` ·
`gate_e004.py` · `train_e005.py` · `stack_e008.py` 일곱 파일의 **모든** `np.load` 와
`save_pool_pred` 호출을 찾아 다음 표로 보고한다.

| 파일 | 읽는 파일명 패턴 | 쓰는 파일명 패턴 |

파일명이 f-string 이면 변수가 취할 수 있는 값을 코드에서 확인해 실제 파일명으로 펼쳐라.
이 표로 **E002·E003 외에 더 끊긴 연결이 있는지** 판정한다. 있으면 그것도 고친다.

### 2. `P2/src/run_all.py` 수정

단계 순서를 아래로 고친다. 각 단계는 기존 모듈의 `main()` 을 호출한다.
학습·피처 로직을 복사하거나 다시 쓰지 마라.

```
E002_train → E002_infer → E003_train → E003_infer → E004 → E005 → E008 → wmix → mix_sub
```

`inference_e002.py` · `inference_e003.py` 가 `C.ROOT` 를 모듈 임포트 시점에 고정하거나
`sys.argv` 를 읽으면, 기존 `stack_e008.ROOT` 를 다루는 방식과 같은 패턴으로 처리한다.

### 3. 각 단계 뒤 산출물 존재를 확인한다

각 단계가 끝난 직후, 다음 단계가 읽을 파일이 실제로 생겼는지 확인하고
없으면 **그 단계 이름과 빠진 파일명을 넣어 즉시 예외를 던진다.**
20분짜리 체인이 마지막에 죽는 것을 막는 것이 목적이다. 표 1의 의존성 표를 그대로 쓴다.
로깅 프레임워크·설정 클래스를 만들지 마라. `raise FileNotFoundError(f"...")` 한 줄이면 된다.

### 4. 학습을 돌리지 마라

**codex 샌드박스에서 전체 체인을 실행하지 마라.** 20~40분이라 시간 제한에 걸린다.
학습이 필요 없는 검사만 실행한다:

```
python P2/src/run_all.py --help
python -c "import ast,sys; [ast.parse(open(p,encoding='utf-8').read()) for p in sys.argv[1:]]" P2/src/run_all.py
python P2/src/bugcheck.py all
```

전체 실행은 사람이 한다.

## 산출물 경로

| 경로 | 내용 |
|---|---|
| `P2/src/run_all.py` | 수정 |
| `P2/preds/` | 체인이 쓰는 곳. 기존 파일을 덮어쓰지 마라 |
| `P2/models/` | 체인이 가중치를 쓰는 곳 |
| `P2/cache/T18/` | 체인 임시 폴더. 그대로 둔다 |

## 기준선 숫자

| 항목 | 현재값 |
|---|---|
| 재현 대상 sha256 | `c4c16318c17efbfd8f1f84c85d9478855b2e999127e2fec02298c61d9e858159` |
| 재현 대상 행 수 | 26,061 (결측 7,766 / 완전 18,295) |
| 실패 단계 | E003 진입 시 `test_e002_E003.npy` 부재 |
| `run_all.py` 현재 | 101줄, 단계 5개 (E002·E003·E004·E005·E008) |
| E005 단계 실측 소요 | hold 4개에 76 / 74 / 86 / 236 초 |
| `bugcheck.py all` 현재 | C1~C5·C7~C9 전부 PASS — 유지해야 한다 |

## 수락 기준

- 의존성 표에서 모든 `읽는 파일`이 앞선 단계의 `쓰는 파일`에 존재한다. 예외가 있으면 보고해라
- `python P2/src/run_all.py --help` 가 `--out`·`--w` 를 보여준다
- `bugcheck.py all` `판정: OK`
- 신규 파일 금지. `run_all.py` 만 고친다. `inference_*.py` 수정이 불가피하면 최소로 하고 보고해라
- `P2/subs/` 의 csv 를 건드리지 마라. E017 은 이미 제출됐다
- `P2/P2_data/` 읽기 전용
- 새 의존성 금지. `pip install` 이 필요하면 설치하지 말고 즉시 멈추고 보고해라

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 의존성 표, 변경 줄 수, `--help` 출력,
bugcheck PASS/FAIL 개수, 사람이 돌릴 명령 한 줄. 산문 금지.
