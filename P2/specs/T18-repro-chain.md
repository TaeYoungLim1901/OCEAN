# T18 — 원자료에서 제출 CSV 까지 한 명령으로 재현되는 체인을 만든다

## 왜 (읽고 넘어가라, 재조사 금지)

대회 규정 `P2/P2CONTEXT.md:22-23`:
> 운영진은 제출된 모델이 업로드한 예측 답안을 실제로 재현하는지, 정상적으로 학습되었는지
> 검증합니다. 검증을 통과한 팀에 한해 리더보드 점수가 최종 성적으로 확정됩니다.

재현 실패 = 점수 0. 현재 상태 (2026-08-26 실측):

- `P2/src/check.py final()` 이 `inference_e002.py` + `P2/models/E002.pkl` + `P2/preds/test_fixed_depth_E002.npy`
  하드코딩이다. E002 제출본만 재현한다. 실제 최고 제출 `P2/subs/E017_antistack_w-2.5.csv` 는 재현 경로가 없다
- `gate_e004.py`·`train_e005.py`·`stack_e008.py` 는 `save_pool_pred` 로 예측 npy 만 저장하고
  모델 가중치를 저장하지 않는다. `P2/models/` 에는 `E001.pkl`·`E002.pkl`·`E003.pkl` 뿐이다
- E017 은 사람이 `wmix.py` → `mix_sub.py` 를 손으로 이어붙여 만들었다. 스크립트가 없다

## 재현해야 하는 대상

`P2/subs/E017_antistack_w-2.5.csv`, sha256
`c4c16318c17efbfd8f1f84c85d9478855b2e999127e2fec02298c61d9e858159`, 26,061행.

구성:

| 행 집합 | 개수 | 값 |
|---|---:|---|
| `missing_public` | 7,766 | `P2/preds/test_maskaug_E005.npy` |
| 나머지 | 18,295 | `3.5*test_gate_robust_blend_E004 - 2.5*test_stack_regime_E008` |

`missing_public` 플래그 정의는 `P2/src/mix_sub.py` 안에 이미 있다. 그대로 쓴다.

## 할 일

### 1. `P2/src/run_all.py` 신규 — 유일한 신규 파일이다

원자료 `P2/P2_data/` 만 입력으로 받아 `P2/subs/<out>.csv` 를 만든다. 한 번의 실행으로 끝난다.

```
python P2/src/run_all.py --out P2/subs/REPRO.csv --w -2.5
```

- 기존 스크립트를 **import 해서 재사용**한다. 학습·피처 로직을 복사하거나 다시 쓰지 마라.
  필요하면 각 스크립트에 `main()` 을 노출시키는 최소 리팩터만 한다
- 단계 순서: E002 → E003 → E004 gate → E005 maskaug → E008 stack → 완전행 w 혼합 → csv
- `--w` 기본값 -2.5. 이 값만 바꿔서 다른 후보를 만들 수 있어야 한다
- 각 단계 시작·종료에 `print` 로 경과 초를 찍는다. 로깅 프레임워크 만들지 마라
- `SEED = 42` 고정. 난수 원천이 여러 개면 전부 고정한다

### 2. 가중치 저장

각 학습 단계가 만든 모델을 `P2/models/{단계}.pkl` 로 저장한다 (`E004`·`E005`·`E008`).
저장 형식은 기존 `E002.pkl` 과 같은 방식을 따른다. 이미 존재하면 덮어쓰지 말고 그대로 둔다.

### 3. `P2/src/check.py` 의 `final()` 을 일반화

E002 하드코딩을 지우고, 빈 임시 폴더에 `P2_data` 와 `src/` 전체·`P2/models/` 전체를 복사한 뒤
`run_all.py` 를 실행해 나온 csv 의 sha256 이 인자로 준 csv 와 같은지 본다.
manifest 항목(코드 sha256·가중치 sha256·seed·생성시각)은 지금 형식을 유지하되
`code_files` 를 실제 사용된 `src/*.py` 전체로 넓힌다.

`final()` 의 호출 형태 `python P2/src/check.py final <csv>` 는 바꾸지 마라.

### 4. 학습 실행은 하지 마라

**codex 샌드박스에서 전체 체인을 돌리지 마라.** 시간 제한에 걸린다.
코드를 만들고, 학습이 필요 없는 검사만 실행한다:

```
python P2/src/run_all.py --help
python -c "import ast,sys; [ast.parse(open(p,encoding='utf-8').read()) for p in sys.argv[1:]]" P2/src/run_all.py P2/src/check.py
python P2/src/bugcheck.py all
```

전체 체인 실행은 사람이 한다. 실행 명령 한 줄을 보고에 적어라.

## 산출물 경로

| 경로 | 내용 |
|---|---|
| `P2/src/run_all.py` | 신규 진입점 |
| `P2/preds/` | 각 단계의 oof/test npy — 기존 파일명 규칙 유지, 덮어쓰지 마라 |
| `P2/subs/` | `--out` 으로 지정된 재현 csv 만 새로 쓴다 |
| `P2/models/` | `E004.pkl`·`E005.pkl`·`E008.pkl` 신규 저장 |
| `P2/cache/T18/` | 중간 산출물. 풀에 남기지 마라 |

## 기준선 숫자

| 항목 | 현재값 |
|---|---|
| 재현 대상 sha256 | `c4c16318c17efbfd8f1f84c85d9478855b2e999127e2fec02298c61d9e858159` |
| 행 수 | 26,061 (missing 7,766 / complete 18,295) |
| E017 mean / range | 23.065605 / [14.156363, 29.287231] |
| `P2/models/` 현재 | E001.pkl · E002.pkl · E003.pkl (E004·E005·E008 없음) |
| `bugcheck.py all` 현재 | C1~C5, C7~C9 전부 PASS — 유지해야 한다 |
| E005 학습 실측 소요 | hold 4개에 76 / 74 / 86 / 236 초 |

## 수락 기준

- `python P2/src/run_all.py --help` 가 `--out`·`--w` 를 보여준다
- `bugcheck.py all` `판정: OK`
- 신규 파일은 `P2/src/run_all.py` 하나뿐이다. 다른 파일은 수정만 한다
- `P2/subs/` 의 기존 csv 를 건드리지 마라. E017 은 이미 제출됐다
- `P2/P2_data/` 는 읽기 전용이다
- 새 의존성 금지. `pip install` 이 필요하면 설치하지 말고 즉시 멈추고 보고해라
- 추상화 금지. 인터페이스·config 클래스·플러그인 구조 만들지 마라

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 변경·신규 파일 경로, 각 파일의 추가/삭제 줄 수,
`run_all.py --help` 출력, bugcheck PASS/FAIL 개수, 사람이 돌릴 실행 명령 한 줄. 산문 금지.
