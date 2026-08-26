# T17 — mix_sub.py 가 혼합 npy 를 저장하도록 고치고, 오염된 풀 자산 2개를 재생성한다

## 배경 (읽고 넘어가라, 재조사 금지)

- `P2/src/wmix.py` 는 델타를 26,061행 전부에 적용한다. 마스크가 없다.
- 완전행만 바꾸는 제출은 `P2/src/mix_sub.py --missing/--complete` 로 조립한다.
- `mix_sub.py` 는 csv 만 쓰고 혼합 결과 npy 를 저장하지 않는다. 그래서 제출 csv 에 대응하는
  풀 예측 npy 가 존재하지 않고, 사람이 wmix 의 무마스크 중간 산출물을 풀에 남겼다.
- 실측 (2026-08-26): `P2/preds/test_antistack_w-1_E014.npy` 와 제출본
  `P2/subs/E014_antistack_w-1.csv` 가 12,215행 다르고 max|Δ| = 3.39015 다.
  csv 쪽이 정답이다 (E009 대비 18,295 완전행에서만 다름 — 검증됨).

## 할 일

### 1. `P2/src/mix_sub.py` 수정

`--out` csv 를 쓸 때 혼합 벡터 `pred` 를 npy 로도 저장한다.
저장 경로는 `--out-npy` 인자로 받되, 주지 않으면 저장하지 않고 **경고를 출력하지 않는다**.
기존 호출 시그니처를 깨지 마라. 저장은 `submission_tools.save_pool_pred` 를 쓴다
(`wmix.py` 가 쓰는 것과 같은 함수). 파일이 이미 있으면 `FileExistsError` 로 멈춘다.

### 2. 오염된 풀 자산 재생성

두 파일을 **제출 csv 의 temp 열과 바이트 수준으로 일치**하도록 다시 만든다.
csv 를 읽어 npy 로 덤프하지 마라 — `mix_sub.py` 의 수정된 경로로 재조립해서
결과가 csv 와 일치하는지를 검증하는 방식이어야 한다.

| npy | missing 입력 | complete 입력 |
|---|---|---|
| `P2/preds/test_antistack_w-1_E014.npy` | `P2/preds/test_maskaug_E005.npy` | `robust_blend + (-1)*(stack_regime - robust_blend)` |
| `P2/preds/test_antistack_w-2.5_E017.npy` | `P2/preds/test_maskaug_E005.npy` | `robust_blend + (-2.5)*(stack_regime - robust_blend)` |

`robust_blend` = `P2/preds/test_gate_robust_blend_E004.npy`,
`stack_regime` = `P2/preds/test_stack_regime_E008.npy`.
complete 입력 벡터는 `wmix.py` 로 만든다. 기존 두 npy 는 덮어쓰기 전에 지워라
(현재 내용은 무마스크 오염본이라 보존 가치 없다).

`P2/preds/test_antistack_w-2.5_E017.npy` 는 지금 wmix 중간 산출물(완전행 벡터)이 들어 있다.
이것도 지우고 위 표대로 다시 만든다. wmix 중간 산출물은 `P2/cache/T17/` 에 두고 풀에 남기지 마라.

### 3. 검증 명령을 직접 실행하고 숫자를 보고

```
python P2/src/submission_tools.py audit P2/subs/E014_antistack_w-1.csv --pred P2/preds/test_antistack_w-1_E014.npy
python P2/src/submission_tools.py audit P2/subs/E017_antistack_w-2.5.csv --pred P2/preds/test_antistack_w-2.5_E017.npy
python P2/src/bugcheck.py all
```

## 기준선 숫자

| 항목 | 현재값 |
|---|---|
| `test_antistack_w-1_E014.npy` vs E014 csv 불일치 행 | 12,215 / 26,061 |
| 같은 쌍 max\|Δ\| | 3.39015 |
| E014 csv vs E009 csv 다른 행 | 18,295 |
| E017 csv vs E009 csv 다른 행 | 18,295 (비율 정확히 2.5) |
| E017 csv | rows 26,061 mean 23.065605 range [14.156363, 29.287231] |
| 수정 후 목표 불일치 행 | 0 (`npy_exact=True`) |
| `bugcheck.py all` 현재 | PASS 19 / FAIL 0 — 유지해야 한다 |

## 수락 기준

- 두 `audit` 모두 `npy_exact=True` 와 `판정: OK`
- `bugcheck.py all` `판정: OK`
- `mix_sub.py` 의 기존 인자만으로 호출해도 동작 (`--out-npy` 는 선택)
- `P2/subs/` 의 csv 를 수정하지 마라. E014 는 이미 제출됐다
- 새 파일·새 모듈을 만들지 마라. `mix_sub.py` 한 파일만 고친다
- 새 의존성 금지. `pip install` 이 필요하면 **설치하지 말고 즉시 멈추고 보고해라**

## 보고 형식

**결론 문장 금지, 숫자와 표만.** 변경 파일 경로, 두 audit 의 sha256·rows·npy_exact,
bugcheck 의 PASS/FAIL 개수만 적어라. 산문 금지.
