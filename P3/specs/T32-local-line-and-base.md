# T32 — 로컬 직선 최적화 검증 + BASE05 채점

**결론 문장 금지. 숫자와 표만 보고한다.** 권고·다음 단계·요약 산문을 쓰지 마라.

**`P3/experiments/log.md` 와 `graph.json` 을 수정하지 마라.** 기록은 Opus 가 한다.

## 배경 · 기준선(baseline) 숫자 — 재조사 금지, 그대로 쓴다

`d = lr − T1F` 직선 위 public 2차형이 제출 3회로 정확히 풀렸다:
`A_pub = mean_pub(d·e) = -0.023914130`, `B_pub = mean_pub(d²) = 0.051323660`,
`alpha*_pub = 0.465947`. public RMSE 0.585693 → **0.577650** (현행 최고).
전체 test 의 `mean(d²) = 0.049337554` 이라 `B_pub/B_full = 1.0403`.

**5멤버 격자탐색은 3연패했다** (T13·T21·T26). **1파라미터 폐형 최적화는 성공했다.**
자유도가 문제였다는 가설을 로컬에서 독립 검증한다.

판정기: 조밀 재가중. `P3/src/judge_dense.py` 가 이미 있다 — 그것을 쓴다.
조밀 재가중 봉인 fold4: T1F `0.611959`, lr `0.598926`, lgbm_t1 `0.614009`.
lr fold4 시드 분산 `0.001803` (42/7/2026).

## 작업 A — 로컬 폐형 직선 최적화

조밀 OOF 로 `d_oof = lr_oof − t1f_oof` 를 만들고, **재가중 내적**으로 폐형 해를 구한다.
격자탐색·반복 최적화 금지. 공식은 `alpha* = −A/B`, `A = <d, e>_w`, `B = <d, d>_w`,
`e = t1f_oof − y`.

폴드별로 따로, 그리고 4폴드 통합으로 각각 `A_local`·`B_local`·`alpha*_local`·
`예상 RMSE(alpha*)` 를 낸다. 시드 42 만 쓴다.

### 수락 기준 A — 결과 보기 전 고정

1. **(대조군)** `alpha=0` 예상 RMSE 가 fold4 에서 `0.611959` 와 `|델타| <= 1e-6`,
   `alpha=1` 이 `0.598926` 과 `|델타| <= 1e-6`. 어긋나면 **중단하고 숫자만 보고한다.**
2. **(방법 검증)** fold4 `alpha*_local` 이 `alpha*_pub = 0.465947` 와 `|델타| <= 0.15` 이면
   「독립 두 표본이 같은 최적을 가리킨다」로 본다. 초과면 어긋난 것이다.
   **이 판정은 보고만 하고 아무 축도 열거나 닫지 마라.**

## 작업 B — BASE05 채점

`BASE05 = 0.5·T1F + 0.5·lr` (현행 최고 제출) 의 조밀 OOF 를 만들어
`judge_dense.py` 로 fold1~4 를 채점한다. 시드 42/7/2026 세 개 전부.

산출: `P3/preds/oof_dense_base05_p3_t32.npy` (시드 42), `_s7`·`_s2026` 2개 더.

## 작업 C — 다음 직선 후보의 로컬 A·B

BASE05 를 기준점으로 두고 아래 방향 각각에 대해 **폴드4 와 4폴드통합**의
`A_local`·`B_local`·`alpha*_local`·`예상 fold4 RMSE(alpha*)` 를 폐형으로 계산한다.
제출 CSV 는 만들지 마라. 학습도 하지 마라. 저장된 OOF 만 쓴다.

| 방향 | `u = ` |
|---|---|
| U1 | `ls_oof − BASE05` |
| U2 | `r_oof − BASE05` |
| U3 | `lgbm_t1_oof − BASE05` |
| U4 | `damped_oof − BASE05` |
| U5 | `arx_oof − BASE05` |
| U6 | `wsswell_oof − BASE05` |
| U7 | `1` (상수 편향보정) |

`alpha` 는 `[-1, 2]` 로 잘라 보고한다 (그 밖이면 경계값과 함께 `범위밖` 표시).

## 보고 표 4개만

1. `대조군 | 값 | 기대값 | 델타`
2. `A: 폴드 | A_local | B_local | alpha*_local | 예상 RMSE(alpha*)` (fold1~4 + 통합)
   + `alpha*_pub 0.465947 와의 델타` 1행
3. `B: 시드 | fold1 | fold2 | fold3 | fold4` (BASE05 3행)
4. `C: 방향 | fold4 A | fold4 B | fold4 alpha* | fold4 예상 RMSE | 통합 alpha*`

## 환경

실행 파이썬은 `python`. 새 의존성 금지. `pip install` 이 필요해지면 **설치하지 말고 즉시 정지**하고
무엇이 없는지만 보고한다.

## 금지

- 학습, 격자탐색, 반복 최적화 (폐형만)
- 제출 CSV 생성
- 기존 npy·CSV 수정
- `log.md`·`graph.json` 수정
- `judge_dense.py` 에 옵션·설정·플러그인 구조 추가
- 추상화·설정 클래스. `print` 로 충분
- `cx.cmd`·`codex exec` 재귀 호출
- 결론·권고 문장
