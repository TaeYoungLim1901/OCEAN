# E013 — bugcheck 오탐 교정 + Sep-Oct 정합 재판정

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).

**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
**재학습 금지.** 저장된 `P2/preds/{oof,test}_*.npy` 재조합과 채점만 한다.
`common.py`·`train*.py`·`inference*.py`·`gate_e004.py`·`check.py` **수정 금지.**

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** 산문 요약·소감·다음 제안 쓰지 마라.

## 1. 배경 (재조사 금지, 확정 사실)

public 실측 2건과 로컬 hold 별 OOF RMSE 대조:

| 후보 | 2024_sepoct | 2024_julaug | 2025_mayjun | 2025_julaug | 4-hold 평균 | public |
|---|---:|---:|---:|---:|---:|---:|
| gate_robust_blend (E004) | 0.495506 | 1.285701 | 1.303431 | 1.271631 | 1.089067 | **0.537099** |
| maskaug (E005 → E009) | 0.492306 | 1.304079 | 1.330035 | 1.248801 | 1.093805 | **0.525759** |

public 절대 수준(0.53)은 **2024_sepoct(0.49)** 와 같은 자리수이고 4-hold 평균(1.09)과는 2배 차이다.
순서도 2024_sepoct 만 맞다(maskaug < robust_blend). test 는 2025-09-01~10-31 로 2024_sepoct 와
같은 계절(성층 붕괴기)이다. **따라서 판정기는 Sep-Oct 정합 창이며 4-hold 균등이 아니다.**

Sep-Oct 정합 OOF 는 이미 있다: `oof_maskaug_S{1,3,4}_E006.npy`(Sep-Oct 포함 창),
`oof_maskaug_S2_E006.npy`(8월 비중 큰 창 — 판정에서 제외 대상). 창 정의는 `P2/src/season_e006.py`.

기타 확정 사실 (재조사 금지):
- 마스크 창 2025-09-01~10-31 안의 학습행은 **0개**. 학습 데이터가 2025-12-10 까지 있는 것은
  창 *이후* 구간(16,884행)이며 보간 과제이므로 규정 위반이 아니다.
- `common.py:125-126` 의 `shift(-36)` lead6h/trend12h 는 **공개층** 피처다. 공개층은 test 기간에도
  관측되므로 미래 누수가 아니다.
- `SEED = 42` 는 `common.py:11` 에 있고 전부 `C.SEED` 로 참조한다.

## 2. 작업 1 — `P2/src/bugcheck.py` 오탐 교정

아래 4개만 고친다. 다른 검사 로직·출력 형식·exit 규약은 그대로 둔다.

- **B1**: 기준을 `test_start` 가 아니라 **마스크 창 `[2025-09-01, 2025-10-31 23:59:59+09:00]` 내부**로
  바꾼다. 창 내부 학습행 수가 0 이 아니면 FAIL. 창 이후 행수는 `INFO` 로 함께 찍는다.
- **B4**: `PUBLIC_LAYERS` 이름이 들어간 lead/trend 피처 라인과 파이썬 내장 `next(` 호출은
  **오탐으로 제외**한다. 제외한 라인 수를 `INFO` 로 찍는다. 타깃층 이름이 들어간 미래 참조만 FAIL.
- **F1**: `SEED\s*=\s*42` 를 `common.py` 를 포함해 찾고, `train*.py` 는 `C.SEED` 또는
  `random_state=` 로 시드를 넘기면 통과로 본다. 시드가 전혀 안 걸린 파일만 FAIL.
- **E3**: 길이 균일성 대신 **각 oof 길이가 그 후보의 평가 인덱스 길이와 맞는지**로 바꾼다.
  맞출 인덱스를 못 찾은 파일만 FAIL 로 나열한다.

교정 후 `python P2/src/bugcheck.py` 를 실행하고 **출력 전문**을 보고한다.

## 3. 작업 2 — Sep-Oct 정합 재판정 (`P2/src/judge_sepoct.py` 신규)

`P2/preds/oof_*.npy` **전 후보**에 대해 아래 3개 평가 집합에서 RMSE 를 낸다.
평가 인덱스가 안 맞는 후보는 건너뛰고 이름만 `skipped` 로 나열한다.

1. `2024_sepoct` hold (전체 / `missing_public` / `complete_public` 3분할)
2. `season_e006.py` 의 S1·S3·S4 창 (Sep-Oct 포함 창. S2 는 제외하고 그 이유를 표 각주로 적지 마라 — 숫자만)
3. 위 4개(2024_sepoct·S1·S3·S4)의 **단순 평균** = `sepoct_mean`

출력 표: 행 = 후보, 열 = 2024_sepoct(전체/결측/완전), S1, S3, S4, `sepoct_mean`.
`sepoct_mean` 오름차순 정렬. 상위 10행만 찍는다.

## 4. 작업 3 — Sep-Oct 판정기로 스택 재적합

`P2/src/stack_e008.py` 의 비음수·합1 가중 적합 코드를 **재사용**해서(복사·재구현 금지,
import 해서 호출), 가중 추정 대상을 4-hold 가 아니라 **Sep-Oct 정합 창 4개**로 바꾼다.
과적합 방지: **leave-one-window-out** — 창 하나를 빼고 나머지 3창에서 가중을 적합해
뺀 창에서 채점한다. 4번 반복.

- `regime` 2벌(missing/complete)과 전역 1벌 둘 다 낸다.
- LOWO 4창 평균 RMSE 를 `maskaug` 단독의 같은 평균과 비교해 표로 낸다.
- **maskaug 를 이기는 구성이 있을 때만** 전체 4창으로 가중을 재적합해
  `P2/preds/test_stack_sepoct_E013.npy` 와 `P2/preds/oof_stack_sepoct_E013.npy` 를 저장한다.
  이기는 구성이 없으면 npy 를 만들지 말고 표만 낸다.

## 5. 작업 4 — 제출 필요성 게이트 재실행

`test_stack_sepoct_E013.npy` 를 만들었다면 `P2/src/mix_sub.py` 로
`P2/subs/E013_stack_sepoct.csv` 를 만들고(결측행·완전행 구성은 §4 에서 이긴 구성 그대로),
`python P2/src/bugcheck.py --only G` 를 실행해 기준선 `E009`(public 0.525759) 대비
`측정폭` 을 보고한다. 만들지 않았다면 이 절은 `해당없음` 한 줄.

## 6. 수락 기준

- 새 파일은 `P2/src/judge_sepoct.py` 하나. `bugcheck.py` 는 §2 의 4개 항목만 수정
- `python P2/src/bugcheck.py` 출력 전문 보고
- §3 표(상위 10행), §4 LOWO 비교표, §5 측정폭 보고
- 다른 기존 파일 무수정, **제출은 하지 않는다**
