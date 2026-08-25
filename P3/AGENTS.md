# P3 — 종합해양과학기지 유의파고 단기 예측

루트 [AGENTS.md](../AGENTS.md) 가 상위 계약이다. 이 파일은 P3 고유 사항만 적는다.

## ★ codex 위임 = 토큰 절약의 전부. 컨텍스트 길어져도 잊지 마라

> **codex 에게: 이 절은 Opus(Claude Code) 전용 지시다. 너는 이미 위임받은 실행자다.**
> `cx.cmd`·`codex exec` 를 다시 호출하지 마라. 사양은 네가 직접 구현한다.
> (2026-08-13 P2 E004: 이 절을 자기 지시로 읽고 codex 를 3회 재귀 호출해 timeout, 산출물 0)

코드 작업은 전부 `scripts\cx.cmd <사양>` (codex `gpt-5.6-luna`, `CX_EFFORT=xhigh`). Opus 직접 구현 금지.

**Claude Code Bash 도구의 샌드박스가 codex 인증·네트워크를 막는다.** 2026-08-13 P2 실측:
샌드박스 상태 호출은 출력 없이 `exit 255`, 산출물 0개. 반드시 아래 형태로 부른다.

```
scripts\cx.cmd P3/specs/<사양>.md > P3/experiments/logs/codex_<ID>.log 2>&1
```

- Bash 도구 인자: `dangerouslyDisableSandbox: true`, `run_in_background: true` (호출이 600초를 넘는다).
- 실패 판단은 로그 `tail -20` 과 산출물(`P3/cache/...`) 존재 여부만으로 한다. 로그 전문 금지.

**사실·규정·물리·실험축은 이 문서가 아니라 `graph.json` 에 있다.**

```bash
python scripts/g.py --p P3 ls axis --status open      # 열린 실험 축
python scripts/g.py --p P3 show f.atmos_gap_is        # 노드 1개 (3줄)
python scripts/g.py --p P3 check "달력 월"            # 닫힌 축이면 exit 1
```

## 과제

각 `case_id` 의 과거 48시간 관측만으로 +3/6/9/12/18/24h 유의파고 예측. 지표 통합 RMSE(m),
1,200건. 지속성 B=0.769455, 정책상수 T=0.624165. 답안 제출 8/20 개시, 하루 1회.
모델 제출은 9/7 마감이며 **제출 즉시 답안 업로드가 끝난다.**

## 하드 제약 (위반 = 실격)

- 추론 피처·정규화·예측은 해당 `case_id`의 289행 컨텍스트만 쓴다. 다른 test case의 행을 섞지 않는다.
  따라서 test 200사례 전체 통계로 정규화하는 transductive 방식도 금지다
- 절대시각 단서가 없다 → 달력 피처 불가 (`a.calendar_features` 닫힘)
- 제출 열 순서 `case_id,station,lead_h,hs_pred`, 값은 0~30m 유한
- 업로드한 예측을 코드가 재현하지 못하면 성적 무효

## 게이트

```bash
python P3/src/check.py data       # 원본 무결성 · test 스키마
python P3/src/check.py pipeline   # 지속성 재현이 baseline_persistence.csv 와 완전일치 (정답 없이 생산경로 검증)
python P3/src/check.py cases      # 학습 컨텍스트 == test 컨텍스트 형태, 사례 선정 분포 대조
python P3/src/check.py cv <fold.csv>  # 폴드 생성 후 정점별 경계의 78h 간격 검사
python P3/src/check.py hashes    # 원본 P3_data 8개 SHA256 대장 생성
python P3/src/metric.py selftest  # 완벽예측 0, 행 가법성, 키 불일치 탐지
python P3/src/submission_tools.py audit <csv> --pred <npy>  # 모든 제출 직전 필수
```

Public 점수는 `python P3/src/submission_tools.py public add <csv> <RMSE> --pred <npy>`로 기록하고 `python P3/src/submission_tools.py public list`로 본다.
숨은 정답 없이는 Public RMSE를 계산할 수 없으므로 리더보드에서 받은 값만 기록한다.

`cv` 게이트는 fold 간 시간 간격만 본다. 달력 피처와 t0 이후 입력 혼입은 사양·코드에서 별도 확인한다.

## 코드 계약

- 피처 함수는 `(289, 10) float32` 컨텍스트 배열 **하나만** 받는다. 학습·추론이 같은 경로다
- 열 순서 `hs tp hmax wvdir wspd gust wdir airt relh caph` 고정 (`cases.COLS`)
- `metric.evaluate(..., by=[...])` — `by` 는 필수. 전체 점수 하나만 보는 실험은 시작하지 마라
- 로컬 평가셋은 `cases.eval_split()` (78h 그리디 시닝). 공식 t0 hs 분위수와 일치함을 확인했다

## 로컬 기준값 (2026-08-13 실측, 78h 평가셋 272건)

| | +3h | +6h | +9h | +12h | +18h | +24h | 통합 |
|---|---|---|---|---|---|---|---|
| 지속성 RMSE | 0.5764 | 0.7525 | 0.8286 | 0.8638 | 0.9526 | 0.9610 | **0.8329** |
| 평균편차 | −0.008 | −0.057 | −0.066 | −0.057 | −0.231 | −0.355 | |

로컬 통합 0.8329 는 공식 0.769455 보다 어렵다(기간·표본 차이). **절대값 비교 금지, 델타만 쓴다.**
평균편차가 음수 = 지속성은 계통적으로 과대예측이다 (`p.selection_regression` 확인됨).
