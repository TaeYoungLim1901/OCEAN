# E008 — 앙상블 풀 LOHO 스택 (재학습 없음)

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 금지).
**새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
(NNLS 는 `scipy.optimize.nnls` 가 있으면 쓰고, 없으면 `numpy` 만으로 사영 경사법 20줄로 푼다.)

## 0.2 예산 규칙

**파일 통독 금지.** `grep -n` 으로 필요한 구간만 읽는다. 읽을 것은 `gate_e004.py` 의
`regimes`·`holdout`·채점 호출부와 `P2/cache/E003/oof_index.csv` 스키마뿐이다. **새 모델 학습 없음.**

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.** §7 표와 판정 1행만 출력한다.

## 1. 배경

E005~E007 로 새 모델 축은 전부 닫혔다(`a.mask_augment`, `a.season_matched_holdout`,
`a.layer4_anchor` — layer_4 는 모든 행이 이미 하부 앵커를 갖고 `common.py:58-64` 가 layer 6·7 을
쓰므로 외삽 가설 자체가 기각). 남은 자산은 **이미 저장된 OOF/test 예측 풀**이다.
E003 의 blend 는 2개 모델 고정가중, E004 는 2개 모델 regime 게이트였고, **풀 전체를 대상으로 한
가중 추정은 아직 없다.**

## 2. 범위

- 새 파일 1개: `P2/src/stack_e008.py`. 기존 파일 수정 금지(import 만).
- **모델 학습 금지.** `P2/preds/oof_*.npy` 와 `P2/preds/test_*.npy` 만 쓴다.
- 풀 구성: `oof_index.csv` 와 행 정렬이 일치하는 모든 저장 예측. 최소한 아래를 포함한다.
  `e002_E003`, `enhanced_E003`, `eof_E003`, `no_depth_E003`, `robust_blend_E003`,
  `regime_gate_E003`, `gate_robust_blend_E004`, `maskaug_E005`, `missing_only_E005`.
  정렬이 다르거나 길이가 다른 예측은 **버리고 그 사실을 표0 에 적는다.**
- 기준선(`base_nom`)도 풀의 한 성분으로 넣는다(수축 효과를 허용하기 위함).
- 가중치는 **비음수, 합 1** 로 제한한다. 절편 금지. `SEED = 42`.
  `sys.stdout.reconfigure(encoding="utf-8")`. `P2_data/` 읽기 전용.

## 3. 후보 2개

- `stack_global` — 전 행 공통 가중 1벌.
- `stack_regime` — `missing_public` / `complete_public` 두 regime 에 각각 가중 1벌.
  regime 플래그는 `gate_e004.py` 의 함수를 import 해서 쓴다. 재정의 금지.

## 4. LOHO 검증 (이 실험의 핵심. 축약 금지)

4개 hold(`2024_julaug`, `2024_sepoct`, `2025_mayjun`, `2025_julaug`)에 대해:

```
for h in holds:
    w = argmin RMSE(가중합, 정답) on (holds - h)      # 비음수, 합 1
    평가 = RMSE on h                                    # h 는 가중 추정에 절대 쓰지 않는다
```

**hold h 의 정답은 h 의 가중치 추정에 어떤 형태로도 들어가지 않는다.** 위반 시 즉시 중단하고 보고.
보고에는 hold 별로 추정된 가중치 상위 3개 성분과 값을 싣는다.

test 예측용 최종 가중은 4개 hold 전부로 다시 추정한 1벌을 쓴다(LOHO 는 판정용, 최종은 전량 적합).

## 4.1 기준선 숫자

기준선은 E004 조립이다: 4-hold 평균 `1.089067`, hold 별 전체
`2024_julaug 1.285701`, `2024_sepoct 0.495506`, `2025_mayjun 1.303431`, `2025_julaug 1.271631`.
재계산해 불일치하면 그 사실을 표에 적어라.

## 5. 수락 기준 — 사전등록 판정 규칙 (결과 보기 전 확정. 사후 변경 금지)

후보가 **아래 5개를 전부** 만족할 때만 채택한다. 판정에 쓰는 값은 **LOHO 값뿐이다**
(전량 적합 값으로 판정하면 즉시 실격 처리하고 기각으로 보고하라).

1. LOHO 4-hold 평균이 E004 대비 **≥ 0.010 개선**
2. **4개 hold 전부에서** LOHO 전체 RMSE 가 E004 대비 악화 없음(≤ 0.000)
3. 어떤 hold 에서도 `missing_public` 세그먼트 악화 **≤ 0.005**
4. test-like hold `2025_julaug` 가 E004 대비 **≥ 0.005 개선**
5. `n ≥ 200` 인 모든 세그먼트에서 E004 대비 악화 **≤ 0.030**

둘 다 통과하면 LOHO 평균이 낮은 쪽. 통과 후보가 없으면 **기각**하고 E004 를 유지한다. 규칙 완화 금지.

## 6. 산출물과 보고

항상 저장:

- `P2/preds/oof_{stack_global,stack_regime}_E008.npy` (LOHO 예측), `P2/preds/test_{...}_E008.npy` (전량 적합 가중)
- `P2/cache/E008/metrics.json`(E004 스키마 + hold 별 가중치), `P2/cache/E008/weights.json`

채택된 경우에만:

- `P2/subs/E008_stack.csv` 와 아래를 직접 실행 후 숫자 보고:
  ```
  python P2/src/submission_tools.py audit P2/subs/E008_stack.csv --pred P2/preds/test_{선택}_E008.npy
  python P2/src/check.py sub P2/subs/E008_stack.csv
  ```

보고 형식 (이 표만):

```
표0 풀 구성: 성분 | 사용여부 | 제외사유
표1 LOHO 전체: cand | 4hold_mean | Δ_vs_E004 | hold별 RMSE와 Δ
표2 LOHO missing_public: cand | hold별 E004 | cand | Δ
표3 가중치: cand | hold | 상위3성분=값
표4 규칙 체크: cand | R1 | R2 | R3 | R4 | R5 (숫자와 pass/fail)
표5 위반 세그먼트: cand | hold | segment | n | E004 | cand | Δ (없으면 "없음")
판정: 채택 <cand> 또는 판정: 기각
```
