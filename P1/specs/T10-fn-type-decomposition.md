# T10 — FN 유형 분해 (읽기 전용 EDA, 모델 변경 없음)

## 배경 숫자 (확정. 재측정 불필요)

| 경로 | F1 |
|---|---|
| forward inner (fold1·2) | 0.697836 |
| 봉인 outer fold3 | 0.815757 |
| **public LB** | **0.717435** |

봉인 fold3 TP 5612 / FP 704 / FN 1831 → precision 0.888, recall 0.754. **recall 제약**이 손실의 전부다.
rule_score 단독 inner F1 0.174346, lgbm 확률과 Spearman 0.055173 → 규칙 검출기가 사실상 죽어 있다 (weight 0.00 선택됨).
`f.synthetic`: 이상은 합성 주입이다. 주입 유형이 알고리즘적이면 유형 정합 검출기는 원리상 강해야 한다. 왜 0.174 인지가 이 작업의 표적이다.

**봉인 fold3 는 이미 3회 판독됐다. 이 작업은 판독 4회차가 아니다 — 새 파라미터를 고르거나 threshold 를 재선택하지 마라. 기존 exp005 예측 배열을 그대로 분해만 한다.**

## 입력

- `P1/P1_data/train.csv` — `station,year,layer,time,temp,psal,depth,label,anomaly_type`
- `P1/preds/oof_exp005.npy` (inner fold1·2 + 봉인 fold3 OOF 확률), `P1/preds/test_exp005.npy`
- exp005 확정 설정: weight 0.00 / threshold 0.62 / hole 48 / r 0.030 / λ 0.00
- 기존 fold 정의·purge 마스크 재사용. 새로 만들지 마라. 코드 경로는 `P1/src/` 에서 찾아 재사용한다.

## 기준선 숫자

exp005 봉인 fold3: TP 5612 / FP 704 / FN 1831, F1 0.815757. inner F1 0.697836. public F1 0.717435.
rule_score 단독 inner F1 0.174346. 이 숫자들이 baseline 이며 B1~B6 합계가 여기에 정합해야 한다.

## 산출 (전부 `P1/experiments/exp006_fn_decomp.json` + 표를 stdout 으로)

### B1 — anomaly_type 별 행 단위 성능
inner(fold1·2) 와 봉인 fold3 **각각** 에서, `anomaly_type` 별로:
`n_rows_true, TP, FN, recall`. 그리고 예측 양성 중 `anomaly_type` 별 분포와 FP 수.
유형 이름은 데이터에서 읽어라 (하드코딩 금지).

### B2 — 세그먼트 단위 분해
세그먼트 = 같은 `(station, year, layer)` stream 안의 연속 `label==1` 구간 (시간 정렬, gap 은 stream 의 최빈 간격 기준으로 끊는다).
유형별로:
`n_segments, 세그먼트 길이 분포 (p10/p50/p90/max), full_miss (TP 0개), partial (0<TP<len), full_hit (TP==len), 세그먼트 내 hit 비율 p50`.

### B3 — 세그먼트 내 위치별 recall
각 세그먼트를 길이 정규화 위치 10분위로 나누고 유형별 분위별 recall. (가장자리를 놓치는가, 통째로 놓치는가를 가른다.)

### B4 — 놓친 세그먼트의 신호 진폭
유형별로 full_miss 세그먼트 vs full_hit 세그먼트에 대해:
세그먼트 내 `temp`·`psal` 이 같은 stream 의 학습집합 climatology 잔차 기준 몇 σ 인가 (`|resid|/σ` 의 세그먼트 p50). full_miss 가 저진폭이라 못 잡는 건지, 진폭은 큰데 feature 가 못 보는 건지 가른다.

### B5 — rule_score 유형별 성능
현재 `rule_score` 를 유형별로 단독 채점 (유형별 최적 threshold 를 grid 로 찾아 F1 최대). 어느 유형 검출기가 죽어 있는지, 어느 유형에서만 lgbm 을 이기는지 표로.
유형별로 rule_score 와 lgbm 확률의 Spearman 도 함께.

### B6 — test 예측의 유형 미상 진단
test 에는 라벨이 없다. exp005 test 확률로 threshold 0.62 예측 양성 세그먼트를 만들고, 세그먼트 길이 분포를 train 의 유형별 길이 분포와 비교한 표. (어떤 유형을 test 에서 아예 못 만들고 있는지의 간접 증거.)

## 제약

- **새 의존성 금지. `pip install` 이 필요하면 실행하지 말고 즉시 멈추고 보고한다.**
- 모델 학습·재학습 없음. 새 feature 없음. 새 제출 파일 없음. `P1_data/` 는 읽기 전용.
- `n_jobs` ≤ 2.
- 스크립트 상단 `sys.stdout.reconfigure(encoding="utf-8")`.
- 새 코드는 `P1/src/exp006_fn_decomp.py` 한 파일.

## 수락 기준

`python P1/src/exp006_fn_decomp.py` 가 오류 없이 끝나고 `P1/experiments/exp006_fn_decomp.json` 이 생성된다.
B1~B6 표를 전부 stdout 에 출력하고, 그 숫자를 보고한다.

## 보고 형식

**결론 문장 금지. 숫자와 표만.** 해석·권고·"~로 보인다" 금지.
