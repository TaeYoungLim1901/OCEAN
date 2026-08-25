# Q1 — 상위권 F1 0.86 격차 원인 조사 (읽기 전용, 웹 검색 사용)

실행 파이썬: `python` (코드 실행은 이 작업에서 하지 않는다).

## 기준선 숫자

P1 은 해양 관측 시계열의 **행 단위 이상 탐지**다. 지표는 label=1 의 F1 (`anomaly_type` 은 순위 미반영).
데이터: `station,year,layer,time,temp,psal,depth,label,anomaly_type`. train 2024~2025, test 2026 상반기, 10분 간격.
이상은 **합성 주입**이다 (spike / noise / flatline / offset / drift 유형).

우리 현황:

| 경로 | F1 |
|---|---|
| forward inner (fold1·2) | 0.697836 |
| 봉인 outer fold3 | 0.815757 |
| public LB | 0.717435 |

리더보드 점수 ≈ 36 × F1. 1위 30.885889 → F1 약 **0.858**. 우리와 **+0.141**.
봉인 fold3 혼동행렬 TP 5612 / FP 704 / FN 1831 → precision 0.888, **recall 0.754**. 손실이 전부 recall 이다.
현재 모델은 LightGBM 행 단위 분류 + 규칙 점수 blend 인데 규칙 점수 가중치가 0 으로 선택됐다
(규칙 점수 단독 inner F1 0.174346, LightGBM 확률과 Spearman 0.055173).

## 배경 — 재조사 금지 사실 (확정. 다시 측정하거나 의심하지 마라)

- 기각된 축: 유형 multiclass blend (봉인 −0.006379), 양성률 앵커링 (inner 최고 0.689435 < 0.697836), 랜덤 행 CV (누수), adversarial drift (source AUC 0.850133, 진단만).
- 확장 격자에서 exp005 선택(threshold 0.62 / hole 48 / weight 0)이 내부 최적임이 확인됐다. threshold·hole 재탐색 제안은 필요 없다.
- 봉인 fold3 는 이미 3회 판독됐다. 추가 판독을 요구하는 제안은 하지 마라.
- 사용 가능 라이브러리는 numpy·pandas·scikit-learn·lightgbm 뿐이다. xgboost·catboost·torch 는 없다.

## 질문

**Q1-1.** 합성 주입 이상 탐지 대회(HAICon, VSB, SKAB, Numenta NAB, Kaggle 시계열 이상탐지 계열)에서
상위권이 F1 0.85+ 를 낼 때 쓴 구조가 무엇인가. 행 단위 GBDT 분류가 상한에 걸리는 지점과,
그 상한을 넘은 팀이 대신 쓴 것(시퀀스 모델 / 잔차 예측 / 매칭 필터 / 세그먼트 후처리)을 출처와 함께.

**Q1-2.** spike·noise·flatline·offset·drift 5유형 각각에 대해, 합성 주입을 거의 완전히 잡아내는
**결정론적 검출기**의 표준 형태가 무엇인가. 특히 flatline(분산 0 구간), offset(계단 변화), drift(느린 기울기)는
GBDT 행 feature 로 왜 어려운지, 어떤 통계량(rolling std, CUSUM, ruptures/PELT 변화점, 1차 차분 run-length)이
표준인지. 우리 규칙 점수가 F1 0.174 로 죽어 있는 전형적 실패 원인 목록.

**Q1-3.** recall 이 0.754 에서 막힐 때 F1 을 올린 **세그먼트 후처리** 기법. hole filling 외에
경계 확장(dilation), 최소 길이 필터, point-adjust 계열 평가와의 차이. 행 단위 F1 지표에서 실제로 이득인 것만.

**Q1-4.** train 2024~2025 → test 2026 처럼 시간 분리된 test 에서, 봉인 fold 점수(0.816)가
public(0.717) 보다 0.098 낙관인 현상의 알려진 원인과, fold 난이도 편차를 보정하는 표준 진단.

## 제약

- **읽기 전용이다. 파일 생성·수정·삭제 금지. 명령 실행 금지.** 조사와 근거 제시만 한다.
- **새 의존성 금지. `pip install` 이 필요한 제안을 하려면 실행하지 말고 그 사실을 명시만 한다.**
- 웹 검색을 쓴다. 주장마다 출처 URL 또는 대회·논문 이름을 붙인다. 출처 없는 일반론은 쓰지 마라.

## 수락 기준

Q1-1~Q1-4 각각에 대해 출처가 붙은 답이 있다. 우리 숫자(F1 0.717435, recall 0.754, 규칙 F1 0.174346)에
직접 연결되는 항목은 그 연결을 한 줄로 명시한다.

## 보고 형식

**결론 문장 금지, 숫자와 표만** — 단 이 작업은 조사이므로 예외적으로 항목당 3줄 이내 요약을 허용한다.
장황한 산문·서론·재확인 금지. 출처는 URL 또는 이름만.
