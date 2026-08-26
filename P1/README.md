# P1 exp001 재현

루트 `D:\hackathon\OCEAN`에서 실행한다.

```bash
python P1/src/check.py data
python P1/src/metric.py
python P1/src/train.py
python P1/src/inference.py
python P1/src/check.py pipeline P1/subs/exp001.csv P1/preds/oof_lgbm_exp001.npy P1/preds/test_lgbm_exp001.npy
python P1/src/check.py sub P1/subs/exp001.csv
```

- 제출: `P1/subs/exp001.csv`
- 모델/기준 통계: `P1/cache/exp001/`
- OOF/test 점수: `P1/preds/`
- `train.py`는 모델 파일이 이미 있으면 재학습하지 않고 저장 모델의 OOF만 재생성한다. 처음부터 재학습하려면 별도 실험 ID를 만들어야 하며 기존 `exp001` 자산은 보존한다.

exp002 최종 후보 검증:

`subs/exp003.csv`는 outer early-stop 누수 발견 전 생성된 기각 산출물이므로 제출하지 않는다.

```bash
python P1/src/exp002.py
python P1/src/check.py reproduce exp002
python P1/src/check.py sub P1/subs/exp002.csv
```
