# T31 — alpha 탐침 제출 CSV + 상설 조밀 채점기 + 시드 분산

**결론 문장 금지. 숫자와 표만 보고한다.** 권고·다음 단계·요약 산문을 쓰지 마라.

**`P3/experiments/log.md` 와 `graph.json` 을 수정하지 마라.** 기록은 Opus 가 한다.

## 배경 · 기준선(baseline) 숫자 — 재조사 금지, 그대로 쓴다

제출 1회 = public 잔차 `e` 의 선형범함수 1개라는 항등식이 **실측으로 정확히** 성립한다
(`f.public_mse_algebra_exact`, 재구성 오차 0.000000).

`d = lr − T1F` 로 두면 `MSE(T1F + α·d) = M0 + 2α·A + α²·B`,
`A = mean_pub(d·e)`, `B = mean_pub(d²)`. 이미 확보한 실측 2점:

| α | 제출 | public RMSE | MSE |
|---|---|---|---|
| 0 | `submission_p3_t1f.csv` | 0.587165 | 0.344763 |
| 1 | `submission_p3_lr.csv` | 0.590134 | 0.348258 |

미지수 2개(A·B)에 방정식 1개(`2A + B = +0.003495`)뿐이다. **α 를 하나 더 내면 정확히 풀린다.**
`mean_pub(d²)` 를 전체 test 값 0.049338 로 근사하면 안 된다 — 그 근사가 T18 을 죽였다
(public 은 200사례 중 ~80 이하 부분집합, `f.public_small_random_subset`).

## 작업

`α = 0.5` 한 점의 제출 CSV 를 만든다. 즉 **`0.5 · T1F + 0.5 · lr`** 이다.

- 입력: `P3/preds/test_final_p3_t1f.npy`, `P3/preds/test_lr_p3_t26.npy`
- 가중은 상수 0.5/0.5 고정. **격자탐색·적합 금지.** 상수 편향보정도 섞지 마라(교란된다)
- `[0, 30]` 클립, 클립행수 보고
- 산출: `P3/preds/test_alpha05_p3_t31.npy`, `P3/subs/submission_p3_alpha05.csv`
- 열 순서 `case_id,station,lead_h,hs_pred`, `P3/P3_data/test_index.csv` 키 정렬 일치

부수로 **전체 test 기준 예측 MSE 를 계산해 출력한다** (정답 없이 계산되는 부분만):
`mean(d)`, `mean(d²)`, 그리고 전체test 근사가 맞다고 가정했을 때의 예측 public RMSE
`sqrt(0.344763 + 2·0.5·A_hat + 0.25·B_hat)` — 여기서 `B_hat = mean_full(d²)`,
`A_hat = (0.003495 − B_hat)/2`. 이건 **예측치일 뿐 판정 근거가 아니다.**


---

## 작업 B — 상설 조밀 재가중 채점기 `P3/src/judge_dense.py`

T28·T30 이 임시 스크립트로 같은 채점을 두 번 했고 둘 다 남지 않았다. 이번이 세 번째다.
**한 번만 만든다.**

```
python P3/src/judge_dense.py <oof_npy> [<oof_npy> ...]
```

각 파일에 대해 조밀 재가중 fold1·2·3·4 RMSE 를 출력한다. 채점 경로는 `P3/src/run_t21.py`
가 쓰는 것을 그대로 import 해서 재사용한다 (복사 금지). 인자 없이 부르면 사용법만 출력.

과한 일반화 금지 — 옵션·설정파일·플러그인 구조를 만들지 마라. 위 한 줄 사용법이 전부다.

## 작업 C — 시드 분산 측정 (학습 0, B 를 쓴다)

시드 42/7/2026 4폴드 실행이 끝나 조밀 OOF 9개가 저장돼 있다:

```
P3/preds/oof_dense_{r,lr,ls}_p3_t26.npy          # 시드 42
P3/preds/oof_dense_{r,lr,ls}_p3_t26_s{7,2026}.npy
```

`judge_dense.py` 로 9개 전부를 채점해 표로 낸다. **학습 금지 — 저장된 npy 만 읽는다.**

### 대조군 C

시드 42 `lr` 의 fold4 가 `0.598926` 과 `|델타| <= 1e-6` 이어야 한다.
어긋나면 채점경로 불일치이므로 **중단하고 숫자만 보고한다.**

(참고: 시드 42 재실행이 `--n-jobs` 2→6 차이로 OOF 를 max 절대차 `1.78e-15` 만큼 바꿨다.
제출에 쓰인 `test_lr_p3_t26.npy` 는 해시 불변이다. 이 크기는 무시한다.)

---

## 수락 기준 A (직접 실행하고 출력 숫자를 보고. 판정이 OK 가 아니면 멈추고 숫자만 보고)

```
python P3/src/submission_tools.py audit P3/subs/submission_p3_alpha05.csv --pred P3/preds/test_alpha05_p3_t31.npy
python scripts/preflight.py sub P3/subs/submission_p3_alpha05.csv --index P3/P3_data/test_index.csv --base P3/P3_data/baseline_persistence.csv
python P3/src/bugcheck.py all
```

추가 대조군 — 기존 제출 CSV SHA256 **불변**:
`submission_p3_t1f.csv` = `2E4984283B047950E5D2DA219EF6D8B3D02D7FEB54CC67DAA2AF75764D3470C1`,
`submission_p3_t1f_bias.csv` = `605BA2BB6FA70CC5811F1ACAEA3F200543C929E37BDF632E7FDAFEF389E3BA1C`,
`submission_p3_lr.csv` = `F1584F54F0BE6808874446A78673677194B041933C621DE8458447018E1CBE24`,
`submission_p3_blend_t21.csv` = `77C4F71C19FBCD8357C3389D2C91DB94A7EDBDA430FB664362BC9B0CADA43BAB`.
달라지면 고치지 말고 숫자를 보고하고 멈춘다.

또 하나 — 재구성 대조군: 생성한 CSV 의 `hs_pred` 가 `0.5*(T1F npy) + 0.5*(lr npy)` 와
max 절대차 `<= 1e-9` 여야 한다.

## 보고 표 3개만

1. `항목 | 값` — rows, hs 평균, sd, 범위, 클립행수, SHA256, npy max delta,
   mean(d), mean_full(d²), A_hat, B_hat, 예측 public RMSE
2. `검증 명령 | 판정 | 핵심 숫자`
3. `변형 | 시드 | fold1 | fold2 | fold3 | fold4` (9행) + `변형 | fold4 최대-최소`(3행)
   + 대조군 C 1행

## 환경

실행 파이썬은 `python`. 새 의존성 금지. `pip install` 이 필요해지면 **설치하지 말고 즉시 정지**하고
무엇이 없는지만 보고한다.

## 금지

- 학습(전부), 가중 적합, 격자탐색, 상수 편향보정 적용
- `judge_dense.py` 에 옵션·설정·플러그인 구조 추가
- `P3/preds/*_lr_p3_t26*.npy` 를 포함한 기존 npy·CSV 수정 (다른 프로세스가 쓰고 있다)
- `log.md`·`graph.json` 수정
- 추상화·설정 클래스. `print` 로 충분
- `cx.cmd`·`codex exec` 재귀 호출
- 결론·권고 문장
