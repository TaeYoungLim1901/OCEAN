# E016 — bugcheck G 섹션 기준선 버그 수정

## 0. 실행자 확인

**너(codex)가 이 사양의 실행자다. 코드를 직접 쓰고 직접 실행한다.**
`cx.cmd`·`codex exec` 를 다시 호출하지 마라(재귀 timeout).
**새 의존성 금지. `pip install` 이 필요하면 멈추고 보고한다. 재학습 금지.**

## 0.1 보고 규칙

**결론 문장 금지. 숫자와 표만.**

## 1. 배경 (재조사 금지, 확정 사실)

`P2/src/bugcheck.py` 의 `check_g1` 이 틀렸다.

- 689행: 비교 **벡터**를 `P2/P2_data/baseline_interp.csv` 에서 읽는다.
- 687행: 비교 **점수** `base_public` 은 `P2/experiments/public_scores.csv` 의 public 최소값에서 읽는다.

둘이 서로 다른 대상이라 `m`·`mse_shift` 가 무의미하다. 실측 증거:
`E014_antistack_w-1.csv` 는 `E009_maskaug_diff.csv` 와 **18,295행**만 다른데
G1 이 `m=26,061 mse_shift=0.894146868` 을 냈다. 결과적으로 모든 후보가 `제출가치있음` 이 된다.

원장 4점(전부 확정): E004_gate 0.537099, **E009_maskaug_diff 0.525759(최소)**,
E010_stackregime_complete 0.528449, E011_missingonly 0.530045.

임계값 `0.002` 도 틀렸다. 리더보드는 소수점 **6자리**(예: 0.537099)로 나오므로 해상도는 1e-6 이다.

## 2. 수정 — `check_g1` 만 고친다

다른 검사·출력 형식·exit 규약은 손대지 마라. 새 파일 만들지 마라.

1. **비교 벡터를 원장 최소 public CSV 로 바꾼다.** `public_scores.csv` 에서 public 최소 행의
   `sha256` 을 얻고, `P2/subs/*.csv` 중 sha256 이 일치하는 파일을 찾아 그 `temp` 를 기준 벡터로 쓴다.
   - 일치 파일이 없으면 `ERROR` 로 `base_csv=not_found` 를 내고 그 섹션만 실패시킨다.
   - 기준 벡터로 쓴 CSV 파일명을 `report.detail` 1행에 찍는다.
2. **키 정렬 검증을 넣는다.** 기준 CSV 와 후보 CSV 의 키 열(`common.py` 의 `C.KEY`)이
   행 순서까지 같은지 확인하고, 다르면 그 후보는 `key_order_mismatch` 로 찍고 건너뛴다.
   (정렬이 어긋난 채 뺄셈하면 `mse_shift` 가 조용히 부풀어 오른다 — 이번 버그의 증상이다.)
3. **임계값을 `0.002` 에서 `0.00002` 로 바꾼다.** 판정 문자열은 그대로
   `제출불필요` / `제출가치있음` 을 쓴다.
4. 기준 벡터와 같은 sha256 인 CSV(=기준선 자신)는 pending 목록에서 제외한다.

## 3. 실행하고 보고

`python P2/src/bugcheck.py --only G` 를 실행하고 **출력 전문**을 보고한다.
이어서 `python P2/src/bugcheck.py` 전체를 실행하고 마지막 2줄만 보고한다.

## 4. 수락 기준

- 기준 벡터 CSV = `E009_maskaug_diff.csv`
- `E014_antistack_w-1.csv` 의 `m` = **18,295**, `mse_shift` ≈ 0.000461344
- `E015_missblend.csv` 의 `m` = **7,766**, `mse_shift` ≈ 0.000246528
- 전체 실행이 `PASS 19 / FAIL 0` 과 `판정: OK` 유지
- `bugcheck.py` 외 파일 무수정, 제출하지 않는다
