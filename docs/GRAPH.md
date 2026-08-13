# 온톨로지 — `P{n}/graph.json`

현재 구현된 그래프는 `P3/graph.json`뿐이다. P1·P2는 그래프를 만든 뒤에 아래 명령을 사용한다.

지식을 산문에 흩어놓으면 "이미 측정했는가"를 사람 기억으로 판정하게 된다. BARAM 2026 에서
**이미 닫힌 축을 다시 연 사고가 5회** 났다. 이 그래프는 그 판정을 명령어로 바꾼다.

## 타입 = 집합

| 타입 | 접두사 | 뜻 |
|---|---|---|
| `rule` | `r.` | 대회 문서에서 유도된 제약. `hard: true` 면 위반 = 실격 |
| `fact` | `f.` | 실측으로 확정. **재조사 금지** |
| `physics` | `p.` | 도메인 지식. 가설의 근거 |
| `axis` | `a.` | 실험 축. `open` / `running` / `closed`. `closed` 면 `evidence` 필수 |
| `asset` | `s.` | 보존된 예측 파일. 앙상블 풀의 원소 |
| `gate` | `g.` | 실행 전 통과해야 할 체크 |

관계: `forbids` · `requires` · `derives_from` · `closes` · `measures`

## 쓰는 법

```bash
python scripts/g.py --p P3 ls axis --status open    # 지금 열려 있는 축
python scripts/g.py --p P3 show p.decay             # 노드 1개 + 1홉 이웃
python scripts/g.py --p P3 check "너울 감쇠 수축"    # 새 실험 착수 전. exit 1 = 닫힌 축
python scripts/g.py --p P3 add a.foo axis "설명" --src "근거"
python scripts/g.py --p P3 close a.foo --evidence "78h셋 Δ=+0.012, 두 폴드 동일 부호"
python scripts/g.py --p P3 edge a.foo derives_from p.decay
python scripts/g.py --p P3 selftest                 # 구조 검사
```

## 규율

1. **실험 착수 전 `check` 를 돌리고 결과를 사양에 붙인다.** 자동 필터를 믿지 말고 출력을 읽어라
2. 축을 닫을 때 `--evidence` 에 **숫자**를 적는다. "안 좋았다" 는 근거가 아니다
3. 새 사실을 측정했으면 `fact` 노드로 추가한다. 다음 세션이 같은 것을 다시 재지 않게
4. `physics` 노드에 `"verified": false` 가 붙어 있으면 쓰기 전에 확인한다
5. 그래프는 문제별로 분리한다. 세 세션이 한 파일에 쓰면 충돌한다

## 왜 토큰이 절약되는가

노드 1개 조회 = 3줄. 같은 정보를 산문 문서에서 찾으면 문서 전체가 컨텍스트에 들어온다.
BARAM 의 `experiments/log.md` 는 139k 토큰까지 자랐고, 결국 "통째로 읽지 마라"는 규칙이 붙었다.
그래프는 그 규칙을 구조로 강제한다.
