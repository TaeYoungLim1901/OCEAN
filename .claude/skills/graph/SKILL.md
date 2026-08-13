---
name: graph
description: 문제 온톨로지(P{n}/graph.json) 조회와 갱신. 새 실험 착수 전 닫힌 축 확인, 도메인 지식·규정·확정 사실 조회, 측정 결과로 축 닫기에 사용한다. "이거 해봤나", "닫힌 축", "규정 확인", "그래프", "축 닫아", "사실 추가" 요청 시 사용.
---

# 온톨로지 조회·갱신

문서를 통독하지 마라. 노드 단위로 꺼낸다. 상세는 [docs/GRAPH.md](../../../docs/GRAPH.md).

```bash
python scripts/g.py --p P3 check "<키워드>"        # 착수 전 필수. exit 1 = 닫힌 축
python scripts/g.py --p P3 ls axis --status open   # 지금 열려 있는 축
python scripts/g.py --p P3 show <노드id>           # 노드 + 1홉 이웃
```

## 언제 무엇을 하나

| 상황 | 명령 |
|---|---|
| 새 가설이 떠올랐다 | `check "<키워드>"` → 출력을 읽고 사양에 붙인다 |
| "이 데이터 이렇더라" 를 측정했다 | `add f.<이름> fact "<숫자 포함 한 줄>" --src "<근거>"` |
| 실험이 끝나 축을 닫는다 | `close a.<축> --evidence "<숫자>"` |
| 새 실험 축을 연다 | `add a.<이름> axis "<한 줄>"` + `edge a.<이름> derives_from p.<근거>` |
| 규정 해석을 확정했다 | `add r.<이름> rule "<제약>" --src "<문서 위치>"` |

## 규율

- 축을 닫을 때 `--evidence` 에 **숫자**를 적는다. "안 좋았다" 는 근거가 아니다
- `check` 의 자동 판정을 맹신하지 마라. 출력을 사람이 읽는다. BARAM 에서 자동 중복필터가
  틀려 닫힌 축을 5회 다시 열었다
- `"verified": false` 인 `physics` 노드는 쓰기 전에 확인한다
- 그래프는 문제별로 분리한다. `--p` 를 틀리면 다른 문제의 지식을 읽는다
- 구조를 바꿨으면 `selftest` 를 돌린다 (id 중복·끊긴 엣지·근거 없는 닫힘을 잡는다)
