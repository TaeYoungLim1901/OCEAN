"""온톨로지 질의 CLI — P{n}/graph.json.

문서 통독 대신 노드 단위로 읽는다. 새 실험 전 `check` 는 필수 게이트다.

  python scripts/g.py show a.gbdt_direct
  python scripts/g.py ls axis --status open
  python scripts/g.py check "달력 월 시각"          # 닫힌 축이면 exit 1
  python scripts/g.py add a.foo axis "설명" --src "근거"
  python scripts/g.py close a.foo --evidence "실측 Δ=-0.01, 2폴드 동일부호"
  python scripts/g.py edge a.foo derives_from p.decay
  python scripts/g.py selftest
"""
import argparse, json, os, sys, pathlib

sys.stdout.reconfigure(encoding="utf-8")  # 콘솔 기본 cp949 회피

ROOT = pathlib.Path(__file__).resolve().parent.parent
TYPES = ["rule", "fact", "physics", "axis", "asset", "gate"]
RELS = ["forbids", "requires", "derives_from", "closes", "measures"]


def gpath(p):
    return ROOT / p / "graph.json"


def load(p):
    f = gpath(p)
    if not f.exists():
        sys.exit(f"그래프 없음: {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def save(p, g):
    gpath(p).write_text(json.dumps(g, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find(g, nid):
    return next((n for n in g["nodes"] if n["id"] == nid), None)


def line(n):
    tag = f"[{n['status']}]" if "status" in n else ""
    ev = f"  근거: {n['evidence']}" if n.get("evidence") else ""
    return f"{n['id']:24} {n['type']:8}{tag:10} {n['t']}{ev}"


def cmd_show(g, a):
    n = find(g, a.id)
    if not n:
        near = [x["id"] for x in g["nodes"] if a.id.lower() in x["id"].lower()]
        sys.exit(f"노드 없음: {a.id}" + (f" — 비슷한 것: {near}" if near else ""))
    print(line(n))
    for k in ("src", "verified"):
        if k in n:
            print(f"  {k}: {n[k]}")
    for s, r, d in g["edges"]:
        if s == n["id"]:
            print(f"  -> {r:12} {d}")
        elif d == n["id"]:
            print(f"  <- {r:12} {s}")


def cmd_ls(g, a):
    ns = [n for n in g["nodes"] if (not a.type or n["type"] == a.type)
          and (not a.status or n.get("status") == a.status)]
    for n in ns:
        print(line(n))
    print(f"-- {len(ns)}개")


def cmd_check(g, a):
    """새 실험 키워드가 닫힌 축·강제 규정과 겹치는지. 겹치면 exit 1."""
    words = [w.lower() for w in " ".join(a.words).split() if w]
    hit = lambda n: any(w in (n["id"] + " " + n["t"] + " " + n.get("evidence", "")).lower() for w in words)
    closed = [n for n in g["nodes"] if n.get("status") == "closed" and hit(n)]
    rules = [n for n in g["nodes"] if n["type"] == "rule" and n.get("hard") and hit(n)]
    for n in closed:
        print("닫힌 축 " + line(n))
    for n in rules:
        print("강제 규정 " + line(n))
    if not closed and not rules:
        print(f"겹침 없음 ({' '.join(words)}) — 진행 가능")
        return 0
    return 1 if closed or rules else 0


def cmd_add(g, a):
    if find(g, a.id):
        sys.exit(f"이미 있다: {a.id}")
    if a.type not in TYPES:
        sys.exit(f"타입은 {TYPES}")
    n = {"id": a.id, "type": a.type, "t": a.text}
    if a.src:
        n["src"] = a.src
    if a.type == "axis":
        n["status"] = a.status or "open"
    g["nodes"].append(n)
    save(a.p, g)
    print("추가 " + line(n))
    return 0


def cmd_close(g, a):
    n = find(g, a.id)
    if not n or n["type"] != "axis":
        sys.exit(f"axis 노드가 아니다: {a.id}")
    n["status"], n["evidence"] = "closed", a.evidence
    save(a.p, g)
    print("닫음 " + line(n))
    return 0


def cmd_set(g, a):
    n = find(g, a.id)
    if not n:
        sys.exit(f"노드 없음: {a.id}")
    n[a.field] = a.value
    save(a.p, g)
    print("수정 " + line(n))
    return 0


def cmd_edge(g, a):
    if a.rel not in RELS:
        sys.exit(f"관계는 {RELS}")
    for x in (a.src, a.dst):
        if not find(g, x):
            sys.exit(f"노드 없음: {x}")
    e = [a.src, a.rel, a.dst]
    if e not in g["edges"]:
        g["edges"].append(e)
        save(a.p, g)
    print(" ".join(e))
    return 0


def cmd_selftest(g, a):
    ids = [n["id"] for n in g["nodes"]]
    assert len(ids) == len(set(ids)), "노드 id 중복"
    for n in g["nodes"]:
        assert n["type"] in TYPES, f"미정의 타입 {n['id']}"
        assert n["id"].split(".")[0] + "." in g["meta"]["id_prefix"], f"접두사 규약 위반 {n['id']}"
        if n.get("status") == "closed":
            assert n.get("evidence"), f"닫힌 축에 근거 없음 {n['id']}"
    for s, r, d in g["edges"]:
        assert r in RELS, f"미정의 관계 {r}"
        assert s in ids and d in ids, f"끊긴 엣지 {s}->{d}"
    # check 가 실제로 닫힌 축을 잡는가
    closed = [n for n in g["nodes"] if n.get("status") == "closed"]
    if closed:
        kw = closed[0]["id"].split(".")[1]
        assert cmd_check(g, argparse.Namespace(words=[kw])) == 1, "check 가 닫힌 축을 못 잡는다"
    probe = {"nodes": [
        {"id": "r.hard", "type": "rule", "hard": True, "t": "hard rule"},
        {"id": "a.closed", "type": "axis", "status": "closed", "t": "closed axis"},
    ]}
    assert cmd_check(probe, argparse.Namespace(words=["hard"])) == 1, "hard rule을 못 잡는다"
    assert cmd_check(probe, argparse.Namespace(words=["closed"])) == 1, "닫힌 축을 못 잡는다"
    assert cmd_check(probe, argparse.Namespace(words=["absent"])) == 0, "miss를 진행 불가로 판정한다"
    print(f"OK — 노드 {len(ids)} 엣지 {len(g['edges'])}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", default=os.environ.get("OCEAN_P", "P3"), help="문제 트리 (기본 P3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("id"); s.set_defaults(f=cmd_show)
    s = sub.add_parser("ls"); s.add_argument("type", nargs="?"); s.add_argument("--status"); s.set_defaults(f=cmd_ls)
    s = sub.add_parser("check"); s.add_argument("words", nargs="+"); s.set_defaults(f=cmd_check)
    s = sub.add_parser("add"); s.add_argument("id"); s.add_argument("type"); s.add_argument("text")
    s.add_argument("--src"); s.add_argument("--status"); s.set_defaults(f=cmd_add)
    s = sub.add_parser("close"); s.add_argument("id"); s.add_argument("--evidence", required=True); s.set_defaults(f=cmd_close)
    s = sub.add_parser("set"); s.add_argument("id"); s.add_argument("field", choices=("t", "src")); s.add_argument("value"); s.set_defaults(f=cmd_set)
    s = sub.add_parser("edge"); s.add_argument("src"); s.add_argument("rel"); s.add_argument("dst"); s.set_defaults(f=cmd_edge)
    s = sub.add_parser("selftest"); s.set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    sys.exit(a.f(load(a.p), a) or 0)


if __name__ == "__main__":
    main()
