"""P1 제출 감사와 Public 점수 원장."""
import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check as C

AUDITS = C.ROOT / "experiments/submission_audits.jsonl"
SCORES = C.ROOT / "experiments/public_scores.csv"
SCORE_COLUMNS = ["sha256", "file", "mtime", "recorded_at", "public", "local", "note"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_pool_pred(path, values):
    """Write one-dimensional continuous pool scores, never binary labels."""
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or np.unique(values).size < 3):
        raise ValueError("pool prediction must be finite floating scores in [0,1] with >=3 unique values")
    stored = values.astype(np.float32)
    if np.unique(stored).size < 3:
        raise ValueError("pool prediction loses score resolution in float32")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, stored)


def audit(path, pred, canonical=None, baseline=None, log_path=AUDITS, use_check=True):
    if pred is None:
        raise ValueError("--pred is required")
    path = Path(path)
    canonical = Path(canonical or C.DATA / "sample_submission.csv")
    baseline = Path(baseline or C.DATA / "baseline_rule.csv")
    fails = []
    try:
        if use_check and C.sub(path):
            fails.append("기존 check.sub 실패")
        sub, canon, base = pd.read_csv(path), pd.read_csv(canonical), pd.read_csv(baseline)
        allowed = [C.KEY + ["label"], C.KEY + ["label", "anomaly_type"]]
        if list(sub.columns) not in allowed:
            fails.append(f"열 이름/순서 {list(sub.columns)}")
        if len(sub) != len(canon):
            fails.append(f"행 수 {len(sub):,} != {len(canon):,}")
        if sub.duplicated(C.KEY).any():
            fails.append(f"키 중복 {int(sub.duplicated(C.KEY).sum()):,}")
        if sub[C.KEY].isna().any().any():
            fails.append("키 결측")
        if len(sub) == len(canon) and not sub[C.KEY].astype(str).equals(canon[C.KEY].astype(str)):
            fails.append("canonical 키/행 순서 불일치")
        y = pd.to_numeric(sub.label, errors="coerce")
        if y.isna().any() or not np.isfinite(y.to_numpy()).all() or not y.isin([0, 1]).all():
            fails.append("label은 결측 없는 유한 정수 0/1이어야 함")
        elif not np.equal(y.to_numpy(), y.astype(int).to_numpy()).all():
            fails.append("label 정수성 오류")
        if len(base) != len(sub) or not base[C.KEY].astype(str).equals(sub[C.KEY].astype(str)):
            fails.append("baseline 키/행 순서 불일치")
            mismatch = pos_delta = None
        else:
            by = pd.to_numeric(base.label, errors="coerce").to_numpy()
            mismatch = int((y.to_numpy() != by).sum())
            pos_delta = int(y.sum() - by.sum())
        p = np.load(pred, allow_pickle=False)
        if p.shape != (len(sub),) or not np.isfinite(p).all():
            fails.append(f"pred shape/유한 오류 {p.shape}")
        elif not np.array_equal((p >= 0.5).astype(int), y.astype(int).to_numpy()):
            fails.append(f"pred→label 불일치 {int(((p >= 0.5).astype(int) != y.astype(int).to_numpy()).sum()):,}")
    except Exception as e:
        fails.append(f"읽기/검증 오류: {e}")
        sub, y, mismatch, pos_delta = pd.DataFrame(), pd.Series(dtype=float), None, None
    if fails:
        for msg in fails:
            print("  FAIL " + msg)
        print("판정: STOP")
        return 1, None
    sha = digest(path)
    record = {
        "sha256": sha, "file": path.name, "path": str(path.resolve()),
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().isoformat(),
        "audited_at": datetime.now().astimezone().isoformat(), "rows": len(sub),
        "positive": int(y.sum()), "positive_rate": float(y.mean()),
        "baseline_mismatch": mismatch, "baseline_mismatch_rate": mismatch / len(sub),
        "baseline_positive_delta": pos_delta, "pred": str(Path(pred).resolve()),
    }
    existing = []
    if Path(log_path).exists():
        existing = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not any(r["sha256"] == sha for r in existing):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  rows={len(sub):,} positive={int(y.sum()):,} ({y.mean():.4%}) baseline_diff={mismatch:,} ({mismatch/len(sub):.4%}) pos_delta={pos_delta:+,}")
    print(f"  sha256={sha}")
    print("판정: OK")
    return 0, record


def public_add(path, score, local=None, note="", ledger=SCORES, **audit_kwargs):
    if not 0 <= score <= 1 or local is not None and not 0 <= local <= 1:
        print("  FAIL P1 F1 public/local은 0~1")
        print("판정: STOP")
        return 1
    code, rec = audit(path, **audit_kwargs)
    if code:
        return code
    row = {"sha256": rec["sha256"], "file": rec["file"], "mtime": rec["mtime"],
           "recorded_at": datetime.now().astimezone().isoformat(), "public": float(score),
           "local": "" if local is None else float(local), "note": note}
    ledger = Path(ledger)
    old = pd.read_csv(ledger, dtype=str).to_dict("records") if ledger.exists() else []
    old = [r for r in old if r["sha256"] != rec["sha256"]] + [row]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(old, columns=SCORE_COLUMNS).to_csv(ledger, index=False)
    print(f"upsert {rec['file']} public={score:.6f} sha256={rec['sha256']}")
    return 0


def public_list(ledger=SCORES):
    if not Path(ledger).exists() or Path(ledger).stat().st_size == 0:
        print("Public 점수 기록 없음")
        return 0
    d = pd.read_csv(ledger)
    if d.empty:
        print("Public 점수 기록 없음")
        return 0
    d = d.sort_values("public", ascending=False)
    best = d.public.iloc[0]
    d["delta_best"] = d.public - best
    print(d[["file", "public", "delta_best", "local", "note", "sha256"]].to_string(index=False))
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        keys = pd.DataFrame({"station": ["S"] * 3, "year": [2026] * 3, "layer": [1] * 3, "time": ["t1", "t2", "t3"]})
        canon = keys.assign(label=0, anomaly_type="")
        base = canon.copy()
        good = keys.assign(label=[0, 1, 0])
        canon.to_csv(td / "canon.csv", index=False); base.to_csv(td / "base.csv", index=False); good.to_csv(td / "good.csv", index=False)
        save_pool_pred(td / "pred.npy", np.array([0.1, 0.9, 0.3]))
        try:
            save_pool_pred(td / "rejected.npy", np.array([0, 1], dtype=np.uint8))
            raise AssertionError("binary writer input was accepted")
        except ValueError:
            assert not (td / "rejected.npy").exists()
        kw = dict(canonical=td / "canon.csv", baseline=td / "base.csv", log_path=td / "audit.jsonl", use_check=False)
        assert audit(td / "good.csv", td / "pred.npy", **kw)[0] == 0
        assert audit(td / "good.csv", td / "pred.npy", **kw)[0] == 0
        assert len((td / "audit.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        np.save(td / "wrong.npy", np.array([0.9, 0.1]))
        assert audit(td / "good.csv", td / "wrong.npy", **kw)[0] == 1
        bad = good.iloc[::-1]; bad.to_csv(td / "bad.csv", index=False)
        assert audit(td / "bad.csv", td / "pred.npy", **kw)[0] == 1
        assert public_add(td / "good.csv", 0.6, pred=td / "pred.npy", ledger=td / "scores.csv", **kw) == 0
        assert public_add(td / "good.csv", 0.7, pred=td / "pred.npy", note="updated", ledger=td / "scores.csv", **kw) == 0
        saved = pd.read_csv(td / "scores.csv")
        assert len(saved) == 1 and saved.public.iloc[0] == 0.7
    print("selftest 판정: OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit"); a.add_argument("csv"); a.add_argument("--pred", required=True)
    p = sub.add_parser("public"); ps = p.add_subparsers(dest="public_cmd", required=True)
    a = ps.add_parser("add"); a.add_argument("csv"); a.add_argument("score", type=float); a.add_argument("--pred", required=True); a.add_argument("--local", type=float); a.add_argument("--note", default="")
    ps.add_parser("list"); sub.add_parser("selftest")
    args = ap.parse_args()
    if args.cmd == "audit": return audit(args.csv, args.pred)[0]
    if args.cmd == "selftest": return selftest()
    if args.public_cmd == "add": return public_add(args.csv, args.score, args.local, args.note, pred=args.pred)
    return public_list()


if __name__ == "__main__":
    raise SystemExit(main())
