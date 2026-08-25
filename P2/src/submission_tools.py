"""P2 제출 감사와 리더보드 점수 원장.

  python src/submission_tools.py audit subs/E002_final.csv --pred preds/test_fixed_depth_E002.npy
  python src/submission_tools.py public add subs/E002_final.csv 1.23 --local 1.30 --note "first"
  python src/submission_tools.py public list
  python src/submission_tools.py selftest
"""
import argparse
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

AUDITS = C.ROOT / "experiments" / "submission_audits.jsonl"
SCORES = C.ROOT / "experiments" / "public_scores.csv"
COLS = C.KEY + ["temp"]
SCORE_COLS = ["sha256", "time", "file", "public", "local", "note"]


def save_pool_pred(path, values):
    path = Path(path)
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("pool 예측은 floating dtype이어야 합니다")
    if not np.isfinite(values).all():
        raise ValueError("pool 예측에 비유한값이 있습니다")
    if np.unique(values).size < 3:
        raise ValueError("pool 예측은 고유값이 3개 이상이어야 합니다")
    stored = values.astype(np.float32)
    if not np.isfinite(stored).all():
        raise ValueError("float32 변환 후 비유한값이 있습니다")
    np.save(path, stored)
    return stored


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _upsert_jsonl(path, row):
    path = Path(path)
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [old for old in rows if old.get("sha256") != row["sha256"]] + [row]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def audit(csv_path, pred_path, ledger=AUDITS, record=True):
    csv_path = Path(csv_path)
    if pred_path is None:
        raise ValueError("audit에는 --pred가 필요합니다")
    fails = []
    try:
        sub = pd.read_csv(csv_path, float_precision="round_trip")
    except Exception as exc:
        print(f"FAIL CSV 읽기: {exc}")
        return 1, None
    idx = pd.read_csv(C.DATA / "test_index.csv")
    if list(sub.columns) != COLS:
        fails.append(f"열 이름/순서 {list(sub.columns)} != {COLS}")
    if len(sub) != len(idx):
        fails.append(f"행 수 {len(sub):,} != {len(idx):,}")
    if set(C.KEY).issubset(sub.columns):
        if sub[C.KEY].isna().any().any():
            fails.append("키 결측")
        if sub.duplicated(C.KEY).any():
            fails.append(f"키 중복 {int(sub.duplicated(C.KEY).sum()):,}")
        if len(sub) == len(idx) and not sub[C.KEY].equals(idx[C.KEY]):
            fails.append("canonical test_index 키/행 순서 불일치")
    if "temp" not in sub:
        fails.append("temp 열 없음")
        values = np.array([], dtype=float)
    else:
        values = pd.to_numeric(sub.temp, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            fails.append(f"temp 결측/비유한 {int((~np.isfinite(values)).sum()):,}")
        elif not ((values >= -5) & (values <= 45)).all():
            fails.append(f"temp 범위 이탈 {int(((values < -5) | (values > 45)).sum()):,}")

    npy_exact = None
    if pred_path:
        try:
            pred = np.load(pred_path, allow_pickle=False)
            npy_exact = pred.shape == values.shape and np.array_equal(pred, values)
            npy_match = npy_exact or (pred.shape == values.shape and np.allclose(pred, values, rtol=0, atol=1e-5))
            if not npy_match:
                diff = np.max(np.abs(pred - values)) if pred.shape == values.shape and len(values) else math.inf
                fails.append(f"NPY exact 불일치 shape={pred.shape}/{values.shape} maxdiff={diff}")
        except Exception as exc:
            fails.append(f"NPY 읽기/비교 실패: {exc}")

    digest = sha256(csv_path)
    stats = {"sha256": digest, "time": datetime.now(timezone.utc).isoformat(),
             "file": str(csv_path.resolve()), "rows": len(sub), "pred": str(Path(pred_path).resolve()) if pred_path else "",
             "npy_exact": npy_exact}
    if len(values) and np.isfinite(values).all():
        base = pd.read_csv(C.DATA / "baseline_interp.csv", float_precision="round_trip").temp.to_numpy(float)
        delta = values - base if len(values) == len(base) else np.array([])
        stats.update({"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()),
                      "baseline_mae": float(np.mean(np.abs(delta))) if len(delta) else None,
                      "baseline_mean_delta": float(delta.mean()) if len(delta) else None})
    for msg in fails:
        print("FAIL " + msg)
    if fails:
        print("판정: STOP")
        return 1, stats
    print(f"sha256={digest} rows={len(sub):,} mean={stats['mean']:.6f} range=[{stats['min']:.6f},{stats['max']:.6f}]")
    print(f"baseline mean|delta|={stats['baseline_mae']:.6f} mean_delta={stats['baseline_mean_delta']:+.6f}")
    if pred_path:
        print("npy_exact=True")
    if record:
        _upsert_jsonl(ledger, stats)
    print("판정: OK")
    return 0, stats


def public_add(csv_path, public, local=None, note="", score_path=SCORES, audit_path=AUDITS):
    if not math.isfinite(public) or public < 0 or local is not None and (not math.isfinite(local) or local < 0):
        print("FAIL RMSE는 0 이상의 유한값이어야 합니다")
        return 1
    with tempfile.TemporaryDirectory() as td:
        csv_values = pd.read_csv(csv_path, usecols=["temp"]).temp.to_numpy(float)
        pred_file = Path(td) / "public_add.npy"
        save_pool_pred(pred_file, csv_values)
        rc, info = audit(csv_path, pred_file, ledger=audit_path)
    if rc:
        return rc
    score_path = Path(score_path)
    old = pd.read_csv(score_path, dtype={"sha256": str}) if score_path.exists() else pd.DataFrame(columns=SCORE_COLS)
    row = pd.DataFrame([[info["sha256"], datetime.now(timezone.utc).isoformat(), str(Path(csv_path).resolve()),
                         public, np.nan if local is None else local, note]], columns=SCORE_COLS)
    out = pd.concat([old[old.sha256 != info["sha256"]], row], ignore_index=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(score_path, index=False)
    print(f"public upsert: {info['sha256']} public={public:.6f}")
    return 0


def public_list(score_path=SCORES):
    score_path = Path(score_path)
    if not score_path.exists() or not len(pd.read_csv(score_path)):
        print("기록 없음")
        return 0
    d = pd.read_csv(score_path).sort_values("public", kind="stable").reset_index(drop=True)
    best = d.public.iat[0]
    for r in d.itertuples(index=False):
        local = "" if pd.isna(r.local) else f" local={r.local:.6f}"
        print(f"public={r.public:.6f} delta_best={r.public-best:+.6f}{local} sha256={r.sha256} file={r.file} note={r.note}")
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        idx = pd.read_csv(C.DATA / "test_index.csv")
        good = idx[C.KEY].assign(temp=np.linspace(20.0, 21.0, len(idx)))
        good_path, pred_path = td / "good.csv", td / "good.npy"
        good.to_csv(good_path, index=False)
        save_pool_pred(pred_path, good.temp.to_numpy())
        audits, scores = td / "audits.jsonl", td / "scores.csv"
        assert audit(good_path, pred_path, audits)[0] == 0
        assert audit(good_path, pred_path, audits)[0] == 0
        assert len(audits.read_text(encoding="utf-8").splitlines()) == 1
        bad = good.iloc[::-1]
        bad_path = td / "bad.csv"
        bad.to_csv(bad_path, index=False)
        assert audit(bad_path, pred_path, ledger=audits)[0] == 1
        blocked = td / "blocked.npy"
        try:
            save_pool_pred(blocked, np.ones(3, dtype=float))
        except ValueError:
            pass
        else:
            raise AssertionError("상수 pool 예측이 저장됨")
        assert not blocked.exists()
        assert public_add(good_path, 1.2, 1.3, "a", scores, audits) == 0
        assert public_add(good_path, 1.1, 1.25, "updated", scores, audits) == 0
        saved = pd.read_csv(scores)
        assert len(saved) == 1 and saved.public.iat[0] == 1.1 and saved.note.iat[0] == "updated"
        assert public_add(good_path, -1, score_path=scores, audit_path=audits) == 1
        assert public_list(scores) == 0
    print("selftest OK")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit"); a.add_argument("csv"); a.add_argument("--pred", required=True)
    p = sub.add_parser("public"); ps = p.add_subparsers(dest="public_cmd", required=True)
    add = ps.add_parser("add"); add.add_argument("csv"); add.add_argument("score", type=float)
    add.add_argument("--local", type=float); add.add_argument("--note", default="")
    ps.add_parser("list")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.cmd == "audit":
        return audit(args.csv, args.pred)[0]
    if args.cmd == "public" and args.public_cmd == "add":
        return public_add(args.csv, args.score, args.local, args.note)
    if args.cmd == "public":
        return public_list()
    return selftest()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
