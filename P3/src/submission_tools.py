"""Audit P3 submissions and keep a small public-score ledger."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
P3 = pathlib.Path(__file__).resolve().parent.parent
KEY = ["case_id", "station", "lead_h"]
COLS = KEY + ["hs_pred"]
LEADS = [3, 6, 9, 12, 18, 24]


class AuditError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_pool_pred(path, values):
    """Validate and save a continuous prediction pool preserving its dtype."""
    path = pathlib.Path(path)
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("prediction pool must have a floating dtype")
    if not values.size or not np.isfinite(values).all():
        raise ValueError("prediction pool must be non-empty and finite")
    if np.unique(values).size < 3:
        raise ValueError("prediction pool must have at least 3 unique values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return path


def _record_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = [item for item in existing if item.get("file") != row["file"]]
    existing.append(row)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in existing), encoding="utf-8")


def audit(csv_path, pred_path, root=P3, record=True, run_preflight=True, quiet=False):
    if pred_path is None:
        raise AuditError("--pred is required")
    csv_path, root = pathlib.Path(csv_path).resolve(), pathlib.Path(root).resolve()
    index_path = root / "P3_data" / "test_index.csv"
    base_path = root / "P3_data" / "baseline_persistence.csv"
    sub, index = pd.read_csv(csv_path), pd.read_csv(index_path)
    errors = []
    if list(sub.columns) != COLS:
        errors.append(f"columns {list(sub.columns)} != {COLS}")
    if len(sub) != len(index):
        errors.append(f"rows {len(sub)} != {len(index)}")
    if all(c in sub for c in KEY):
        if sub[KEY].isna().any().any():
            errors.append("key null")
        if sub.duplicated(KEY).any():
            errors.append(f"duplicate keys {int(sub.duplicated(KEY).sum())}")
        if len(sub) == len(index) and not sub[KEY].equals(index[KEY]):
            errors.append("canonical key order mismatch")
        bad_cases = [cid for cid, g in sub.groupby("case_id", sort=False)
                     if g.station.nunique() != 1 or g.lead_h.tolist() != LEADS]
        if bad_cases:
            errors.append(f"case lead completeness/order {len(bad_cases)}")
    if "hs_pred" in sub:
        values = pd.to_numeric(sub.hs_pred, errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            errors.append("hs_pred null/non-finite/non-numeric")
        elif ((values < 0) | (values > 30)).any():
            errors.append(f"hs_pred outside [0,30] {int(((values < 0) | (values > 30)).sum())}")
    else:
        values = np.array([])
    expected_cases = index.case_id.drop_duplicates().tolist()
    canonical_bad = [cid for cid, g in index.groupby("case_id", sort=False)
                     if g.station.nunique() != 1 or g.lead_h.tolist() != LEADS]
    if canonical_bad:
        errors.append("canonical index case structure invalid")
    pred_delta = None
    if not errors:
        pred = np.load(pred_path, allow_pickle=False)
        expected_shapes = [(len(index),), (len(expected_cases), len(LEADS))]
        if pred.shape not in expected_shapes:
            errors.append(f"pred shape {pred.shape} not in {expected_shapes}")
        elif not np.isfinite(pred).all():
            errors.append("pred null/non-finite")
        else:
            pred_delta = float(np.max(np.abs(pred.reshape(-1).astype(float) - values)))
            if pred_delta > 1e-12:
                errors.append(f"pred mapping mismatch max_delta={pred_delta:.3g}")
    if errors:
        raise AuditError("; ".join(errors))
    base = pd.read_csv(base_path)
    if list(base.columns) != COLS or not base[KEY].equals(index[KEY]):
        raise AuditError("baseline schema/key order mismatch")
    delta = values - base.hs_pred.to_numpy(float)
    digest = sha256(csv_path)
    if run_preflight and (root.parent / "scripts" / "preflight.py").exists():
        proc = subprocess.run([sys.executable, str(root.parent / "scripts" / "preflight.py"), "sub",
                               str(csv_path), "--index", str(index_path), "--base", str(base_path)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode:
            raise AuditError("preflight sub failed: " + (proc.stdout + proc.stderr).strip())
    row = {"sha256": digest, "time": datetime.now(timezone.utc).isoformat(),
           "file": str(csv_path), "rows": len(sub), "mean": float(values.mean()),
           "min": float(values.min()), "max": float(values.max()),
           "baseline_mean_abs_delta": float(np.mean(np.abs(delta))),
           "baseline_mean_delta": float(delta.mean()), "pred": str(pathlib.Path(pred_path).resolve()),
           "pred_max_delta": pred_delta}
    if record:
        _record_jsonl(root / "experiments" / "submission_audits.jsonl", row)
    if not quiet:
        print(f"audit=OK rows={row['rows']} sha256={digest}")
        print(f"hs mean={row['mean']:.6f} range=[{row['min']:.6f},{row['max']:.6f}] "
              f"baseline_dist={row['baseline_mean_abs_delta']:.6f} mean_delta={row['baseline_mean_delta']:+.6f}")
        print(f"pred_shape={np.load(pred_path, mmap_mode='r').shape} max_delta={pred_delta:.3g}")
    return row


def public_add(csv_path, score, local=None, note="", root=P3, run_preflight=True, pred_path=None):
    scores = [score] + ([] if local is None else [local])
    if not all(np.isfinite(scores)) or any(x < 0 for x in scores):
        raise ValueError("RMSE scores must be finite and >= 0")
    result = audit(csv_path, pred_path, root=root, run_preflight=run_preflight)
    ledger = pathlib.Path(root) / "experiments" / "public_scores.csv"
    cols = ["sha256", "time", "file", "public", "local", "note"]
    old = pd.read_csv(ledger) if ledger.exists() else pd.DataFrame(columns=cols)
    row = {"sha256": result["sha256"], "time": datetime.now(timezone.utc).isoformat(),
           "file": result["file"], "public": float(score),
           "local": "" if local is None else float(local), "note": note}
    old = old[old.sha256 != result["sha256"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[cols].to_csv(ledger, index=False)
    print(f"public upsert sha256={result['sha256']} score={score:.6f}")


def public_list(root=P3):
    ledger = pathlib.Path(root) / "experiments" / "public_scores.csv"
    if not ledger.exists() or not len(d := pd.read_csv(ledger)):
        print("public scores: empty")
        return
    d.public = pd.to_numeric(d.public)
    d = d.sort_values("public").reset_index(drop=True)
    d.insert(0, "rank", np.arange(1, len(d) + 1))
    d.insert(2, "delta", d.public - d.public.iloc[0])
    print(f"best={d.public.iloc[0]:.6f}")
    print(d[["rank", "public", "delta", "local", "file", "note"]].to_string(index=False))


def selftest():
    real = [P3 / "experiments" / "submission_audits.jsonl", P3 / "experiments" / "public_scores.csv"]
    before = {p: p.read_bytes() if p.exists() else None for p in real}
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td) / "P3"
        (root / "P3_data").mkdir(parents=True)
        rows = [(f"C{i:04d}", "G-ORS", lead) for i in (1, 2) for lead in LEADS]
        index = pd.DataFrame(rows, columns=KEY)
        index.to_csv(root / "P3_data" / "test_index.csv", index=False)
        sub = index.assign(hs_pred=np.arange(len(index), dtype=np.float32) + 1)
        sub.to_csv(root / "ok.csv", index=False)
        index.assign(hs_pred=1.0).to_csv(root / "P3_data" / "baseline_persistence.csv", index=False)
        pred_values = np.arange(len(index), dtype=np.float32).reshape(2, 6) + 1
        save_pool_pred(root / "pred.npy", pred_values)
        save_pool_pred(root / "pred_flat.npy", pred_values.reshape(-1))
        first = audit(root / "ok.csv", root / "pred.npy", root=root, run_preflight=False, quiet=True)
        audit(root / "ok.csv", root / "pred_flat.npy", root=root, run_preflight=False, quiet=True)
        audit(root / "ok.csv", root / "pred.npy", root=root, run_preflight=False, quiet=True)
        bad = sub.iloc[::-1]
        bad.to_csv(root / "bad.csv", index=False)
        try:
            audit(root / "bad.csv", root / "pred.npy", root=root, run_preflight=False, quiet=True)
            raise AssertionError("bad order accepted")
        except AuditError:
            pass
        public_add(root / "ok.csv", 0.7, 0.8, "first", root, False, root / "pred.npy")
        public_add(root / "ok.csv", 0.6, 0.8, "updated", root, False, root / "pred.npy")
        audits = (root / "experiments" / "submission_audits.jsonl").read_text().splitlines()
        ledger = pd.read_csv(root / "experiments" / "public_scores.csv")
        assert len(audits) == 1 and len(ledger) == 1 and ledger.public.iat[0] == 0.6
        assert ledger.sha256.iat[0] == first["sha256"]
    assert before == {p: p.read_bytes() if p.exists() else None for p in real}
    print("selftest=OK success/failure/upsert actual_ledger_unchanged=1")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("csv"); a.add_argument("--pred", required=True)
    public = sub.add_parser("public").add_subparsers(dest="public_command", required=True)
    add = public.add_parser("add"); add.add_argument("csv"); add.add_argument("score", type=float)
    add.add_argument("--local", type=float); add.add_argument("--note", default=""); add.add_argument("--pred", required=True)
    public.add_parser("list")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.csv, args.pred)
    elif args.command == "selftest":
        selftest()
    elif args.public_command == "add":
        public_add(args.csv, args.score, args.local, args.note, pred_path=args.pred)
    else:
        public_list()


if __name__ == "__main__":
    try:
        main()
    except (AuditError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"audit=FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
