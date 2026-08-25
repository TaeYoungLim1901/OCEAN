import argparse
import csv
import datetime as dt
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from rules_runtime import KST, load


def _today_count(path, today):
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        field = "recorded_at" if "recorded_at" in (reader.fieldnames or []) else "time"
        if field not in (reader.fieldnames or []):
            raise RuntimeError(f"{path} recorded_at 없음")
        count = 0
        for row in reader:
            stamp = row.get(field, "").strip().replace("Z", "+00:00")
            try:
                parsed = dt.datetime.fromisoformat(stamp)
            except ValueError as e:
                raise RuntimeError(f"{path} timestamp 오류") from e
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=KST)
            count += int(parsed.astimezone(KST).date() == today)
        return count


def context():
    rules = load()
    now = dt.datetime.now(KST)
    today = now.date()
    quota = rules["quota_per_problem_per_day"]
    counts = {
        problem: _today_count(ROOT / problem / "experiments" / "public_scores.csv", today)
        for problem in ("P1", "P2", "P3")
    }
    lines = [
        f"오늘 {today.isoformat()} | 예선 마감까지 {(rules['deadline_pred'].date() - today).days}일 | 모델 마감까지 {(rules['deadline_model'] - today).days}일",
        "오늘 제출: " + " ".join(f"{p} {counts[p]}/{quota}" for p in ("P1", "P2", "P3")),
    ]
    exhausted = [p for p in ("P1", "P2", "P3") if counts[p] >= quota]
    if exhausted:
        lines.append("경고: 오늘 제출 quota 소진 — " + " ".join(exhausted))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    try:
        lines = context()
        if not args.plain:
            print("OCEAN session context")
        print("\n".join(lines))
        if args.selftest:
            print("판정: OK")
        return 0
    except Exception as e:
        if args.selftest:
            print(f"판정: STOP ({type(e).__name__}: {e})")
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
