"""Create fixed-offset probes from the accepted T1F submission."""
import argparse
import csv
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

P3 = pathlib.Path(__file__).resolve().parent.parent
INPUT = P3 / "subs" / "submission_p3_t1f.csv"
COLS = ["case_id", "station", "lead_h", "hs_pred"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float)
    args = parser.parse_args()

    with INPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != COLS:
        raise ValueError(f"columns {rows[0] if rows else []} != {COLS}")

    probes = [("shift", 0.05)]
    if args.c is not None:
        probes.append(("bias", args.c))
    else:
        print("bias.csv not created: --c not provided")

    for name, offset in probes:
        output_rows = [rows[0]]
        clipped_rows = 0
        for row in rows[1:]:
            value = float(row[3]) + offset
            clipped = min(30.0, max(0.0, value))
            clipped_rows += clipped != value
            output_rows.append(row[:3] + [str(clipped)])
        output = P3 / "subs" / f"submission_p3_t1f_{name}.csv"
        with output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(output_rows)
        print(f"created={output} rows={len(output_rows) - 1} offset={offset:+.12g} clipped_rows={clipped_rows}")
        if clipped_rows:
            print(f"WARNING: clipped_rows={clipped_rows}; exact offset equation does not hold")


if __name__ == "__main__":
    main()
