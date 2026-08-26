import argparse
import csv
from itertools import combinations
import math
from pathlib import Path


KEY = ["station", "layer", "time"]
SCORE_A = 33.3335
SCORE_B = 12.54789


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != KEY + ["temp"]:
        raise ValueError(f"columns=STOP {rows[0] if rows else []}")
    keys, values = [], []
    for row in rows[1:]:
        if len(row) != 4:
            raise ValueError(f"row=STOP {row}")
        keys.append(tuple(row[:3]))
        values.append(float(row[3]))
    return keys, values


def parallel(delta, direction):
    if len(delta) != len(direction) or [x != 0 for x in delta] != [x != 0 for x in direction]:
        return None
    nonzero = [i for i, x in enumerate(direction) if x != 0]
    if not nonzero:
        return None
    x = [delta[i] for i in nonzero]
    y = [direction[i] for i in nonzero]
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    scale = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    corr = covariance / scale if scale else 0.0
    if not math.isfinite(corr) or min(abs(corr - 1), abs(corr + 1)) > 1e-9:
        return None
    weight = sum(a * b for a, b in zip(delta, direction)) / sum(x * x for x in direction)
    if max(abs(a - weight * b) for a, b in zip(delta, direction)) > 1e-9:
        return None
    return weight


def score(rmse):
    return SCORE_A - SCORE_B * rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--base")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ledger_path = root / "experiments" / "public_scores.csv"
    with ledger_path.open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    if not ledger:
        raise ValueError("ledger=STOP empty")

    if args.base:
        base_path = Path(args.base)
    else:
        base_path = Path(min(ledger, key=lambda row: float(row["public"]))["file"])
        if not base_path.is_absolute():
            base_path = root / base_path

    candidate_keys, candidate_values = read_csv(args.candidate)
    base_keys, base_values = read_csv(base_path)
    if candidate_keys != base_keys:
        raise ValueError("key_order=STOP candidate/base keys differ")

    direction = [candidate - base for candidate, base in zip(candidate_values, base_values)]
    base_public = None
    entries = []
    base_resolved = base_path.resolve()
    for row in ledger:
        path = Path(row["file"])
        if not path.is_absolute():
            path = root / path
        if path.resolve() == base_resolved:
            base_public = float(row["public"])
        if not path.exists():
            continue
        keys, values = read_csv(path)
        if keys == base_keys:
            entries.append((path, float(row["public"]), values))
    if base_public is None:
        raise ValueError("base_public=STOP base is not in public ledger")

    pairs = []
    for left, right in combinations(entries, 2):
        delta = [b - a for a, b in zip(left[2], right[2])]
        if parallel(delta, direction) is not None:
            pairs.append((left, right))

    points = []
    for path, public, values in entries:
        if path.resolve() == base_resolved:
            weight = 0.0
        else:
            weight = parallel([value - base for value, base in zip(values, base_values)], direction)
        if weight is not None:
            points.append((weight, public, path.name))

    exact = False
    point_rmse = point_score = e_de = residual = None
    nonzero_points = [(w, p) for w, p, _ in points if abs(w) > 1e-12]
    if pairs and len(nonzero_points) >= 2 and any(direction):
        weights = [w for w, _ in nonzero_points]
        observed = [p**2 for _, p in nonzero_points]
        base_mse = base_public**2
        values = [value - base_mse for value in observed]
        a11 = sum(w**4 for w in weights)
        a12 = sum(-2 * w**3 for w in weights)
        a22 = sum(4 * w**2 for w in weights)
        b1 = sum(w**2 * value for w, value in zip(weights, values))
        b2 = sum(-2 * w * value for w, value in zip(weights, values))
        determinant = a11 * a22 - a12**2
        if abs(determinant) > 1e-20:
            e_d2_public = (b1 * a22 - b2 * a12) / determinant
            e_de = (a11 * b2 - a12 * b1) / determinant
            fitted = [
                base_mse + w**2 * e_d2_public - 2 * w * e_de
                for w, _, _ in points
            ]
            measured = [p**2 for _, p, _ in points]
            residual = max(abs(measured - fitted) for measured, fitted in zip(measured, fitted))
            point_mse = base_mse + e_d2_public - 2 * e_de
            if point_mse >= 0:
                exact = True
                point_rmse = math.sqrt(point_mse)
                point_score = score(point_rmse)

    base_mse = base_public**2
    e_d2 = sum(value**2 for value in direction) / len(direction)
    radius = 2 * math.sqrt(e_d2 * base_mse)
    range_mse = sorted((max(0.0, base_mse + e_d2 - radius), base_mse + e_d2 + radius))
    range_rmse = [math.sqrt(x) for x in range_mse]
    range_score = sorted((score(range_rmse[0]), score(range_rmse[1])))

    print(f"base={base_path} base_public={base_public:.6f} base_score={score(base_public):.6f}")
    print(
        f"changed_rows={sum(value != 0 for value in direction)} E[d2]={e_d2:.9f} "
        f"max|d|={max(abs(value) for value in direction):.9f} mean_d={sum(direction) / len(direction):.9f}"
    )
    print(f"exact={exact}", end="")
    if exact:
        names = ",".join(name for _, _, name in sorted(points))
        print(
            f" parallel_dir={names} w=1.000000 E[d.e]={e_de:.9f} "
            f"residual={residual:.9f}"
        )
        print(f"point_rmse={point_rmse:.6f} point_score={point_score:.6f}")
    else:
        print()
    print(f"range_rmse=[{range_rmse[0]:.6f},{range_rmse[1]:.6f}] range_score=[{range_score[0]:.6f},{range_score[1]:.6f}]")
    print(f"measurable={range_rmse[1] - range_rmse[0] >= 0.000002}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"STOP {exc}")
        raise SystemExit(1)
