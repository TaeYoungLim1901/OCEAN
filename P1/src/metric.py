"""P1 F1과 필수 세그먼트 평가."""
import numpy as np
import pandas as pd


def f1(y, p):
    y, p = np.asarray(y, int), np.asarray(p, int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn), tp, fp, fn


def evaluate(frame, pred, by):
    if not by:
        raise ValueError("by는 필수")
    d = frame.assign(_pred=np.asarray(pred, int))
    rows = []
    keys = by[0] if len(by) == 1 else by
    for key, g in d.groupby(keys, dropna=False):
        score, tp, fp, fn = f1(g.label, g._pred)
        key = (key,) if len(by) == 1 else key
        rows.append((*key, len(g), int(g.label.sum()), score, tp, fp, fn))
    return pd.DataFrame(rows, columns=by + ["n", "positive", "f1", "tp", "fp", "fn"])


if __name__ == "__main__":
    y = np.array([0, 1, 1])
    assert f1(y, y)[0] == 1.0
    print("판정: OK")

