"""P1 exp016: add transferred neighbor-regression residual features."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp013_shift import BASE_FEATURES, REFERENCE, fit_full, save_submission
from features import make_features
from q3_neighbor import (
    _derive_neighbor_signals,
    _fit_residual,
    _impute_neighbors,
    _neighbor_residuals,
)


EXTRA = ("n1", "n2_432", "n3", "n4")


def _add_ids(raw):
    out = raw.copy()
    out["_time"] = pd.to_datetime(out.time, utc=True)
    out["_row"] = np.arange(len(out), dtype=np.int64)
    return out


def _test_neighbor_residuals(train, test):
    """Apply train-fitted q3 regressions to test neighbor observations."""
    train_wide = train.pivot_table(
        index=["station", "_time"], columns="layer", values="temp", aggfunc="median"
    )
    test_wide = test.pivot_table(
        index=["station", "_time"], columns="layer", values="temp", aggfunc="median"
    )
    test_layers = {
        station: set(group.layer.unique())
        for station, group in test.groupby("station", sort=False)
    }
    train_groups = {
        key: group
        for key, group in train.sort_values(
            ["station", "layer", "_time"], kind="stable"
        ).groupby(["station", "layer"], sort=False)
    }
    n1 = np.full(len(test), np.nan, dtype=float)
    work = test.sort_values(["station", "layer", "_time"], kind="stable")
    for (station, layer), group in work.groupby(["station", "layer"], sort=False):
        neighbors = sorted(test_layers.get(station, set()) - {layer})
        train_group = train_groups.get((station, layer))
        if not neighbors or train_group is None:
            continue

        train_index = pd.MultiIndex.from_frame(train_group[["station", "_time"]])
        x_train = train_wide.reindex(columns=neighbors).reindex(train_index).to_numpy(float)
        x_train, usable, _ = _impute_neighbors(x_train)
        fit, _ = _fit_residual(x_train, train_group.temp.to_numpy(float), usable)
        if fit is None:
            continue

        test_index = pd.MultiIndex.from_frame(group[["station", "_time"]])
        x_test = test_wide.reindex(columns=neighbors).reindex(test_index).to_numpy(float)
        x_test, usable, _ = _impute_neighbors(x_test)
        target = group.temp.to_numpy(float)
        valid = usable & np.isfinite(target) & np.isfinite(x_test).all(axis=1)
        if valid.any():
            intercept, coef = fit
            rows = group.loc[valid, "_row"].to_numpy(np.int64)
            n1[rows] = target[valid] - (intercept + x_test[valid] @ coef)
    return n1


def _append_neighbor_features(x_train, x_test, train, test):
    train_n1, _ = _neighbor_residuals(train, test)
    test_n1 = _test_neighbor_residuals(train, test)
    train_signals = _derive_neighbor_signals(train, train_n1)
    test_signals = _derive_neighbor_signals(test, test_n1)

    out_train, out_test = x_train.copy(), x_test.copy()
    for name in EXTRA:
        out_train[name] = train_signals[name]
        out_test[name] = test_signals[name]
    columns = list(out_train.columns)
    assert columns == list(out_test.columns) and len(columns) == 46
    return out_train, out_test, columns, train_n1, test_n1


def _print_nan_rows(train, test, train_n1, test_n1):
    g_train = int(train.station.eq("G-ORS").sum())
    g_test = int(test.station.eq("G-ORS").sum())
    print(
        "nan_rows | G-ORS train={} n1={} match={} | G-ORS test={} n1={} match={}".format(
            g_train, int(np.isnan(train_n1).sum()), int(g_train == np.isnan(train_n1).sum()),
            g_test, int(np.isnan(test_n1).sum()), int(g_test == np.isnan(test_n1).sum()),
        )
    )


def _print_importance(model, columns):
    gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    order = np.argsort(-gain, kind="stable")
    ranks = {columns[index]: rank for rank, index in enumerate(order, 1)}
    shown = list(order[:10])
    shown_names = {columns[index] for index in shown}
    shown.extend(index for index in order if columns[index] in EXTRA and columns[index] not in shown_names)
    print("rank | feature | gain")
    for index in shown:
        print(f"{ranks[columns[index]]} | {columns[index]} | {gain[index]:.6f}")


def main():
    train = _add_ids(pd.read_csv(ROOT / "P1_data/train.csv"))
    test = _add_ids(pd.read_csv(ROOT / "P1_data/test.csv"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    x_train = pd.read_pickle(BASE_FEATURES)
    x_test = make_features(test, reference)[0]
    assert len(x_train) == len(train) and len(x_test) == len(test)
    assert list(x_train.columns) == list(x_test.columns) and len(x_train.columns) == 42

    x_train, x_test, columns, train_n1, test_n1 = _append_neighbor_features(
        x_train, x_test, train, test
    )
    _print_nan_rows(train, test, train_n1, test_n1)

    y = train.label.to_numpy(np.int8)
    model = fit_full(x_train, y, columns, np.ones(len(train), dtype=bool))
    model_path = ROOT / "models/exp016_full.txt"
    model_path.parent.mkdir(exist_ok=True)
    model.save_model(model_path)
    _print_importance(model, columns)
    raw = model.predict(x_test[columns]).astype(np.float64, copy=False)
    save_submission("exp016", raw, columns, len(train))


if __name__ == "__main__":
    main()
