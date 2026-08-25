"""exp003: strictly-forward validation, type specialist, drift and pool diagnostics."""
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features
from metric import evaluate, f1
from submission_tools import save_pool_pred
from train import fill_short_gaps, segment_ids

SEED = 42
KEY = ["station", "layer"]
CLASSES = ["normal", "drift", "flatline", "noise", "offset", "spike"]
BASE_OUTER = 0.7664233576642335
BASE_INNER = 0.7201325043808061


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def forward_mask(raw, folds, held_out, hours=87):
    """Only earlier blocks are training data; purge the validation left boundary."""
    tm = pd.to_datetime(raw.time, utc=True)
    keep = folds < held_out
    for _, idx in raw.groupby(KEY, sort=False).groups.items():
        idx = np.asarray(list(idx))
        va = idx[folds[idx] == held_out]
        if not len(va):
            continue
        cutoff = tm.iloc[va].min() - pd.Timedelta(hours=hours)
        keep[idx[tm.iloc[idx].ge(cutoff).to_numpy()]] = False
    return keep


def robust_reference(raw, mask):
    d = raw.loc[mask, ["station", "layer", "time", "temp"]].copy()
    d["month"] = pd.to_datetime(d.time, utc=True).dt.month
    med = d.groupby(["station", "layer", "month"]).temp.median()
    z = d.join(med.rename("med"), on=["station", "layer", "month"])
    mad = z.assign(dev=(z.temp - z.med).abs()).groupby(["station", "layer", "month"]).dev.median()
    return {
        "median": {"|".join(map(str, k)): float(v) for k, v in med.items()},
        "mad": {"|".join(map(str, k)): float(v) for k, v in mad.items()},
        "global_median": float(d.temp.median()),
    }


def with_reference(base, raw, ref):
    out = base.copy()
    month = pd.to_datetime(raw.time, utc=True).dt.month
    keys = pd.DataFrame({"station": raw.station, "layer": raw.layer, "month": month}).astype(str).agg("|".join, axis=1)
    med = keys.map(ref["median"]).fillna(ref["global_median"]).to_numpy(float)
    mad = keys.map(ref["mad"]).fillna(0.5).clip(lower=0.01).to_numpy(float)
    out["clim_resid"] = ((raw.temp.to_numpy(float) - med) / mad).astype(np.float32)
    return out


def type_target(raw):
    first = raw.anomaly_type.fillna("normal").str.split("+").str[0]
    first = first.where(raw.label.eq(1), "normal")
    return first.map({v: i for i, v in enumerate(CLASSES)}).fillna(0).to_numpy(np.int8)


def params(objective, n_estimators):
    common = dict(
        learning_rate=0.05, num_leaves=63, min_child_samples=80,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        n_estimators=n_estimators, random_state=SEED, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    if objective == "binary":
        return dict(objective="binary", scale_pos_weight=2.0, **common)
    return dict(objective="multiclass", num_class=len(CLASSES), **common)


def fit_fold(x, y, yt, tr, va, fold):
    outputs = {}
    for family, target, n_est in (("binary", y, 700), ("multiclass", yt, 500)):
        model = lgb.LGBMClassifier(**params(family, n_est))
        weight = None if family == "binary" else np.where(target[tr] == 0, 1.0, 2.0)
        model.fit(
            x.loc[tr], target[tr], sample_weight=weight,
            eval_set=[(x.loc[va], target[va])],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        path = ROOT / f"models/exp003_{family}_fold{fold}.txt"
        model.booster_.save_model(path)
        booster = lgb.Booster(model_file=path)
        p = booster.predict(x.loc[va])
        outputs[family] = (p if family == "binary" else 1.0 - p[:, 0], int(model.best_iteration_ or n_est))
    return outputs


def fit_final(x, y, yt, iterations):
    for family, target in (("binary", y), ("multiclass", yt)):
        model = lgb.LGBMClassifier(**params(family, int(iterations[family])))
        weight = None if family == "binary" else np.where(target == 0, 1.0, 2.0)
        model.fit(x, target, sample_weight=weight, callbacks=[lgb.log_evaluation(0)])
        model.booster_.save_model(ROOT / f"models/exp003_{family}_full.txt")


def choose(y, binary, multi, raw, folds):
    use = np.isin(folds, [1, 2])
    seg = segment_ids(raw)
    rows = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        score = (1 - w) * binary + w * multi
        for threshold in np.arange(0.20, 0.701, 0.02):
            for hole in (0, 1, 2):
                pred = fill_short_gaps(score >= threshold, raw, hole, seg)
                rows.append((f1(y[use], pred[use])[0], w, float(threshold), hole))
    return max(rows, key=lambda z: z[0])


def store_score(score, pred):
    z = np.clip(score, 0, 1).astype(np.float32)
    z[pred] = np.maximum(z[pred], 0.5)
    z[~pred] = np.minimum(z[~pred], np.nextafter(np.float32(0.5), np.float32(0)))
    return z


def event_metrics(raw, pred, mask):
    d = raw.loc[mask].copy()
    p = np.asarray(pred)[mask]
    seg = segment_ids(raw)[mask]
    y = d.label.to_numpy(bool)
    ystart = y & np.r_[True, (~y[:-1]) | (seg[1:] != seg[:-1])]
    pstart = p & np.r_[True, (~p[:-1]) | (seg[1:] != seg[:-1])]
    yid = np.cumsum(ystart)
    pid = np.cumsum(pstart)
    true_hit = sum(bool(p[yid == i].any()) for i in range(1, int(yid.max()) + 1))
    pred_hit = sum(bool(y[pid == i].any()) for i in range(1, int(pid.max()) + 1))
    return {
        "true_events": int(ystart.sum()), "pred_events": int(pstart.sum()),
        "hit_true_events": int(true_hit), "overlap_pred_events": int(pred_hit),
        "event_recall": true_hit / max(1, int(ystart.sum())),
        "event_precision": pred_hit / max(1, int(pstart.sum())),
        "row_recall": int((y & p).sum()) / max(1, int(y.sum())),
    }


def adversarial_drift(train, test, xtr, xte):
    def half(raw):
        pos = raw.groupby(KEY, sort=False).cumcount()
        size = raw.groupby(KEY, sort=False).temp.transform("size")
        return (pos / size).to_numpy() < 0.5

    early_tr, early_te = half(train), half(test)
    rng = np.random.default_rng(SEED)

    def take(mask, n=70000):
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, min(n, len(idx)), replace=False))

    tr_a, tr_b = take(early_tr), take(early_te)
    va_a, va_b = take(~early_tr), take(~early_te)
    xx = pd.concat([xtr.iloc[tr_a], xte.iloc[tr_b]], ignore_index=True)
    yy = np.r_[np.zeros(len(tr_a), np.int8), np.ones(len(tr_b), np.int8)]
    xv = pd.concat([xtr.iloc[va_a], xte.iloc[va_b]], ignore_index=True)
    yv = np.r_[np.zeros(len(va_a), np.int8), np.ones(len(va_b), np.int8)]
    model = lgb.LGBMClassifier(**params("binary", 350))
    model.set_params(scale_pos_weight=1.0)
    model.fit(xx, yy, eval_set=[(xv, yv)], callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    model.booster_.save_model(ROOT / "models/exp003_adversarial.txt")
    prob = model.predict_proba(xv, num_iteration=model.best_iteration_)[:, 1]
    imp = pd.DataFrame({"feature": xtr.columns, "gain": model.booster_.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(ROOT / "experiments/exp003_adversarial_importance.csv", index=False)
    return {"auc": float(roc_auc_score(yv, prob)), "train_rows": len(xx), "valid_rows": len(xv),
            "best_iteration": int(model.best_iteration_), "top_features": imp.head(10).to_dict("records")}


def exp002_outer_score(train, base, folds):
    tr = forward_mask(train, folds, 3)
    ref = robust_reference(train, tr)
    x = with_reference(base, train, ref)
    model = lgb.Booster(model_file=ROOT / "models/exp002_outer_honest.txt")
    raw = model.predict(x)
    score = 0.85 * raw + 0.15 * x.rule_score.to_numpy()
    pred = fill_short_gaps(score >= 0.44, train, 2)
    return score, pred


def train_main():
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    y, yt = train.label.to_numpy(np.int8), type_target(train)
    oof_b = np.zeros(len(train), np.float32)
    oof_m = np.zeros(len(train), np.float32)
    refs, best = {}, {"binary": [], "multiclass": []}

    for fold in (1, 2, 3):
        tr, va = forward_mask(train, folds, fold), folds == fold
        ref = robust_reference(train, tr)
        refs[str(fold)] = ref
        x = with_reference(base, train, ref)
        out = fit_fold(x, y, yt, tr, va, fold)
        oof_b[va], oof_m[va] = out["binary"][0], out["multiclass"][0]
        best["binary"].append(out["binary"][1])
        best["multiclass"].append(out["multiclass"][1])
        print(f"fold {fold}: train={tr.sum():,} valid={va.sum():,} bin_iter={out['binary'][1]} multi_iter={out['multiclass'][1]}")

    # Lock tree counts from inner folds before scoring outer. The saved outer boosters
    # may contain more trees, but their fixed prefix never used outer labels for gradients.
    iterations = {k: int(np.median(v[:2])) for k, v in best.items()}
    outer = folds == 3
    x_outer = with_reference(base, train, refs["3"])
    oof_b[outer] = lgb.Booster(model_file=ROOT / "models/exp003_binary_fold3.txt").predict(
        x_outer.loc[outer], num_iteration=iterations["binary"]
    )
    pm_outer = lgb.Booster(model_file=ROOT / "models/exp003_multiclass_fold3.txt").predict(
        x_outer.loc[outer], num_iteration=iterations["multiclass"]
    )
    oof_m[outer] = 1.0 - pm_outer[:, 0]

    inner_f1, multi_weight, threshold, hole = choose(y, oof_b, oof_m, train, folds)
    combined = (1 - multi_weight) * oof_b + multi_weight * oof_m
    pred = fill_short_gaps(combined >= threshold, train, hole)
    inner = np.isin(folds, [1, 2])
    inner_folds = {str(k): f1(y[folds == k], pred[folds == k])[0] for k in (1, 2)}
    outer_stats = f1(y[outer], pred[outer])

    base_outer_score, base_outer_pred = exp002_outer_score(train, base, folds)
    base_outer = f1(y[outer], base_outer_pred[outer])[0]
    pearson = float(np.corrcoef(base_outer_score[outer], combined[outer])[0, 1])
    spearman = float(spearmanr(base_outer_score[outer], combined[outer]).statistic)
    residual_corr = float(np.corrcoef(y[outer] - base_outer_score[outer], y[outer] - combined[outer])[0, 1])

    ref_full = robust_reference(train, np.ones(len(train), bool))
    xfull = with_reference(base, train, ref_full)
    fit_final(xfull, y, yt, iterations)
    xte, _ = make_features(test, {"median": ref_full["median"], "mad": ref_full["mad"]})

    save_pool_pred(ROOT / "preds/oof_binary_exp003.npy", oof_b)
    save_pool_pred(ROOT / "preds/oof_multiclass_exp003.npy", oof_m)

    drift = adversarial_drift(train, test, xfull, xte)
    by_type = evaluate(train.loc[outer].assign(anomaly_type=train.loc[outer].anomaly_type.fillna("normal")), pred[outer], ["anomaly_type"])
    by_sl = evaluate(train.loc[outer], pred[outer], ["station", "layer"])
    by_type.to_csv(ROOT / "experiments/exp003_outer_by_type.csv", index=False)
    by_sl.to_csv(ROOT / "experiments/exp003_outer_by_station_layer.csv", index=False)
    events = event_metrics(train, pred, outer)

    promote = outer_stats[0] > BASE_OUTER or (inner_f1 > BASE_INNER and residual_corr < 0.98 and outer_stats[0] >= BASE_OUTER - 0.002)
    if promote:
        oof_store = store_score(combined - threshold + 0.5, pred)
        save_pool_pred(ROOT / "preds/oof_exp003.npy", oof_store)

    pool = {
        "outer_pearson_exp002_exp003": pearson,
        "outer_spearman_exp002_exp003": spearman,
        "outer_residual_corr_exp002_exp003": residual_corr,
        "exp002_outer_reproduced": base_outer,
    }
    (ROOT / "experiments/exp003_pool.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
    meta = {
        "inner_f1": inner_f1, "inner_fold_f1": inner_folds,
        "outer_f1": outer_stats[0], "outer_tp": outer_stats[1], "outer_fp": outer_stats[2], "outer_fn": outer_stats[3],
        "exp002_outer": base_outer, "outer_delta": outer_stats[0] - base_outer,
        "multi_weight": multi_weight, "threshold": threshold, "hole": hole,
        "iterations": iterations, "fold_best_iterations": best, "fold_references": refs, "full_reference": ref_full,
        "events": events, "drift": drift, "pool": pool, "promoted": promote,
        "submission_sha256": None, "valid_oof_folds": [1, 2, 3], "classes": CLASSES,
    }
    (ROOT / "cache/exp003_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("fold_references", "full_reference", "drift")}, indent=2))
    return 0


def reproduce():
    meta = json.loads((ROOT / "cache/exp003_meta.json").read_text(encoding="utf-8"))
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    rb = np.zeros(len(train), np.float32)
    rm = np.zeros(len(train), np.float32)
    for fold in (1, 2, 3):
        va = folds == fold
        x = with_reference(base, train, meta["fold_references"][str(fold)])
        itb = meta["iterations"]["binary"] if fold == 3 else None
        itm = meta["iterations"]["multiclass"] if fold == 3 else None
        rb[va] = lgb.Booster(model_file=ROOT / f"models/exp003_binary_fold{fold}.txt").predict(x.loc[va], num_iteration=itb).astype(np.float32)
        p = lgb.Booster(model_file=ROOT / f"models/exp003_multiclass_fold{fold}.txt").predict(x.loc[va], num_iteration=itm)
        rm[va] = (1.0 - p[:, 0]).astype(np.float32)
    xte, _ = make_features(test, {"median": meta["full_reference"]["median"], "mad": meta["full_reference"]["mad"]})
    tb = lgb.Booster(model_file=ROOT / "models/exp003_binary_full.txt").predict(xte, num_iteration=meta["iterations"]["binary"]).astype(np.float32)
    tm = (1.0 - lgb.Booster(model_file=ROOT / "models/exp003_multiclass_full.txt").predict(xte, num_iteration=meta["iterations"]["multiclass"])[:, 0]).astype(np.float32)
    checks = {
        "oof_binary": np.array_equal(rb, np.load(ROOT / "preds/oof_binary_exp003.npy")),
        "oof_multiclass": np.array_equal(rm, np.load(ROOT / "preds/oof_multiclass_exp003.npy")),
        "test_binary": np.array_equal(tb, np.load(ROOT / "preds/test_binary_exp003.npy")),
        "test_multiclass": np.array_equal(tm, np.load(ROOT / "preds/test_multiclass_exp003.npy")),
    }
    if meta["promoted"]:
        w, threshold, hole = meta["multi_weight"], meta["threshold"], meta["hole"]
        score = (1 - w) * tb + w * tm
        pred = fill_short_gaps(score >= threshold, test, hole)
        stored = store_score(score - threshold + 0.5, pred)
        sub = pd.read_csv(ROOT / "subs/exp003.csv")
        checks["test_final"] = np.array_equal(stored, np.load(ROOT / "preds/test_exp003.npy"))
        checks["submission"] = np.array_equal(pred.astype(int), sub.label.to_numpy())
        checks["sha256"] = sha(ROOT / "subs/exp003.csv") == meta["submission_sha256"]
    print(json.dumps(checks, indent=2))
    print("판정: " + ("OK" if all(checks.values()) else "STOP"))
    return int(not all(checks.values()))


def reseal_saved():
    """Repair the initial outer early-stop leak using inner-locked booster prefixes; no fit."""
    meta_path = ROOT / "cache/exp003_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train = pd.read_csv(ROOT / "P1_data/train.csv")
    test = pd.read_csv(ROOT / "P1_data/test.csv")
    base = pd.read_pickle(ROOT / "cache/exp001/train_features.pkl")
    folds = np.load(ROOT / "cache/exp001/folds.npy")
    y = train.label.to_numpy(np.int8)
    iterations = {k: int(np.median(v[:2])) for k, v in meta["fold_best_iterations"].items()}
    outer = folds == 3
    x = with_reference(base, train, meta["fold_references"]["3"])
    oof_b = np.load(ROOT / "preds/oof_binary_exp003.npy")
    oof_m = np.load(ROOT / "preds/oof_multiclass_exp003.npy")
    oof_b[outer] = lgb.Booster(model_file=ROOT / "models/exp003_binary_fold3.txt").predict(
        x.loc[outer], num_iteration=iterations["binary"]
    )
    pm = lgb.Booster(model_file=ROOT / "models/exp003_multiclass_fold3.txt").predict(
        x.loc[outer], num_iteration=iterations["multiclass"]
    )
    oof_m[outer] = 1.0 - pm[:, 0]
    save_pool_pred(ROOT / "preds/oof_binary_exp003.npy", oof_b)
    save_pool_pred(ROOT / "preds/oof_multiclass_exp003.npy", oof_m)
    score = (1 - meta["multi_weight"]) * oof_b + meta["multi_weight"] * oof_m
    pred = fill_short_gaps(score >= meta["threshold"], train, meta["hole"])
    stats = f1(y[outer], pred[outer])
    save_pool_pred(ROOT / "preds/oof_exp003.npy", store_score(score - meta["threshold"] + 0.5, pred))
    evaluate(train.loc[outer].assign(anomaly_type=train.loc[outer].anomaly_type.fillna("normal")), pred[outer], ["anomaly_type"]).to_csv(
        ROOT / "experiments/exp003_outer_by_type.csv", index=False
    )
    evaluate(train.loc[outer], pred[outer], ["station", "layer"]).to_csv(
        ROOT / "experiments/exp003_outer_by_station_layer.csv", index=False
    )
    base_score, base_pred = exp002_outer_score(train, base, folds)
    pool = {
        "outer_pearson_exp002_exp003": float(np.corrcoef(base_score[outer], score[outer])[0, 1]),
        "outer_spearman_exp002_exp003": float(spearmanr(base_score[outer], score[outer]).statistic),
        "outer_residual_corr_exp002_exp003": float(np.corrcoef(y[outer] - base_score[outer], y[outer] - score[outer])[0, 1]),
        "exp002_outer_reproduced": f1(y[outer], base_pred[outer])[0],
    }
    promote = stats[0] > BASE_OUTER or (meta["inner_f1"] > BASE_INNER and pool["outer_residual_corr_exp002_exp003"] < 0.98 and stats[0] >= BASE_OUTER - 0.002)
    meta.update(iterations=iterations, outer_f1=stats[0], outer_tp=stats[1], outer_fp=stats[2], outer_fn=stats[3],
                outer_delta=stats[0] - BASE_OUTER, events=event_metrics(train, pred, outer), pool=pool,
                promoted=promote, submission_sha256=None, outer_tree_count_source="inner folds 1-2 median; fixed before outer score")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Existing full boosters have at least the locked prefix; inference truncates them.
    from foundation_inference import main as infer
    infer()
    print(json.dumps({"iterations": iterations, "outer_f1": stats[0], "tp": stats[1], "fp": stats[2], "fn": stats[3],
                      "delta": stats[0] - BASE_OUTER, "promoted": promote, "pool": pool}, indent=2))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    raise SystemExit(reproduce() if cmd == "reproduce" else reseal_saved() if cmd == "reseal" else train_main())
