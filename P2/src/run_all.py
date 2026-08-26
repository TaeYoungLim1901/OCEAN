"""E002부터 제출 CSV까지 한 번에 재현한다."""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
SEED = 42


def _stage(name, fn):
    started = time.perf_counter()
    print(f"[{name}] start", flush=True)
    result = fn()
    print(f"[{name}] end seconds={time.perf_counter() - started:.2f}", flush=True)
    return result


def _invoke(fn, args):
    old = sys.argv
    sys.argv = [fn.__module__] + args
    try:
        return fn()
    finally:
        sys.argv = old


def _copy_if_absent(source_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)


def _require(stage, *paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{stage} missing: {', '.join(missing)}")


def _require_any(stage, *paths):
    if not any(path.exists() for path in paths):
        raise FileNotFoundError(f"{stage} missing: {', '.join(map(str, paths))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="제출 CSV 경로")
    parser.add_argument("--w", type=float, default=-2.5, help="완전행 anti-stack 가중치")
    args = parser.parse_args()

    import common as C
    import gate_e004
    import inference_e002
    import inference_e003
    import mix_sub
    import stack_e008
    import train_e002
    import train_e003
    import train_e005
    import wmix

    root = Path(__file__).resolve().parents[1]
    C.SEED = SEED
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t18 = root / "cache" / "T18"
    t18.mkdir(parents=True, exist_ok=True)
    final_name = f"test_antistack_w{args.w:g}_E017.npy"

    original_root = C.ROOT
    with tempfile.TemporaryDirectory(prefix="run_", dir=t18) as td_name:
        work = Path(td_name) / "P2"
        for name in ("cache/E002", "cache/E003", "cache/E004", "cache/E005", "cache/E008",
                     "models", "preds", "subs"):
            (work / name).mkdir(parents=True, exist_ok=True)
        C.ROOT = work
        old_stack_root = stack_e008.ROOT
        stack_e008.ROOT = work
        try:
            _stage("E002_train", train_e002.main)
            _require(
                "E002_train",
                work / "models" / "E002.pkl",
                *(work / "preds" / f"oof_{name}_{label}_E002.npy"
                  for name in ("bug_depth", "fixed_depth", "fixed_no_target_depth")
                  for label in ("2024", "2025")),
            )
            _stage("E002_infer", inference_e002.main)
            e002_test = [work / "preds" / f"test_{name}_E002.npy"
                         for name in ("bug_depth", "fixed_depth", "fixed_no_target_depth")]
            _require_any("E002_infer", *e002_test)
            _stage("E003_train", train_e003.main)
            _require(
                "E003_train",
                work / "models" / "E003.pkl",
                *(work / "preds" / f"oof_{name}_E003.npy"
                  for name in ("e002", "no_depth", "enhanced", "robust_blend",
                               "regime_gate", "eof")),
            )
            _stage("E003_infer", inference_e003.main)
            _require(
                "E003_infer",
                *(work / "preds" / f"oof_{name}_E003.npy"
                  for name in ("e002", "enhanced", "robust_blend", "eof")),
                *(work / "preds" / f"test_{name}_E003.npy"
                  for name in ("e002", "enhanced", "no_depth", "eof",
                               "robust_blend", "regime_gate")),
            )
            _stage("E004", gate_e004.main)
            _require(
                "E004",
                *(work / "preds" / f"{kind}_gate_{name}_E004.npy"
                  for kind in ("oof", "test")
                  for name in ("eof", "enhanced", "robust_blend")),
                work / "cache" / "E004" / "gate_stats.json",
            )
            _stage("E005", train_e005.main)
            _require(
                "E005",
                *(work / "preds" / f"{kind}_{name}_E005.npy"
                  for kind in ("oof", "test")
                  for name in ("maskaug", "missing_only")),
            )
            _stage("E008", stack_e008.main)
            _require(
                "E008",
                *(work / "preds" / f"{kind}_stack_{name}_E008.npy"
                  for kind in ("oof", "test")
                  for name in ("global", "regime")),
                work / "preds" / "test_gate_robust_blend_E004.npy",
            )

            complete = work / "preds" / f"test_antistack_w{args.w:g}_complete_T18.npy"
            _stage("wmix", lambda: _invoke(wmix.main, [
                "--base", str(work / "preds" / "test_gate_robust_blend_E004.npy"),
                "--delta-from", str(work / "preds" / "test_gate_robust_blend_E004.npy"),
                "--delta-to", str(work / "preds" / "test_stack_regime_E008.npy"),
                "--w", str(args.w), "--out", str(complete),
            ]))
            _require("wmix", complete)
            _stage("mix_sub", lambda: _invoke(mix_sub.main, [
                "--missing", str(work / "preds" / "test_maskaug_E005.npy"),
                "--complete", str(complete), "--out", str(out),
                "--out-npy", str(work / "preds" / final_name),
            ]))
            _require("mix_sub", out, work / "preds" / final_name)
            _copy_if_absent(work / "models", root / "models")
            _copy_if_absent(work / "preds", root / "preds")
        finally:
            stack_e008.ROOT = old_stack_root
            C.ROOT = original_root
    print(f"out={out} w={args.w:g}")


if __name__ == "__main__":
    main()
