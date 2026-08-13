"""Static guard for P3 context-only T2 feature and inference paths."""
import ast
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parent


def main():
    feature_path = ROOT / "features_t2.py"
    inference_path = ROOT / "inference_t2.py"
    feature_text = feature_path.read_text(encoding="utf-8")
    inference_text = inference_path.read_text(encoding="utf-8")
    tree = ast.parse(feature_text)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "extract"]
    assert len(funcs) == 1 and [arg.arg for arg in funcs[0].args.args] == ["ctx"]
    forbidden_feature = ("read_csv", "read_parquet", "datetime", "timestamp", "month", "year", "case_id")
    assert not any(token in feature_text.lower() for token in forbidden_feature)
    assert "C.iter_test_contexts()" in inference_text and "extract(ctx)" in inference_text
    forbidden_inference = (".groupby(", ".mean(", ".median(", ".std(", ".quantile(", ".fit(", ".fit_transform(")
    assert not any(token in inference_text for token in forbidden_inference)
    print("context_static_check=OK extract_args=1 absolute_time=0 cross_case_aggregate=0")


if __name__ == "__main__":
    main()
