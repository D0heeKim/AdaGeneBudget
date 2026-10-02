#!/usr/bin/env python3
"""Evaluate the four Geneformer Kang scoring/allocation ablation methods.

Methods
-------
1. Fixed Expression
2. Adaptive Expression
3. Fixed TF-IDF
4. AdaGeneBudget

The script reuses validated token caches and the original frozen Geneformer
quality protocol: FP32, hidden_states[5], mean non-padding pooling, and exact
5-NN cosine classification.

This script evaluates annotation quality only; it does not rerun systems timing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support


PROJECT_DIR = Path(".")
DEFAULT_BASE_CACHE = (
    PROJECT_DIR
    / "data/processed/kang_2018/patient_1015_holdout/geneformer_cache/base"
)
DEFAULT_METHOD_CACHE_ROOT = (
    PROJECT_DIR
    / "data/processed/kang_2018/patient_1015_holdout/geneformer_cache/methods"
)
DEFAULT_CHECKPOINT = Path("./checkpoints/Geneformer-V1-10M")
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs/geneformer/kang_patient1015/expression_ablation"
)
DEFAULT_BENCHMARK_SCRIPT = (
    PROJECT_DIR
    / "experiments/geneformer/kang/run_main.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument(
        "--method-cache-root",
        type=Path,
        default=DEFAULT_METHOD_CACHE_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=DEFAULT_BENCHMARK_SCRIPT,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--similarity-chunk", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "geneformer_kang_main_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def per_class_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    labels = np.unique(true_labels.astype(str))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels.astype(str),
        predictions.astype(str),
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "cell_type": labels,
            "support": support.astype(int),
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
        }
    )


def resolve_cache_names(method_cache_root: Path) -> dict[str, str]:
    main_summary_path = method_cache_root / "summary.csv"
    expression_summary_path = (
        method_cache_root / "expression_ablation_summary.csv"
    )
    if not main_summary_path.is_file():
        raise FileNotFoundError(main_summary_path)
    if not expression_summary_path.is_file():
        raise FileNotFoundError(expression_summary_path)

    main_summary = pd.read_csv(main_summary_path)
    expression_summary = pd.read_csv(expression_summary_path)

    mapping: dict[str, str] = {}
    for method in ("fixed_tfidf", "adaptive"):
        rows = main_summary.loc[main_summary["method"] == method]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one {method} row in {main_summary_path}; "
                f"found {len(rows)}."
            )
        mapping[method] = str(rows.iloc[0]["cache"])

    for method in ("fixed_expression", "adaptive_expression"):
        rows = expression_summary.loc[
            expression_summary["method"] == method
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one {method} row in {expression_summary_path}; "
                f"found {len(rows)}."
            )
        mapping[method] = str(rows.iloc[0]["cache"])

    return mapping


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")

    benchmark = load_module(args.benchmark_script)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_names = resolve_cache_names(args.method_cache_root)
    method_specs = [
        (
            "fixed_expression",
            "Fixed Expression",
            "expression_only_fixed_budget",
        ),
        (
            "adaptive_expression",
            "Adaptive Expression",
            "expression_only_adaptive_mass",
        ),
        (
            "fixed_tfidf",
            "Fixed TF-IDF",
            "tfidf_fixed_budget",
        ),
        (
            "adaptive",
            "AdaGeneBudget",
            "tfidf_adaptive_mass",
        ),
    ]

    metadata = benchmark.load_base_metadata(args.base_cache)
    train_labels = metadata["train_labels"]
    test_labels = metadata["test_labels"]

    print("=" * 104, flush=True)
    print("GENEFORMER KANG EXPRESSION × BUDGET ABLATION", flush=True)
    print("=" * 104, flush=True)
    print("DEVICE:", device, flush=True)
    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("CHECKPOINT:", args.checkpoint, flush=True)
    print("METHOD CACHE ROOT:", args.method_cache_root, flush=True)
    print("OUTPUT:", args.output_dir, flush=True)

    benchmark.seed_all(0)
    model, hidden_state_index = benchmark.load_model(
        args.checkpoint,
        device,
    )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for method, method_label, design_cell in method_specs:
        method_dir = args.output_dir / method
        summary_path = method_dir / "summary.json"

        if summary_path.is_file() and not args.overwrite:
            print(f"[resume] {method}", flush=True)
            summary_rows.append(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
            per_path = method_dir / "per_class.csv"
            if not per_path.is_file():
                raise FileNotFoundError(per_path)
            per_class_rows.extend(
                pd.read_csv(per_path).to_dict(orient="records")
            )
            continue

        cache_dir = args.method_cache_root / cache_names[method]
        print("\n" + "=" * 104, flush=True)
        print(f"METHOD: {method_label}", flush=True)
        print("CACHE:", cache_dir, flush=True)
        print("=" * 104, flush=True)

        train_cache = benchmark.load_token_cache(cache_dir, "train")
        test_cache = benchmark.load_token_cache(cache_dir, "test")
        train_dataset = benchmark.CachedTokenDataset(train_cache)
        test_dataset = benchmark.CachedTokenDataset(test_cache)
        train_loader, _ = benchmark.make_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            "length_sorted",
        )
        test_loader, _ = benchmark.make_loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            "length_sorted",
        )

        train_embeddings, _ = benchmark.extract_embeddings(
            model,
            hidden_state_index,
            train_loader,
            len(train_dataset),
            device,
            measure=False,
        )
        test_embeddings, _ = benchmark.extract_embeddings(
            model,
            hidden_state_index,
            test_loader,
            len(test_dataset),
            device,
            measure=False,
        )
        predictions = benchmark.exact_cosine_knn_predict(
            train_embeddings,
            train_labels,
            test_embeddings,
            args.knn_k,
            device,
            args.similarity_chunk,
        )
        metrics = benchmark.classification_metrics(
            test_labels,
            predictions,
        )

        per_class = per_class_metrics(test_labels, predictions)
        per_class.insert(0, "method_label", method_label)
        per_class.insert(0, "method", method)

        summary = {
            "method": method,
            "method_label": method_label,
            "design_cell": design_cell,
            "cache": cache_names[method],
            "train_mean_selected_genes": float(
                np.mean(train_cache.lengths)
            ),
            "test_mean_selected_genes": float(
                np.mean(test_cache.lengths)
            ),
            "train_median_selected_genes": float(
                np.median(train_cache.lengths)
            ),
            "test_median_selected_genes": float(
                np.median(test_cache.lengths)
            ),
            **metrics,
        }

        method_dir.mkdir(parents=True, exist_ok=True)
        benchmark.stable_json_dump(summary, summary_path)
        per_class.to_csv(method_dir / "per_class.csv", index=False)
        pd.DataFrame(
            {
                "cell_type": test_labels.astype(str),
                "prediction": predictions.astype(str),
            }
        ).to_csv(method_dir / "predictions.csv", index=False)

        summary_rows.append(summary)
        per_class_rows.extend(per_class.to_dict(orient="records"))

        print(
            f"[quality] macro_f1={metrics['macro_f1']:.6f}, "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"balanced={metrics['balanced_accuracy']:.6f}, "
            f"test_mean_genes={summary['test_mean_selected_genes']:.3f}",
            flush=True,
        )

        del (
            train_embeddings,
            test_embeddings,
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
        )
        torch.cuda.empty_cache()

    order = [item[0] for item in method_specs]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["_order"] = summary_df["method"].map(
        {name: i for i, name in enumerate(order)}
    )
    summary_df = (
        summary_df.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    per_class_df = pd.DataFrame(per_class_rows)

    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    per_class_df.to_csv(args.output_dir / "per_class.csv", index=False)

    print("\n" + "=" * 104, flush=True)
    print("FINAL ABLATION SUMMARY", flush=True)
    print("=" * 104, flush=True)
    print(
        summary_df[
            [
                "method_label",
                "train_mean_selected_genes",
                "test_mean_selected_genes",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("SUMMARY:", args.output_dir / "summary.csv", flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
