#!/usr/bin/env python3
"""Build expression-only Geneformer Kang ablation caches.

This script adds two method caches without touching the existing main caches:

1. Fixed Expression:
   select the top K genes by raw expression, where K is the existing
   AdaGeneBudget-matched fixed budget from the main cache summary.

2. Adaptive Expression:
   remove IDF from AdaGeneBudget and apply the same cumulative-mass rule,
   tau, Kmin, and Kmax to raw expression values.

After selection, every subset is reordered by Geneformer's native rank-value
score, exactly as in the validated main cache builder.

This script is CPU-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_DIR = Path(".")
DEFAULT_BASE_CACHE = (
    PROJECT_DIR
    / "data/processed/kang_2018/patient_1015_holdout/geneformer_cache/base"
)
DEFAULT_METHOD_CACHE_ROOT = (
    PROJECT_DIR
    / "data/processed/kang_2018/patient_1015_holdout/geneformer_cache/methods"
)
DEFAULT_MAIN_BUILDER = (
    PROJECT_DIR
    / "experiments/geneformer/kang/build_method_caches.py"
)

EXPECTED_TRAIN_CELLS = 19_583
EXPECTED_TEST_CELLS = 5_090
EXPECTED_SUPPORTED_GENES = 12_288
DEFAULT_TAU = 0.90
DEFAULT_K_MIN = 128
DEFAULT_K_MAX = 600


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
    parser.add_argument(
        "--main-builder",
        type=Path,
        default=DEFAULT_MAIN_BUILDER,
    )
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace only the two expression-ablation cache directories.",
    )
    return parser.parse_args()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "geneformer_kang_main_cache_builder",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def deterministic_expression_order(
    row_values: np.ndarray,
    row_positions: np.ndarray,
    token_ids: np.ndarray,
    main: Any,
) -> np.ndarray:
    scores = np.asarray(row_values, dtype=np.float64)
    if np.any(scores < 0) or not np.isfinite(scores).all():
        raise RuntimeError("Raw expression scores must be finite and nonnegative.")
    return main.deterministic_descending_order(
        scores,
        token_ids[row_positions],
    )


def choose_fixed_expression_subset(
    row_positions: np.ndarray,
    row_values: np.ndarray,
    token_ids: np.ndarray,
    budget: int,
    main: Any,
) -> tuple[np.ndarray, float]:
    n_genes = int(row_positions.size)
    if n_genes == 0:
        return np.empty(0, dtype=np.int64), 1.0

    order = deterministic_expression_order(
        row_values,
        row_positions,
        token_ids,
        main,
    )
    selected_local = order[: min(n_genes, int(budget))]
    total_mass = float(np.sum(row_values, dtype=np.float64))
    retained_mass = float(
        np.sum(row_values[selected_local], dtype=np.float64)
        / max(total_mass, 1e-12)
    )
    return selected_local.astype(np.int64, copy=False), retained_mass


def choose_adaptive_expression_subset(
    row_positions: np.ndarray,
    row_values: np.ndarray,
    token_ids: np.ndarray,
    tau: float,
    k_min: int,
    k_max: int,
    main: Any,
) -> tuple[np.ndarray, float, int, bool, bool]:
    n_genes = int(row_positions.size)
    if n_genes == 0:
        return (
            np.empty(0, dtype=np.int64),
            1.0,
            0,
            False,
            False,
        )

    order = deterministic_expression_order(
        row_values,
        row_positions,
        token_ids,
        main,
    )
    total_mass = float(np.sum(row_values, dtype=np.float64))

    if total_mass <= 0:
        raw_k = n_genes
    else:
        cumulative = np.cumsum(
            row_values[order],
            dtype=np.float64,
        )
        raw_k = int(
            np.searchsorted(
                cumulative,
                float(tau) * total_mass,
                side="left",
            )
            + 1
        )

    lower_clipped = bool(n_genes >= k_min and raw_k < k_min)
    upper_clipped = bool(raw_k > k_max)
    selected_k = min(
        n_genes,
        int(k_max),
        max(int(k_min), raw_k),
    )
    selected_local = order[:selected_k]
    retained_mass = float(
        np.sum(row_values[selected_local], dtype=np.float64)
        / max(total_mass, 1e-12)
    )

    return (
        selected_local.astype(np.int64, copy=False),
        retained_mass,
        raw_k,
        lower_clipped,
        upper_clipped,
    )


def build_expression_selection(
    *,
    matrix: sp.csr_matrix,
    n_counts: np.ndarray,
    token_ids: np.ndarray,
    gene_medians: np.ndarray,
    method: str,
    split_name: str,
    fixed_k: int,
    tau: float,
    k_min: int,
    k_max: int,
    progress_every: int,
    main: Any,
) -> Any:
    if method not in {"fixed_expression", "adaptive_expression"}:
        raise ValueError(method)
    if matrix.shape[0] != n_counts.size:
        raise RuntimeError("matrix and n_counts row count mismatch.")

    n_rows = int(matrix.shape[0])
    lengths = np.empty(n_rows, dtype=np.int32)
    retained_mass = np.empty(n_rows, dtype=np.float32)

    is_adaptive = method == "adaptive_expression"
    raw_adaptive_k = (
        np.empty(n_rows, dtype=np.int32) if is_adaptive else None
    )
    lower_clipped = (
        np.empty(n_rows, dtype=np.bool_) if is_adaptive else None
    )
    upper_clipped = (
        np.empty(n_rows, dtype=np.bool_) if is_adaptive else None
    )

    position_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    start_time = time.perf_counter()

    for row_index in range(n_rows):
        start = int(matrix.indptr[row_index])
        stop = int(matrix.indptr[row_index + 1])
        row_positions = matrix.indices[start:stop].astype(
            np.int64,
            copy=False,
        )
        row_values = matrix.data[start:stop].astype(
            np.float64,
            copy=False,
        )

        if is_adaptive:
            (
                selected_local,
                retained,
                raw_k,
                was_lower_clipped,
                was_upper_clipped,
            ) = choose_adaptive_expression_subset(
                row_positions,
                row_values,
                token_ids,
                tau,
                k_min,
                k_max,
                main,
            )
            assert raw_adaptive_k is not None
            assert lower_clipped is not None
            assert upper_clipped is not None
            raw_adaptive_k[row_index] = raw_k
            lower_clipped[row_index] = was_lower_clipped
            upper_clipped[row_index] = was_upper_clipped
        else:
            selected_local, retained = choose_fixed_expression_subset(
                row_positions,
                row_values,
                token_ids,
                fixed_k,
                main,
            )

        retained_mass[row_index] = retained

        # Preserve the validated Geneformer input ordering after selection.
        selected_local = main.reorder_subset_by_native_rank(
            selected_local,
            row_positions,
            row_values,
            float(n_counts[row_index]),
            token_ids,
            gene_medians,
        )

        selected_positions = row_positions[selected_local].astype(
            np.int32,
            copy=False,
        )
        selected_token_ids = token_ids[selected_positions].astype(
            np.int32,
            copy=False,
        )
        lengths[row_index] = int(selected_positions.size)
        position_parts.append(selected_positions)
        token_parts.append(selected_token_ids)

        completed = row_index + 1
        if (
            completed == 1
            or completed == n_rows
            or completed % progress_every == 0
        ):
            print(
                f"[select] {method}/{split_name}: "
                f"{completed:,}/{n_rows:,} "
                f"({100.0 * completed / n_rows:.1f}%)",
                flush=True,
            )

    compute_time_s = time.perf_counter() - start_time
    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths.astype(np.int64), out=indptr[1:])

    gene_positions = np.concatenate(position_parts).astype(
        np.int32,
        copy=False,
    )
    flattened_token_ids = np.concatenate(token_parts).astype(
        np.int32,
        copy=False,
    )

    if int(indptr[-1]) != gene_positions.size:
        raise RuntimeError("indptr and gene_positions length mismatch.")
    if gene_positions.size != flattened_token_ids.size:
        raise RuntimeError("gene_positions and token_ids length mismatch.")

    return main.SelectionResult(
        indptr=indptr,
        gene_positions=gene_positions,
        token_ids=flattened_token_ids,
        lengths=lengths,
        retained_mass=retained_mass,
        raw_adaptive_k=raw_adaptive_k,
        lower_clipped=lower_clipped,
        upper_clipped=upper_clipped,
        compute_time_s=float(compute_time_s),
    )


def build_one_method(
    *,
    resources: Any,
    method_cache_root: Path,
    method: str,
    directory_name: str,
    fixed_k: int,
    tau: float,
    k_min: int,
    k_max: int,
    progress_every: int,
    overwrite_existing: bool,
    main: Any,
) -> dict[str, Any]:
    method_dir = method_cache_root / directory_name
    if method_dir.exists():
        if not overwrite_existing:
            raise FileExistsError(
                f"{method_dir} already exists. "
                "Use --overwrite-existing only after confirming replacement."
            )
        shutil.rmtree(method_dir)

    method_dir.mkdir(parents=True, exist_ok=False)

    settings = {
        "selection_score": "raw_expression",
        "budget_policy": (
            f"fixed_k={fixed_k}"
            if method == "fixed_expression"
            else f"adaptive_mass_tau={tau},k_min={k_min},k_max={k_max}"
        ),
        "tau": tau if method == "adaptive_expression" else None,
        "k_min": k_min if method == "adaptive_expression" else None,
        "k_max": k_max if method == "adaptive_expression" else None,
        "fixed_k": fixed_k if method == "fixed_expression" else None,
        "idf_used": False,
        "n_counts_source": (
            "original uncompressed .X row sum before vocabulary filtering"
        ),
    }

    train_result = build_expression_selection(
        matrix=resources.train_matrix,
        n_counts=resources.train_n_counts,
        token_ids=resources.token_ids,
        gene_medians=resources.gene_medians,
        method=method,
        split_name="train",
        fixed_k=fixed_k,
        tau=tau,
        k_min=k_min,
        k_max=k_max,
        progress_every=progress_every,
        main=main,
    )
    test_result = build_expression_selection(
        matrix=resources.test_matrix,
        n_counts=resources.test_n_counts,
        token_ids=resources.token_ids,
        gene_medians=resources.gene_medians,
        method=method,
        split_name="test",
        fixed_k=fixed_k,
        tau=tau,
        k_min=k_min,
        k_max=k_max,
        progress_every=progress_every,
        main=main,
    )

    train_meta = main.save_selection(
        method_dir / "train",
        train_result,
        method,
        "train",
        settings,
    )
    test_meta = main.save_selection(
        method_dir / "test",
        test_result,
        method,
        "test",
        settings,
    )

    maximum_length = (
        fixed_k if method == "fixed_expression" else k_max
    )
    main.validate_saved_method(
        method_dir,
        EXPECTED_TRAIN_CELLS,
        EXPECTED_TEST_CELLS,
        maximum_length,
    )

    root_meta = {
        "status": "PASS",
        "method": method,
        "directory_name": directory_name,
        "directory": str(method_dir.resolve()),
        "settings": settings,
        "train": train_meta,
        "test": test_meta,
        "combined_selection_compute_time_s": float(
            train_result.compute_time_s + test_result.compute_time_s
        ),
    }
    main.stable_json_dump(root_meta, method_dir / "meta.json")
    return root_meta


def main() -> int:
    args = parse_args()

    if not args.main_builder.is_file():
        raise FileNotFoundError(args.main_builder)
    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must be in (0, 1].")
    if args.k_min <= 0 or args.k_max < args.k_min:
        raise ValueError("Require 0 < k_min <= k_max.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    main_builder = load_module(args.main_builder)
    resources = main_builder.load_base_resources(args.base_cache)

    summary_path = args.method_cache_root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    main_summary = pd.read_csv(summary_path)

    adaptive_rows = main_summary.loc[
        main_summary["method"] == "adaptive"
    ]
    fixed_rows = main_summary.loc[
        main_summary["method"] == "fixed_tfidf"
    ]
    if len(adaptive_rows) != 1 or len(fixed_rows) != 1:
        raise RuntimeError(
            "Expected exactly one adaptive and one fixed_tfidf row "
            "in the main cache summary."
        )

    fixed_k = int(round(float(fixed_rows.iloc[0]["fixed_k"])))
    adaptive_train_mean = float(
        adaptive_rows.iloc[0]["train_mean_genes"]
    )
    matched_expected_mean = float(
        np.minimum(
            np.diff(resources.train_matrix.indptr),
            fixed_k,
        ).mean()
    )

    recorded_fixed_mean = float(
        fixed_rows.iloc[0]["train_mean_genes"]
    )
    if not math.isclose(
        matched_expected_mean,
        recorded_fixed_mean,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Computed fixed-budget mean disagrees with the existing "
            "fixed-TF-IDF cache summary: "
            f"K={fixed_k}, computed={matched_expected_mean:.9f}, "
            f"recorded={recorded_fixed_mean:.9f}."
        )
    match_difference = matched_expected_mean - adaptive_train_mean

    print("=" * 96, flush=True)
    print("KANG GENEFORMER EXPRESSION-ONLY ABLATION CACHE BUILD", flush=True)
    print("=" * 96, flush=True)
    print("BASE CACHE:", args.base_cache, flush=True)
    print("METHOD CACHE ROOT:", args.method_cache_root, flush=True)
    print(
        f"REFERENCE: Ada mean={adaptive_train_mean:.6f}, "
        f"matched fixed K={fixed_k}, "
        f"fixed-minus-Ada mean={match_difference:+.6f}",
        flush=True,
    )
    print(
        f"ADAPTIVE EXPRESSION: tau={args.tau}, "
        f"Kmin={args.k_min}, Kmax={args.k_max}",
        flush=True,
    )

    args.method_cache_root.mkdir(parents=True, exist_ok=True)

    fixed_meta = build_one_method(
        resources=resources,
        method_cache_root=args.method_cache_root,
        method="fixed_expression",
        directory_name=f"matched_fixed_expression_k{fixed_k}",
        fixed_k=fixed_k,
        tau=args.tau,
        k_min=args.k_min,
        k_max=args.k_max,
        progress_every=args.progress_every,
        overwrite_existing=args.overwrite_existing,
        main=main_builder,
    )
    adaptive_meta = build_one_method(
        resources=resources,
        method_cache_root=args.method_cache_root,
        method="adaptive_expression",
        directory_name=(
            f"adaptive_expression_tau{str(args.tau).replace('.', 'p')}"
            f"_k{args.k_min}_{args.k_max}"
        ),
        fixed_k=fixed_k,
        tau=args.tau,
        k_min=args.k_min,
        k_max=args.k_max,
        progress_every=args.progress_every,
        overwrite_existing=args.overwrite_existing,
        main=main_builder,
    )

    rows = []
    for meta in (fixed_meta, adaptive_meta):
        row = {
            "cache": meta["directory_name"],
            "method": meta["method"],
            "fixed_k": meta["settings"].get("fixed_k"),
            "tau": meta["settings"].get("tau"),
            "k_min": meta["settings"].get("k_min"),
            "k_max": meta["settings"].get("k_max"),
            "idf_used": False,
            "train_mean_genes": meta["train"]["length_statistics"]["mean"],
            "test_mean_genes": meta["test"]["length_statistics"]["mean"],
            "train_median_genes": meta["train"]["length_statistics"]["median"],
            "test_median_genes": meta["test"]["length_statistics"]["median"],
            "train_retained_expression_mass_mean": meta["train"][
                "retained_mass_statistics"
            ]["mean"],
            "test_retained_expression_mass_mean": meta["test"][
                "retained_mass_statistics"
            ]["mean"],
            "selection_compute_time_s": meta[
                "combined_selection_compute_time_s"
            ],
        }
        if meta["method"] == "adaptive_expression":
            row.update(
                {
                    "train_lower_clipped_fraction": meta["train"][
                        "lower_clipped_fraction"
                    ],
                    "test_lower_clipped_fraction": meta["test"][
                        "lower_clipped_fraction"
                    ],
                    "train_upper_clipped_fraction": meta["train"][
                        "upper_clipped_fraction"
                    ],
                    "test_upper_clipped_fraction": meta["test"][
                        "upper_clipped_fraction"
                    ],
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = (
        args.method_cache_root / "expression_ablation_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 96, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 96, flush=True)
    print(summary.to_string(index=False), flush=True)
    print("SUMMARY:", summary_path, flush=True)
    print("NOTE: This cache-build step is CPU-only.", flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
