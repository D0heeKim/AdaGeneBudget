#!/usr/bin/env python3
"""Remeasure scGPT PBMC P5 efficiency on the held-out test split only.

This script does NOT recompute annotation quality and does NOT overwrite the
existing six-method benchmark. It reuses validated PBMC/scGPT caches and the
same model/collator code as the legacy benchmark, but changes efficiency scope:

    held-out P5 test selection
    + held-out P5 test bucketing
    + held-out P5 test embedding

Outputs
-------
outputs/scgpt/pbmc_p5/test_only_timing/
├── timing_repeats.csv
├── summary_timing_only.csv
├── legacy_summary_with_test_only_efficiency.csv
└── config.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse


PROJECT_DIR = Path(".")
EXPECTED_TEST_CELLS = 19_957


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-script",
        type=Path,
        default=PROJECT_DIR / "experiments/scgpt/pbmc/run_reference_pipeline.py",
    )
    parser.add_argument(
        "--legacy-output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/scgpt/pbmc_p5/main",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            PROJECT_DIR
            / "data/processed/pbmc_seurat_v4/p5_holdout/scgpt_cache"
        ),
    )
    parser.add_argument("--repo", type=Path, default=PROJECT_DIR / "scGPT")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./checkpoints/scgpt"),
    )
    parser.add_argument(
        "--full-script",
        type=Path,
        default=PROJECT_DIR / "experiments/scgpt/_shared/full_input.py",
    )
    parser.add_argument(
        "--fixed-script",
        type=Path,
        default=PROJECT_DIR / "experiments/scgpt/_shared/fixed_selection.py",
    )
    parser.add_argument(
        "--corrected-script",
        type=Path,
        default=PROJECT_DIR / "experiments/scgpt/_shared/runtime_utils.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/scgpt/pbmc_p5/test_only_timing",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument(
        "--selection-timing-repeats",
        type=int,
        default=1,
        help="Use 1 to match the current Geneformer PBMC script default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def recompute_selection_once(
    *,
    method: str,
    matrix: sparse.csr_matrix,
    idf: np.ndarray,
    fixed_k: int,
    tau: float,
    k_min: int,
    k_max: int,
    top_order_fn,
) -> int:
    """Recompute P5 selection without disk writes."""
    selected_total = 0

    for row in range(matrix.shape[0]):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        positions = matrix.indices[start:stop]
        values = matrix.data[start:stop].astype(np.float32, copy=False)
        n_expressed = int(positions.size)

        if n_expressed == 0:
            continue

        scores = values * idf[positions]

        if method == "matched_fixed_tfidf":
            k = min(n_expressed, fixed_k)
            chosen = top_order_fn(scores, k)

        elif method == "adaptive":
            order = top_order_fn(scores, min(n_expressed, k_max))
            total_mass = float(np.sum(scores, dtype=np.float64))
            if total_mass <= 0:
                raw_k = min(n_expressed, k_max)
            else:
                cumulative = np.cumsum(scores[order], dtype=np.float64)
                raw_k = int(
                    np.searchsorted(
                        cumulative,
                        tau * total_mass,
                        side="left",
                    )
                    + 1
                )
            k = min(n_expressed, k_max, max(k_min, raw_k))
            chosen = order[:k]

        else:
            raise ValueError(f"Unsupported selection method: {method}")

        selected_total += int(len(chosen))

    return selected_total


def measure_selection(
    *,
    method: str,
    matrix: sparse.csr_matrix,
    idf: np.ndarray,
    fixed_k: int,
    tau: float,
    k_min: int,
    k_max: int,
    top_order_fn,
    repeats: int,
    expected_selected_total: int,
) -> list[float]:
    times: list[float] = []

    for repeat in range(repeats):
        started = time.perf_counter()
        selected_total = recompute_selection_once(
            method=method,
            matrix=matrix,
            idf=idf,
            fixed_k=fixed_k,
            tau=tau,
            k_min=k_min,
            k_max=k_max,
            top_order_fn=top_order_fn,
        )
        elapsed = float(time.perf_counter() - started)

        if selected_total != expected_selected_total:
            raise RuntimeError(
                f"{method}: recomputed selected_total={selected_total:,}, "
                f"cache selected_total={expected_selected_total:,}"
            )

        times.append(elapsed)
        print(
            f"    selection repeat {repeat + 1}/{repeats}: {elapsed:.4f}s",
            flush=True,
        )

    return times


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.warmup_steps <= 0:
        raise ValueError("--warmup-steps must be positive")
    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive")
    if args.selection_timing_repeats <= 0:
        raise ValueError("--selection-timing-repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    required_files = [
        args.legacy_script,
        args.legacy_output_dir / "config.json",
        args.legacy_output_dir / "summary.csv",
        args.legacy_output_dir / "matched_genes.npy",
        args.cache_dir / "test_matched_csr.npz",
        args.cache_dir / "idf.npy",
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.full_script,
        args.fixed_script,
        args.corrected_script,
    ]
    for path in required_files:
        require_file(path)
    require_dir(args.repo)
    require_dir(args.model_dir)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists. Use --overwrite only "
                "after reviewing the existing output."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    legacy = load_module("legacy_scgpt_pbmc_main", args.legacy_script)
    full_module = load_module("pbmc_full_module_test_only", args.full_script)
    fixed_module = load_module("pbmc_fixed_module_test_only", args.fixed_script)
    corrected = load_module(
        "pbmc_corrected_module_test_only",
        args.corrected_script,
    )

    legacy_config = json.loads(
        (args.legacy_output_dir / "config.json").read_text(encoding="utf-8")
    )
    fixed_k = int(legacy_config["fixed_k"])
    tau = float(legacy_config.get("tau", 0.90))
    k_min = int(legacy_config.get("k_min", 128))
    k_max = int(legacy_config.get("k_max", 600))
    native_max_length = int(
        legacy_config.get("native_max_length", 1200)
    )

    print("=" * 118)
    print("scGPT PBMC P5 TEST-ONLY TIMING")
    print("=" * 118)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Test cells:", f"{EXPECTED_TEST_CELLS:,}")
    print("Batch size:", args.batch_size)
    print("Timing repeats:", args.timing_repeats)
    print("Selection timing repeats:", args.selection_timing_repeats)
    print(
        "E2E: test selection + test bucketing + test embedding",
        flush=True,
    )

    test_matrix = sparse.load_npz(
        args.cache_dir / "test_matched_csr.npz"
    ).tocsr()
    test_matrix.sum_duplicates()
    test_matrix.eliminate_zeros()
    test_matrix.sort_indices()

    if test_matrix.shape[0] != EXPECTED_TEST_CELLS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_CELLS:,} test cells, "
            f"found {test_matrix.shape[0]:,}"
        )

    matched_genes = np.load(
        args.legacy_output_dir / "matched_genes.npy",
        allow_pickle=True,
    ).astype(str)
    if test_matrix.shape[1] != len(matched_genes):
        raise RuntimeError(
            "test matrix columns and matched_genes length do not match: "
            f"{test_matrix.shape[1]} vs {len(matched_genes)}"
        )

    idf = np.load(args.cache_dir / "idf.npy", allow_pickle=False)
    if idf.shape != (test_matrix.shape[1],):
        raise RuntimeError(
            f"Unexpected IDF shape {idf.shape}; "
            f"expected {(test_matrix.shape[1],)}"
        )

    adaptive_test_dir = args.cache_dir / "adaptive_test"
    fixed_test_dir = args.cache_dir / f"fixed_tfidf_k{fixed_k}_test"
    require_dir(adaptive_test_dir)
    require_dir(fixed_test_dir)

    model_args = SimpleNamespace(repo=args.repo, model_dir=args.model_dir)
    model, vocab, cfg, pad_id, pad_value, device = legacy.load_model(
        model_args,
        fixed_module,
    )
    gene_ids = np.asarray(vocab(list(matched_genes)), dtype=np.int64)

    full_test = full_module.FullCellDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
    )
    fixed_test = legacy.PreselectedTokenDataset(
        fixed_test_dir,
        gene_ids,
        vocab["<cls>"],
        pad_value,
    )
    adaptive_test = legacy.PreselectedTokenDataset(
        adaptive_test_dir,
        gene_ids,
        vocab["<cls>"],
        pad_value,
    )

    if not (
        len(full_test)
        == len(fixed_test)
        == len(adaptive_test)
        == EXPECTED_TEST_CELLS
    ):
        raise RuntimeError("Test dataset row counts are inconsistent")

    from scgpt.data_collator import DataCollator

    max_full_length = int(np.max(full_test.sequence_lengths))
    collator_full = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=max_full_length,
        sampling=False,
        keep_first_n_tokens=1,
    )
    collator_native = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=native_max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    collator_random = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=fixed_k + 1,
        sampling=True,
        keep_first_n_tokens=1,
    )
    collator_fixed = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=fixed_k + 1,
        sampling=False,
        keep_first_n_tokens=1,
    )
    collator_adaptive = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=k_max + 1,
        sampling=False,
        keep_first_n_tokens=1,
    )

    specs: list[dict[str, Any]] = [
        {
            "name": "full_bucketed",
            "dataset": full_test,
            "collator": collator_full,
            "batching": "length_sorted",
            "effective_max_length": None,
            "sampling": "none",
            "fixed_k": None,
        },
        {
            "name": "scgpt_native_sequential",
            "dataset": full_test,
            "collator": collator_native,
            "batching": "sequential",
            "effective_max_length": native_max_length,
            "sampling": "random_in_dataloader",
            "fixed_k": native_max_length - 1,
        },
        {
            "name": "scgpt_native_bucketed",
            "dataset": full_test,
            "collator": collator_native,
            "batching": "length_sorted",
            "effective_max_length": native_max_length,
            "sampling": "random_in_dataloader",
            "fixed_k": native_max_length - 1,
        },
        {
            "name": "matched_random_fixed_k",
            "dataset": full_test,
            "collator": collator_random,
            "batching": "length_sorted",
            "effective_max_length": fixed_k + 1,
            "sampling": "random_in_dataloader",
            "fixed_k": fixed_k,
        },
        {
            "name": "matched_fixed_tfidf",
            "dataset": fixed_test,
            "collator": collator_fixed,
            "batching": "length_sorted",
            "effective_max_length": fixed_k + 1,
            "sampling": "fixed_tfidf",
            "fixed_k": fixed_k,
        },
        {
            "name": "adaptive",
            "dataset": adaptive_test,
            "collator": collator_adaptive,
            "batching": "length_sorted",
            "effective_max_length": k_max + 1,
            "sampling": "adaptive_tfidf_mass",
            "fixed_k": None,
        },
    ]

    timing_rows: list[dict[str, Any]] = []

    for spec in specs:
        name = str(spec["name"])
        dataset = spec["dataset"]

        print()
        print("-" * 118)
        print("METHOD:", name)
        print("-" * 118)

        legacy.seed_all(0)
        loader, bucketing_time = legacy.make_loader(
            dataset,
            spec["collator"],
            args.batch_size,
            args.num_workers,
            999002,
            spec["batching"],
            spec["effective_max_length"],
        )

        if name in {"matched_fixed_tfidf", "adaptive"}:
            expected_selected_total = int(
                np.sum(legacy.get_lengths(dataset) - 1, dtype=np.int64)
            )
            selection_times = measure_selection(
                method=name,
                matrix=test_matrix,
                idf=idf,
                fixed_k=fixed_k,
                tau=tau,
                k_min=k_min,
                k_max=k_max,
                top_order_fn=legacy.top_order,
                repeats=args.selection_timing_repeats,
                expected_selected_total=expected_selected_total,
            )
        else:
            selection_times = [0.0]

        selection_mean = float(np.mean(selection_times))
        selection_std = float(np.std(selection_times, ddof=0))

        corrected.warmup(
            model,
            loader,
            pad_id,
            device,
            args.warmup_steps,
        )

        for repeat in range(args.timing_repeats):
            embeddings, timing = corrected.extract_once(
                model,
                loader,
                dataset,
                pad_id,
                device,
                cfg["embsize"],
            )
            embedding_time = float(timing["embedding_time_s"])
            end_to_end_time = (
                selection_mean
                + float(bucketing_time)
                + embedding_time
            )
            actual_tokens = int(timing["actual_sequence_tokens"])
            padded_tokens = int(timing["padded_tokens_processed"])
            padding_ratio = float(
                padded_tokens / max(actual_tokens, 1)
            )

            row = {
                "method": name,
                "repeat": repeat,
                "timing_scope": "held-out P5 test only",
                "num_cells": EXPECTED_TEST_CELLS,
                "batch_size": args.batch_size,
                "selection_in_dataloader": (
                    spec["sampling"] == "random_in_dataloader"
                ),
                "selection_time_s": selection_mean,
                "selection_time_std_s": selection_std,
                "bucketing_time_s": float(bucketing_time),
                "embedding_time_s": embedding_time,
                "end_to_end_time_s": end_to_end_time,
                "cells_per_s": float(
                    EXPECTED_TEST_CELLS / end_to_end_time
                ),
                "embedding_only_cells_per_s": float(
                    EXPECTED_TEST_CELLS / embedding_time
                ),
                "peak_gpu_memory_gb": float(
                    timing["peak_gpu_memory_gb"]
                ),
                "actual_sequence_tokens": actual_tokens,
                "padded_tokens_processed": padded_tokens,
                "padding_overhead_ratio": padding_ratio,
            }
            timing_rows.append(row)

            print(
                f"    timing repeat {repeat + 1}/{args.timing_repeats}: "
                f"E2E={end_to_end_time:.4f}s, "
                f"{row['cells_per_s']:.2f} cells/s, "
                f"embedding-only={row['embedding_only_cells_per_s']:.2f}, "
                f"memory={row['peak_gpu_memory_gb']:.3f} GB, "
                f"padding={padding_ratio:.4f}",
                flush=True,
            )

            del embeddings
            gc.collect()
            torch.cuda.empty_cache()

        del loader
        gc.collect()
        torch.cuda.empty_cache()

    timing_frame = pd.DataFrame(timing_rows)
    timing_frame.to_csv(
        args.output_dir / "timing_repeats.csv",
        index=False,
    )

    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        subset = timing_frame.loc[
            timing_frame["method"] == name
        ].copy()
        dataset = spec["dataset"]

        gene_lengths = legacy.get_lengths(dataset) - 1
        if spec["sampling"] == "random_in_dataloader":
            gene_lengths = np.minimum(
                gene_lengths,
                int(spec["fixed_k"]),
            )

        summary_rows.append(
            {
                "method": name,
                "timing_scope": "held-out P5 test only",
                "num_cells": EXPECTED_TEST_CELLS,
                "batch_size": args.batch_size,
                "timing_repeats": args.timing_repeats,
                "selection_timing_repeats": (
                    args.selection_timing_repeats
                ),
                "selection_in_dataloader": bool(
                    spec["sampling"] == "random_in_dataloader"
                ),
                "test_mean_selected_genes": float(
                    np.mean(gene_lengths)
                ),
                "test_median_selected_genes": float(
                    np.median(gene_lengths)
                ),
                "selection_time_s": float(
                    subset["selection_time_s"].mean()
                ),
                "selection_time_std_s": float(
                    subset["selection_time_std_s"].mean()
                ),
                "bucketing_time_s": float(
                    subset["bucketing_time_s"].mean()
                ),
                "embedding_time_mean_s": float(
                    subset["embedding_time_s"].mean()
                ),
                "embedding_time_std_s": sample_std(
                    subset["embedding_time_s"].tolist()
                ),
                "end_to_end_time_mean_s": float(
                    subset["end_to_end_time_s"].mean()
                ),
                "end_to_end_time_std_s": sample_std(
                    subset["end_to_end_time_s"].tolist()
                ),
                "cells_per_s_mean": float(
                    subset["cells_per_s"].mean()
                ),
                "cells_per_s_std": sample_std(
                    subset["cells_per_s"].tolist()
                ),
                "embedding_only_cells_per_s_mean": float(
                    subset["embedding_only_cells_per_s"].mean()
                ),
                "embedding_only_cells_per_s_std": sample_std(
                    subset[
                        "embedding_only_cells_per_s"
                    ].tolist()
                ),
                "peak_gpu_memory_gb": float(
                    subset["peak_gpu_memory_gb"].max()
                ),
                "padding_overhead_ratio_mean": float(
                    subset["padding_overhead_ratio"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    native_reference = summary.loc[
        summary["method"] == "scgpt_native_bucketed"
    ].iloc[0]
    full_reference = summary.loc[
        summary["method"] == "full_bucketed"
    ].iloc[0]

    summary["speedup_vs_native_bucketed"] = (
        native_reference["end_to_end_time_mean_s"]
        / summary["end_to_end_time_mean_s"]
    )
    summary["memory_reduction_vs_native_bucketed"] = (
        1.0
        - summary["peak_gpu_memory_gb"]
        / native_reference["peak_gpu_memory_gb"]
    )
    summary["speedup_vs_full_bucketed"] = (
        full_reference["end_to_end_time_mean_s"]
        / summary["end_to_end_time_mean_s"]
    )
    summary["memory_reduction_vs_full_bucketed"] = (
        1.0
        - summary["peak_gpu_memory_gb"]
        / full_reference["peak_gpu_memory_gb"]
    )

    summary.to_csv(
        args.output_dir / "summary_timing_only.csv",
        index=False,
    )

    legacy_summary = pd.read_csv(
        args.legacy_output_dir / "summary.csv"
    )
    if set(legacy_summary["method"]) != set(summary["method"]):
        raise RuntimeError(
            "Legacy summary methods do not match timing-only methods"
        )

    replacement_columns = [
        "selection_time_s",
        "selection_time_std_s",
        "bucketing_time_s",
        "embedding_time_mean_s",
        "embedding_time_std_s",
        "end_to_end_time_mean_s",
        "end_to_end_time_std_s",
        "cells_per_s_mean",
        "cells_per_s_std",
        "embedding_only_cells_per_s_mean",
        "embedding_only_cells_per_s_std",
        "peak_gpu_memory_gb",
        "padding_overhead_ratio_mean",
        "speedup_vs_native_bucketed",
        "memory_reduction_vs_native_bucketed",
        "speedup_vs_full_bucketed",
        "memory_reduction_vs_full_bucketed",
    ]

    updated = legacy_summary.copy()
    timing_by_method = summary.set_index("method")

    for row_index, method in enumerate(updated["method"].astype(str)):
        for column in replacement_columns:
            updated.loc[row_index, column] = timing_by_method.loc[
                method,
                column,
            ]
        updated.loc[row_index, "timing_scope"] = "held-out P5 test only"
        updated.loc[row_index, "timing_num_cells"] = EXPECTED_TEST_CELLS
        updated.loc[row_index, "timing_repeats"] = args.timing_repeats

    updated.to_csv(
        args.output_dir
        / "legacy_summary_with_test_only_efficiency.csv",
        index=False,
    )

    config = {
        "status": "PASS",
        "timing_scope": "held-out P5 test only",
        "num_cells": EXPECTED_TEST_CELLS,
        "legacy_script": str(args.legacy_script.resolve()),
        "legacy_output_dir": str(args.legacy_output_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "repo": str(args.repo.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "timing_repeats": args.timing_repeats,
        "selection_timing_repeats": args.selection_timing_repeats,
        "dtype": "AMP FP16 via existing corrected.extract_once",
        "fixed_k": fixed_k,
        "tau": tau,
        "k_min": k_min,
        "k_max": k_max,
        "native_max_length": native_max_length,
        "e2e_definition": (
            "test-only selection + test-only bucketing "
            "+ test-only embedding"
        ),
        "selection_definition": (
            "Fixed TF-IDF and AdaGeneBudget test selection are recomputed "
            "from the cached P5 CSR matrix without disk writes. Native and "
            "random selection occur inside DataCollator and are included "
            "in embedding wall time."
        ),
        "quality_metrics_reused_from": str(
            (args.legacy_output_dir / "summary.csv").resolve()
        ),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_columns = [
        "method",
        "test_mean_selected_genes",
        "end_to_end_time_mean_s",
        "cells_per_s_mean",
        "cells_per_s_std",
        "peak_gpu_memory_gb",
        "speedup_vs_native_bucketed",
        "memory_reduction_vs_native_bucketed",
    ]

    print()
    print("=" * 118)
    print("scGPT PBMC P5 TEST-ONLY TIMING SUMMARY")
    print("=" * 118)
    print(summary[display_columns].to_string(index=False))
    print()
    print("Output:", args.output_dir)
    print("FINAL STATUS: PASS")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
