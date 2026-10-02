#!/usr/bin/env python3
"""Recheck PBMC Geneformer efficiency under the unified systems protocol.

Unified protocol
----------------
- Scope: train/reference + held-out test/query
- Batch size: 64
- Warm-up: 3 batches per split
- Embedding timing: 5 repeats
- Selection timing: once over train + test
- FP32
- Physical GPU 1 is selected externally with CUDA_VISIBLE_DEVICES=1
- Process-local device: cuda:0
- Quality metrics and embeddings are not recomputed
- Existing main outputs are never modified

Methods
-------
1. Geneformer Native Sequential
2. Geneformer Native Bucketed
3. Native-Rank Top-599
4. Matched Random, timing seed 0
5. Fixed TF-IDF
6. AdaGeneBudget
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(".")

MAIN_SCRIPT = (
    PROJECT / "experiments/geneformer/pbmc/run_main.py"
)

BASE_CACHE = (
    PROJECT
    / "data"
    / "processed"
    / "pbmc_seurat_v4"
    / "p5_holdout"
    / "geneformer_cache"
    / "base"
)

METHOD_ROOT = BASE_CACHE.parent / "methods"

CHECKPOINT = Path(
    "./checkpoints/Geneformer-V1-10M"
)

DEFAULT_OUTPUT = (
    PROJECT
    / "outputs"
    / "geneformer"
    / "pbmc_p5"
    / "timing_train_test_b64_r5"
)

TRAIN_CELLS = 132_137
TEST_CELLS = 19_957
TOTAL_CELLS = TRAIN_CELLS + TEST_CELLS

BATCH_SIZE = 64
WARMUP_BATCHES = 3
TIMING_REPEATS = 5
NATIVE_RANK_K = 599


METHOD_SPECS = [
    {
        "method": "geneformer_native_sequential",
        "method_label": "Geneformer Native Sequential",
        "cache": METHOD_ROOT / "native_max2048",
        "batching": "sequential",
        "selection_key": "native",
        "module_method": "geneformer_native_sequential",
        "seed": None,
    },
    {
        "method": "geneformer_native_bucketed",
        "method_label": "Geneformer Native Bucketed",
        "cache": METHOD_ROOT / "native_max2048",
        "batching": "bucketed",
        "selection_key": "native",
        "module_method": "geneformer_native_bucketed",
        "seed": None,
    },
    {
        "method": "native_rank_fixed_k",
        "method_label": "Native-Rank Top-599",
        "cache": METHOD_ROOT / "native_rank_fixed_k599",
        "batching": "bucketed",
        "selection_key": "native_rank_fixed_k",
        "module_method": None,
        "seed": None,
    },
    {
        "method": "matched_random_fixed_k",
        "method_label": "Matched Random",
        "cache": (
            METHOD_ROOT
            / "matched_random_fixed_k599"
            / "seed0"
        ),
        "batching": "bucketed",
        "selection_key": "matched_random_fixed_k",
        "module_method": "matched_random_fixed_k",
        "seed": 0,
    },
    {
        "method": "matched_fixed_tfidf",
        "method_label": "Fixed TF-IDF",
        "cache": METHOD_ROOT / "matched_fixed_tfidf_k599",
        "batching": "bucketed",
        "selection_key": "matched_fixed_tfidf",
        "module_method": "matched_fixed_tfidf",
        "seed": None,
    },
    {
        "method": "adaptive",
        "method_label": "AdaGeneBudget",
        "cache": METHOD_ROOT / "adaptive_tau0p90_k128_k600",
        "batching": "bucketed",
        "selection_key": "adaptive",
        "module_method": "adaptive",
        "seed": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_main_module() -> Any:
    require(MAIN_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "geneformer_pbmc_unified_timing",
        MAIN_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {MAIN_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expected_selected_total(
    train_cache: dict[str, np.ndarray],
    test_cache: dict[str, np.ndarray],
) -> int:
    return int(
        np.sum(
            train_cache["lengths"],
            dtype=np.int64,
        )
        + np.sum(
            test_cache["lengths"],
            dtype=np.int64,
        )
    )


def recompute_native_rank_topk_split(
    module: Any,
    base_split: dict[str, np.ndarray],
    gene_median: np.ndarray,
) -> int:
    selected_total = 0

    for row in range(len(base_split["lengths"])):
        positions, counts = module.base_row(
            base_split,
            row,
        )

        order = module.native_order(
            positions,
            counts,
            gene_median,
        )

        selected_total += min(
            NATIVE_RANK_K,
            len(order),
        )

    return int(selected_total)


def measure_selection_once(
    module: Any,
    spec: dict[str, Any],
    inputs: dict[str, Any],
    train_cache: dict[str, np.ndarray],
    test_cache: dict[str, np.ndarray],
) -> float:
    expected = expected_selected_total(
        train_cache,
        test_cache,
    )

    started = time.perf_counter()

    if spec["selection_key"] == "native_rank_fixed_k":
        train_total = recompute_native_rank_topk_split(
            module,
            inputs["base_splits"]["train"],
            inputs["gene_median"],
        )
        test_total = recompute_native_rank_topk_split(
            module,
            inputs["base_splits"]["test"],
            inputs["gene_median"],
        )
    else:
        train_total = module.recompute_selection_once(
            method=spec["module_method"],
            base_split=inputs["base_splits"]["train"],
            gene_median=inputs["gene_median"],
            idf=inputs["idf"],
            seed=spec["seed"],
        )
        test_total = module.recompute_selection_once(
            method=spec["module_method"],
            base_split=inputs["base_splits"]["test"],
            gene_median=inputs["gene_median"],
            idf=inputs["idf"],
            seed=spec["seed"],
        )

    elapsed = float(
        time.perf_counter() - started
    )
    observed = int(train_total + test_total)

    if observed != expected:
        raise RuntimeError(
            f"{spec['method']}: selection total mismatch: "
            f"observed={observed}, expected={expected}"
        )

    return elapsed


def make_order(
    module: Any,
    cache: dict[str, np.ndarray],
    batching: str,
) -> tuple[np.ndarray, float, float, int]:
    started = time.perf_counter()
    order = module.batching_order(
        cache["lengths"],
        batching,
    )
    bucketing_time = float(
        time.perf_counter() - started
    )

    padding_ratio = module.padding_overhead_ratio(
        lengths=cache["lengths"],
        order=order,
        batch_size=BATCH_SIZE,
    )

    actual_tokens = int(
        np.sum(
            cache["lengths"],
            dtype=np.int64,
        )
    )

    return (
        order,
        bucketing_time,
        float(padding_ratio),
        actual_tokens,
    )


def measure_split_once(
    module: Any,
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    order: np.ndarray,
    inputs: dict[str, Any],
    pad_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    elapsed = module.time_embedding_pass(
        model=model,
        cache=cache,
        order=order,
        token_ids=inputs["token_ids"],
        pad_token_id=pad_token_id,
        device=device,
        batch_size=BATCH_SIZE,
    )

    peak_memory = float(
        torch.cuda.max_memory_allocated(device)
        / (1024**3)
    )

    return float(elapsed), peak_memory


def main() -> int:
    args = parse_args()

    for path in (
        MAIN_SCRIPT,
        BASE_CACHE,
        METHOD_ROOT,
        CHECKPOINT,
    ):
        require(path)

    for spec in METHOD_SPECS:
        require(spec["cache"])

    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. "
                "Inspect it before using --overwrite."
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    module = load_main_module()
    inputs = module.load_inputs(
        BASE_CACHE,
        METHOD_ROOT,
    )

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    model, pad_token_id = module.load_model(
        CHECKPOINT,
        device,
    )

    print("=" * 118, flush=True)
    print(
        "PBMC GENEFORMER UNIFIED TRAIN+TEST TIMING",
        flush=True,
    )
    print("=" * 118, flush=True)
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        flush=True,
    )
    print("process device:", device, flush=True)
    print(
        "GPU:",
        torch.cuda.get_device_name(device),
        flush=True,
    )
    print(
        f"scope=train+test, cells={TOTAL_CELLS:,}",
        flush=True,
    )
    print(
        f"batch={BATCH_SIZE}, warmup={WARMUP_BATCHES}, "
        f"repeats={TIMING_REPEATS}",
        flush=True,
    )

    selection_times: dict[str, float] = {}
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method_index, spec in enumerate(
        METHOD_SPECS,
        start=1,
    ):
        print("\n" + "=" * 118, flush=True)
        print(
            f"[{method_index}/{len(METHOD_SPECS)}] "
            f"{spec['method']} ({spec['method_label']})",
            flush=True,
        )
        print(
            "cache:",
            spec["cache"],
            flush=True,
        )
        print("=" * 118, flush=True)

        train_cache = module.load_method_split(
            spec["cache"],
            "train",
        )
        test_cache = module.load_method_split(
            spec["cache"],
            "test",
        )

        if len(train_cache["lengths"]) != TRAIN_CELLS:
            raise RuntimeError(
                f"{spec['method']}: invalid train row count."
            )
        if len(test_cache["lengths"]) != TEST_CELLS:
            raise RuntimeError(
                f"{spec['method']}: invalid test row count."
            )

        selection_key = str(spec["selection_key"])

        if selection_key not in selection_times:
            selection_times[selection_key] = (
                measure_selection_once(
                    module=module,
                    spec=spec,
                    inputs=inputs,
                    train_cache=train_cache,
                    test_cache=test_cache,
                )
            )

        selection_time = selection_times[
            selection_key
        ]

        (
            train_order,
            train_bucketing_time,
            train_padding_ratio,
            train_actual_tokens,
        ) = make_order(
            module,
            train_cache,
            spec["batching"],
        )

        (
            test_order,
            test_bucketing_time,
            test_padding_ratio,
            test_actual_tokens,
        ) = make_order(
            module,
            test_cache,
            spec["batching"],
        )

        bucketing_time = float(
            train_bucketing_time
            + test_bucketing_time
        )

        combined_padding_ratio = float(
            (
                train_padding_ratio
                * train_actual_tokens
                + test_padding_ratio
                * test_actual_tokens
            )
            / (
                train_actual_tokens
                + test_actual_tokens
            )
        )

        module.warmup_model(
            model=model,
            cache=train_cache,
            order=train_order,
            token_ids=inputs["token_ids"],
            pad_token_id=pad_token_id,
            device=device,
            batch_size=BATCH_SIZE,
            warmup_batches=WARMUP_BATCHES,
        )
        module.warmup_model(
            model=model,
            cache=test_cache,
            order=test_order,
            token_ids=inputs["token_ids"],
            pad_token_id=pad_token_id,
            device=device,
            batch_size=BATCH_SIZE,
            warmup_batches=WARMUP_BATCHES,
        )

        method_runs: list[dict[str, Any]] = []

        for repeat in range(TIMING_REPEATS):
            train_time, train_peak = measure_split_once(
                module=module,
                model=model,
                cache=train_cache,
                order=train_order,
                inputs=inputs,
                pad_token_id=pad_token_id,
                device=device,
            )

            test_time, test_peak = measure_split_once(
                module=module,
                model=model,
                cache=test_cache,
                order=test_order,
                inputs=inputs,
                pad_token_id=pad_token_id,
                device=device,
            )

            embedding_time = float(
                train_time + test_time
            )
            end_to_end_time = float(
                selection_time
                + bucketing_time
                + embedding_time
            )
            peak_memory = float(
                max(train_peak, test_peak)
            )

            row = {
                "method": spec["method"],
                "method_label": spec["method_label"],
                "repeat": repeat,
                "timing_scope": "train_plus_test",
                "train_cells": TRAIN_CELLS,
                "test_cells": TEST_CELLS,
                "total_cells": TOTAL_CELLS,
                "batch_size": BATCH_SIZE,
                "warmup_batches": WARMUP_BATCHES,
                "timing_repeats": TIMING_REPEATS,
                "batching": spec["batching"],
                "selection_time_s": selection_time,
                "selection_timing_repeats": 1,
                "bucketing_time_s": bucketing_time,
                "train_embedding_time_s": train_time,
                "test_embedding_time_s": test_time,
                "embedding_time_s": embedding_time,
                "end_to_end_time_s": end_to_end_time,
                "cells_per_s": float(
                    TOTAL_CELLS / end_to_end_time
                ),
                "embedding_only_cells_per_s": float(
                    TOTAL_CELLS / embedding_time
                ),
                "peak_gpu_memory_gb": peak_memory,
                "padding_overhead_ratio": (
                    combined_padding_ratio
                ),
            }

            method_runs.append(row)
            run_rows.append(row)

            print(
                f"repeat={repeat + 1}/{TIMING_REPEATS} "
                f"train={train_time:.3f}s "
                f"test={test_time:.3f}s "
                f"E2E={end_to_end_time:.3f}s "
                f"throughput={row['cells_per_s']:.2f} cells/s "
                f"peak={peak_memory:.3f} GB",
                flush=True,
            )

        frame = pd.DataFrame(method_runs)

        summary = {
            "method": spec["method"],
            "method_label": spec["method_label"],
            "cache": str(spec["cache"].resolve()),
            "timing_scope": "train_plus_test",
            "train_cells": TRAIN_CELLS,
            "test_cells": TEST_CELLS,
            "total_cells": TOTAL_CELLS,
            "batch_size": BATCH_SIZE,
            "warmup_batches": WARMUP_BATCHES,
            "timing_repeats": TIMING_REPEATS,
            "selection_timing_repeats": 1,
            "batching": spec["batching"],
            "selection_time_s": selection_time,
            "bucketing_time_s": bucketing_time,
            "train_mean_selected_genes": float(
                np.mean(train_cache["lengths"])
            ),
            "test_mean_selected_genes": float(
                np.mean(test_cache["lengths"])
            ),
            "embedding_time_mean_s": float(
                frame["embedding_time_s"].mean()
            ),
            "embedding_time_std_s": float(
                frame["embedding_time_s"].std(ddof=1)
            ),
            "end_to_end_time_mean_s": float(
                frame["end_to_end_time_s"].mean()
            ),
            "end_to_end_time_std_s": float(
                frame["end_to_end_time_s"].std(ddof=1)
            ),
            "cells_per_s_mean": float(
                frame["cells_per_s"].mean()
            ),
            "cells_per_s_std": float(
                frame["cells_per_s"].std(ddof=1)
            ),
            "embedding_only_cells_per_s_mean": float(
                frame[
                    "embedding_only_cells_per_s"
                ].mean()
            ),
            "embedding_only_cells_per_s_std": float(
                frame[
                    "embedding_only_cells_per_s"
                ].std(ddof=1)
            ),
            "peak_gpu_memory_gb": float(
                frame["peak_gpu_memory_gb"].max()
            ),
            "padding_overhead_ratio_mean": (
                combined_padding_ratio
            ),
        }

        summary_rows.append(summary)

        pd.DataFrame(run_rows).to_csv(
            output_dir / "timing_runs_partial.csv",
            index=False,
        )
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "timing_summary_partial.csv",
            index=False,
        )

        del (
            train_cache,
            test_cache,
            train_order,
            test_order,
        )
        gc.collect()
        torch.cuda.empty_cache()

    run_frame = pd.DataFrame(run_rows)
    summary_frame = pd.DataFrame(summary_rows)

    native = summary_frame.loc[
        summary_frame["method"]
        == "geneformer_native_bucketed"
    ]

    if len(native) != 1:
        raise RuntimeError(
            "Native Bucketed reference row is missing."
        )

    native_row = native.iloc[0]

    summary_frame[
        "speedup_vs_native_bucketed"
    ] = (
        summary_frame["cells_per_s_mean"]
        / float(native_row["cells_per_s_mean"])
    )

    summary_frame[
        "memory_reduction_vs_native_bucketed"
    ] = (
        1.0
        - summary_frame["peak_gpu_memory_gb"]
        / float(native_row["peak_gpu_memory_gb"])
    )

    run_path = output_dir / "timing_runs.csv"
    summary_path = output_dir / "timing_summary.csv"

    run_frame.to_csv(
        run_path,
        index=False,
    )
    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "status": "PASS",
        "purpose": (
            "PBMC Geneformer unified train+test "
            "batch-64 five-repeat timing"
        ),
        "main_results_modified": False,
        "physical_gpu_requested": 1,
        "cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "process_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "timing_scope": "train_plus_test",
        "train_cells": TRAIN_CELLS,
        "test_cells": TEST_CELLS,
        "total_cells": TOTAL_CELLS,
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "timing_repeats": TIMING_REPEATS,
        "selection_timing_repeats": 1,
        "selection_time_policy": (
            "Fresh CPU recomputation over train and test; "
            "no disk writes."
        ),
        "methods": [
            spec["method"]
            for spec in METHOD_SPECS
        ],
        "timing_runs_csv": str(run_path.resolve()),
        "timing_summary_csv": str(
            summary_path.resolve()
        ),
    }

    atomic_json(
        output_dir / "manifest.json",
        manifest,
    )

    print("\n" + "=" * 118, flush=True)
    print("FINAL UNIFIED TIMING SUMMARY", flush=True)
    print("=" * 118, flush=True)

    display_columns = [
        "method",
        "train_mean_selected_genes",
        "test_mean_selected_genes",
        "end_to_end_time_mean_s",
        "end_to_end_time_std_s",
        "cells_per_s_mean",
        "cells_per_s_std",
        "peak_gpu_memory_gb",
        "speedup_vs_native_bucketed",
        "memory_reduction_vs_native_bucketed",
    ]

    print(
        summary_frame[
            display_columns
        ].to_string(index=False),
        flush=True,
    )

    print("\nSaved:", summary_path, flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
