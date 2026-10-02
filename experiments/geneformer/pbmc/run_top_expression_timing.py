#!/usr/bin/env python3
"""Table-1-matched timing for Geneformer-PBMC Fixed Top-Expression.

Matches 36_recheck_geneformer_pbmc_train_test_b64_r5.py:
- scope: train/reference + held-out P5 test/query
- batch size 64
- warm-up 3 batches per split
- 5 embedding timing repeats
- selection timing once over train + test, without disk writes
- FP32
- length-sorted/bucketed timing
- E2E = selection + bucketing + embedding

Quality is NOT recomputed.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

TRAIN_CELLS = 132_137
TEST_CELLS = 19_957
TOTAL_CELLS = TRAIN_CELLS + TEST_CELLS
BATCH_SIZE = 64
WARMUP_BATCHES = 3
TIMING_REPEATS = 5


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main-script", type=Path,
                   default=Path("experiments/geneformer/pbmc/run_main.py"))
    p.add_argument("--base-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base"))
    p.add_argument("--methods-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/methods"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/Geneformer-V1-10M"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/geneformer_pbmc_timing_table1"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def expression_selection_once(module: Any, base_split: dict[str, np.ndarray],
                              gene_median: np.ndarray, matched_k: int) -> int:
    selected_total = 0
    for row in range(len(base_split["lengths"])):
        positions, counts = module.base_row(base_split, row)
        k = min(matched_k, len(positions))
        expression_order = np.lexsort((positions, -counts))
        idx = expression_order[:k]
        selected_positions = positions[idx]
        selected_counts = counts[idx]
        native_scores = selected_counts.astype(np.float64, copy=False) / gene_median[selected_positions]
        final_order = np.lexsort((selected_positions, -native_scores))
        _ = selected_positions[final_order]
        selected_total += k
    return int(selected_total)


def expected_total(train_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> int:
    return int(np.sum(train_cache["lengths"], dtype=np.int64) +
               np.sum(test_cache["lengths"], dtype=np.int64))


def make_order(module: Any, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, float, float, int]:
    started = time.perf_counter()
    order = module.batching_order(cache["lengths"], "bucketed")
    bucketing_time = float(time.perf_counter() - started)
    padding_ratio = module.padding_overhead_ratio(
        lengths=cache["lengths"], order=order, batch_size=BATCH_SIZE
    )
    actual_tokens = int(np.sum(cache["lengths"], dtype=np.int64))
    return order, bucketing_time, float(padding_ratio), actual_tokens


def measure_split_once(module: Any, model: torch.nn.Module,
                       cache: dict[str, np.ndarray], order: np.ndarray,
                       inputs: dict[str, Any], pad_token_id: int,
                       device: torch.device) -> tuple[float, float]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
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
    peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return float(elapsed), peak


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.main_script, args.base_dir, args.methods_dir, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    module = load_module("gf_pbmc_top_expr_timing", args.main_script)
    matched_meta = module.read_json(args.methods_dir / "matched_budget.json")
    matched_k = int(matched_meta["matched_fixed_k"])
    cache_dir = args.methods_dir / f"matched_fixed_expression_k{matched_k}"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Run 01c first: {cache_dir}")

    inputs = module.load_inputs(args.base_dir, args.methods_dir)
    train_cache = module.load_method_split(cache_dir, "train")
    test_cache = module.load_method_split(cache_dir, "test")
    if len(train_cache["lengths"]) != TRAIN_CELLS or len(test_cache["lengths"]) != TEST_CELLS:
        raise RuntimeError("Unexpected train/test row counts")

    expected = expected_total(train_cache, test_cache)
    started = time.perf_counter()
    train_selected = expression_selection_once(
        module, inputs["base_splits"]["train"], inputs["gene_median"], matched_k
    )
    test_selected = expression_selection_once(
        module, inputs["base_splits"]["test"], inputs["gene_median"], matched_k
    )
    selection_time_s = float(time.perf_counter() - started)
    if train_selected + test_selected != expected:
        raise RuntimeError(
            f"Selection total mismatch: observed={train_selected + test_selected}, expected={expected}"
        )

    train_order, train_bucket, train_pad, train_actual = make_order(module, train_cache)
    test_order, test_bucket, test_pad, test_actual = make_order(module, test_cache)
    bucketing_time_s = float(train_bucket + test_bucket)
    combined_padding_ratio = float(
        (train_pad * train_actual + test_pad * test_actual) / (train_actual + test_actual)
    )

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    model, pad_token_id = module.load_model(args.checkpoint, device)
    module.warmup_model(
        model=model, cache=train_cache, order=train_order,
        token_ids=inputs["token_ids"], pad_token_id=pad_token_id,
        device=device, batch_size=BATCH_SIZE, warmup_batches=WARMUP_BATCHES,
    )
    module.warmup_model(
        model=model, cache=test_cache, order=test_order,
        token_ids=inputs["token_ids"], pad_token_id=pad_token_id,
        device=device, batch_size=BATCH_SIZE, warmup_batches=WARMUP_BATCHES,
    )

    rows = []
    for repeat in range(TIMING_REPEATS):
        train_time, train_peak = measure_split_once(
            module, model, train_cache, train_order, inputs, pad_token_id, device
        )
        test_time, test_peak = measure_split_once(
            module, model, test_cache, test_order, inputs, pad_token_id, device
        )
        embedding_time_s = float(train_time + test_time)
        e2e = float(selection_time_s + bucketing_time_s + embedding_time_s)
        peak = float(max(train_peak, test_peak))
        row = {
            "method": "matched_fixed_expression",
            "method_label": "Fixed Top-Expression",
            "repeat": repeat,
            "timing_scope": "train_plus_test",
            "train_cells": TRAIN_CELLS,
            "test_cells": TEST_CELLS,
            "total_cells": TOTAL_CELLS,
            "batch_size": BATCH_SIZE,
            "warmup_batches": WARMUP_BATCHES,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
            "embedding_time_s": embedding_time_s,
            "end_to_end_time_s": e2e,
            "cells_per_s": float(TOTAL_CELLS / e2e),
            "embedding_only_cells_per_s": float(TOTAL_CELLS / embedding_time_s),
            "peak_gpu_memory_gb": peak,
            "padding_overhead_ratio": combined_padding_ratio,
        }
        rows.append(row)
        print(
            f"repeat={repeat+1}/{TIMING_REPEATS}: "
            f"{row['cells_per_s']:.2f} cells/s, peak={peak:.3f} GB",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "timing_repeats.csv", index=False)
    s = {
        "method": "matched_fixed_expression",
        "method_label": "Fixed Top-Expression",
        "fixed_k": matched_k,
        "cache": str(cache_dir.resolve()),
        "timing_scope": "train_plus_test",
        "batch_size": BATCH_SIZE,
        "warmup_batches": WARMUP_BATCHES,
        "timing_repeats": TIMING_REPEATS,
        "selection_timing_repeats": 1,
        "train_mean_selected_genes": float(np.mean(train_cache["lengths"])),
        "test_mean_selected_genes": float(np.mean(test_cache["lengths"])),
        "selection_time_s": selection_time_s,
        "bucketing_time_s": bucketing_time_s,
        "embedding_time_mean_s": float(frame["embedding_time_s"].mean()),
        "embedding_time_std_s": float(frame["embedding_time_s"].std(ddof=1)),
        "end_to_end_time_mean_s": float(frame["end_to_end_time_s"].mean()),
        "end_to_end_time_std_s": float(frame["end_to_end_time_s"].std(ddof=1)),
        "cells_per_s_mean": float(frame["cells_per_s"].mean()),
        "cells_per_s_std": float(frame["cells_per_s"].std(ddof=1)),
        "embedding_only_cells_per_s_mean": float(frame["embedding_only_cells_per_s"].mean()),
        "embedding_only_cells_per_s_std": float(frame["embedding_only_cells_per_s"].std(ddof=1)),
        "peak_gpu_memory_gb": float(frame["peak_gpu_memory_gb"].max()),
        "padding_overhead_ratio_mean": combined_padding_ratio,
    }
    pd.DataFrame([s]).to_csv(args.output_dir / "summary_timing_only.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "status": "PASS",
        "protocol": "Geneformer-PBMC Table 1 unified timing",
        "timing_scope": "train_plus_test",
        "batch_size": 64,
        "warmup_batches": 3,
        "timing_repeats": 5,
        "selection_timing_repeats": 1,
        "dtype": "FP32",
        "batching": "bucketed/length-sorted",
        "end_to_end": "selection + bucketing + embedding",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([s]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
