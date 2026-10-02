#!/usr/bin/env python3
"""Train+test Table-1 timing for the scGPT-PBMC Fixed Top-Expression baseline.

Protocol:
- scope: PBMC P5 train + held-out P5 test
- batch size 64, workers 4
- warm-up 3 per split
- 5 embedding timing repeats
- 1 selection timing repeat
- CSR cache
- length-sorted batching
- E2E = train+test selection + train+test bucketing + train+test embedding

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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957


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
    p.add_argument("--legacy-script", type=Path,
                   default=Path("experiments/scgpt/pbmc/run_reference_pipeline.py"))
    p.add_argument("--fixed-script", type=Path,
                   default=Path("experiments/scgpt/_shared/fixed_selection.py"))
    p.add_argument("--corrected-script", type=Path,
                   default=Path("experiments/scgpt/_shared/runtime_utils.py"))
    p.add_argument("--legacy-output-dir", type=Path,
                   default=Path("outputs/scgpt/pbmc_p5/main"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/scgpt_cache"))
    p.add_argument("--repo", type=Path, default=Path("scGPT"))
    p.add_argument("--model-dir", type=Path, default=Path("checkpoints/scgpt"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/scgpt_pbmc_timing_table1_train_plus_test"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--selection-timing-repeats", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def recompute_expression_once(matrix: sparse.csr_matrix, fixed_k: int) -> int:
    selected_total = 0
    for row in range(matrix.shape[0]):
        start = int(matrix.indptr[row]); stop = int(matrix.indptr[row + 1])
        indices = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        n = int(indices.size)
        if n <= fixed_k:
            chosen = indices
        else:
            positions = np.argpartition(values, -fixed_k)[-fixed_k:]
            chosen = np.sort(indices[positions])
        selected_total += int(len(chosen))
    return int(selected_total)


def main() -> int:
    args = parse_args()
    frozen = (args.batch_size, args.num_workers, args.warmup_steps,
              args.timing_repeats, args.selection_timing_repeats)
    if frozen != (64, 4, 3, 5, 1):
        raise RuntimeError(
            "Table-1 protocol is frozen at batch_size=64, num_workers=4, "
            "warmup_steps=3, timing_repeats=5, selection_timing_repeats=1."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    required = [
        args.legacy_script,
        args.fixed_script,
        args.corrected_script,
        args.legacy_output_dir / "config.json",
        args.legacy_output_dir / "matched_genes.npy",
        args.cache_dir / "train_matched_csr.npz",
        args.cache_dir / "test_matched_csr.npz",
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    legacy = load_module("scgpt_pbmc_top_expr_timing_main", args.legacy_script)
    fixed = load_module("scgpt_pbmc_top_expr_timing_fixed", args.fixed_script)
    corrected = load_module("scgpt_pbmc_top_expr_timing_corrected", args.corrected_script)

    config = json.loads((args.legacy_output_dir / "config.json").read_text(encoding="utf-8"))
    fixed_k = int(config["fixed_k"])
    train_matrix = sparse.load_npz(args.cache_dir / "train_matched_csr.npz").tocsr()
    test_matrix = sparse.load_npz(args.cache_dir / "test_matched_csr.npz").tocsr()
    for matrix in (train_matrix, test_matrix):
        matrix.sum_duplicates(); matrix.eliminate_zeros(); matrix.sort_indices()
    if train_matrix.shape[0] != EXPECTED_TRAIN_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_CELLS} train cells, found {train_matrix.shape[0]}")
    if test_matrix.shape[0] != EXPECTED_TEST_CELLS:
        raise RuntimeError(f"Expected {EXPECTED_TEST_CELLS} test cells, found {test_matrix.shape[0]}")

    matched_genes = np.load(args.legacy_output_dir / "matched_genes.npy", allow_pickle=True).astype(str)
    model_args = SimpleNamespace(repo=args.repo, model_dir=args.model_dir)
    model, vocab, cfg, pad_id, pad_value, device = legacy.load_model(model_args, fixed)
    gene_ids = np.asarray(vocab(list(matched_genes)), dtype=np.int64)

    expr_train = fixed.FixedSelectionDataset(
        matrix=train_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
        method="expression",
        budget=fixed_k,
        seed=0,
        idf=None,
    )
    expr_test = fixed.FixedSelectionDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
        method="expression",
        budget=fixed_k,
        seed=0,
        idf=None,
    )

    from scgpt.data_collator import DataCollator
    collator = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=fixed_k + 1,
        sampling=False,
        keep_first_n_tokens=1,
    )
    legacy.seed_all(0)
    train_loader, train_bucketing_time_s = legacy.make_loader(
        expr_train,
        collator,
        args.batch_size,
        args.num_workers,
        999001,
        "length_sorted",
        fixed_k + 1,
    )
    test_loader, test_bucketing_time_s = legacy.make_loader(
        expr_test,
        collator,
        args.batch_size,
        args.num_workers,
        999002,
        "length_sorted",
        fixed_k + 1,
    )
    bucketing_time_s = float(train_bucketing_time_s + test_bucketing_time_s)

    expected_total = int(
        np.sum(expr_train.gene_lengths, dtype=np.int64)
        + np.sum(expr_test.gene_lengths, dtype=np.int64)
    )
    selection_times = []
    for repeat in range(args.selection_timing_repeats):
        started = time.perf_counter()
        observed = (
            recompute_expression_once(train_matrix, fixed_k)
            + recompute_expression_once(test_matrix, fixed_k)
        )
        elapsed = float(time.perf_counter() - started)
        if observed != expected_total:
            raise RuntimeError(f"Selection total mismatch: {observed} != {expected_total}")
        selection_times.append(elapsed)
    selection_time_s = float(np.mean(selection_times))

    corrected.warmup(model, train_loader, pad_id, device, args.warmup_steps)
    corrected.warmup(model, test_loader, pad_id, device, args.warmup_steps)
    total_cells = EXPECTED_TRAIN_CELLS + EXPECTED_TEST_CELLS
    rows = []
    for repeat in range(args.timing_repeats):
        train_embeddings, train_timing = corrected.extract_once(
            model, train_loader, expr_train, pad_id, device, cfg["embsize"]
        )
        test_embeddings, test_timing = corrected.extract_once(
            model, test_loader, expr_test, pad_id, device, cfg["embsize"]
        )
        embedding_time_s = float(
            train_timing["embedding_time_s"] + test_timing["embedding_time_s"]
        )
        e2e = float(selection_time_s + bucketing_time_s + embedding_time_s)
        actual = int(
            train_timing["actual_sequence_tokens"] + test_timing["actual_sequence_tokens"]
        )
        padded = int(
            train_timing["padded_tokens_processed"] + test_timing["padded_tokens_processed"]
        )
        peak_memory = max(
            float(train_timing["peak_gpu_memory_gb"]),
            float(test_timing["peak_gpu_memory_gb"]),
        )
        row = {
            "method": "matched_fixed_expression",
            "method_label": "Fixed Top-Expression",
            "repeat": repeat,
            "timing_scope": "train_plus_test",
            "num_cells": total_cells,
            "batch_size": args.batch_size,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": float(bucketing_time_s),
            "embedding_time_s": embedding_time_s,
            "end_to_end_time_s": e2e,
            "cells_per_s": float(total_cells / e2e),
            "embedding_only_cells_per_s": float(total_cells / embedding_time_s),
            "peak_gpu_memory_gb": peak_memory,
            "actual_sequence_tokens": actual,
            "padded_tokens_processed": padded,
            "padding_overhead_ratio": float(padded / actual),
        }
        rows.append(row)
        print(
            f"repeat={repeat+1}/{args.timing_repeats}: "
            f"{row['cells_per_s']:.2f} cells/s, peak={row['peak_gpu_memory_gb']:.3f} GB",
            flush=True,
        )
        del train_embeddings, test_embeddings
        gc.collect(); torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "timing_repeats.csv", index=False)
    summary = {
        "method": "matched_fixed_expression",
        "method_label": "Fixed Top-Expression",
        "fixed_k": fixed_k,
        "timing_scope": "train_plus_test",
        "train_mean_selected_genes": float(np.mean(expr_train.gene_lengths)),
        "test_mean_selected_genes": float(np.mean(expr_test.gene_lengths)),
        "selection_time_s": selection_time_s,
        "selection_time_std_s": float(np.std(selection_times, ddof=0)),
        "bucketing_time_s": float(bucketing_time_s),
        "embedding_time_mean_s": float(frame["embedding_time_s"].mean()),
        "embedding_time_std_s": float(frame["embedding_time_s"].std(ddof=1)),
        "end_to_end_time_mean_s": float(frame["end_to_end_time_s"].mean()),
        "end_to_end_time_std_s": float(frame["end_to_end_time_s"].std(ddof=1)),
        "cells_per_s_mean": float(frame["cells_per_s"].mean()),
        "cells_per_s_std": float(frame["cells_per_s"].std(ddof=1)),
        "embedding_only_cells_per_s_mean": float(frame["embedding_only_cells_per_s"].mean()),
        "embedding_only_cells_per_s_std": float(frame["embedding_only_cells_per_s"].std(ddof=1)),
        "peak_gpu_memory_gb": float(frame["peak_gpu_memory_gb"].max()),
        "padding_overhead_ratio_mean": float(frame["padding_overhead_ratio"].mean()),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary_timing_only.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "status": "PASS",
        "fixed_k": fixed_k,
        "protocol": "scGPT-PBMC Table 1 train+test timing",
        "timing_scope": "train_plus_test",
        "batch_size": 64,
        "num_workers": 4,
        "warmup_steps": 3,
        "timing_repeats": 5,
        "selection_timing_repeats": 1,
        "batching": "length_sorted",
        "end_to_end": "train+test selection + train+test bucketing + train+test embedding",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
