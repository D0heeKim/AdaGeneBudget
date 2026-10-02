#!/usr/bin/env python3
"""Table-1-matched timing for Geneformer-Kang Fixed Top-Expression.

Matches 37_recheck_geneformer_kang_train_test_b64_r5_scopefix.py:
- train + test scope
- batch size 64, workers 4
- warm-up 3 per split
- 5 timing repeats
- FP32
- length-sorted batching
- hidden_states[5], non-padding mean pooling
- selection time taken from the validated expression-cache build summary
- E2E = selection + bucketing + embedding

Quality is NOT recomputed.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

TRAIN_CELLS = 19_583
TEST_CELLS = 5_090
TOTAL_CELLS = TRAIN_CELLS + TEST_CELLS


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
    p.add_argument("--benchmark-script", type=Path,
                   default=Path("experiments/geneformer/kang/run_main.py"))
    p.add_argument("--method-cache-root", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/geneformer_cache/methods"))
    p.add_argument("--expression-summary", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/geneformer_cache/methods/expression_ablation_summary.csv"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/Geneformer-V1-10M"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/geneformer_kang_timing_table1"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def measure_split_once(benchmark: Any, model: torch.nn.Module,
                       hidden_state_index: int, loader: Any,
                       device: torch.device) -> dict[str, float]:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    actual_tokens = 0; padded_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            batch_embeddings = benchmark.model_batch_embeddings(
                model=model,
                hidden_state_index=hidden_state_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths,
            )
            actual_tokens += int(lengths.sum().item())
            padded_tokens += int(input_ids.numel())
            del batch_embeddings, input_ids, attention_mask, lengths
    torch.cuda.synchronize(device)
    elapsed = float(perf_counter() - started)
    peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    return {
        "embedding_time_s": elapsed,
        "peak_gpu_memory_gb": peak,
        "actual_sequence_tokens": float(actual_tokens),
        "padded_tokens_processed": float(padded_tokens),
    }


def main() -> int:
    args = parse_args()
    if (args.batch_size, args.num_workers, args.warmup_steps, args.timing_repeats) != (64, 4, 3, 5):
        raise RuntimeError("Frozen Table-1 protocol is batch=64, workers=4, warmup=3, repeats=5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.benchmark_script, args.expression_summary, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; use --overwrite")
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    summary = pd.read_csv(args.expression_summary)
    rows = summary.loc[summary["method"] == "fixed_expression"]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one fixed_expression row, found {len(rows)}")
    qrow = rows.iloc[0]
    cache_path = Path(str(qrow["cache"]))
    if not cache_path.is_absolute():
        cache_path = args.method_cache_root / cache_path
    if not cache_path.is_dir():
        raise FileNotFoundError(cache_path)
    selection_time_s = float(qrow["selection_compute_time_s"])

    benchmark = load_module("gf_kang_top_expr_timing", args.benchmark_script)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    benchmark.seed_all(0)
    model, hidden_state_index = benchmark.load_model(args.checkpoint, device)

    train_cache = benchmark.load_token_cache(cache_path, "train")
    test_cache = benchmark.load_token_cache(cache_path, "test")
    train_dataset = benchmark.CachedTokenDataset(train_cache)
    test_dataset = benchmark.CachedTokenDataset(test_cache)
    if len(train_dataset) != TRAIN_CELLS or len(test_dataset) != TEST_CELLS:
        raise RuntimeError("Unexpected train/test row counts")

    train_loader, train_bucket = benchmark.make_loader(
        train_dataset, args.batch_size, args.num_workers, "length_sorted"
    )
    test_loader, test_bucket = benchmark.make_loader(
        test_dataset, args.batch_size, args.num_workers, "length_sorted"
    )
    bucketing_time_s = float(train_bucket + test_bucket)

    benchmark.warmup(model, hidden_state_index, train_loader, device, args.warmup_steps)
    benchmark.warmup(model, hidden_state_index, test_loader, device, args.warmup_steps)

    out_rows = []
    for repeat in range(args.timing_repeats):
        train_t = measure_split_once(benchmark, model, hidden_state_index, train_loader, device)
        test_t = measure_split_once(benchmark, model, hidden_state_index, test_loader, device)
        embedding_time_s = float(train_t["embedding_time_s"] + test_t["embedding_time_s"])
        e2e = float(selection_time_s + bucketing_time_s + embedding_time_s)
        actual = float(train_t["actual_sequence_tokens"] + test_t["actual_sequence_tokens"])
        padded = float(train_t["padded_tokens_processed"] + test_t["padded_tokens_processed"])
        peak = float(max(train_t["peak_gpu_memory_gb"], test_t["peak_gpu_memory_gb"]))
        row = {
            "method": "matched_fixed_expression",
            "method_label": "Fixed Top-Expression",
            "repeat": repeat,
            "timing_scope": "train_plus_test",
            "batch_size": 64,
            "num_workers": 4,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
            "embedding_time_s": embedding_time_s,
            "end_to_end_time_s": e2e,
            "cells_per_s": float(TOTAL_CELLS / e2e),
            "embedding_only_cells_per_s": float(TOTAL_CELLS / embedding_time_s),
            "peak_gpu_memory_gb": peak,
            "actual_sequence_tokens": actual,
            "padded_tokens_processed": padded,
            "padding_overhead_ratio": float(padded / actual),
        }
        out_rows.append(row)
        print(
            f"repeat={repeat+1}/{args.timing_repeats}: "
            f"{row['cells_per_s']:.2f} cells/s, peak={peak:.3f} GB",
            flush=True,
        )
        gc.collect(); torch.cuda.empty_cache()

    frame = pd.DataFrame(out_rows)
    frame.to_csv(args.output_dir / "timing_repeats.csv", index=False)
    train_lengths = np.asarray(train_cache.lengths, dtype=np.int64)
    test_lengths = np.asarray(test_cache.lengths, dtype=np.int64)
    s = {
        "method": "matched_fixed_expression",
        "method_label": "Fixed Top-Expression",
        "fixed_k": int(round(float(qrow["fixed_k"]))),
        "cache": str(cache_path.resolve()),
        "timing_scope": "train_plus_test",
        "train_mean_selected_genes": float(np.mean(train_lengths)),
        "test_mean_selected_genes": float(np.mean(test_lengths)),
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
        "padding_overhead_ratio_mean": float(frame["padding_overhead_ratio"].mean()),
    }
    pd.DataFrame([s]).to_csv(args.output_dir / "summary_timing_only.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "status": "PASS",
        "protocol": "Geneformer-Kang Table 1 scope-fixed timing",
        "timing_scope": "train_plus_test",
        "batch_size": 64,
        "num_workers": 4,
        "warmup_steps": 3,
        "timing_repeats": 5,
        "selection_time_source": str(args.expression_summary.resolve()),
        "batching": "length_sorted",
        "dtype": "FP32",
        "end_to_end": "selection + bucketing + embedding",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([s]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
