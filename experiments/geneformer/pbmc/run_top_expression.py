#!/usr/bin/env python3
"""Evaluate the Geneformer-PBMC matched fixed Top-Expression baseline.

Reuses the validated PBMC benchmark's model loading, embedding extraction,
exact 5-NN reference mapping, fidelity metrics, and timing utilities. Results
are written only under outputs/release_r1.
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
from sklearn.preprocessing import LabelEncoder


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("geneformer_pbmc_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-script", type=Path,
                   default=Path("experiments/geneformer/pbmc/run_main.py"))
    p.add_argument("--base-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base"))
    p.add_argument("--methods-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/methods"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/Geneformer-V1-10M"))
    p.add_argument("--canonical-output", type=Path,
                   default=Path("outputs/geneformer/pbmc_p5/main"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/geneformer_pbmc"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--timing-repeats", type=int, default=3)
    p.add_argument("--selection-timing-repeats", type=int, default=1)
    p.add_argument("--knn-query-batch-size", type=int, default=512)
    p.add_argument("--neighbor-query-batch-size", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def expression_selection_once(base_split: dict[str, np.ndarray], gene_median: np.ndarray,
                              matched_k: int) -> int:
    selected_total = 0
    for row in range(len(base_split["lengths"])):
        s, e = int(base_split["indptr"][row]), int(base_split["indptr"][row + 1])
        positions = np.asarray(base_split["gene_pos"][s:e], dtype=np.int64)
        counts = np.asarray(base_split["counts"][s:e], dtype=np.float64)
        k = min(matched_k, len(positions))
        order = np.lexsort((positions, -counts))
        idx = order[:k]
        selected_positions = positions[idx]
        selected_counts = counts[idx]
        native_scores = selected_counts / gene_median[selected_positions]
        _ = selected_positions[np.lexsort((selected_positions, -native_scores))]
        selected_total += k
    return int(selected_total)


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume/--overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bench = load_module(args.benchmark_script)

    matched_meta = bench.read_json(args.methods_dir / "matched_budget.json")
    matched_k = int(matched_meta["matched_fixed_k"])
    cache_dir = args.methods_dir / f"matched_fixed_expression_k{matched_k}"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Run 01c first: {cache_dir}")

    if args.output_dir.exists():
        if args.overwrite:
            import shutil
            shutil.rmtree(args.output_dir)
        elif not args.resume:
            raise FileExistsError(f"{args.output_dir} exists; use --resume or --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = bench.load_inputs(args.base_dir, args.methods_dir)
    top_cache = {
        split: bench.load_method_split(cache_dir, split)
        for split in ("train", "test")
    }
    device = torch.device("cuda:0")
    model, pad_token_id = bench.load_model(args.checkpoint, device)

    # Timing on held-out test, matching canonical Geneformer-PBMC timing scope.
    timing = bench.measure_embedding_timing(
        model=model, cache=top_cache["test"], token_ids=inputs["token_ids"],
        pad_token_id=pad_token_id, device=device, batch_size=args.batch_size,
        warmup_batches=args.warmup_batches, timing_repeats=args.timing_repeats,
        batching="bucketed",
    )
    expected_total = int(np.sum(top_cache["test"]["lengths"], dtype=np.int64))
    selection_times = []
    for r in range(args.selection_timing_repeats):
        t0 = time.perf_counter()
        total = expression_selection_once(inputs["base_splits"]["test"], inputs["gene_median"], matched_k)
        elapsed = time.perf_counter() - t0
        if total != expected_total:
            raise RuntimeError(f"Selection total mismatch: {total} != {expected_total}")
        selection_times.append(float(elapsed))
        print(f"[selection] {r+1}/{args.selection_timing_repeats}: {elapsed:.3f}s", flush=True)
    selection_mean = float(np.mean(selection_times))

    train_path = args.output_dir / "train_embeddings.npy"
    test_path = args.output_dir / "test_embeddings.npy"
    for split, path in (("train", train_path), ("test", test_path)):
        bench.extract_embeddings_to_file(
            model=model, cache=top_cache[split], token_ids=inputs["token_ids"],
            pad_token_id=pad_token_id, device=device, batch_size=args.batch_size,
            output_path=path, resume=args.resume,
        )
    del model
    gc.collect(); torch.cuda.empty_cache()

    train_labels_text = np.asarray(inputs["base_splits"]["train"]["labels_l1"]).astype(str)
    test_labels_text = np.asarray(inputs["base_splits"]["test"]["labels_l1"]).astype(str)
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_labels_text)
    test_labels = encoder.transform(test_labels_text)
    pred = bench.predict_exact_knn_cosine(
        train_embeddings_path=train_path, test_embeddings_path=test_path,
        train_labels=train_labels, device=device,
        query_batch_size=args.knn_query_batch_size, k=bench.CLASSIFIER_K,
        n_classes=len(encoder.classes_),
    )
    metrics = bench.quality_metrics(test_labels, pred)
    pred_text = encoder.inverse_transform(pred)

    cosine_mean = cosine_median = cosine_min = recall = float("nan")
    native_test = args.canonical_output / "geneformer_native_bucketed" / "test_embeddings.npy"
    native_neighbors = args.canonical_output / "native_reference_neighbors10.npy"
    if native_test.is_file():
        cos = bench.rowwise_embedding_cosine(native_test, test_path)
        cosine_mean = float(np.mean(cos)); cosine_median = float(np.median(cos)); cosine_min = float(np.min(cos))
        if native_neighbors.is_file():
            ref_n = np.load(native_neighbors, allow_pickle=False)
            method_n = bench.exact_neighbors_cosine(
                test_path, device, args.neighbor_query_batch_size, bench.NEIGHBOR_K
            )
            recall = bench.neighbor_recall(ref_n, method_n)

    per_class = bench.per_class_metrics(
        y_true=test_labels, y_pred=pred, label_encoder=encoder,
        variant=bench.Variant(
            variant_id="matched_fixed_expression", method="matched_fixed_expression",
            seed=None, cache_dir=cache_dir, batching="bucketed", sampling="expression",
        ),
    ) if "matched_fixed_expression" in bench.METHOD_LABEL else None

    # per_class_metrics indexes METHOD_LABEL; create our own compact per-class frame instead.
    from sklearn.metrics import f1_score
    rows = []
    for cid, cname in enumerate(encoder.classes_):
        yt = (test_labels == cid).astype(np.int64)
        yp = (pred == cid).astype(np.int64)
        rows.append({
            "method": "matched_fixed_expression", "method_label": "Fixed Top-Expression",
            "cell_type": str(cname), "support": int(yt.sum()),
            "f1": float(f1_score(yt, yp, zero_division=0)),
        })
    pd.DataFrame(rows).to_csv(args.output_dir / "per_class_f1.csv", index=False)

    bench.save_quality_npz(
        args.output_dir / "quality_seed0.npz", inputs, pred_text, test_path
    )

    embedding_times = np.asarray(timing["embedding_times_s"], dtype=float)
    e2e = selection_mean + float(timing["bucketing_time_s"]) + embedding_times
    throughput = bench.EXPECTED_TEST_CELLS / e2e
    summary = {
        "method": "matched_fixed_expression",
        "method_label": "Fixed Top-Expression",
        "fixed_k": matched_k,
        "batching": "bucketed",
        "sampling": "expression",
        "selection_time_s": selection_mean,
        "selection_time_std_s": float(np.std(selection_times, ddof=0)),
        "bucketing_time_s": float(timing["bucketing_time_s"]),
        "train_mean_selected_genes": float(np.mean(top_cache["train"]["lengths"])),
        "test_mean_selected_genes": float(np.mean(top_cache["test"]["lengths"])),
        "cells_per_s_mean": float(np.mean(throughput)),
        "cells_per_s_std": float(np.std(throughput, ddof=0)),
        "embedding_only_cells_per_s_mean": float(bench.EXPECTED_TEST_CELLS / np.mean(embedding_times)),
        "peak_gpu_memory_gb": float(timing["peak_gpu_memory_gb"]),
        **metrics,
        "embedding_cosine_mean": cosine_mean,
        "embedding_cosine_median": cosine_median,
        "embedding_cosine_min": cosine_min,
        "neighbor_recall_10": recall,
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame({
        "repeat": np.arange(len(embedding_times)),
        "selection_time_s": selection_mean,
        "bucketing_time_s": float(timing["bucketing_time_s"]),
        "embedding_time_s": embedding_times,
        "end_to_end_time_s": e2e,
        "cells_per_s": throughput,
        "peak_gpu_memory_gb": float(timing["peak_gpu_memory_gb"]),
    }).to_csv(args.output_dir / "timing_runs.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "matched_fixed_k": matched_k,
        "selection_rule": "top raw expression among expressed supported genes",
        "final_sequence_order": "official Geneformer median-scaled expression rank",
        "budget_source": str((args.methods_dir / "matched_budget.json").resolve()),
        "query_statistics_used_for_K": False,
    }, indent=2), encoding="utf-8")

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
