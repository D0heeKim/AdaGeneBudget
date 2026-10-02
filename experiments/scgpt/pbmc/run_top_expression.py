#!/usr/bin/env python3
"""Fixed-budget Top-Expression baseline for scGPT-PBMC.

The implementation deliberately reuses the canonical PBMC pipeline and the
existing scGPT FixedSelectionDataset(method="expression"). The fixed K and
reference-cell subset are read from the canonical scGPT-PBMC run, so no query
statistics are used to choose the budget.

Outputs are written under outputs/release_r1 and never modify canonical runs.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse


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
                   default=Path("experiments/scgpt/pbmc/run_reference_pipeline.py"))
    p.add_argument("--fixed-script", type=Path,
                   default=Path("experiments/scgpt/_shared/fixed_selection.py"))
    p.add_argument("--corrected-script", type=Path,
                   default=Path("experiments/scgpt/_shared/runtime_utils.py"))
    p.add_argument("--repo", type=Path, default=Path("scGPT"))
    p.add_argument("--model-dir", type=Path, default=Path("checkpoints/scgpt"))
    p.add_argument("--source", type=Path,
                   default=Path("data/raw/pbmc_seurat_v4/pbmc_seurat_v4_rna_only.h5ad"))
    p.add_argument("--split", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/split_indices.npz"))
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/scgpt_cache"))
    p.add_argument("--main-output", type=Path,
                   default=Path("outputs/scgpt/pbmc_p5/main"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/scgpt_pbmc"))
    p.add_argument("--label-col", default="celltype.l1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--similarity-chunk", type=int, default=512)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    for path in (args.main_script, args.fixed_script, args.corrected_script,
                 args.repo, args.model_dir / "args.json", args.model_dir / "vocab.json",
                 args.model_dir / "best_model.pt", args.source, args.split,
                 args.main_output / "config.json",
                 args.main_output / "reference_train_indices.npy"):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.output_dir.exists() and (args.output_dir / "summary.csv").exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_mod = load_module("scgpt_pbmc_main_for_top_expression", args.main_script)
    fixed_mod = load_module("scgpt_fixed_for_top_expression", args.fixed_script)
    corrected = load_module("scgpt_corrected_for_top_expression", args.corrected_script)

    config = json.loads((args.main_output / "config.json").read_text(encoding="utf-8"))
    fixed_k = int(config["fixed_k"])
    print(f"Resolved scGPT-PBMC matched fixed K = {fixed_k}", flush=True)

    sys.path.insert(0, str(args.repo))
    from scgpt.data_collator import DataCollator
    from scgpt.tokenizer import GeneVocab

    split = np.load(args.split)
    train_idx = split["train_indices"].astype(np.int64)
    test_idx = split["test_indices"].astype(np.int64)

    source = ad.read_h5ad(args.source, backed="r")
    if args.label_col not in source.obs:
        raise KeyError(f"Missing obs column: {args.label_col}")
    train_labels = source.obs.iloc[train_idx][args.label_col].astype(str).to_numpy()
    test_labels = source.obs.iloc[test_idx][args.label_col].astype(str).to_numpy()

    vocab_match = GeneVocab.from_file(args.model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab_match:
            vocab_match.append_token(token)
    genes = np.asarray(source.var_names.astype(str))
    matched_mask = np.asarray([g in vocab_match for g in genes])
    matched_genes = genes[matched_mask]

    train_cache = args.cache_dir / "train_matched_csr.npz"
    test_cache = args.cache_dir / "test_matched_csr.npz"
    if not train_cache.exists() or not test_cache.exists():
        raise FileNotFoundError(
            "Canonical PBMC CSR cache is missing. Run 16_scgpt_pbmc_p5_all_methods.py first."
        )
    train_matrix = sparse.load_npz(train_cache).tocsr()
    test_matrix = sparse.load_npz(test_cache).tocsr()
    source.file.close()

    # load_model needs an argparse-like object with repo/model_dir.
    class A: pass
    load_args = A()
    load_args.repo = args.repo
    load_args.model_dir = args.model_dir
    model, vocab, cfg, pad_id, pad_value, device = main_mod.load_model(load_args, fixed_mod)
    gene_ids = np.asarray(vocab(list(matched_genes)), dtype=np.int64)

    seed_all(0)
    expr_train = fixed_mod.FixedSelectionDataset(
        matrix=train_matrix, gene_ids=gene_ids, cls_id=vocab["<cls>"],
        cls_value=pad_value, method="expression", budget=fixed_k, seed=0, idf=None,
    )
    expr_test = fixed_mod.FixedSelectionDataset(
        matrix=test_matrix, gene_ids=gene_ids, cls_id=vocab["<cls>"],
        cls_value=pad_value, method="expression", budget=fixed_k, seed=0, idf=None,
    )
    ref_idx = np.load(args.main_output / "reference_train_indices.npy").astype(np.int64)
    ref_labels = train_labels[ref_idx]
    expr_train_ref = main_mod.RemappedSubsetDataset(expr_train, ref_idx)

    collator = DataCollator(
        do_padding=True, pad_token_id=pad_id, pad_value=pad_value,
        do_mlm=False, do_binning=True, max_length=fixed_k + 1,
        sampling=False, keep_first_n_tokens=1,
    )

    tr_loader, _ = main_mod.make_loader(
        expr_train_ref, collator, args.batch_size, args.num_workers,
        seed=0, batching="length_sorted", effective_max_length=fixed_k + 1,
    )
    te_loader, _ = main_mod.make_loader(
        expr_test, collator, args.batch_size, args.num_workers,
        seed=100000, batching="length_sorted", effective_max_length=fixed_k + 1,
    )
    tr_emb, _ = corrected.extract_once(
        model, tr_loader, expr_train_ref, pad_id, device, cfg["embsize"]
    )
    te_emb, _ = corrected.extract_once(
        model, te_loader, expr_test, pad_id, device, cfg["embsize"]
    )
    pred = main_mod.exact_cosine_knn_predict(
        tr_emb, ref_labels, te_emb, args.knn_k, device, args.similarity_chunk
    )
    metrics = main_mod.classification_metrics(test_labels, pred)

    fidelity = {
        "embedding_cosine_mean": float("nan"),
        "embedding_cosine_median": float("nan"),
        "embedding_cosine_min": float("nan"),
        "neighbor_recall_10": float("nan"),
    }
    full_npz = args.main_output / "full_bucketed" / "quality_seed0.npz"
    full_neighbors_path = args.main_output / "full_bucketed" / "full_neighbors_10.npy"
    if full_npz.exists() and full_neighbors_path.exists():
        full = np.load(full_npz, allow_pickle=True)
        full_test_emb = full["test_embeddings"].astype(np.float32)
        full_neighbors = np.load(full_neighbors_path)
        fidelity = main_mod.fidelity_metrics(
            full_test_emb, te_emb, full_neighbors, 10, device, args.similarity_chunk
        )

    # Timing uses the complete train+test selection, matching the original scGPT-PBMC main script.
    tr_time_loader, tr_bucket = main_mod.make_loader(
        expr_train, collator, args.batch_size, args.num_workers,
        seed=999001, batching="length_sorted", effective_max_length=fixed_k + 1,
    )
    te_time_loader, te_bucket = main_mod.make_loader(
        expr_test, collator, args.batch_size, args.num_workers,
        seed=999002, batching="length_sorted", effective_max_length=fixed_k + 1,
    )
    corrected.warmup(model, tr_time_loader, pad_id, device, args.warmup_steps)
    corrected.warmup(model, te_time_loader, pad_id, device, args.warmup_steps)
    selection_time = float(expr_train.selection_time_s + expr_test.selection_time_s)
    bucket_time = float(tr_bucket + te_bucket)
    timing_rows = []
    total_cells = len(expr_train) + len(expr_test)
    for repeat in range(args.timing_repeats):
        _, tr_t = corrected.extract_once(
            model, tr_time_loader, expr_train, pad_id, device, cfg["embsize"]
        )
        _, te_t = corrected.extract_once(
            model, te_time_loader, expr_test, pad_id, device, cfg["embsize"]
        )
        emb_time = float(tr_t["embedding_time_s"] + te_t["embedding_time_s"])
        e2e = selection_time + bucket_time + emb_time
        timing_rows.append({
            "repeat": repeat,
            "selection_time_s": selection_time,
            "bucketing_time_s": bucket_time,
            "embedding_time_s": emb_time,
            "end_to_end_time_s": e2e,
            "cells_per_s": total_cells / e2e,
            "peak_gpu_memory_gb": max(tr_t["peak_gpu_memory_gb"], te_t["peak_gpu_memory_gb"]),
        })
        print(f"[timing] {repeat+1}/{args.timing_repeats}: {total_cells/e2e:.2f} cells/s", flush=True)

    timing = pd.DataFrame(timing_rows)
    summary = {
        "method": "matched_fixed_expression",
        "method_label": "Fixed Top-Expression",
        "fixed_k": fixed_k,
        "train_mean_selected_genes": float(np.mean(expr_train.gene_lengths)),
        "test_mean_selected_genes": float(np.mean(expr_test.gene_lengths)),
        "selection_time_s": selection_time,
        "bucketing_time_s": bucket_time,
        "cells_per_s_mean": float(timing["cells_per_s"].mean()),
        "cells_per_s_std": float(timing["cells_per_s"].std(ddof=1)) if len(timing) > 1 else 0.0,
        "peak_gpu_memory_gb": float(timing["peak_gpu_memory_gb"].max()),
        **metrics,
        **fidelity,
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.csv", index=False)
    timing.to_csv(args.output_dir / "timing_repeats.csv", index=False)
    np.savez_compressed(
        args.output_dir / "quality_seed0.npz",
        train_reference_indices=ref_idx,
        train_reference_labels=ref_labels,
        test_labels=test_labels,
        test_predictions=pred,
        test_embeddings=te_emb.astype(np.float32),
    )
    (args.output_dir / "config.json").write_text(json.dumps({
        "fixed_k": fixed_k,
        "main_output": str(args.main_output),
        "selection_rule": "top raw expression among expressed supported genes",
        "budget_source": "matched fixed K from canonical scGPT-PBMC run",
        "query_statistics_used_for_K": False,
    }, indent=2), encoding="utf-8")

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    del model, tr_emb, te_emb
    gc.collect(); torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
