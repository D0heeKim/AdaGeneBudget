#!/usr/bin/env python3
"""Build the Geneformer-PBMC matched fixed Top-Expression cache.

Selection: top min(K, n_expressed) genes by raw count/expression.
After subset selection, genes are reordered by the exact Geneformer native
rank (count / official gene median), matching the validated PBMC cache builder.
This script is CPU-only and writes only a new auxiliary cache directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("geneformer_pbmc_fixed_random", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-builder", type=Path,
                   default=Path("experiments/geneformer/pbmc/build_fixed_random_cache.py"))
    p.add_argument("--base-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base"))
    p.add_argument("--methods-dir", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/methods"))
    p.add_argument("--matched-budget-json", type=Path,
                   default=Path("data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/methods/matched_budget.json"))
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.canonical_builder.is_file():
        raise FileNotFoundError(args.canonical_builder)
    canonical = load_module(args.canonical_builder)
    base = canonical.load_base(args.base_dir)
    matched_k, matched_meta = canonical.load_matched_k(args.matched_budget_json)
    final_dir = args.methods_dir / f"matched_fixed_expression_k{matched_k}"
    building_dir = final_dir.with_name(final_dir.name + ".building")

    for p in (final_dir, building_dir):
        if p.exists():
            if not args.overwrite:
                raise FileExistsError(f"{p} exists; use --overwrite")
            shutil.rmtree(p)
    building_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 100)
    print("GENEFORMER-PBMC MATCHED FIXED TOP-EXPRESSION CACHE")
    print("=" * 100)
    print("Matched K:", matched_k)

    split_meta = {}
    started_all = time.perf_counter()
    try:
        for split_name, expected_cells in (
            ("train", canonical.EXPECTED_TRAIN_CELLS),
            ("test", canonical.EXPECTED_TEST_CELLS),
        ):
            split = base["splits"][split_name]
            base_indptr = split["indptr"]
            base_gene_pos = split["gene_pos"]
            base_counts = split["counts"]
            base_lengths = split["lengths"]
            selected_lengths = np.minimum(base_lengths, matched_k).astype(canonical.SELECTED_K_DTYPE)

            split_dir = building_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=False)
            out_indptr = canonical.write_indptr(selected_lengths, split_dir / "indptr.npy")
            np.save(split_dir / "selected_k.npy", selected_lengths, allow_pickle=False)
            out = np.lib.format.open_memmap(
                split_dir / "gene_pos.npy", mode="w+", dtype=canonical.GENE_POS_DTYPE,
                shape=(int(out_indptr[-1]),),
            )

            started = time.perf_counter()
            for row in range(len(base_lengths)):
                s, e = int(base_indptr[row]), int(base_indptr[row + 1])
                positions = np.asarray(base_gene_pos[s:e], dtype=np.int64)
                counts = np.asarray(base_counts[s:e], dtype=np.float64)
                k = int(selected_lengths[row])
                if k <= 0 or k > len(positions):
                    raise RuntimeError(f"{split_name} row {row}: invalid k={k}")

                # Primary: descending raw expression. Tie-break: ascending gene position.
                expression_order = np.lexsort((positions, -counts))
                selected_idx = expression_order[:k]
                selected_positions = positions[selected_idx]
                selected_counts = counts[selected_idx]

                # Preserve the validated Geneformer post-selection ordering.
                selected_positions = canonical.native_reorder(
                    selected_positions, selected_counts, base["gene_median"]
                )
                os_, oe_ = int(out_indptr[row]), int(out_indptr[row + 1])
                out[os_:oe_] = selected_positions.astype(canonical.GENE_POS_DTYPE, copy=False)

                done = row + 1
                if done == 1 or done == len(base_lengths) or done % args.progress_every == 0:
                    print(f"[{split_name}] {done:,}/{len(base_lengths):,} "
                          f"({100*done/len(base_lengths):.1f}%)", flush=True)
            out.flush(); del out

            canonical.validate_method_split(
                split_dir, expected_cells=expected_cells,
                expected_lengths=selected_lengths, matched_k=matched_k,
            )
            meta = {
                "status": "PASS",
                "method": "matched_fixed_expression",
                "split": split_name,
                "matched_fixed_k": int(matched_k),
                "cells": int(len(base_lengths)),
                "mean_selected_genes": float(np.mean(selected_lengths)),
                "median_selected_genes": float(np.median(selected_lengths)),
                "minimum_selected_genes": int(np.min(selected_lengths)),
                "maximum_selected_genes": int(np.max(selected_lengths)),
                "fraction_cells_with_fewer_than_k_expressed_genes": float(np.mean(base_lengths < matched_k)),
                "selection_rule": "top raw expression among expressed supported genes",
                "final_sequence_order": "Official Geneformer median-scaled expression rank after subset selection.",
                "selection_compute_time_s": float(time.perf_counter() - started),
            }
            (split_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            split_meta[split_name] = meta

        root_meta = {
            "status": "PASS",
            "method": "matched_fixed_expression",
            "method_name": f"matched_fixed_expression_k{matched_k}",
            "matched_fixed_k": int(matched_k),
            "matched_budget_json": str(args.matched_budget_json.resolve()),
            "matched_budget_fit_split": matched_meta.get("fit_split", "train_only"),
            "test_statistics_used_for_k": False,
            "base_cache": str(args.base_dir.resolve()),
            "selection_rule": "top raw expression among expressed supported genes",
            "final_sequence_order": "Official Geneformer median-scaled expression rank after subset selection.",
            "train": split_meta["train"],
            "test": split_meta["test"],
            "elapsed_s": float(time.perf_counter() - started_all),
        }
        (building_dir / "meta.json").write_text(json.dumps(root_meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(building_dir, final_dir)
    except Exception:
        print("Build failed; incomplete output remains in .building directory", flush=True)
        raise

    print("Output:", final_dir)
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
