#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def resolve_from_config(root: Path, x: str | Path) -> Path:
    p = Path(x)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main-script", type=Path,
                   default=Path("experiments/scgpt/pbmc/run_reference_pipeline.py"))
    p.add_argument("--main-output", type=Path,
                   default=Path("outputs/scgpt/pbmc_p5/main"))
    p.add_argument("--outdir", type=Path,
                   default=Path("outputs/scgpt/pbmc_p5/adagenebudget_corrected"))
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--timing-repeats", type=int, default=None)
    p.add_argument("--similarity-chunk", type=int, default=None)
    p.add_argument("--overwrite-cache", action="store_true")
    return p.parse_args()


def full_stable_native_cache(matrix, idf, cache_dir, tau, k_min, k_max, rebuild=False):
    """Canonical candidate for scGPT:
      1) full stable TF-IDF ranking over all expressed supported genes
      2) shortest prefix reaching tau mass, clipped to [Kmin,Kmax]
      3) restore selected genes to original/native CSR gene-position order
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    required = [cache_dir / "indptr.npy", cache_dir / "gene_pos.npy",
                cache_dir / "values.npy", meta_path]
    if all(p.exists() for p in required) and not rebuild:
        return json.loads(meta_path.read_text())

    t0 = time.perf_counter()
    pos_parts, val_parts = [], []
    lengths = np.empty(matrix.shape[0], dtype=np.int64)
    retained_sum = 0.0
    upper = lower = 0

    for row in range(matrix.shape[0]):
        s, e = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
        pos = matrix.indices[s:e].astype(np.int64, copy=False)
        vals = matrix.data[s:e].astype(np.float32, copy=False)
        n = pos.size

        if n == 0:
            chosen_native = np.empty(0, dtype=np.int64)
            retained = 1.0
        else:
            scores = vals * idf[pos]
            total = float(scores.sum(dtype=np.float64))
            # Full deterministic ranking; stable ties follow native/input gene position.
            order = np.argsort(-scores, kind="stable")
            if total <= 0:
                raw_k = min(n, k_max)
            else:
                raw_k = int(np.searchsorted(
                    np.cumsum(scores[order], dtype=np.float64),
                    tau * total,
                    side="left",
                ) + 1)
            if n >= k_min and raw_k < k_min:
                lower += 1
            if raw_k > k_max:
                upper += 1
            k = min(n, k_max, max(k_min, raw_k))
            chosen = order[:k]
            retained = float(scores[chosen].sum(dtype=np.float64) / max(total, 1e-12))
            # Restore backbone/native input ordering after selecting the subset.
            chosen_native = chosen[np.argsort(pos[chosen], kind="stable")]

        lengths[row] = chosen_native.size
        pos_parts.append(pos[chosen_native].astype(np.int32, copy=False))
        val_parts.append(vals[chosen_native].astype(np.float32, copy=False))
        retained_sum += retained

        if (row + 1) % 10000 == 0 or row + 1 == matrix.shape[0]:
            print(f"[selection] {cache_dir.name}: {row+1:,}/{matrix.shape[0]:,}", flush=True)

    indptr = np.empty(matrix.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    gene_pos = np.concatenate(pos_parts) if pos_parts else np.empty(0, dtype=np.int32)
    values = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.float32)
    np.save(cache_dir / "indptr.npy", indptr)
    np.save(cache_dir / "gene_pos.npy", gene_pos)
    np.save(cache_dir / "values.npy", values)

    meta = {
        "method": "adaptive_full_stable_native_order",
        "tau": float(tau),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "rows": int(matrix.shape[0]),
        "mean_selected_genes": float(lengths.mean()),
        "median_selected_genes": float(np.median(lengths)),
        "selection_time_s": float(time.perf_counter() - t0),
        "retained_mass_mean": float(retained_sum / matrix.shape[0]),
        "upper_clipped_fraction": float(upper / matrix.shape[0]),
        "lower_clipped_fraction": float(lower / matrix.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def compare_caches(old_dir: Path, new_dir: Path):
    old_indptr = np.load(old_dir / "indptr.npy", mmap_mode="r")
    new_indptr = np.load(new_dir / "indptr.npy", mmap_mode="r")
    old_pos = np.load(old_dir / "gene_pos.npy", mmap_mode="r")
    new_pos = np.load(new_dir / "gene_pos.npy", mmap_mode="r")
    if not np.array_equal(old_indptr, new_indptr):
        raise RuntimeError("K/indptr changed unexpectedly; audit said K should be identical.")
    set_equal = 0
    exact_equal = 0
    n = len(old_indptr) - 1
    for i in range(n):
        a, b = int(old_indptr[i]), int(old_indptr[i+1])
        x = np.asarray(old_pos[a:b], dtype=np.int64)
        y = np.asarray(new_pos[a:b], dtype=np.int64)
        exact_equal += int(np.array_equal(x, y))
        set_equal += int(np.array_equal(np.sort(x), np.sort(y)))
    return {
        "cells": n,
        "exact_order_equal_fraction": exact_equal / n,
        "selected_set_equal_fraction": set_equal / n,
    }


def main():
    a = parse_args()
    root = Path.cwd()
    if not a.main_script.exists():
        raise FileNotFoundError(a.main_script)
    cfg_path = a.main_output / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text())

    mainmod = load_module("pbmc16_for_validation", a.main_script)

    source_path = resolve_from_config(root, cfg["source"])
    split_path = resolve_from_config(root, cfg["split"])
    cache_dir = resolve_from_config(root, cfg["cache_dir"])
    repo = resolve_from_config(root, cfg["repo"])
    model_dir = resolve_from_config(root, cfg["model_dir"])
    fixed_script = resolve_from_config(root, cfg["fixed_script"])
    corrected_script = resolve_from_config(root, cfg["corrected_script"])

    batch_size = a.batch_size or int(cfg.get("batch_size", 64))
    num_workers = a.num_workers if a.num_workers is not None else int(cfg.get("num_workers", 4))
    warmup_steps = a.warmup_steps if a.warmup_steps is not None else int(cfg.get("warmup_steps", 3))
    timing_repeats = a.timing_repeats if a.timing_repeats is not None else int(cfg.get("timing_repeats", 5))
    similarity_chunk = a.similarity_chunk or int(cfg.get("similarity_chunk", 512))
    tau = float(cfg.get("tau", 0.9))
    k_min = int(cfg.get("k_min", 128))
    k_max = int(cfg.get("k_max", 600))
    label_col = cfg.get("label_col", "celltype.l1")
    knn_k = int(cfg.get("knn_k", 5))

    for p in [source_path, split_path, cache_dir / "train_matched_csr.npz",
              cache_dir / "test_matched_csr.npz", cache_dir / "idf.npy",
              a.main_output / "matched_genes.npy", a.main_output / "reference_train_indices.npy",
              fixed_script, corrected_script, model_dir / "best_model.pt"]:
        if not p.exists():
            raise FileNotFoundError(p)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    a.outdir.mkdir(parents=True, exist_ok=True)
    corrected_cache = a.outdir / "selection_cache"
    tr_new_dir = corrected_cache / "adaptive_train"
    te_new_dir = corrected_cache / "adaptive_test"

    train_matrix = sparse.load_npz(cache_dir / "train_matched_csr.npz").tocsr()
    test_matrix = sparse.load_npz(cache_dir / "test_matched_csr.npz").tocsr()
    train_matrix.sort_indices(); test_matrix.sort_indices()
    idf = np.load(cache_dir / "idf.npy")

    print("[1/5] Building corrected selection caches", flush=True)
    tr_meta = full_stable_native_cache(train_matrix, idf, tr_new_dir, tau, k_min, k_max, a.overwrite_cache)
    te_meta = full_stable_native_cache(test_matrix, idf, te_new_dir, tau, k_min, k_max, a.overwrite_cache)

    old_tr = cache_dir / "adaptive_train"
    old_te = cache_dir / "adaptive_test"
    comparison = {
        "train": compare_caches(old_tr, tr_new_dir),
        "test": compare_caches(old_te, te_new_dir),
    }
    print("Cache comparison:", json.dumps(comparison, indent=2), flush=True)

    print("[2/5] Loading labels/model", flush=True)
    split = np.load(split_path)
    train_idx = split["train_indices"].astype(np.int64)
    test_idx = split["test_indices"].astype(np.int64)
    src = ad.read_h5ad(source_path, backed="r")
    train_labels = src.obs.iloc[train_idx][label_col].astype(str).to_numpy()
    test_labels = src.obs.iloc[test_idx][label_col].astype(str).to_numpy()
    src.file.close()
    ref_idx = np.load(a.main_output / "reference_train_indices.npy").astype(np.int64)
    ref_labels = train_labels[ref_idx]
    matched_genes = np.load(a.main_output / "matched_genes.npy", allow_pickle=True).astype(str)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scgpt.data_collator import DataCollator

    fixedmod = load_module("pbmc_fixed_for_validation", fixed_script)
    corrected = load_module("pbmc_corrected_for_validation", corrected_script)
    ns = SimpleNamespace(repo=repo, model_dir=model_dir)
    model, vocab, model_cfg, pad_id, pad_value, device = mainmod.load_model(ns, fixedmod)
    gene_ids = np.asarray(vocab(list(matched_genes)), dtype=np.int64)

    train_ds = mainmod.PreselectedTokenDataset(tr_new_dir, gene_ids, vocab["<cls>"], pad_value)
    test_ds = mainmod.PreselectedTokenDataset(te_new_dir, gene_ids, vocab["<cls>"], pad_value)
    train_ref = mainmod.RemappedSubsetDataset(train_ds, ref_idx)
    collator = DataCollator(
        do_padding=True, pad_token_id=pad_id, pad_value=pad_value,
        do_mlm=False, do_binning=True, max_length=k_max + 1,
        sampling=False, keep_first_n_tokens=1,
    )

    print("[3/5] Corrected Ada quality", flush=True)
    mainmod.seed_all(0)
    tr_loader, _ = mainmod.make_loader(train_ref, collator, batch_size, num_workers, 0,
                                       "length_sorted", k_max + 1)
    te_loader, _ = mainmod.make_loader(test_ds, collator, batch_size, num_workers, 100000,
                                       "length_sorted", k_max + 1)
    tr_emb, _ = corrected.extract_once(model, tr_loader, train_ref, pad_id, device, model_cfg["embsize"])
    te_emb, _ = corrected.extract_once(model, te_loader, test_ds, pad_id, device, model_cfg["embsize"])
    pred = mainmod.exact_cosine_knn_predict(tr_emb, ref_labels, te_emb, knn_k, device, similarity_chunk)
    metrics = mainmod.classification_metrics(test_labels, pred)
    print("Corrected metrics:", metrics, flush=True)

    old_npz = np.load(a.main_output / "adaptive" / "quality_seed0.npz", allow_pickle=True)
    old_pred = old_npz["test_predictions"].astype(str)
    old_metrics = mainmod.classification_metrics(test_labels, old_pred)
    prediction_agreement = float(np.mean(pred.astype(str) == old_pred))
    print("Old metrics:", old_metrics, flush=True)
    print(f"Prediction agreement old vs corrected: {prediction_agreement:.8f}", flush=True)

    np.savez_compressed(
        a.outdir / "quality_seed0.npz",
        train_reference_indices=ref_idx,
        train_reference_labels=ref_labels,
        test_labels=test_labels,
        test_predictions=pred,
        test_embeddings=te_emb.astype(np.float32),
    )
    pd.DataFrame([{
        "version": "old_pbmc",
        **old_metrics,
    }, {
        "version": "corrected_full_stable_native_order",
        **metrics,
    }]).to_csv(a.outdir / "quality_comparison.csv", index=False)

    del tr_loader, te_loader, tr_emb, te_emb
    gc.collect(); torch.cuda.empty_cache()

    print("[4/5] Corrected Ada train+test timing", flush=True)
    mainmod.seed_all(0)
    tr_time_loader, tr_bucket = mainmod.make_loader(train_ds, collator, batch_size, num_workers,
                                                    999001, "length_sorted", k_max + 1)
    te_time_loader, te_bucket = mainmod.make_loader(test_ds, collator, batch_size, num_workers,
                                                    999002, "length_sorted", k_max + 1)
    bucket_time = tr_bucket + te_bucket
    corrected.warmup(model, tr_time_loader, pad_id, device, warmup_steps)
    corrected.warmup(model, te_time_loader, pad_id, device, warmup_steps)
    selection_time = float(tr_meta["selection_time_s"] + te_meta["selection_time_s"])
    total_cells = len(train_ds) + len(test_ds)
    timing_rows = []
    for r in range(timing_repeats):
        tr_e, tr_t = corrected.extract_once(model, tr_time_loader, train_ds, pad_id, device, model_cfg["embsize"])
        te_e, te_t = corrected.extract_once(model, te_time_loader, test_ds, pad_id, device, model_cfg["embsize"])
        emb_time = float(tr_t["embedding_time_s"] + te_t["embedding_time_s"])
        e2e = selection_time + bucket_time + emb_time
        row = {
            "repeat": r,
            "selection_time_s": selection_time,
            "bucketing_time_s": float(bucket_time),
            "embedding_time_s": emb_time,
            "end_to_end_time_s": e2e,
            "cells_per_s": total_cells / e2e,
            "embedding_only_cells_per_s": total_cells / emb_time,
            "peak_gpu_memory_gb": max(float(tr_t["peak_gpu_memory_gb"]), float(te_t["peak_gpu_memory_gb"])),
            "actual_sequence_tokens": int(tr_t["actual_sequence_tokens"] + te_t["actual_sequence_tokens"]),
            "padded_tokens_processed": int(tr_t["padded_tokens_processed"] + te_t["padded_tokens_processed"]),
        }
        row["padding_overhead_ratio"] = row["padded_tokens_processed"] / row["actual_sequence_tokens"]
        timing_rows.append(row)
        print(f"repeat {r+1}/{timing_repeats}: {row['cells_per_s']:.2f} cells/s", flush=True)
        del tr_e, te_e
        gc.collect(); torch.cuda.empty_cache()

    td = pd.DataFrame(timing_rows)
    td.to_csv(a.outdir / "timing_repeats.csv", index=False)
    old_summary = pd.read_csv(a.main_output / "summary.csv")
    old_ada = old_summary.loc[old_summary["method"].astype(str) == "adaptive"].iloc[0]
    summary = {
        "method": "adaptive_corrected_full_stable_native_order",
        "tau": tau,
        "k_min": k_min,
        "k_max": k_max,
        "train_mean_selected_genes": tr_meta["mean_selected_genes"],
        "test_mean_selected_genes": te_meta["mean_selected_genes"],
        "selection_time_s": selection_time,
        "bucketing_time_s": float(bucket_time),
        "embedding_time_mean_s": float(td["embedding_time_s"].mean()),
        "end_to_end_time_mean_s": float(td["end_to_end_time_s"].mean()),
        "end_to_end_time_std_s": float(td["end_to_end_time_s"].std(ddof=1)),
        "cells_per_s_mean": float(td["cells_per_s"].mean()),
        "cells_per_s_std": float(td["cells_per_s"].std(ddof=1)),
        "peak_gpu_memory_gb": float(td["peak_gpu_memory_gb"].max()),
        "padding_overhead_ratio_mean": float(td["padding_overhead_ratio"].mean()),
        **metrics,
        "prediction_agreement_with_old": prediction_agreement,
        "old_accuracy": float(old_ada["accuracy"]),
        "old_macro_f1": float(old_ada["macro_f1"]),
        "old_balanced_accuracy": float(old_ada["balanced_accuracy"]),
        "old_cells_per_s_mean": float(old_ada["cells_per_s_mean"]),
        "old_peak_gpu_memory_gb": float(old_ada["peak_gpu_memory_gb"]),
        "train_selected_set_equal_fraction": comparison["train"]["selected_set_equal_fraction"],
        "test_selected_set_equal_fraction": comparison["test"]["selected_set_equal_fraction"],
        "train_exact_order_equal_fraction": comparison["train"]["exact_order_equal_fraction"],
        "test_exact_order_equal_fraction": comparison["test"]["exact_order_equal_fraction"],
    }
    pd.DataFrame([summary]).to_csv(a.outdir / "summary.csv", index=False)
    (a.outdir / "run_meta.json").write_text(json.dumps({
        "status": "PASS",
        "purpose": "Validate only the scGPT-PBMC Ada row under full stable ranking + native-order restoration without overwriting canonical outputs.",
        "main_output": str(a.main_output),
        "corrected_cache": str(corrected_cache),
    }, indent=2))

    print("[5/5] FINAL STATUS: PASS")
    print(pd.DataFrame([summary]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
