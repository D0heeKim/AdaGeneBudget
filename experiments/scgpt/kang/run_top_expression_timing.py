#!/usr/bin/env python3
"""Table-1-matched timing for the scGPT-Kang Fixed Top-Expression baseline.

Matches the canonical scGPT-Kang compressed-method timing protocol used by
11_scgpt_kang_main.py:
- scope: train/reference + held-out test/query
- batch size 64
- DataLoader workers 4
- warm-up 3 batches per split
- 5 timing repeats
- AMP FP16
- CSR row access
- length-sorted batching
- E2E = selection + bucketing + embedding

Quality is NOT recomputed. Existing top-expression quality outputs are left untouched.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
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
    p.add_argument("--main-output", type=Path,
                   default=Path("outputs/scgpt/kang_patient1015/main_csr"))
    p.add_argument("--fixed-script", type=Path,
                   default=Path("experiments/scgpt/_shared/fixed_selection.py"))
    p.add_argument("--corrected-script", type=Path,
                   default=Path("experiments/scgpt/_shared/runtime_utils.py"))
    p.add_argument("--repo", type=Path, default=Path("scGPT"))
    p.add_argument("--model-dir", type=Path, default=Path("checkpoints/scgpt"))
    p.add_argument("--train", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/train.h5ad"))
    p.add_argument("--test", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/test.h5ad"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/scgpt_kang_timing_table1"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    frozen = (args.batch_size, args.num_workers, args.warmup_steps, args.timing_repeats)
    if frozen != (64, 4, 3, 5):
        raise RuntimeError(
            "Table-1 protocol is frozen at batch_size=64, num_workers=4, "
            "warmup_steps=3, timing_repeats=5."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    required = [
        args.main_output / "config.json",
        args.fixed_script,
        args.corrected_script,
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.train,
        args.test,
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

    fixed = load_module("scgpt_fixed_top_expr_timing", args.fixed_script)
    corrected = load_module("scgpt_corrected_top_expr_timing", args.corrected_script)

    config = json.loads((args.main_output / "config.json").read_text(encoding="utf-8"))
    fixed_k = int(config["matched_fixed_k"])

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)
    if not np.array_equal(train.var_names.astype(str), test.var_names.astype(str)):
        raise RuntimeError("Train/test gene ordering differs")

    sys.path.insert(0, str(args.repo))
    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    vocab = GeneVocab.from_file(args.model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)

    model_config = json.loads((args.model_dir / "args.json").read_text(encoding="utf-8"))
    pad_token = model_config["pad_token"]
    pad_id = vocab[pad_token]
    pad_value = float(model_config["pad_value"])
    vocab.set_default_index(pad_id)

    genes = np.asarray(train.var_names.astype(str))
    matched_mask = np.asarray([gene in vocab for gene in genes])
    matched_genes = genes[matched_mask]
    gene_ids = np.asarray(vocab(list(matched_genes)), dtype=np.int64)

    train_matrix = train.X[:, matched_mask]
    test_matrix = test.X[:, matched_mask]
    if sparse.issparse(train_matrix):
        train_matrix = train_matrix.tocsr(copy=True)
        train_matrix.sum_duplicates(); train_matrix.eliminate_zeros(); train_matrix.sort_indices()
    else:
        train_matrix = np.asarray(train_matrix)
    if sparse.issparse(test_matrix):
        test_matrix = test_matrix.tocsr(copy=True)
        test_matrix.sum_duplicates(); test_matrix.eliminate_zeros(); test_matrix.sort_indices()
    else:
        test_matrix = np.asarray(test_matrix)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_config["embsize"],
        nhead=model_config["nheads"],
        d_hid=model_config["d_hid"],
        nlayers=model_config["nlayers"],
        nlayers_cls=model_config["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_config["dropout"],
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    checkpoint = fixed.load_checkpoint(args.model_dir / "best_model.pt")
    load_pretrained(model, checkpoint, strict=False, verbose=False)
    device = torch.device("cuda:0")
    model.to(device); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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
    selection_time_s = float(expr_train.selection_time_s + expr_test.selection_time_s)

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
    train_loader, train_bucket = corrected.build_bucketed_loader(
        expr_train, collator, args.batch_size, args.num_workers
    )
    test_loader, test_bucket = corrected.build_bucketed_loader(
        expr_test, collator, args.batch_size, args.num_workers
    )
    bucketing_time_s = float(train_bucket + test_bucket)

    corrected.warmup(model, train_loader, pad_id, device, args.warmup_steps)
    corrected.warmup(model, test_loader, pad_id, device, args.warmup_steps)

    total_cells = len(expr_train) + len(expr_test)
    rows = []
    for repeat in range(args.timing_repeats):
        train_emb, train_t = corrected.extract_once(
            model, train_loader, expr_train, pad_id, device, model_config["embsize"]
        )
        test_emb, test_t = corrected.extract_once(
            model, test_loader, expr_test, pad_id, device, model_config["embsize"]
        )
        embedding_time_s = float(train_t["embedding_time_s"] + test_t["embedding_time_s"])
        e2e = float(selection_time_s + bucketing_time_s + embedding_time_s)
        peak = float(max(train_t["peak_gpu_memory_gb"], test_t["peak_gpu_memory_gb"]))
        actual = int(train_t["actual_sequence_tokens"] + test_t["actual_sequence_tokens"])
        padded = int(train_t["padded_tokens_processed"] + test_t["padded_tokens_processed"])
        row = {
            "method": "matched_fixed_expression",
            "method_label": "Fixed Top-Expression",
            "repeat": repeat,
            "timing_scope": "train_plus_test",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
            "embedding_time_s": embedding_time_s,
            "end_to_end_time_s": e2e,
            "cells_per_s": float(total_cells / e2e),
            "peak_gpu_memory_gb": peak,
            "actual_sequence_tokens": actual,
            "padded_tokens_processed": padded,
            "padding_overhead_ratio": float(padded / actual),
        }
        rows.append(row)
        print(
            f"repeat={repeat+1}/{args.timing_repeats}: "
            f"{row['cells_per_s']:.2f} cells/s, peak={peak:.3f} GB",
            flush=True,
        )
        del train_emb, test_emb
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
        "bucketing_time_s": bucketing_time_s,
        "embedding_time_mean_s": float(frame["embedding_time_s"].mean()),
        "embedding_time_std_s": float(frame["embedding_time_s"].std(ddof=1)),
        "end_to_end_time_mean_s": float(frame["end_to_end_time_s"].mean()),
        "end_to_end_time_std_s": float(frame["end_to_end_time_s"].std(ddof=1)),
        "cells_per_s_mean": float(frame["cells_per_s"].mean()),
        "cells_per_s_std": float(frame["cells_per_s"].std(ddof=1)),
        "peak_gpu_memory_gb": float(frame["peak_gpu_memory_gb"].max()),
        "padding_overhead_ratio_mean": float(frame["padding_overhead_ratio"].mean()),
    }
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary_timing_only.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "status": "PASS",
        "fixed_k": fixed_k,
        "protocol": "scGPT-Kang Table 1 protocol",
        "timing_scope": "train_plus_test",
        "batch_size": 64,
        "num_workers": 4,
        "warmup_steps": 3,
        "timing_repeats": 5,
        "matrix_format": "CSR",
        "batching": "length_sorted",
        "end_to_end": "selection + bucketing + embedding",
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
