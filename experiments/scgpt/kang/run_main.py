#!/usr/bin/env python3
"""Kang 2018 donor-held-out generalization experiment.

Methods:
1. scGPT native cell-embedding policy:
   max sequence length 1200, random sampling, sequential batching
2. Full input with length-sorted batching
3. Mean-token-matched fixed TF-IDF
4. AdaGeneBudget with frozen parameters from Pancreas

All methods use:
- frozen scGPT whole-human checkpoint
- PyTorch-native Transformer
- AMP FP16
- batch size 64
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("./scGPT"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "./checkpoints/scgpt"
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path(
            "./data/processed/"
            "kang_2018/patient_1015_holdout/train.h5ad"
        ),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path(
            "./data/processed/"
            "kang_2018/patient_1015_holdout/test.h5ad"
        ),
    )

    parser.add_argument(
        "--full-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/full_input.py"
        ),
    )
    parser.add_argument(
        "--fixed-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/fixed_selection.py"
        ),
    )
    parser.add_argument(
        "--adaptive-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/adaptive_selection.py"
        ),
    )
    parser.add_argument(
        "--corrected-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/runtime_utils.py"
        ),
    )

    parser.add_argument("--label-col", default="cell_type")
    parser.add_argument("--condition-col", default="label")
    parser.add_argument("--donor-col", default="replicate")

    parser.add_argument("--tau", type=float, default=0.90)
    parser.add_argument("--k-min", type=int, default=128)
    parser.add_argument("--k-max", type=int, default=600)

    parser.add_argument(
        "--native-max-length",
        type=int,
        default=1200,
        help=(
            "Official scGPT default total sequence length, "
            "including the first CLS token."
        ),
    )
    parser.add_argument(
        "--native-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "kang_patient1015/main_csr"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NativeCollatorWrapper:
    """Retain original cell IDs while using official DataCollator."""

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(
        self,
        examples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        original_ids = torch.stack(
            [example["id"] for example in examples]
        )

        batch = self.collator(examples)
        batch["original_id"] = original_ids

        return batch


def build_native_loader(
    dataset: Any,
    collator: Any,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": SequentialSampler(dataset),
        "collate_fn": NativeCollatorWrapper(collator),
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": generator,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
    }


def mean_and_sample_std(
    values: list[float],
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)

    mean = float(np.mean(array))
    std = (
        float(np.std(array, ddof=1))
        if len(array) > 1
        else 0.0
    )

    return mean, std


def run_native_policy(
    *,
    corrected: Any,
    dataset_train: Any,
    dataset_test: Any,
    model: torch.nn.Module,
    data_collator_class: Any,
    pad_id: int,
    pad_value: float,
    device: torch.device,
    embedding_dim: int,
    max_length: int,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    timing_repeats: int,
    seeds: list[int],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    full_test_embeddings: np.ndarray,
    reference_neighbors: np.ndarray,
    output_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print()
    print("=" * 80)
    print("scGPT NATIVE POLICY")
    print("=" * 80)
    print(f"Max total sequence length: {max_length}")
    print(f"Maximum sampled genes: {max_length - 1}")
    print(f"Quality seeds: {seeds}")
    print("Sampling: official DataCollator random sampling")
    print("Batching: sequential")

    collator = data_collator_class(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )

    quality_rows = []
    quality_values: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "weighted_f1": [],
        "balanced_accuracy": [],
        "embedding_cosine_mean": [],
        "embedding_cosine_median": [],
        "embedding_cosine_min": [],
        "neighbor_recall_10": [],
    }

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_loader = build_native_loader(
            dataset_train,
            collator,
            batch_size,
            num_workers,
            seed,
        )
        test_loader = build_native_loader(
            dataset_test,
            collator,
            batch_size,
            num_workers,
            seed + 100_000,
        )

        train_embeddings, _ = corrected.extract_once(
            model,
            train_loader,
            dataset_train,
            pad_id,
            device,
            embedding_dim,
        )
        test_embeddings, _ = corrected.extract_once(
            model,
            test_loader,
            dataset_test,
            pad_id,
            device,
            embedding_dim,
        )

        downstream, predictions = (
            corrected.evaluate_knn(
                train_embeddings,
                train_labels,
                test_embeddings,
                test_labels,
            )
        )

        fidelity = corrected.fidelity_metrics(
            full_test_embeddings,
            test_embeddings,
            reference_neighbors,
        )

        row = {
            "method": "scgpt_native_policy",
            "seed": seed,
            **downstream,
            **fidelity,
        }
        quality_rows.append(row)

        for key in quality_values:
            quality_values[key].append(
                float(row[key])
            )

        np.savez_compressed(
            output_dir
            / f"native_policy_seed{seed}.npz",
            train_labels=train_labels,
            test_labels=test_labels,
            test_predictions=predictions,
            test_embeddings=test_embeddings,
        )

        print(
            f"seed={seed}: "
            f"accuracy={row['accuracy']:.6f}, "
            f"macro-F1={row['macro_f1']:.6f}, "
            f"cosine={row['embedding_cosine_mean']:.6f}, "
            f"neighbor-R@10={row['neighbor_recall_10']:.6f}"
        )

    # Timing is measured separately from quality evaluation.
    timing_train_loader = build_native_loader(
        dataset_train,
        collator,
        batch_size,
        num_workers,
        seed=999_001,
    )
    timing_test_loader = build_native_loader(
        dataset_test,
        collator,
        batch_size,
        num_workers,
        seed=999_002,
    )

    corrected.warmup(
        model,
        timing_train_loader,
        pad_id,
        device,
        warmup_steps,
    )
    corrected.warmup(
        model,
        timing_test_loader,
        pad_id,
        device,
        warmup_steps,
    )

    timing_rows = []
    throughputs = []
    total_times = []
    peak_memories = []
    padding_ratios = []

    total_cells = (
        len(dataset_train) + len(dataset_test)
    )

    for repeat in range(timing_repeats):
        _, train_timing = corrected.extract_once(
            model,
            timing_train_loader,
            dataset_train,
            pad_id,
            device,
            embedding_dim,
        )
        _, test_timing = corrected.extract_once(
            model,
            timing_test_loader,
            dataset_test,
            pad_id,
            device,
            embedding_dim,
        )

        end_to_end_time_s = (
            train_timing["embedding_time_s"]
            + test_timing["embedding_time_s"]
        )
        throughput = total_cells / end_to_end_time_s

        actual_tokens = (
            train_timing["actual_sequence_tokens"]
            + test_timing["actual_sequence_tokens"]
        )
        padded_tokens = (
            train_timing["padded_tokens_processed"]
            + test_timing["padded_tokens_processed"]
        )
        padding_ratio = padded_tokens / actual_tokens

        peak_memory = max(
            train_timing["peak_gpu_memory_gb"],
            test_timing["peak_gpu_memory_gb"],
        )

        throughputs.append(throughput)
        total_times.append(end_to_end_time_s)
        peak_memories.append(peak_memory)
        padding_ratios.append(padding_ratio)

        timing_rows.append(
            {
                "method": "scgpt_native_policy",
                "repeat": repeat,
                "end_to_end_time_s": end_to_end_time_s,
                "cells_per_s": throughput,
                "peak_gpu_memory_gb": peak_memory,
                "actual_sequence_tokens": actual_tokens,
                "padded_tokens_processed": padded_tokens,
                "padding_overhead_ratio": padding_ratio,
            }
        )

        print(
            f"native timing repeat={repeat + 1}: "
            f"{throughput:.2f} cells/s, "
            f"padding={padding_ratio:.3f}"
        )

    train_native_gene_lengths = np.minimum(
        dataset_train.gene_lengths,
        max_length - 1,
    )
    test_native_gene_lengths = np.minimum(
        dataset_test.gene_lengths,
        max_length - 1,
    )

    summary: dict[str, Any] = {
        "method": "scgpt_native_policy",
        "tau": None,
        "fixed_k": max_length - 1,
        "batching": "sequential",
        "sampling": "random",
        "quality_num_seeds": len(seeds),
        "train_mean_selected_genes": float(
            np.mean(train_native_gene_lengths)
        ),
        "test_mean_selected_genes": float(
            np.mean(test_native_gene_lengths)
        ),
        "train_median_selected_genes": float(
            np.median(train_native_gene_lengths)
        ),
        "test_median_selected_genes": float(
            np.median(test_native_gene_lengths)
        ),
        "selection_time_s": 0.0,
        "selection_in_dataloader": True,
        "end_to_end_time_mean_s": float(
            np.mean(total_times)
        ),
        "end_to_end_time_std_s": float(
            np.std(total_times, ddof=1)
        ),
        "cells_per_s_mean": float(
            np.mean(throughputs)
        ),
        "cells_per_s_std": float(
            np.std(throughputs, ddof=1)
        ),
        "peak_gpu_memory_gb": float(
            np.max(peak_memories)
        ),
        "padding_overhead_ratio_mean": float(
            np.mean(padding_ratios)
        ),
    }

    for metric, values in quality_values.items():
        metric_mean, metric_std = (
            mean_and_sample_std(values)
        )
        summary[metric] = metric_mean
        summary[f"{metric}_std"] = metric_std

    return summary, quality_rows, timing_rows


def main() -> int:
    args = parse_args()

    required_paths = [
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.train,
        args.test,
        args.full_script,
        args.fixed_script,
        args.adaptive_script,
        args.corrected_script,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    summary_path = args.output_dir / "summary.csv"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {summary_path}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_module = load_module(
        "kang_full_module",
        args.full_script,
    )
    fixed_module = load_module(
        "kang_fixed_module",
        args.fixed_script,
    )
    adaptive_module = load_module(
        "kang_adaptive_module",
        args.adaptive_script,
    )
    corrected = load_module(
        "kang_corrected_module",
        args.corrected_script,
    )

    sys.path.insert(0, str(args.repo))

    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda:0")

    print("=" * 80)
    print("scGPT KANG 2018 CROSS-DONOR GENERALIZATION")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("Visible GPU index: 0")
    print("Test donor: patient_1015")
    print("Backend: PyTorch native Transformer")
    print("AMP FP16: ON")
    print(f"Tau: {args.tau}")
    print(f"K_min={args.k_min}, K_max={args.k_max}")

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)

    if not np.array_equal(
        train.var_names.astype(str),
        test.var_names.astype(str),
    ):
        raise RuntimeError(
            "Train/test gene ordering differs"
        )

    for column in (
        args.label_col,
        args.condition_col,
        args.donor_col,
    ):
        if column not in train.obs:
            raise KeyError(
                f"Train obs missing column: {column}"
            )
        if column not in test.obs:
            raise KeyError(
                f"Test obs missing column: {column}"
            )

    train_labels = (
        train.obs[args.label_col]
        .astype(str)
        .to_numpy()
    )
    test_labels = (
        test.obs[args.label_col]
        .astype(str)
        .to_numpy()
    )

    train_conditions = (
        train.obs[args.condition_col]
        .astype(str)
        .to_numpy()
    )
    test_conditions = (
        test.obs[args.condition_col]
        .astype(str)
        .to_numpy()
    )

    vocab = GeneVocab.from_file(
        args.model_dir / "vocab.json"
    )

    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)

    with (
        args.model_dir / "args.json"
    ).open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    pad_token = model_config["pad_token"]
    pad_id = vocab[pad_token]
    pad_value = float(model_config["pad_value"])
    vocab.set_default_index(pad_id)

    genes = np.asarray(
        train.var_names.astype(str)
    )
    matched_mask = np.asarray(
        [gene in vocab for gene in genes]
    )

    matched_genes = genes[matched_mask]
    gene_ids = np.asarray(
        vocab(list(matched_genes)),
        dtype=np.int64,
    )

    print(
        f"Matched genes: "
        f"{int(np.sum(matched_mask))}/{len(matched_mask)} "
        f"({np.mean(matched_mask):.4f})"
    )
    print(f"Train cells: {train.n_obs}")
    print(f"Test cells: {test.n_obs}")

    train_matrix = train.X[:, matched_mask]
    test_matrix = test.X[:, matched_mask]

    # All datasets are accessed cell-by-cell. CSR provides efficient
    # row access, whereas the original Kang matrices are CSC and make
    # repeated getrow() calls extremely expensive.
    if sparse.issparse(train_matrix):
        train_matrix = train_matrix.tocsr(copy=True)
        train_matrix.eliminate_zeros()
        train_matrix.sort_indices()
    else:
        train_matrix = np.asarray(train_matrix)

    if sparse.issparse(test_matrix):
        test_matrix = test_matrix.tocsr(copy=True)
        test_matrix.eliminate_zeros()
        test_matrix.sort_indices()
    else:
        test_matrix = np.asarray(test_matrix)

    print(
        "Train matrix format:",
        train_matrix.getformat()
        if sparse.issparse(train_matrix)
        else "dense",
    )
    print(
        "Test matrix format:",
        test_matrix.getformat()
        if sparse.issparse(test_matrix)
        else "dense",
    )

    idf = fixed_module.compute_idf(train_matrix)

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

    checkpoint = fixed_module.load_checkpoint(
        args.model_dir / "best_model.pt"
    )

    load_pretrained(
        model,
        checkpoint,
        strict=False,
        verbose=False,
    )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # Full bucketed reference
    # ------------------------------------------------------------------
    full_train = full_module.FullCellDataset(
        matrix=train_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
    )
    full_test = full_module.FullCellDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
    )

    full_max_length = int(
        max(
            np.max(full_train.sequence_lengths),
            np.max(full_test.sequence_lengths),
        )
    )

    print()
    print("=" * 80)
    print("FULL BUCKETED REFERENCE")
    print("=" * 80)
    print(f"Maximum sequence length: {full_max_length}")

    (
        full_summary,
        full_train_embeddings,
        full_test_embeddings,
        full_predictions,
        full_timing_rows,
    ) = corrected.run_config(
        method="full_bucketed",
        tau=None,
        fixed_k=None,
        train_dataset=full_train,
        test_dataset=full_test,
        max_length=full_max_length,
        selection_time_s=0.0,
        model=model,
        data_collator_class=DataCollator,
        pad_id=pad_id,
        pad_value=pad_value,
        device=device,
        embedding_dim=model_config["embsize"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        timing_repeats=args.timing_repeats,
        train_labels=train_labels,
        test_labels=test_labels,
        reference_test_embeddings=None,
        reference_neighbors=None,
    )

    full_summary["batching"] = "length_sorted"
    full_summary["sampling"] = "none"
    full_summary["quality_num_seeds"] = 1

    reference_neighbors = corrected.nearest_neighbors(
        full_test_embeddings,
        k=10,
    )

    np.savez_compressed(
        args.output_dir / "full_bucketed.npz",
        train_embeddings=full_train_embeddings,
        test_embeddings=full_test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        test_predictions=full_predictions,
    )

    # ------------------------------------------------------------------
    # scGPT native policy
    # ------------------------------------------------------------------
    (
        native_summary,
        native_quality_rows,
        native_timing_rows,
    ) = run_native_policy(
        corrected=corrected,
        dataset_train=full_train,
        dataset_test=full_test,
        model=model,
        data_collator_class=DataCollator,
        pad_id=pad_id,
        pad_value=pad_value,
        device=device,
        embedding_dim=model_config["embsize"],
        max_length=args.native_max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        timing_repeats=args.timing_repeats,
        seeds=args.native_seeds,
        train_labels=train_labels,
        test_labels=test_labels,
        full_test_embeddings=full_test_embeddings,
        reference_neighbors=reference_neighbors,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # AdaGeneBudget and exact mean-token-matched fixed TF-IDF
    # ------------------------------------------------------------------
    adaptive_train = (
        adaptive_module.AdaptiveSelectionDataset(
            matrix=train_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            idf=idf,
            tau=args.tau,
            k_min=args.k_min,
            k_max=args.k_max,
            get_row_function=fixed_module.get_row,
        )
    )
    adaptive_test = (
        adaptive_module.AdaptiveSelectionDataset(
            matrix=test_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            idf=idf,
            tau=args.tau,
            k_min=args.k_min,
            k_max=args.k_max,
            get_row_function=fixed_module.get_row,
        )
    )

    adaptive_selection_time = (
        adaptive_train.selection_time_s
        + adaptive_test.selection_time_s
    )

    target_train_mean = float(
        np.mean(adaptive_train.gene_lengths)
    )

    train_nonzero_counts = (
        corrected.row_nonzero_counts(train_matrix)
    )

    (
        matched_fixed_k,
        expected_fixed_mean,
        mean_difference,
    ) = corrected.find_exact_matched_fixed_k(
        train_nonzero_counts=train_nonzero_counts,
        target_mean=target_train_mean,
        max_k=args.k_max,
    )

    print()
    print("=" * 80)
    print("MATCHED TOKEN BUDGET")
    print("=" * 80)
    print(
        f"Adaptive train mean: {target_train_mean:.4f}"
    )
    print(f"Matched fixed K: {matched_fixed_k}")
    print(
        f"Expected fixed train mean: "
        f"{expected_fixed_mean:.4f}"
    )
    print(f"Mean difference: {mean_difference:.6f}")

    fixed_train = fixed_module.FixedSelectionDataset(
        matrix=train_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
        method="tfidf",
        budget=matched_fixed_k,
        seed=0,
        idf=idf,
    )
    fixed_test = fixed_module.FixedSelectionDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
        method="tfidf",
        budget=matched_fixed_k,
        seed=0,
        idf=idf,
    )

    fixed_selection_time = (
        fixed_train.selection_time_s
        + fixed_test.selection_time_s
    )

    (
        adaptive_summary,
        adaptive_train_embeddings,
        adaptive_test_embeddings,
        adaptive_predictions,
        adaptive_timing_rows,
    ) = corrected.run_config(
        method="adaptive",
        tau=args.tau,
        fixed_k=None,
        train_dataset=adaptive_train,
        test_dataset=adaptive_test,
        max_length=args.k_max + 1,
        selection_time_s=adaptive_selection_time,
        model=model,
        data_collator_class=DataCollator,
        pad_id=pad_id,
        pad_value=pad_value,
        device=device,
        embedding_dim=model_config["embsize"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        timing_repeats=args.timing_repeats,
        train_labels=train_labels,
        test_labels=test_labels,
        reference_test_embeddings=full_test_embeddings,
        reference_neighbors=reference_neighbors,
    )

    (
        fixed_summary,
        fixed_train_embeddings,
        fixed_test_embeddings,
        fixed_predictions,
        fixed_timing_rows,
    ) = corrected.run_config(
        method="matched_fixed_tfidf",
        tau=args.tau,
        fixed_k=matched_fixed_k,
        train_dataset=fixed_train,
        test_dataset=fixed_test,
        max_length=matched_fixed_k + 1,
        selection_time_s=fixed_selection_time,
        model=model,
        data_collator_class=DataCollator,
        pad_id=pad_id,
        pad_value=pad_value,
        device=device,
        embedding_dim=model_config["embsize"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_steps=args.warmup_steps,
        timing_repeats=args.timing_repeats,
        train_labels=train_labels,
        test_labels=test_labels,
        reference_test_embeddings=full_test_embeddings,
        reference_neighbors=reference_neighbors,
    )

    for summary, batching, sampling in (
        (adaptive_summary, "length_sorted", "adaptive_tfidf_mass"),
        (fixed_summary, "length_sorted", "fixed_tfidf"),
    ):
        summary["batching"] = batching
        summary["sampling"] = sampling
        summary["quality_num_seeds"] = 1
        summary["target_adaptive_train_mean"] = (
            target_train_mean
        )
        summary["matched_fixed_k"] = (
            matched_fixed_k
        )
        summary["matched_fixed_expected_mean"] = (
            expected_fixed_mean
        )
        summary["speedup_vs_full_bucketed"] = (
            summary["cells_per_s_mean"]
            / full_summary["cells_per_s_mean"]
        )
        summary["memory_reduction_vs_full_bucketed"] = (
            1.0
            - summary["peak_gpu_memory_gb"]
            / full_summary["peak_gpu_memory_gb"]
        )

    adaptive_summary.update(
        {
            "train_retained_mass_mean": float(
                np.mean(adaptive_train.retained_mass)
            ),
            "test_retained_mass_mean": float(
                np.mean(adaptive_test.retained_mass)
            ),
            "train_upper_clipped_fraction": float(
                np.mean(adaptive_train.upper_clipped)
            ),
            "test_upper_clipped_fraction": float(
                np.mean(adaptive_test.upper_clipped)
            ),
            "train_lower_clipped_fraction": float(
                np.mean(adaptive_train.lower_clipped)
            ),
            "test_lower_clipped_fraction": float(
                np.mean(adaptive_test.lower_clipped)
            ),
        }
    )

    native_summary["speedup_vs_full_bucketed"] = (
        native_summary["cells_per_s_mean"]
        / full_summary["cells_per_s_mean"]
    )
    native_summary[
        "memory_reduction_vs_full_bucketed"
    ] = (
        1.0
        - native_summary["peak_gpu_memory_gb"]
        / full_summary["peak_gpu_memory_gb"]
    )

    np.savez_compressed(
        args.output_dir / "adaptive.npz",
        train_embeddings=adaptive_train_embeddings,
        test_embeddings=adaptive_test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        test_predictions=adaptive_predictions,
        train_selected_genes=adaptive_train.gene_lengths,
        test_selected_genes=adaptive_test.gene_lengths,
        train_retained_mass=adaptive_train.retained_mass,
        test_retained_mass=adaptive_test.retained_mass,
    )

    np.savez_compressed(
        args.output_dir / "matched_fixed_tfidf.npz",
        train_embeddings=fixed_train_embeddings,
        test_embeddings=fixed_test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
        test_predictions=fixed_predictions,
        train_selected_genes=fixed_train.gene_lengths,
        test_selected_genes=fixed_test.gene_lengths,
    )

    summary_rows = [
        full_summary,
        native_summary,
        fixed_summary,
        adaptive_summary,
    ]

    all_timing_rows = (
        full_timing_rows
        + native_timing_rows
        + fixed_timing_rows
        + adaptive_timing_rows
    )

    corrected.write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
    )
    corrected.write_csv(
        args.output_dir / "timing_repeats.csv",
        all_timing_rows,
    )
    corrected.write_csv(
        args.output_dir / "native_seed_metrics.csv",
        native_quality_rows,
    )

    config = vars(args).copy()
    config["matched_genes"] = int(
        np.sum(matched_mask)
    )
    config["total_genes"] = int(
        len(matched_mask)
    )
    config["matched_fixed_k"] = int(
        matched_fixed_k
    )

    with (
        args.output_dir / "config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            config,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print(f"SUMMARY: {args.output_dir / 'summary.csv'}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
