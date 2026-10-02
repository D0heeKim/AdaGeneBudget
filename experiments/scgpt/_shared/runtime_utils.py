#!/usr/bin/env python3
"""Corrected AdaGeneBudget comparison.

Fixes:
1. Match fixed-K by actual mean selected genes on the train split.
2. Apply identical length-sorted batching to Full, Adaptive, and Fixed.
3. Restore embeddings to the original cell order.
4. Include selection and bucketing overhead in end-to-end timing.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestNeighbors,
)
from torch.utils.data import DataLoader, Sampler


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
        default=Path("./checkpoints/scgpt"),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path(
            "./data/raw/pancreas/demo_train.h5ad"
        ),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path(
            "./data/raw/pancreas/demo_test.h5ad"
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
        "--full-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/full_input.py"
        ),
    )

    parser.add_argument("--label-col", default="Celltype")
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95],
    )
    parser.add_argument("--k-min", type=int, default=128)
    parser.add_argument("--k-max", type=int, default=600)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "pancreas/adaptive_corrected"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


class SortedLengthBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        sequence_lengths: np.ndarray,
        batch_size: int,
    ) -> None:
        self.batch_size = int(batch_size)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.sorted_indices = np.argsort(
            np.asarray(sequence_lengths),
            kind="stable",
        ).astype(np.int64)

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(
            0,
            len(self.sorted_indices),
            self.batch_size,
        ):
            yield self.sorted_indices[
                start : start + self.batch_size
            ].tolist()

    def __len__(self) -> int:
        return (
            len(self.sorted_indices)
            + self.batch_size
            - 1
        ) // self.batch_size


def build_bucketed_loader(
    dataset: Any,
    data_collator: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, float]:
    start = time.perf_counter()

    batch_sampler = SortedLengthBatchSampler(
        dataset.sequence_lengths,
        batch_size,
    )

    def collate_with_ids(
        examples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        ids = torch.stack(
            [example["id"] for example in examples]
        )
        batch = data_collator(examples)
        batch["original_id"] = ids
        return batch

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "collate_fn": collate_with_ids,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    loader = DataLoader(**kwargs)
    bucketing_time_s = time.perf_counter() - start

    return loader, bucketing_time_s


def warmup(
    model: torch.nn.Module,
    loader: DataLoader,
    pad_id: int,
    device: torch.device,
    steps: int,
) -> None:
    batch = next(iter(loader))

    for _ in range(steps):
        gene_ids = batch["gene"].to(
            device,
            non_blocking=True,
        )
        expression = batch["expr"].to(
            device,
            non_blocking=True,
        )
        padding_mask = gene_ids.eq(pad_id)

        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            output = model._encode(
                gene_ids,
                expression,
                src_key_padding_mask=padding_mask,
                batch_labels=None,
            )

        _ = output[:, 0].float().sum()

    torch.cuda.synchronize(device)


def extract_once(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: Any,
    pad_id: int,
    device: torch.device,
    embedding_dim: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    embeddings = np.empty(
        (len(dataset), embedding_dim),
        dtype=np.float32,
    )

    actual_tokens = 0
    padded_tokens = 0

    start = time.perf_counter()

    for batch in loader:
        original_ids = (
            batch["original_id"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        gene_ids = batch["gene"].to(
            device,
            non_blocking=True,
        )
        expression = batch["expr"].to(
            device,
            non_blocking=True,
        )
        padding_mask = gene_ids.eq(pad_id)

        with torch.inference_mode(), torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):
            encoded = model._encode(
                gene_ids,
                expression,
                src_key_padding_mask=padding_mask,
                batch_labels=None,
            )
            cell_embeddings = encoded[:, 0, :]

        embeddings[original_ids] = (
            cell_embeddings.float()
            .detach()
            .cpu()
            .numpy()
        )

        actual_tokens += int(
            (~padding_mask).sum().item()
        )
        padded_tokens += int(padding_mask.numel())

    torch.cuda.synchronize(device)
    embedding_time_s = time.perf_counter() - start

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0):
        raise RuntimeError("Zero-norm embedding found")

    embeddings /= norms

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embedding found")

    return embeddings, {
        "embedding_time_s": embedding_time_s,
        "actual_sequence_tokens": actual_tokens,
        "padded_tokens_processed": padded_tokens,
        "padding_overhead_ratio": (
            padded_tokens / actual_tokens
        ),
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        ),
    }


def row_nonzero_counts(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(
            matrix.getnnz(axis=1)
        ).reshape(-1)

    return np.count_nonzero(
        np.asarray(matrix),
        axis=1,
    )


def find_exact_matched_fixed_k(
    train_nonzero_counts: np.ndarray,
    target_mean: float,
    max_k: int,
) -> tuple[int, float, float]:
    best_k = 1
    best_mean = float(
        np.mean(np.minimum(train_nonzero_counts, 1))
    )
    best_difference = abs(best_mean - target_mean)

    for candidate_k in range(2, max_k + 1):
        candidate_mean = float(
            np.mean(
                np.minimum(
                    train_nonzero_counts,
                    candidate_k,
                )
            )
        )
        difference = abs(candidate_mean - target_mean)

        if difference < best_difference:
            best_k = candidate_k
            best_mean = candidate_mean
            best_difference = difference

    return best_k, best_mean, best_difference


def evaluate_knn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    classifier = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)

    metrics = {
        "accuracy": float(
            accuracy_score(test_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                test_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                test_labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                test_labels,
                predictions,
            )
        ),
    }

    return metrics, predictions


def nearest_neighbors(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(embeddings)

    indices = model.kneighbors(
        embeddings,
        return_distance=False,
    )

    return indices[:, 1 : k + 1]


def fidelity_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_neighbors: np.ndarray,
) -> dict[str, float]:
    cosine = np.sum(reference * candidate, axis=1)

    candidate_neighbors = nearest_neighbors(
        candidate,
        k=reference_neighbors.shape[1],
    )

    recalls = []

    for expected, observed in zip(
        reference_neighbors,
        candidate_neighbors,
    ):
        recalls.append(
            len(
                set(expected.tolist())
                & set(observed.tolist())
            )
            / reference_neighbors.shape[1]
        )

    return {
        "embedding_cosine_mean": float(
            np.mean(cosine)
        ),
        "embedding_cosine_median": float(
            np.median(cosine)
        ),
        "embedding_cosine_min": float(
            np.min(cosine)
        ),
        "neighbor_recall_10": float(
            np.mean(recalls)
        ),
    }


def run_config(
    *,
    method: str,
    tau: float | None,
    fixed_k: int | None,
    train_dataset: Any,
    test_dataset: Any,
    max_length: int,
    selection_time_s: float,
    model: torch.nn.Module,
    data_collator_class: Any,
    pad_id: int,
    pad_value: float,
    device: torch.device,
    embedding_dim: int,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    timing_repeats: int,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    reference_test_embeddings: np.ndarray | None,
    reference_neighbors: np.ndarray | None,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    collator = data_collator_class(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=False,
        keep_first_n_tokens=1,
    )

    train_loader, train_bucket_time = (
        build_bucketed_loader(
            train_dataset,
            collator,
            batch_size,
            num_workers,
        )
    )
    test_loader, test_bucket_time = (
        build_bucketed_loader(
            test_dataset,
            collator,
            batch_size,
            num_workers,
        )
    )

    bucketing_time_s = (
        train_bucket_time + test_bucket_time
    )

    warmup(
        model,
        train_loader,
        pad_id,
        device,
        warmup_steps,
    )
    warmup(
        model,
        test_loader,
        pad_id,
        device,
        warmup_steps,
    )

    total_cells = (
        len(train_dataset) + len(test_dataset)
    )

    timing_rows = []
    throughputs = []
    total_times = []
    peak_memories = []
    padding_ratios = []

    saved_train = None
    saved_test = None

    for repeat in range(timing_repeats):
        train_embeddings, train_timing = extract_once(
            model,
            train_loader,
            train_dataset,
            pad_id,
            device,
            embedding_dim,
        )
        test_embeddings, test_timing = extract_once(
            model,
            test_loader,
            test_dataset,
            pad_id,
            device,
            embedding_dim,
        )

        if repeat == 0:
            saved_train = train_embeddings
            saved_test = test_embeddings

        embedding_time_s = (
            train_timing["embedding_time_s"]
            + test_timing["embedding_time_s"]
        )

        overhead_time_s = (
            selection_time_s + bucketing_time_s
        )
        end_to_end_time_s = (
            overhead_time_s + embedding_time_s
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

        peak_memory = max(
            train_timing["peak_gpu_memory_gb"],
            test_timing["peak_gpu_memory_gb"],
        )

        padding_ratio = padded_tokens / actual_tokens

        throughputs.append(throughput)
        total_times.append(end_to_end_time_s)
        peak_memories.append(peak_memory)
        padding_ratios.append(padding_ratio)

        timing_rows.append(
            {
                "method": method,
                "tau": tau,
                "fixed_k": fixed_k,
                "repeat": repeat,
                "selection_time_s": selection_time_s,
                "bucketing_time_s": bucketing_time_s,
                "embedding_time_s": embedding_time_s,
                "end_to_end_time_s": end_to_end_time_s,
                "cells_per_s": throughput,
                "peak_gpu_memory_gb": peak_memory,
                "actual_sequence_tokens": actual_tokens,
                "padded_tokens_processed": padded_tokens,
                "padding_overhead_ratio": padding_ratio,
            }
        )

        print(
            f"{method}, tau={tau}, repeat={repeat + 1}: "
            f"{throughput:.2f} cells/s, "
            f"padding={padding_ratio:.3f}"
        )

    if saved_train is None or saved_test is None:
        raise RuntimeError("No embeddings generated")

    downstream, predictions = evaluate_knn(
        saved_train,
        train_labels,
        saved_test,
        test_labels,
    )

    summary: dict[str, Any] = {
        "method": method,
        "tau": tau,
        "fixed_k": fixed_k,
        "train_mean_selected_genes": float(
            np.mean(train_dataset.gene_lengths)
        ),
        "test_mean_selected_genes": float(
            np.mean(test_dataset.gene_lengths)
        ),
        "train_median_selected_genes": float(
            np.median(train_dataset.gene_lengths)
        ),
        "test_median_selected_genes": float(
            np.median(test_dataset.gene_lengths)
        ),
        "selection_time_s": selection_time_s,
        "bucketing_time_s": bucketing_time_s,
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
        **downstream,
    }

    if (
        reference_test_embeddings is not None
        and reference_neighbors is not None
    ):
        summary.update(
            fidelity_metrics(
                reference_test_embeddings,
                saved_test,
                reference_neighbors,
            )
        )
    else:
        summary.update(
            {
                "embedding_cosine_mean": 1.0,
                "embedding_cosine_median": 1.0,
                "embedding_cosine_min": 1.0,
                "neighbor_recall_10": 1.0,
            }
        )

    return (
        summary,
        saved_train,
        saved_test,
        predictions,
        timing_rows,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    # Rows may contain method-specific fields.
    # Preserve every column in first-appearance order.
    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    for path in (
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.train,
        args.test,
        args.fixed_script,
        args.adaptive_script,
        args.full_script,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if (
        (args.output_dir / "summary.csv").exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output exists: {args.output_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_module = load_module(
        "fixed_grid_module",
        args.fixed_script,
    )
    adaptive_module = load_module(
        "adaptive_module",
        args.adaptive_script,
    )
    full_module = load_module(
        "full_module",
        args.full_script,
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
    print("CORRECTED ADAGENEBUDGET COMPARISON")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("GPU index: 0")
    print("Length-sorted batching: ON")
    print("Exact mean-token matching: ON")
    print(f"Taus: {args.taus}")
    print(f"K_min={args.k_min}, K_max={args.k_max}")

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)

    if not np.array_equal(
        train.var_names.astype(str),
        test.var_names.astype(str),
    ):
        raise RuntimeError("Train/test gene order differs")

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

    train_matrix = train.X[:, matched_mask]
    test_matrix = test.X[:, matched_mask]

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

    summary_rows = []
    timing_rows = []

    # Bucketed Full reference
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

    (
        full_summary,
        full_train_embeddings,
        full_test_embeddings,
        full_predictions,
        full_timings,
    ) = run_config(
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

    summary_rows.append(full_summary)
    timing_rows.extend(full_timings)

    reference_neighbors = nearest_neighbors(
        full_test_embeddings,
        k=10,
    )

    np.savez_compressed(
        args.output_dir / "full_bucketed.npz",
        train_embeddings=full_train_embeddings,
        test_embeddings=full_test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        test_predictions=full_predictions,
    )

    train_nonzero = row_nonzero_counts(
        train_matrix
    )

    for tau in args.taus:
        print()
        print("=" * 80)
        print(f"TAU={tau:.2f}")
        print("=" * 80)

        adaptive_train = (
            adaptive_module.AdaptiveSelectionDataset(
                matrix=train_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                idf=idf,
                tau=tau,
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
                tau=tau,
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

        (
            matched_k,
            expected_fixed_mean,
            mean_difference,
        ) = find_exact_matched_fixed_k(
            train_nonzero_counts=train_nonzero,
            target_mean=target_train_mean,
            max_k=args.k_max,
        )

        print(
            f"Adaptive train mean={target_train_mean:.4f}; "
            f"matched fixed K={matched_k}; "
            f"expected fixed mean={expected_fixed_mean:.4f}; "
            f"difference={mean_difference:.6f}"
        )

        fixed_train = (
            fixed_module.FixedSelectionDataset(
                matrix=train_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                method="tfidf",
                budget=matched_k,
                seed=args.seed,
                idf=idf,
            )
        )
        fixed_test = (
            fixed_module.FixedSelectionDataset(
                matrix=test_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                method="tfidf",
                budget=matched_k,
                seed=args.seed,
                idf=idf,
            )
        )

        fixed_selection_time = (
            fixed_train.selection_time_s
            + fixed_test.selection_time_s
        )

        adaptive_result = run_config(
            method="adaptive",
            tau=tau,
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

        fixed_result = run_config(
            method="matched_fixed_tfidf",
            tau=tau,
            fixed_k=matched_k,
            train_dataset=fixed_train,
            test_dataset=fixed_test,
            max_length=matched_k + 1,
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

        (
            adaptive_summary,
            adaptive_train_embeddings,
            adaptive_test_embeddings,
            adaptive_predictions,
            adaptive_timings,
        ) = adaptive_result

        (
            fixed_summary,
            fixed_train_embeddings,
            fixed_test_embeddings,
            fixed_predictions,
            fixed_timings,
        ) = fixed_result

        for summary in (
            adaptive_summary,
            fixed_summary,
        ):
            summary["target_adaptive_train_mean"] = (
                target_train_mean
            )
            summary["matched_fixed_k"] = matched_k
            summary["matched_fixed_expected_mean"] = (
                expected_fixed_mean
            )
            summary["train_mean_token_gap_vs_adaptive"] = (
                summary["train_mean_selected_genes"]
                - target_train_mean
            )
            summary["speedup_vs_bucketed_full"] = (
                summary["cells_per_s_mean"]
                / full_summary["cells_per_s_mean"]
            )
            summary["memory_reduction_vs_bucketed_full"] = (
                1.0
                - summary["peak_gpu_memory_gb"]
                / full_summary["peak_gpu_memory_gb"]
            )

        adaptive_summary.update(
            {
                "train_retained_mass_mean": float(
                    np.mean(
                        adaptive_train.retained_mass
                    )
                ),
                "test_retained_mass_mean": float(
                    np.mean(
                        adaptive_test.retained_mass
                    )
                ),
                "train_upper_clipped_fraction": float(
                    np.mean(
                        adaptive_train.upper_clipped
                    )
                ),
                "test_upper_clipped_fraction": float(
                    np.mean(
                        adaptive_test.upper_clipped
                    )
                ),
            }
        )

        summary_rows.extend(
            [adaptive_summary, fixed_summary]
        )
        timing_rows.extend(
            adaptive_timings + fixed_timings
        )

        tag = f"tau{int(round(tau * 100)):03d}"

        np.savez_compressed(
            args.output_dir / f"{tag}_adaptive.npz",
            train_embeddings=adaptive_train_embeddings,
            test_embeddings=adaptive_test_embeddings,
            train_labels=train_labels,
            test_labels=test_labels,
            test_predictions=adaptive_predictions,
            train_selected_genes=(
                adaptive_train.gene_lengths
            ),
            test_selected_genes=(
                adaptive_test.gene_lengths
            ),
        )

        np.savez_compressed(
            args.output_dir / f"{tag}_fixed.npz",
            train_embeddings=fixed_train_embeddings,
            test_embeddings=fixed_test_embeddings,
            train_labels=train_labels,
            test_labels=test_labels,
            test_predictions=fixed_predictions,
            train_selected_genes=(
                fixed_train.gene_lengths
            ),
            test_selected_genes=(
                fixed_test.gene_lengths
            ),
        )

    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
    )
    write_csv(
        args.output_dir / "timing_repeats.csv",
        timing_rows,
    )

    with (
        args.output_dir / "config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            vars(args),
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
