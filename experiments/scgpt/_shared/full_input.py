#!/usr/bin/env python3
"""Run the true Full/native scGPT Pancreas baseline.

Full/native means:
- all non-zero genes matched to the pretrained vocabulary are retained;
- no random sampling or token selection is applied;
- <cls> is prepended;
- expression binning follows scGPT DataCollator;
- batches are dynamically padded to the longest sequence in each batch.

The script saves embeddings, downstream metrics, per-class metrics,
timing repetitions, token statistics, and environment/config metadata.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset, SequentialSampler


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
    parser.add_argument("--label-col", default="Celltype")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Smoke-test options
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./"
            "outputs/scgpt/pancreas/full_native"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_torch_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint root: {type(checkpoint).__name__}"
        )

    return checkpoint


def nonzero_counts(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.getnnz(axis=1)).reshape(-1)

    return np.count_nonzero(np.asarray(matrix), axis=1)


class FullCellDataset(Dataset):
    def __init__(
        self,
        matrix: Any,
        gene_ids: np.ndarray,
        cls_id: int,
        cls_value: float,
    ) -> None:
        self.matrix = matrix
        self.gene_ids = gene_ids
        self.cls_id = int(cls_id)
        self.cls_value = float(cls_value)

        counts = nonzero_counts(matrix)
        self.gene_lengths = counts.astype(np.int64)
        self.sequence_lengths = self.gene_lengths + 1

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if sparse.issparse(self.matrix):
            row = self.matrix.getrow(index)
            selected_idx = row.indices.astype(np.int64, copy=False)
            values = row.data.astype(np.float32, copy=False)

            # Preserve a stable gene-column order.
            order = np.argsort(selected_idx)
            selected_idx = selected_idx[order]
            values = values[order]
        else:
            row = np.asarray(self.matrix[index])
            selected_idx = np.flatnonzero(row).astype(
                np.int64,
                copy=False,
            )
            values = row[selected_idx].astype(
                np.float32,
                copy=False,
            )

        genes = self.gene_ids[selected_idx]

        genes = np.concatenate(
            [
                np.asarray([self.cls_id], dtype=np.int64),
                genes.astype(np.int64, copy=False),
            ]
        )
        values = np.concatenate(
            [
                np.asarray([self.cls_value], dtype=np.float32),
                values,
            ]
        )

        return {
            "id": torch.tensor(index, dtype=torch.long),
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(values),
        }


def build_loader(
    dataset: FullCellDataset,
    collator: Any,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": SequentialSampler(dataset),
        "collate_fn": collator,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def encode_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    pad_id: int,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    gene_ids = batch["gene"].to(
        device,
        non_blocking=True,
    )
    values = batch["expr"].to(
        device,
        non_blocking=True,
    )
    padding_mask = gene_ids.eq(pad_id)

    with torch.inference_mode(), torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        encoded = model._encode(
            gene_ids,
            values,
            src_key_padding_mask=padding_mask,
            batch_labels=None,
        )
        cell_embeddings = encoded[:, 0, :]

    return cell_embeddings, padding_mask


def warmup(
    model: torch.nn.Module,
    loader: DataLoader,
    pad_id: int,
    device: torch.device,
    amp_enabled: bool,
    steps: int,
) -> None:
    if steps <= 0:
        return

    iterator = iter(loader)
    first_batch = next(iterator)

    for _ in range(steps):
        embeddings, _ = encode_batch(
            model=model,
            batch=first_batch,
            pad_id=pad_id,
            device=device,
            amp_enabled=amp_enabled,
        )
        _ = embeddings.float().sum()

    torch.cuda.synchronize(device)
    del iterator, first_batch, embeddings


def extract_once(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: FullCellDataset,
    pad_id: int,
    device: torch.device,
    embedding_dim: int,
    amp_enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    outputs = np.empty(
        (len(dataset), embedding_dim),
        dtype=np.float32,
    )

    forward_events: list[
        tuple[torch.cuda.Event, torch.cuda.Event]
    ] = []

    padded_tokens = 0
    actual_sequence_tokens = 0
    offset = 0

    loop_start = time.perf_counter()

    for batch in loader:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        embeddings, padding_mask = encode_batch(
            model=model,
            batch=batch,
            pad_id=pad_id,
            device=device,
            amp_enabled=amp_enabled,
        )

        end_event.record()
        forward_events.append((start_event, end_event))

        embeddings_np = (
            embeddings.float()
            .detach()
            .cpu()
            .numpy()
        )

        batch_size = embeddings_np.shape[0]
        outputs[offset : offset + batch_size] = embeddings_np
        offset += batch_size

        padded_tokens += int(padding_mask.numel())
        actual_sequence_tokens += int(
            (~padding_mask).sum().item()
        )

    torch.cuda.synchronize(device)
    loop_time_s = time.perf_counter() - loop_start

    forward_time_s = sum(
        start.elapsed_time(end)
        for start, end in forward_events
    ) / 1000.0

    if offset != len(dataset):
        raise RuntimeError(
            f"Embedding count mismatch: {offset} != {len(dataset)}"
        )

    post_start = time.perf_counter()

    norms = np.linalg.norm(outputs, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("Zero-norm cell embedding detected")

    outputs /= norms

    postprocess_time_s = time.perf_counter() - post_start
    total_time_s = loop_time_s + postprocess_time_s

    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / 1024**3
    )

    if not np.isfinite(outputs).all():
        raise RuntimeError("NaN or Inf detected in cell embeddings")

    metrics = {
        "num_cells": len(dataset),
        "loop_time_s": loop_time_s,
        "model_forward_time_s": forward_time_s,
        "postprocess_time_s": postprocess_time_s,
        "total_embedding_time_s": total_time_s,
        "cells_per_s": len(dataset) / total_time_s,
        "peak_gpu_memory_gb": peak_memory_gb,
        "actual_sequence_tokens": actual_sequence_tokens,
        "padded_tokens_processed": padded_tokens,
        "padding_overhead_ratio": (
            padded_tokens / actual_sequence_tokens
        ),
        "mean_gene_tokens": float(
            np.mean(dataset.gene_lengths)
        ),
        "median_gene_tokens": float(
            np.median(dataset.gene_lengths)
        ),
        "max_gene_tokens": int(
            np.max(dataset.gene_lengths)
        ),
        "mean_sequence_tokens": float(
            np.mean(dataset.sequence_lengths)
        ),
        "max_sequence_tokens": int(
            np.max(dataset.sequence_lengths)
        ),
    }

    return outputs, metrics


def evaluate_knn(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, Any]]]:
    classifier = KNeighborsClassifier(
        n_neighbors=5,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(test_embeddings)

    labels = np.unique(
        np.concatenate([train_labels, test_labels])
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            test_labels,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    per_class = []

    for label, p, r, class_f1, n in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):
        per_class.append(
            {
                "cell_type": str(label),
                "precision": float(p),
                "recall": float(r),
                "f1": float(class_f1),
                "support": int(n),
            }
        )

    metrics = {
        "accuracy": float(
            accuracy_score(test_labels, predictions)
        ),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                test_labels,
                predictions,
            )
        ),
    }

    return metrics, predictions, per_class


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")

    required_paths = [
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.train,
        args.test,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if (
        args.output_dir.exists()
        and (args.output_dir / "metrics.json").exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {args.output_dir}. "
            "Use --overwrite to replace it."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo))

    import scgpt
    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    device = torch.device("cuda:0")

    print("=" * 80)
    print("scGPT PANCREAS TRUE FULL BASELINE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"scGPT: {Path(scgpt.__file__).resolve()}")
    print("External FlashAttention backend: OFF")
    print(f"AMP FP16: {args.amp}")

    train_adata = ad.read_h5ad(args.train)
    test_adata = ad.read_h5ad(args.test)

    if args.train_limit is not None:
        train_adata = train_adata[: args.train_limit].copy()

    if args.test_limit is not None:
        test_adata = test_adata[: args.test_limit].copy()

    if not np.array_equal(
        train_adata.var_names.astype(str),
        test_adata.var_names.astype(str),
    ):
        raise RuntimeError(
            "Train/test gene names or ordering differ"
        )

    train_labels = (
        train_adata.obs[args.label_col]
        .astype(str)
        .to_numpy()
    )
    test_labels = (
        test_adata.obs[args.label_col]
        .astype(str)
        .to_numpy()
    )

    vocab = GeneVocab.from_file(
        args.model_dir / "vocab.json"
    )

    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)

    with (args.model_dir / "args.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_config = json.load(handle)

    pad_token = model_config["pad_token"]
    pad_id = vocab[pad_token]
    pad_value = float(model_config["pad_value"])
    vocab.set_default_index(pad_id)

    all_genes = np.asarray(
        train_adata.var_names.astype(str)
    )
    matched_mask = np.asarray(
        [gene in vocab for gene in all_genes]
    )

    matched_genes = all_genes[matched_mask]
    gene_ids = np.asarray(
        vocab(list(matched_genes)),
        dtype=np.int64,
    )

    train_matrix = train_adata.X[:, matched_mask]
    test_matrix = test_adata.X[:, matched_mask]

    train_dataset = FullCellDataset(
        matrix=train_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
    )
    test_dataset = FullCellDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=vocab["<cls>"],
        cls_value=pad_value,
    )

    full_max_length = int(
        max(
            np.max(train_dataset.sequence_lengths),
            np.max(test_dataset.sequence_lengths),
        )
    )

    print(f"Train cells: {len(train_dataset):,}")
    print(f"Test cells: {len(test_dataset):,}")
    print(
        f"Matched genes: "
        f"{len(matched_genes):,}/{len(all_genes):,}"
    )
    print(f"True Full max length: {full_max_length}")
    print(
        "Train mean/max gene tokens: "
        f"{np.mean(train_dataset.gene_lengths):.2f}/"
        f"{np.max(train_dataset.gene_lengths)}"
    )
    print(
        "Test mean/max gene tokens: "
        f"{np.mean(test_dataset.gene_lengths):.2f}/"
        f"{np.max(test_dataset.gene_lengths)}"
    )

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

    checkpoint = load_torch_checkpoint(
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

    if not all(
        not parameter.requires_grad
        for parameter in model.parameters()
    ):
        raise RuntimeError("Model is not fully frozen")

    collator = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=full_max_length,
        sampling=False,
        keep_first_n_tokens=1,
    )

    train_loader = build_loader(
        dataset=train_dataset,
        collator=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        dataset=test_dataset,
        collator=collator,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Warming up train loader...")
    warmup(
        model=model,
        loader=train_loader,
        pad_id=pad_id,
        device=device,
        amp_enabled=args.amp,
        steps=args.warmup_steps,
    )

    print("Warming up test loader...")
    warmup(
        model=model,
        loader=test_loader,
        pad_id=pad_id,
        device=device,
        amp_enabled=args.amp,
        steps=args.warmup_steps,
    )

    timing_rows: list[dict[str, Any]] = []
    saved_train_embeddings = None
    saved_test_embeddings = None

    for repeat in range(args.timing_repeats):
        print()
        print(
            f"[Timing repeat {repeat + 1}/"
            f"{args.timing_repeats}]"
        )

        train_embeddings, train_timing = extract_once(
            model=model,
            loader=train_loader,
            dataset=train_dataset,
            pad_id=pad_id,
            device=device,
            embedding_dim=model_config["embsize"],
            amp_enabled=args.amp,
        )

        test_embeddings, test_timing = extract_once(
            model=model,
            loader=test_loader,
            dataset=test_dataset,
            pad_id=pad_id,
            device=device,
            embedding_dim=model_config["embsize"],
            amp_enabled=args.amp,
        )

        if repeat == 0:
            saved_train_embeddings = train_embeddings
            saved_test_embeddings = test_embeddings

        for split, values in (
            ("train", train_timing),
            ("test", test_timing),
        ):
            timing_rows.append(
                {
                    "repeat": repeat,
                    "split": split,
                    **values,
                }
            )

        combined_time = (
            train_timing["total_embedding_time_s"]
            + test_timing["total_embedding_time_s"]
        )
        combined_cells = (
            train_timing["num_cells"]
            + test_timing["num_cells"]
        )

        timing_rows.append(
            {
                "repeat": repeat,
                "split": "combined",
                "num_cells": combined_cells,
                "loop_time_s": (
                    train_timing["loop_time_s"]
                    + test_timing["loop_time_s"]
                ),
                "model_forward_time_s": (
                    train_timing["model_forward_time_s"]
                    + test_timing["model_forward_time_s"]
                ),
                "postprocess_time_s": (
                    train_timing["postprocess_time_s"]
                    + test_timing["postprocess_time_s"]
                ),
                "total_embedding_time_s": combined_time,
                "cells_per_s": combined_cells / combined_time,
                "peak_gpu_memory_gb": max(
                    train_timing["peak_gpu_memory_gb"],
                    test_timing["peak_gpu_memory_gb"],
                ),
                "actual_sequence_tokens": (
                    train_timing["actual_sequence_tokens"]
                    + test_timing["actual_sequence_tokens"]
                ),
                "padded_tokens_processed": (
                    train_timing["padded_tokens_processed"]
                    + test_timing["padded_tokens_processed"]
                ),
                "padding_overhead_ratio": (
                    (
                        train_timing["padded_tokens_processed"]
                        + test_timing["padded_tokens_processed"]
                    )
                    /
                    (
                        train_timing["actual_sequence_tokens"]
                        + test_timing["actual_sequence_tokens"]
                    )
                ),
                "mean_gene_tokens": np.nan,
                "median_gene_tokens": np.nan,
                "max_gene_tokens": max(
                    train_timing["max_gene_tokens"],
                    test_timing["max_gene_tokens"],
                ),
                "mean_sequence_tokens": np.nan,
                "max_sequence_tokens": max(
                    train_timing["max_sequence_tokens"],
                    test_timing["max_sequence_tokens"],
                ),
            }
        )

        print(
            "Combined throughput: "
            f"{combined_cells / combined_time:.2f} cells/s"
        )

    if (
        saved_train_embeddings is None
        or saved_test_embeddings is None
    ):
        raise RuntimeError("No embeddings were collected")

    downstream_metrics: dict[str, Any] = {}
    predictions = np.asarray([], dtype=str)
    per_class_rows: list[dict[str, Any]] = []

    if not args.skip_eval:
        (
            downstream_metrics,
            predictions,
            per_class_rows,
        ) = evaluate_knn(
            train_embeddings=saved_train_embeddings,
            train_labels=train_labels,
            test_embeddings=saved_test_embeddings,
            test_labels=test_labels,
        )

        print()
        print("[Held-out 5-NN]")
        for key, value in downstream_metrics.items():
            print(f"{key}: {value:.6f}")

    combined_rows = [
        row
        for row in timing_rows
        if row["split"] == "combined"
    ]

    throughput_values = np.asarray(
        [row["cells_per_s"] for row in combined_rows]
    )
    total_time_values = np.asarray(
        [
            row["total_embedding_time_s"]
            for row in combined_rows
        ]
    )

    summary = {
        "model": "scGPT-whole-human",
        "dataset": "Pancreas",
        "split": "official_train_test",
        "selector": "full_native",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "amp_dtype": "float16" if args.amp else None,
        "use_fast_transformer": False,
        "external_flash_attention": False,
        "torch_compile": False,
        "matched_genes": int(len(matched_genes)),
        "total_genes": int(len(all_genes)),
        "full_max_sequence_length": full_max_length,
        "selection_time_s": 0.0,
        "timing_repeats": args.timing_repeats,
        "combined_throughput_mean": float(
            np.mean(throughput_values)
        ),
        "combined_throughput_std": float(
            np.std(throughput_values, ddof=1)
            if len(throughput_values) > 1
            else 0.0
        ),
        "combined_total_time_mean_s": float(
            np.mean(total_time_values)
        ),
        "combined_total_time_std_s": float(
            np.std(total_time_values, ddof=1)
            if len(total_time_values) > 1
            else 0.0
        ),
        "peak_gpu_memory_gb": float(
            max(
                row["peak_gpu_memory_gb"]
                for row in combined_rows
            )
        ),
        "train_mean_gene_tokens": float(
            np.mean(train_dataset.gene_lengths)
        ),
        "test_mean_gene_tokens": float(
            np.mean(test_dataset.gene_lengths)
        ),
        "train_max_gene_tokens": int(
            np.max(train_dataset.gene_lengths)
        ),
        "test_max_gene_tokens": int(
            np.max(test_dataset.gene_lengths)
        ),
        **downstream_metrics,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "anndata": package_version("anndata"),
            "scanpy": package_version("scanpy"),
            "scikit_learn": package_version(
                "scikit-learn"
            ),
            "scgpt_import_path": str(
                Path(scgpt.__file__).resolve()
            ),
            "scgpt_repo_commit": git_commit(args.repo),
            "flash_attn_package": (
                package_version("flash-attn")
                if importlib.util.find_spec("flash_attn")
                else "NOT_INSTALLED"
            ),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
        },
    }

    np.savez_compressed(
        args.output_dir / "embeddings_and_labels.npz",
        train_embeddings=saved_train_embeddings,
        test_embeddings=saved_test_embeddings,
        train_labels=train_labels,
        test_labels=test_labels,
        test_predictions=predictions,
    )

    write_csv(
        args.output_dir / "timing_repeats.csv",
        timing_rows,
    )
    write_csv(
        args.output_dir / "per_celltype.csv",
        per_class_rows,
    )

    with (args.output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    with (args.output_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
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
    print(f"Output: {args.output_dir}")
    print(
        "Throughput: "
        f"{summary['combined_throughput_mean']:.2f} "
        f"± {summary['combined_throughput_std']:.2f} cells/s"
    )
    print(
        "Peak GPU memory: "
        f"{summary['peak_gpu_memory_gb']:.3f} GB"
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
