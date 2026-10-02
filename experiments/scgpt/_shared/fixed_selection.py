#!/usr/bin/env python3
"""Fixed-budget gene-selection benchmark for scGPT Pancreas."""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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
    parser.add_argument(
        "--full-reference",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "pancreas/full_native_clean/"
            "embeddings_and_labels.npz"
        ),
    )

    parser.add_argument("--label-col", default="Celltype")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "expression", "tfidf"],
        choices=["random", "expression", "tfidf"],
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[1200, 600, 300, 150],
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./"
            "outputs/scgpt/pancreas/fixed_grid_seed0"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
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
            f"Unexpected checkpoint type: {type(checkpoint).__name__}"
        )

    return checkpoint


def compute_idf(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        df = np.asarray((matrix > 0).sum(axis=0)).reshape(-1)
    else:
        df = np.count_nonzero(np.asarray(matrix) > 0, axis=0)

    n_cells = matrix.shape[0]

    return (
        np.log((1.0 + n_cells) / (1.0 + df)) + 1.0
    ).astype(np.float32)


def get_row(
    matrix: Any,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sparse.issparse(matrix):
        row = matrix.getrow(index)

        indices = row.indices.astype(np.int64, copy=False)
        values = row.data.astype(np.float32, copy=False)

        order = np.argsort(indices)
        return indices[order], values[order]

    row = np.asarray(matrix[index])

    indices = np.flatnonzero(row).astype(
        np.int64,
        copy=False,
    )
    values = row[indices].astype(
        np.float32,
        copy=False,
    )

    return indices, values


class FixedSelectionDataset(Dataset):
    def __init__(
        self,
        matrix: Any,
        gene_ids: np.ndarray,
        cls_id: int,
        cls_value: float,
        method: str,
        budget: int,
        seed: int,
        idf: np.ndarray | None,
    ) -> None:
        self.matrix = matrix
        self.gene_ids = gene_ids
        self.cls_id = int(cls_id)
        self.cls_value = float(cls_value)
        self.method = method
        self.budget = int(budget)
        self.seed = int(seed)
        self.idf = idf

        if self.budget <= 0:
            raise ValueError("budget must be positive")

        if self.method == "tfidf" and self.idf is None:
            raise ValueError("TF-IDF requires train-derived IDF")

        start = time.perf_counter()
        self.selected_indices = self._select_all()
        self.selection_time_s = time.perf_counter() - start

        self.gene_lengths = np.asarray(
            [len(indices) for indices in self.selected_indices],
            dtype=np.int64,
        )
        self.sequence_lengths = self.gene_lengths + 1

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def _select_all(self) -> list[np.ndarray]:
        selected: list[np.ndarray] = []

        for cell_index in range(len(self)):
            indices, values = get_row(
                self.matrix,
                cell_index,
            )

            if len(indices) <= self.budget:
                selected.append(indices)
                continue

            if self.method == "random":
                # Cell-specific RNG avoids dependence on iteration order.
                rng = np.random.default_rng(
                    self.seed * 1_000_003 + cell_index
                )

                positions = rng.choice(
                    len(indices),
                    size=self.budget,
                    replace=False,
                )

            elif self.method == "expression":
                positions = np.argpartition(
                    values,
                    -self.budget,
                )[-self.budget :]

            elif self.method == "tfidf":
                assert self.idf is not None

                scores = values * self.idf[indices]

                positions = np.argpartition(
                    scores,
                    -self.budget,
                )[-self.budget :]

            else:
                raise ValueError(
                    f"Unknown selection method: {self.method}"
                )

            chosen = np.sort(indices[positions])
            selected.append(chosen)

        return selected

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        all_indices, all_values = get_row(
            self.matrix,
            index,
        )

        selected = self.selected_indices[index]

        # all_indices is sorted; recover selected expression values.
        positions = np.searchsorted(all_indices, selected)
        values = all_values[positions]
        genes = self.gene_ids[selected]

        genes = np.concatenate(
            [
                np.asarray([self.cls_id], dtype=np.int64),
                genes.astype(np.int64, copy=False),
            ]
        )
        values = np.concatenate(
            [
                np.asarray([self.cls_value], dtype=np.float32),
                values.astype(np.float32, copy=False),
            ]
        )

        return {
            "id": torch.tensor(index, dtype=torch.long),
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(values),
        }


def build_loader(
    dataset: Dataset,
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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    return encoded[:, 0, :], padding_mask


def warmup(
    model: torch.nn.Module,
    loader: DataLoader,
    pad_id: int,
    device: torch.device,
    steps: int,
) -> None:
    batch = next(iter(loader))

    for _ in range(steps):
        embedding, _ = encode_batch(
            model,
            batch,
            pad_id,
            device,
        )
        _ = embedding.float().sum()

    torch.cuda.synchronize(device)


def extract_once(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: FixedSelectionDataset,
    pad_id: int,
    device: torch.device,
    embedding_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    embeddings = np.empty(
        (len(dataset), embedding_dim),
        dtype=np.float32,
    )

    padded_tokens = 0
    actual_tokens = 0
    offset = 0

    start = time.perf_counter()

    for batch in loader:
        output, padding_mask = encode_batch(
            model,
            batch,
            pad_id,
            device,
        )

        output_np = (
            output.float()
            .detach()
            .cpu()
            .numpy()
        )

        batch_size = len(output_np)

        embeddings[offset : offset + batch_size] = output_np
        offset += batch_size

        padded_tokens += int(padding_mask.numel())
        actual_tokens += int(
            (~padding_mask).sum().item()
        )

    torch.cuda.synchronize(device)
    embedding_time_s = time.perf_counter() - start

    if offset != len(dataset):
        raise RuntimeError(
            f"Output count mismatch: {offset} != {len(dataset)}"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0):
        raise RuntimeError("Zero-norm embedding detected")

    embeddings /= norms

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embedding detected")

    metrics = {
        "embedding_time_s": embedding_time_s,
        "cells_per_s_embedding_only": (
            len(dataset) / embedding_time_s
        ),
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        ),
        "actual_sequence_tokens": actual_tokens,
        "padded_tokens_processed": padded_tokens,
        "padding_overhead_ratio": (
            padded_tokens / actual_tokens
        ),
    }

    return embeddings, metrics


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
        "balanced_accuracy": float(
            balanced_accuracy_score(
                test_labels,
                predictions,
            )
        ),
    }

    return metrics, predictions


def cosine_to_reference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    cosine = np.sum(reference * candidate, axis=1)

    return {
        "embedding_cosine_mean": float(np.mean(cosine)),
        "embedding_cosine_median": float(np.median(cosine)),
        "embedding_cosine_min": float(np.min(cosine)),
    }


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

    # The first item should be the cell itself.
    return indices[:, 1 : k + 1]


def neighbor_recall(
    reference_neighbors: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int,
) -> float:
    candidate_neighbors = nearest_neighbors(
        candidate_embeddings,
        k,
    )

    recalls = []

    for ref, candidate in zip(
        reference_neighbors,
        candidate_neighbors,
    ):
        recalls.append(
            len(set(ref.tolist()) & set(candidate.tolist())) / k
        )

    return float(np.mean(recalls))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
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
        raise RuntimeError("CUDA is required")

    if (
        args.output_dir.exists()
        and (args.output_dir / "summary.csv").exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output exists: {args.output_dir}. "
            "Use --overwrite to replace it."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sys.path.insert(0, str(args.repo))

    import scgpt
    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    device = torch.device("cuda:0")

    print("=" * 80)
    print("scGPT PANCREAS FIXED-BUDGET GRID")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"scGPT: {Path(scgpt.__file__).resolve()}")
    print("External FlashAttention backend: OFF")
    print("AMP FP16: True")
    print(f"Methods: {args.methods}")
    print(f"Budgets: {args.budgets}")
    print(f"Seed: {args.seed}")

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)

    if not np.array_equal(
        train.var_names.astype(str),
        test.var_names.astype(str),
    ):
        raise RuntimeError(
            "Train/test gene order mismatch"
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
    pad_value = float(model_config["pad_value"])
    pad_id = vocab[pad_token]

    vocab.set_default_index(pad_id)

    genes = np.asarray(train.var_names.astype(str))
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

    print(
        f"Matched genes: "
        f"{len(matched_genes):,}/{len(genes):,}"
    )
    print("Computing IDF from train only...")
    idf = compute_idf(train_matrix)

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

    checkpoint = load_checkpoint(
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

    full_data = np.load(
        args.full_reference,
        allow_pickle=True,
    )

    full_train_embeddings = (
        full_data["train_embeddings"]
        .astype(np.float32)
    )
    full_test_embeddings = (
        full_data["test_embeddings"]
        .astype(np.float32)
    )

    if full_train_embeddings.shape[0] != train.n_obs:
        raise RuntimeError(
            "Full train embedding count mismatch"
        )

    if full_test_embeddings.shape[0] != test.n_obs:
        raise RuntimeError(
            "Full test embedding count mismatch"
        )

    reference_neighbors_10 = nearest_neighbors(
        full_test_embeddings,
        k=10,
    )

    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for method in args.methods:
        for budget in args.budgets:
            print()
            print("=" * 80)
            print(f"METHOD={method}, K={budget}")
            print("=" * 80)

            set_seed(args.seed)

            train_dataset = FixedSelectionDataset(
                matrix=train_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                method=method,
                budget=budget,
                seed=args.seed,
                idf=idf if method == "tfidf" else None,
            )

            test_dataset = FixedSelectionDataset(
                matrix=test_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                method=method,
                budget=budget,
                seed=args.seed,
                idf=idf if method == "tfidf" else None,
            )

            selection_time_s = (
                train_dataset.selection_time_s
                + test_dataset.selection_time_s
            )

            max_length = budget + 1

            collator = DataCollator(
                do_padding=True,
                pad_token_id=pad_id,
                pad_value=pad_value,
                do_mlm=False,
                do_binning=True,
                max_length=max_length,
                sampling=False,
                keep_first_n_tokens=1,
            )

            train_loader = build_loader(
                train_dataset,
                collator,
                args.batch_size,
                args.num_workers,
            )
            test_loader = build_loader(
                test_dataset,
                collator,
                args.batch_size,
                args.num_workers,
            )

            warmup(
                model,
                train_loader,
                pad_id,
                device,
                args.warmup_steps,
            )
            warmup(
                model,
                test_loader,
                pad_id,
                device,
                args.warmup_steps,
            )

            saved_train = None
            saved_test = None
            repeat_total_times: list[float] = []
            repeat_throughputs: list[float] = []
            repeat_memories: list[float] = []

            for repeat in range(args.timing_repeats):
                train_embeddings, train_timing = extract_once(
                    model,
                    train_loader,
                    train_dataset,
                    pad_id,
                    device,
                    model_config["embsize"],
                )

                test_embeddings, test_timing = extract_once(
                    model,
                    test_loader,
                    test_dataset,
                    pad_id,
                    device,
                    model_config["embsize"],
                )

                if repeat == 0:
                    saved_train = train_embeddings
                    saved_test = test_embeddings

                embedding_time_s = (
                    train_timing["embedding_time_s"]
                    + test_timing["embedding_time_s"]
                )
                total_time_s = (
                    selection_time_s + embedding_time_s
                )
                total_cells = (
                    len(train_dataset) + len(test_dataset)
                )
                throughput = total_cells / total_time_s
                peak_memory = max(
                    train_timing["peak_gpu_memory_gb"],
                    test_timing["peak_gpu_memory_gb"],
                )

                repeat_total_times.append(total_time_s)
                repeat_throughputs.append(throughput)
                repeat_memories.append(peak_memory)

                timing_rows.append(
                    {
                        "method": method,
                        "budget": budget,
                        "seed": args.seed,
                        "repeat": repeat,
                        "selection_time_s": selection_time_s,
                        "embedding_time_s": embedding_time_s,
                        "end_to_end_time_s": total_time_s,
                        "cells_per_s": throughput,
                        "peak_gpu_memory_gb": peak_memory,
                        "train_actual_tokens": (
                            train_timing[
                                "actual_sequence_tokens"
                            ]
                        ),
                        "test_actual_tokens": (
                            test_timing[
                                "actual_sequence_tokens"
                            ]
                        ),
                        "train_padded_tokens": (
                            train_timing[
                                "padded_tokens_processed"
                            ]
                        ),
                        "test_padded_tokens": (
                            test_timing[
                                "padded_tokens_processed"
                            ]
                        ),
                    }
                )

                print(
                    f"repeat={repeat + 1}: "
                    f"{throughput:.2f} cells/s"
                )

            assert saved_train is not None
            assert saved_test is not None

            downstream, predictions = evaluate_knn(
                saved_train,
                train_labels,
                saved_test,
                test_labels,
            )

            fidelity = cosine_to_reference(
                full_test_embeddings,
                saved_test,
            )

            neighbor_recall_10 = neighbor_recall(
                reference_neighbors_10,
                saved_test,
                k=10,
            )

            row = {
                "model": "scGPT",
                "dataset": "Pancreas",
                "selector": method,
                "fixed_k": budget,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "num_cells": (
                    len(train_dataset) + len(test_dataset)
                ),
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
                "end_to_end_time_mean_s": float(
                    np.mean(repeat_total_times)
                ),
                "end_to_end_time_std_s": float(
                    np.std(
                        repeat_total_times,
                        ddof=1,
                    )
                ),
                "cells_per_s_mean": float(
                    np.mean(repeat_throughputs)
                ),
                "cells_per_s_std": float(
                    np.std(
                        repeat_throughputs,
                        ddof=1,
                    )
                ),
                "peak_gpu_memory_gb": float(
                    np.max(repeat_memories)
                ),
                **downstream,
                **fidelity,
                "neighbor_recall_10": neighbor_recall_10,
            }

            summary_rows.append(row)

            config_dir = (
                args.output_dir
                / f"{method}_k{budget}"
            )
            config_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                config_dir / "embeddings_and_labels.npz",
                train_embeddings=saved_train,
                test_embeddings=saved_test,
                train_labels=train_labels,
                test_labels=test_labels,
                test_predictions=predictions,
            )

            with (
                config_dir / "metrics.json"
            ).open("w", encoding="utf-8") as handle:
                json.dump(
                    row,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"accuracy={downstream['accuracy']:.6f}, "
                f"macro_f1={downstream['macro_f1']:.6f}, "
                f"neighbor_recall@10={neighbor_recall_10:.6f}"
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
