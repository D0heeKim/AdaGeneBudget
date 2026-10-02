#!/usr/bin/env python3
"""Evaluate AdaGeneBudget and train-mean-matched fixed TF-IDF baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset


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
    parser.add_argument(
        "--full-metrics",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "pancreas/full_native_clean/metrics.json"
        ),
    )
    parser.add_argument(
        "--fixed-grid-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/fixed_selection.py"
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
            "./"
            "outputs/scgpt/pancreas/adaptive_main"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_base_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "scgpt_fixed_grid_base",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


class AdaptiveSelectionDataset(Dataset):
    def __init__(
        self,
        matrix: Any,
        gene_ids: np.ndarray,
        cls_id: int,
        cls_value: float,
        idf: np.ndarray,
        tau: float,
        k_min: int,
        k_max: int,
        get_row_function: Any,
    ) -> None:
        self.matrix = matrix
        self.gene_ids = gene_ids
        self.cls_id = int(cls_id)
        self.cls_value = float(cls_value)
        self.idf = idf
        self.tau = float(tau)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.get_row = get_row_function

        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")

        if not 0 < self.k_min <= self.k_max:
            raise ValueError("Require 0 < k_min <= k_max")

        start = time.perf_counter()

        (
            self.selected_indices,
            self.raw_budgets,
            self.retained_mass,
            self.lower_clipped,
            self.upper_clipped,
        ) = self._precompute()

        self.selection_time_s = time.perf_counter() - start

        self.gene_lengths = np.asarray(
            [len(indices) for indices in self.selected_indices],
            dtype=np.int64,
        )
        self.sequence_lengths = self.gene_lengths + 1

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def _precompute(
        self,
    ) -> tuple[
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        selected: list[np.ndarray] = []

        raw_budgets = np.zeros(len(self), dtype=np.int32)
        retained_mass = np.ones(len(self), dtype=np.float32)
        lower_clipped = np.zeros(len(self), dtype=bool)
        upper_clipped = np.zeros(len(self), dtype=bool)

        for cell_index in range(len(self)):
            indices, values = self.get_row(
                self.matrix,
                cell_index,
            )

            n_genes = len(indices)

            if n_genes == 0:
                selected.append(indices)
                raw_budgets[cell_index] = 0
                retained_mass[cell_index] = 1.0
                continue

            scores = values * self.idf[indices]

            if np.any(scores < 0):
                raise RuntimeError(
                    f"Negative score in cell {cell_index}"
                )

            total_score = float(np.sum(scores))

            if total_score <= 0:
                selected.append(indices)
                raw_budgets[cell_index] = n_genes
                retained_mass[cell_index] = 1.0
                continue

            # Stable sorting gives deterministic tie handling.
            score_order = np.argsort(
                -scores,
                kind="stable",
            )

            sorted_scores = scores[score_order]
            cumulative = np.cumsum(sorted_scores)

            raw_k = int(
                np.searchsorted(
                    cumulative,
                    self.tau * total_score,
                    side="left",
                )
                + 1
            )

            raw_k = min(raw_k, n_genes)
            raw_budgets[cell_index] = raw_k

            final_k = raw_k

            if n_genes >= self.k_min and final_k < self.k_min:
                final_k = self.k_min
                lower_clipped[cell_index] = True

            if final_k > self.k_max:
                final_k = self.k_max
                upper_clipped[cell_index] = True

            final_k = min(final_k, n_genes)

            selected_positions = score_order[:final_k]
            chosen_indices = np.sort(
                indices[selected_positions]
            )

            selected.append(chosen_indices)

            retained_mass[cell_index] = float(
                np.sum(scores[selected_positions])
                / total_score
            )

        return (
            selected,
            raw_budgets,
            retained_mass,
            lower_clipped,
            upper_clipped,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        all_indices, all_values = self.get_row(
            self.matrix,
            index,
        )

        selected = self.selected_indices[index]

        positions = np.searchsorted(
            all_indices,
            selected,
        )
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
                np.asarray(
                    [self.cls_value],
                    dtype=np.float32,
                ),
                values.astype(np.float32, copy=False),
            ]
        )

        return {
            "id": torch.tensor(index, dtype=torch.long),
            "genes": torch.from_numpy(genes),
            "expressions": torch.from_numpy(values),
        }


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


def per_celltype_rows(
    tau: float,
    method: str,
    test_labels: np.ndarray,
    predictions: np.ndarray,
    gene_lengths: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.unique(
        np.concatenate([test_labels, predictions])
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            test_labels,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    rows = []

    for label, p, r, class_f1, n in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):
        true_mask = test_labels == label

        rows.append(
            {
                "tau": tau,
                "method": method,
                "cell_type": str(label),
                "precision": float(p),
                "recall": float(r),
                "f1": float(class_f1),
                "support": int(n),
                "mean_selected_genes": (
                    float(np.mean(gene_lengths[true_mask]))
                    if np.any(true_mask)
                    else np.nan
                ),
                "median_selected_genes": (
                    float(np.median(gene_lengths[true_mask]))
                    if np.any(true_mask)
                    else np.nan
                ),
            }
        )

    return rows


def run_configuration(
    *,
    base: Any,
    name: str,
    tau: float,
    dataset_train: Any,
    dataset_test: Any,
    max_length: int,
    selection_time_s: float,
    model: torch.nn.Module,
    vocab: Any,
    model_config: dict[str, Any],
    pad_id: int,
    pad_value: float,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    warmup_steps: int,
    timing_repeats: int,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    full_test_embeddings: np.ndarray,
    reference_neighbors_10: np.ndarray,
    full_metrics: dict[str, Any],
    output_dir: Path,
    additional_summary: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scgpt.data_collator import DataCollator

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

    train_loader = base.build_loader(
        dataset_train,
        collator,
        batch_size,
        num_workers,
    )
    test_loader = base.build_loader(
        dataset_test,
        collator,
        batch_size,
        num_workers,
    )

    base.warmup(
        model,
        train_loader,
        pad_id,
        device,
        warmup_steps,
    )
    base.warmup(
        model,
        test_loader,
        pad_id,
        device,
        warmup_steps,
    )

    timing_rows = []
    throughputs = []
    total_times = []
    memories = []
    padding_ratios = []

    saved_train = None
    saved_test = None

    total_cells = len(dataset_train) + len(dataset_test)

    for repeat in range(timing_repeats):
        train_embeddings, train_timing = base.extract_once(
            model,
            train_loader,
            dataset_train,
            pad_id,
            device,
            model_config["embsize"],
        )

        test_embeddings, test_timing = base.extract_once(
            model,
            test_loader,
            dataset_test,
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
        total_time_s = selection_time_s + embedding_time_s
        throughput = total_cells / total_time_s

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
        total_times.append(total_time_s)
        memories.append(peak_memory)
        padding_ratios.append(padding_ratio)

        timing_rows.append(
            {
                "tau": tau,
                "method": name,
                "repeat": repeat,
                "selection_time_s": selection_time_s,
                "embedding_time_s": embedding_time_s,
                "end_to_end_time_s": total_time_s,
                "cells_per_s": throughput,
                "peak_gpu_memory_gb": peak_memory,
                "actual_sequence_tokens": actual_tokens,
                "padded_tokens_processed": padded_tokens,
                "padding_overhead_ratio": padding_ratio,
            }
        )

        print(
            f"{name}, tau={tau:.2f}, repeat={repeat + 1}: "
            f"{throughput:.2f} cells/s"
        )

    if saved_train is None or saved_test is None:
        raise RuntimeError("No embeddings were generated")

    downstream, predictions = base.evaluate_knn(
        saved_train,
        train_labels,
        saved_test,
        test_labels,
    )

    downstream["weighted_f1"] = float(
        f1_score(
            test_labels,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    fidelity = base.cosine_to_reference(
        full_test_embeddings,
        saved_test,
    )

    neighbor_recall_10 = base.neighbor_recall(
        reference_neighbors_10,
        saved_test,
        k=10,
    )

    throughput_mean = float(np.mean(throughputs))
    peak_memory = float(np.max(memories))

    summary = {
        "model": "scGPT",
        "dataset": "Pancreas",
        "method": name,
        "tau": tau,
        "k_min": additional_summary.get("k_min"),
        "k_max": additional_summary.get("k_max"),
        "matched_fixed_k": additional_summary.get(
            "matched_fixed_k"
        ),
        "train_mean_selected_genes": float(
            np.mean(dataset_train.gene_lengths)
        ),
        "test_mean_selected_genes": float(
            np.mean(dataset_test.gene_lengths)
        ),
        "train_median_selected_genes": float(
            np.median(dataset_train.gene_lengths)
        ),
        "test_median_selected_genes": float(
            np.median(dataset_test.gene_lengths)
        ),
        "selection_time_s": selection_time_s,
        "end_to_end_time_mean_s": float(
            np.mean(total_times)
        ),
        "end_to_end_time_std_s": float(
            np.std(total_times, ddof=1)
        ),
        "cells_per_s_mean": throughput_mean,
        "cells_per_s_std": float(
            np.std(throughputs, ddof=1)
        ),
        "speedup_vs_full": (
            throughput_mean
            / full_metrics["combined_throughput_mean"]
        ),
        "peak_gpu_memory_gb": peak_memory,
        "memory_reduction_vs_full": (
            1.0
            - peak_memory
            / full_metrics["peak_gpu_memory_gb"]
        ),
        "padding_overhead_ratio_mean": float(
            np.mean(padding_ratios)
        ),
        **downstream,
        **fidelity,
        "neighbor_recall_10": neighbor_recall_10,
        **{
            key: value
            for key, value in additional_summary.items()
            if key
            not in {"k_min", "k_max", "matched_fixed_k"}
        },
    }

    config_dir = output_dir / (
        f"tau{int(round(tau * 100)):03d}_{name}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        config_dir / "embeddings_and_labels.npz",
        train_embeddings=saved_train,
        test_embeddings=saved_test,
        train_labels=train_labels,
        test_labels=test_labels,
        test_predictions=predictions,
        train_selected_genes=dataset_train.gene_lengths,
        test_selected_genes=dataset_test.gene_lengths,
    )

    with (
        config_dir / "metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    class_rows = per_celltype_rows(
        tau=tau,
        method=name,
        test_labels=test_labels,
        predictions=predictions,
        gene_lengths=dataset_test.gene_lengths,
    )

    write_csv(
        config_dir / "per_celltype.csv",
        class_rows,
    )

    print(
        f"{name}, tau={tau:.2f}: "
        f"accuracy={summary['accuracy']:.6f}, "
        f"macro-F1={summary['macro_f1']:.6f}, "
        f"cosine={summary['embedding_cosine_mean']:.6f}, "
        f"neighbor-R@10={summary['neighbor_recall_10']:.6f}"
    )

    return summary, timing_rows


def main() -> int:
    args = parse_args()

    for path in (
        args.repo,
        args.model_dir / "args.json",
        args.model_dir / "vocab.json",
        args.model_dir / "best_model.pt",
        args.train,
        args.test,
        args.full_reference,
        args.full_metrics,
        args.fixed_grid_script,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if (
        (args.output_dir / "summary.csv").exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output already exists: {args.output_dir}. "
            "Use --overwrite to replace it."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_module(args.fixed_grid_script)

    sys.path.insert(0, str(args.repo))

    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda:0")

    print("=" * 80)
    print("scGPT PANCREAS ADAGENEBUDGET")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("External FlashAttention: OFF")
    print("AMP FP16: ON")
    print(f"Taus: {args.taus}")
    print(f"K_min={args.k_min}, K_max={args.k_max}")
    print("Matched fixed-K source: train adaptive mean only")

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)

    if not np.array_equal(
        train.var_names.astype(str),
        test.var_names.astype(str),
    ):
        raise RuntimeError("Train/test gene order mismatch")

    train_labels = (
        train.obs[args.label_col].astype(str).to_numpy()
    )
    test_labels = (
        test.obs[args.label_col].astype(str).to_numpy()
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

    idf = base.compute_idf(train_matrix)

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

    checkpoint = base.load_checkpoint(
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
    full_test_embeddings = (
        full_data["test_embeddings"]
        .astype(np.float32)
    )

    with args.full_metrics.open(
        "r",
        encoding="utf-8",
    ) as handle:
        full_metrics = json.load(handle)

    reference_neighbors_10 = base.nearest_neighbors(
        full_test_embeddings,
        k=10,
    )

    summary_rows = []
    timing_rows = []

    for tau in args.taus:
        print()
        print("=" * 80)
        print(f"TAU={tau:.2f}")
        print("=" * 80)

        adaptive_train = AdaptiveSelectionDataset(
            matrix=train_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            idf=idf,
            tau=tau,
            k_min=args.k_min,
            k_max=args.k_max,
            get_row_function=base.get_row,
        )
        adaptive_test = AdaptiveSelectionDataset(
            matrix=test_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            idf=idf,
            tau=tau,
            k_min=args.k_min,
            k_max=args.k_max,
            get_row_function=base.get_row,
        )

        adaptive_selection_time = (
            adaptive_train.selection_time_s
            + adaptive_test.selection_time_s
        )

        matched_fixed_k = int(
            np.rint(
                np.mean(adaptive_train.gene_lengths)
            )
        )

        print(
            "Adaptive budgets: "
            f"train mean={np.mean(adaptive_train.gene_lengths):.2f}, "
            f"test mean={np.mean(adaptive_test.gene_lengths):.2f}, "
            f"matched fixed K={matched_fixed_k}"
        )

        adaptive_summary_extra = {
            "k_min": args.k_min,
            "k_max": args.k_max,
            "matched_fixed_k": matched_fixed_k,
            "train_retained_mass_mean": float(
                np.mean(adaptive_train.retained_mass)
            ),
            "test_retained_mass_mean": float(
                np.mean(adaptive_test.retained_mass)
            ),
            "train_retained_mass_min": float(
                np.min(adaptive_train.retained_mass)
            ),
            "test_retained_mass_min": float(
                np.min(adaptive_test.retained_mass)
            ),
            "train_lower_clipped_fraction": float(
                np.mean(adaptive_train.lower_clipped)
            ),
            "test_lower_clipped_fraction": float(
                np.mean(adaptive_test.lower_clipped)
            ),
            "train_upper_clipped_fraction": float(
                np.mean(adaptive_train.upper_clipped)
            ),
            "test_upper_clipped_fraction": float(
                np.mean(adaptive_test.upper_clipped)
            ),
        }

        adaptive_summary, adaptive_timings = (
            run_configuration(
                base=base,
                name="adaptive",
                tau=tau,
                dataset_train=adaptive_train,
                dataset_test=adaptive_test,
                max_length=args.k_max + 1,
                selection_time_s=adaptive_selection_time,
                model=model,
                vocab=vocab,
                model_config=model_config,
                pad_id=pad_id,
                pad_value=pad_value,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup_steps=args.warmup_steps,
                timing_repeats=args.timing_repeats,
                train_labels=train_labels,
                test_labels=test_labels,
                full_test_embeddings=full_test_embeddings,
                reference_neighbors_10=reference_neighbors_10,
                full_metrics=full_metrics,
                output_dir=args.output_dir,
                additional_summary=adaptive_summary_extra,
            )
        )

        fixed_train = base.FixedSelectionDataset(
            matrix=train_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            method="tfidf",
            budget=matched_fixed_k,
            seed=args.seed,
            idf=idf,
        )
        fixed_test = base.FixedSelectionDataset(
            matrix=test_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            method="tfidf",
            budget=matched_fixed_k,
            seed=args.seed,
            idf=idf,
        )

        fixed_selection_time = (
            fixed_train.selection_time_s
            + fixed_test.selection_time_s
        )

        fixed_summary, fixed_timings = run_configuration(
            base=base,
            name="matched_fixed_tfidf",
            tau=tau,
            dataset_train=fixed_train,
            dataset_test=fixed_test,
            max_length=matched_fixed_k + 1,
            selection_time_s=fixed_selection_time,
            model=model,
            vocab=vocab,
            model_config=model_config,
            pad_id=pad_id,
            pad_value=pad_value,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            warmup_steps=args.warmup_steps,
            timing_repeats=args.timing_repeats,
            train_labels=train_labels,
            test_labels=test_labels,
            full_test_embeddings=full_test_embeddings,
            reference_neighbors_10=reference_neighbors_10,
            full_metrics=full_metrics,
            output_dir=args.output_dir,
            additional_summary={
                "k_min": None,
                "k_max": matched_fixed_k,
                "matched_fixed_k": matched_fixed_k,
                "train_retained_mass_mean": np.nan,
                "test_retained_mass_mean": np.nan,
                "train_retained_mass_min": np.nan,
                "test_retained_mass_min": np.nan,
                "train_lower_clipped_fraction": np.nan,
                "test_lower_clipped_fraction": np.nan,
                "train_upper_clipped_fraction": np.nan,
                "test_upper_clipped_fraction": np.nan,
            },
        )

        summary_rows.extend(
            [adaptive_summary, fixed_summary]
        )
        timing_rows.extend(
            adaptive_timings + fixed_timings
        )

        np.savez_compressed(
            args.output_dir
            / f"tau{int(round(tau * 100)):03d}_budgets.npz",
            train_adaptive_k=adaptive_train.gene_lengths,
            test_adaptive_k=adaptive_test.gene_lengths,
            train_raw_k=adaptive_train.raw_budgets,
            test_raw_k=adaptive_test.raw_budgets,
            train_retained_mass=adaptive_train.retained_mass,
            test_retained_mass=adaptive_test.retained_mass,
            train_labels=train_labels,
            test_labels=test_labels,
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
