#!/usr/bin/env python3
"""Run only the compute-matched random fixed-K baseline on Kang 2018.

The budget is read from the existing matched_fixed_tfidf result.
No Full, Native, TF-IDF, or AdaGeneBudget experiment is rerun.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
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
        "--existing-result-dir",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "kang_patient1015/main_csr"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./outputs/scgpt/"
            "kang_patient1015/matched_random_k399_csr"
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
        "--corrected-script",
        type=Path,
        default=Path(
            "./"
            "experiments/scgpt/_shared/runtime_utils.py"
        ),
    )

    parser.add_argument("--label-col", default="cell_type")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help=(
            "Optional override. By default, read the fixed K from "
            "matched_fixed_tfidf in the existing summary."
        ),
    )
    parser.add_argument(
        "--quality-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


class FixedOrderSampler(Sampler[int]):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices.tolist())

    def __len__(self) -> int:
        return int(len(self.indices))


class CollatorWithOriginalIDs:
    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        original_ids = torch.as_tensor(
            [
                int(example["id"].item())
                if torch.is_tensor(example["id"])
                else int(example["id"])
                for example in examples
            ],
            dtype=torch.long,
        )

        batch = self.collator(examples)
        batch["original_id"] = original_ids
        return batch


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_sequence_lengths(dataset: Any) -> np.ndarray:
    if hasattr(dataset, "sequence_lengths"):
        return np.asarray(
            dataset.sequence_lengths,
            dtype=np.int64,
        )

    if hasattr(dataset, "gene_lengths"):
        return (
            np.asarray(
                dataset.gene_lengths,
                dtype=np.int64,
            )
            + 1
        )

    raise AttributeError(
        "Dataset has neither sequence_lengths nor gene_lengths"
    )


def make_bucketed_loader(
    *,
    dataset: Any,
    collator: Any,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, float]:
    start = time.perf_counter()

    sequence_lengths = get_sequence_lengths(dataset)
    sorted_indices = np.argsort(
        sequence_lengths,
        kind="stable",
    )

    bucketing_time_s = time.perf_counter() - start

    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": FixedOrderSampler(sorted_indices),
        "collate_fn": CollatorWithOriginalIDs(collator),
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": generator,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs), bucketing_time_s


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    return float(
        np.std(
            np.asarray(values, dtype=np.float64),
            ddof=1,
        )
    )


def make_random_datasets(
    *,
    fixed_module: Any,
    train_matrix: Any,
    test_matrix: Any,
    gene_ids: np.ndarray,
    cls_id: int,
    cls_value: float,
    budget: int,
    seed: int,
) -> tuple[Any, Any]:
    # Random selection does not use IDF, but the existing dataset
    # constructor accepts this argument.
    dummy_idf = np.ones(
        len(gene_ids),
        dtype=np.float32,
    )

    train_dataset = fixed_module.FixedSelectionDataset(
        matrix=train_matrix,
        gene_ids=gene_ids,
        cls_id=cls_id,
        cls_value=cls_value,
        method="random",
        budget=budget,
        seed=seed,
        idf=dummy_idf,
    )
    test_dataset = fixed_module.FixedSelectionDataset(
        matrix=test_matrix,
        gene_ids=gene_ids,
        cls_id=cls_id,
        cls_value=cls_value,
        method="random",
        budget=budget,
        seed=seed + 100_000,
        idf=dummy_idf,
    )

    return train_dataset, test_dataset


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
        args.corrected_script,
        args.existing_result_dir / "summary.csv",
        args.existing_result_dir / "full_bucketed.npz",
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

    existing_summary = pd.read_csv(
        args.existing_result_dir / "summary.csv"
    )

    fixed_rows = existing_summary.loc[
        existing_summary["method"]
        == "matched_fixed_tfidf"
    ]

    if len(fixed_rows) != 1:
        raise RuntimeError(
            "Expected exactly one matched_fixed_tfidf row"
        )

    matched_fixed_k = int(
        fixed_rows.iloc[0]["fixed_k"]
    )
    budget = (
        int(args.budget)
        if args.budget is not None
        else matched_fixed_k
    )

    if budget != matched_fixed_k:
        print(
            "WARNING: budget override differs from "
            f"matched fixed K={matched_fixed_k}"
        )

    full_module = load_module(
        "kang_random_full_module",
        args.full_script,
    )
    fixed_module = load_module(
        "kang_random_fixed_module",
        args.fixed_script,
    )
    corrected = load_module(
        "kang_random_corrected_module",
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

    print("=" * 90)
    print("KANG COMPUTE-MATCHED RANDOM FIXED-K")
    print("=" * 90)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Budget:", budget)
    print("Selection: random nonzero genes")
    print("Batching: length-aware")
    print("Quality seeds:", args.quality_seeds)
    print("Timing repeats:", args.timing_repeats)

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)

    if not np.array_equal(
        train.var_names.astype(str),
        test.var_names.astype(str),
    ):
        raise RuntimeError(
            "Train/test gene ordering differs"
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
        "Matched genes:",
        f"{int(matched_mask.sum())}/{len(matched_mask)}",
    )
    print("Train cells:", train.n_obs)
    print("Test cells:", test.n_obs)

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

    collator = DataCollator(
        do_padding=True,
        pad_token_id=pad_id,
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=budget + 1,
        sampling=False,
        keep_first_n_tokens=1,
    )

    full_reference = np.load(
        args.existing_result_dir / "full_bucketed.npz",
        allow_pickle=True,
    )
    full_test_embeddings = (
        full_reference["test_embeddings"]
        .astype(np.float32)
    )

    if not np.array_equal(
        full_reference["test_labels"].astype(str),
        test_labels,
    ):
        raise RuntimeError(
            "Full reference labels do not match test labels"
        )

    reference_neighbors = corrected.nearest_neighbors(
        full_test_embeddings,
        k=10,
    )

    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "embedding_cosine_median",
        "embedding_cosine_min",
        "neighbor_recall_10",
    ]
    metric_values = {
        metric: []
        for metric in metric_names
    }
    quality_rows: list[dict[str, Any]] = []

    print()
    print("=" * 90)
    print("QUALITY EVALUATION")
    print("=" * 90)

    for seed in args.quality_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_dataset, test_dataset = (
            make_random_datasets(
                fixed_module=fixed_module,
                train_matrix=train_matrix,
                test_matrix=test_matrix,
                gene_ids=gene_ids,
                cls_id=vocab["<cls>"],
                cls_value=pad_value,
                budget=budget,
                seed=seed,
            )
        )

        train_loader, train_bucket_time = (
            make_bucketed_loader(
                dataset=train_dataset,
                collator=collator,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
            )
        )
        test_loader, test_bucket_time = (
            make_bucketed_loader(
                dataset=test_dataset,
                collator=collator,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed + 100_000,
            )
        )

        train_embeddings, _ = corrected.extract_once(
            model,
            train_loader,
            train_dataset,
            pad_id,
            device,
            model_config["embsize"],
        )
        test_embeddings, _ = corrected.extract_once(
            model,
            test_loader,
            test_dataset,
            pad_id,
            device,
            model_config["embsize"],
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

        selection_time_s = float(
            train_dataset.selection_time_s
            + test_dataset.selection_time_s
        )
        bucketing_time_s = float(
            train_bucket_time + test_bucket_time
        )

        row = {
            "method": "matched_random_fixed_k",
            "seed": seed,
            "fixed_k": budget,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
            **downstream,
            **fidelity,
        }
        quality_rows.append(row)

        for metric in metric_names:
            metric_values[metric].append(
                float(row[metric])
            )

        np.savez_compressed(
            args.output_dir
            / f"matched_random_seed{seed}.npz",
            train_labels=train_labels,
            test_labels=test_labels,
            test_predictions=predictions,
            test_embeddings=test_embeddings,
            train_selected_genes=train_dataset.gene_lengths,
            test_selected_genes=test_dataset.gene_lengths,
        )

        print(
            f"seed={seed}: "
            f"accuracy={row['accuracy']:.6f}, "
            f"macro-F1={row['macro_f1']:.6f}, "
            f"balanced={row['balanced_accuracy']:.6f}, "
            f"cosine={row['embedding_cosine_mean']:.6f}, "
            f"neighbor-R@10={row['neighbor_recall_10']:.6f}"
        )

    # Timing uses one fixed random realization, while quality
    # is averaged over all five random seeds.
    timing_seed = int(args.quality_seeds[0])

    timing_train_dataset, timing_test_dataset = (
        make_random_datasets(
            fixed_module=fixed_module,
            train_matrix=train_matrix,
            test_matrix=test_matrix,
            gene_ids=gene_ids,
            cls_id=vocab["<cls>"],
            cls_value=pad_value,
            budget=budget,
            seed=timing_seed,
        )
    )

    timing_train_loader, train_bucket_time = (
        make_bucketed_loader(
            dataset=timing_train_dataset,
            collator=collator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=999_001,
        )
    )
    timing_test_loader, test_bucket_time = (
        make_bucketed_loader(
            dataset=timing_test_dataset,
            collator=collator,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=999_002,
        )
    )

    selection_time_s = float(
        timing_train_dataset.selection_time_s
        + timing_test_dataset.selection_time_s
    )
    bucketing_time_s = float(
        train_bucket_time + test_bucket_time
    )

    corrected.warmup(
        model,
        timing_train_loader,
        pad_id,
        device,
        args.warmup_steps,
    )
    corrected.warmup(
        model,
        timing_test_loader,
        pad_id,
        device,
        args.warmup_steps,
    )

    total_cells = (
        len(timing_train_dataset)
        + len(timing_test_dataset)
    )

    timing_rows: list[dict[str, Any]] = []
    total_times: list[float] = []
    throughputs: list[float] = []
    peak_memories: list[float] = []
    padding_ratios: list[float] = []

    print()
    print("=" * 90)
    print("TIMING EVALUATION")
    print("=" * 90)

    for repeat in range(args.timing_repeats):
        _, train_timing = corrected.extract_once(
            model,
            timing_train_loader,
            timing_train_dataset,
            pad_id,
            device,
            model_config["embsize"],
        )
        _, test_timing = corrected.extract_once(
            model,
            timing_test_loader,
            timing_test_dataset,
            pad_id,
            device,
            model_config["embsize"],
        )

        embedding_time_s = float(
            train_timing["embedding_time_s"]
            + test_timing["embedding_time_s"]
        )
        end_to_end_time_s = (
            selection_time_s
            + bucketing_time_s
            + embedding_time_s
        )
        throughput = total_cells / end_to_end_time_s

        actual_tokens = int(
            train_timing["actual_sequence_tokens"]
            + test_timing["actual_sequence_tokens"]
        )
        padded_tokens = int(
            train_timing["padded_tokens_processed"]
            + test_timing["padded_tokens_processed"]
        )
        padding_ratio = (
            padded_tokens / actual_tokens
        )

        peak_memory = float(
            max(
                train_timing["peak_gpu_memory_gb"],
                test_timing["peak_gpu_memory_gb"],
            )
        )

        total_times.append(end_to_end_time_s)
        throughputs.append(throughput)
        peak_memories.append(peak_memory)
        padding_ratios.append(padding_ratio)

        timing_rows.append(
            {
                "method": "matched_random_fixed_k",
                "repeat": repeat,
                "fixed_k": budget,
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
            f"repeat={repeat + 1}: "
            f"{throughput:.2f} cells/s, "
            f"memory={peak_memory:.3f} GB, "
            f"padding={padding_ratio:.4f}"
        )

    train_lengths = np.asarray(
        timing_train_dataset.gene_lengths,
        dtype=np.int64,
    )
    test_lengths = np.asarray(
        timing_test_dataset.gene_lengths,
        dtype=np.int64,
    )

    summary: dict[str, Any] = {
        "method": "matched_random_fixed_k",
        "fixed_k": budget,
        "batching": "length_sorted",
        "sampling": "random_fixed_subset",
        "quality_num_seeds": len(args.quality_seeds),
        "timing_seed": timing_seed,
        "selection_in_dataloader": False,
        "selection_time_s": selection_time_s,
        "bucketing_time_s": bucketing_time_s,
        "train_mean_selected_genes": float(
            np.mean(train_lengths)
        ),
        "test_mean_selected_genes": float(
            np.mean(test_lengths)
        ),
        "train_median_selected_genes": float(
            np.median(train_lengths)
        ),
        "test_median_selected_genes": float(
            np.median(test_lengths)
        ),
        "end_to_end_time_mean_s": float(
            np.mean(total_times)
        ),
        "end_to_end_time_std_s": sample_std(
            total_times
        ),
        "cells_per_s_mean": float(
            np.mean(throughputs)
        ),
        "cells_per_s_std": sample_std(
            throughputs
        ),
        "peak_gpu_memory_gb": float(
            np.max(peak_memories)
        ),
        "padding_overhead_ratio_mean": float(
            np.mean(padding_ratios)
        ),
    }

    for metric in metric_names:
        values = metric_values[metric]
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_std"] = sample_std(values)

    full_rows = existing_summary.loc[
        existing_summary["method"] == "full_bucketed"
    ]

    if len(full_rows) != 1:
        raise RuntimeError(
            "Expected exactly one full_bucketed row"
        )

    full_throughput = float(
        full_rows.iloc[0]["cells_per_s_mean"]
    )
    full_memory = float(
        full_rows.iloc[0]["peak_gpu_memory_gb"]
    )

    summary["speedup_vs_full_bucketed"] = (
        summary["cells_per_s_mean"]
        / full_throughput
    )
    summary["memory_reduction_vs_full_bucketed"] = (
        1.0
        - summary["peak_gpu_memory_gb"]
        / full_memory
    )

    pd.DataFrame([summary]).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )
    pd.DataFrame(quality_rows).to_csv(
        args.output_dir / "seed_metrics.csv",
        index=False,
    )
    pd.DataFrame(timing_rows).to_csv(
        args.output_dir / "timing_repeats.csv",
        index=False,
    )

    with (
        args.output_dir / "config.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "resolved_budget": budget,
                "matched_fixed_k": matched_fixed_k,
                "matched_genes": int(matched_mask.sum()),
                "total_genes": int(len(matched_mask)),
            },
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    combined = pd.concat(
        [
            existing_summary,
            pd.DataFrame([summary]),
        ],
        ignore_index=True,
        sort=False,
    )

    columns = [
        "method",
        "fixed_k",
        "train_mean_selected_genes",
        "test_mean_selected_genes",
        "cells_per_s_mean",
        "peak_gpu_memory_gb",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "neighbor_recall_10",
    ]

    print()
    print("=" * 120)
    print("COMBINED RESULTS")
    print("=" * 120)
    print(
        combined[
            [
                column
                for column in columns
                if column in combined.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("FINAL STATUS: PASS")
    print("SUMMARY:", args.output_dir / "summary.csv")


if __name__ == "__main__":
    main()
