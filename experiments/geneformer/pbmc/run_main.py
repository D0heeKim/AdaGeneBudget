#!/usr/bin/env python3
"""Run the full Geneformer PBMC P5-held-out benchmark.

Frozen protocol
---------------
Checkpoint
    Geneformer V1-10M
    BertForMaskedLM, FP32
    hidden_states[5]
    mean non-padding pooling

Split
    Train donors: P1, P2, P3, P4, P6, P7, P8
    Test donor: P5

Methods
    1. Geneformer Native Sequential, max 2048
    2. Geneformer Native Bucketed, max 2048
    3. Matched Random fixed K=599, seeds 0--4
    4. Matched Fixed TF-IDF, K=599
    5. AdaGeneBudget, tau=0.90, Kmin=128, Kmax=600

Quality
    - Exact 5-NN cosine classification:
          train embeddings -> held-out test donor
    - Accuracy, macro-F1, weighted-F1, balanced accuracy
    - Per-cell-type F1
    - Row-wise cosine similarity to Native Bucketed
    - Exact held-out-test neighbor recall@10 relative to Native Bucketed

Timing
    - Test split only
    - Common batch size 32
    - FP32
    - Embedding-only time: model forward and CPU output transfer using
      precomputed method sequences.
    - Selection time: CPU recomputation of the method's actual selection
      rule from the common base cache, without disk writes.
    - End-to-end estimate:
          selection + bucketing + embedding
    - Both embedding-only and end-to-end throughput are reported.
    - Cache construction time and disk writing are not counted.

Important
---------
The method caches are used to guarantee the exact same selected sequences
for timing and quality. Selection time is nevertheless recomputed separately
and included in the end-to-end estimate, so precomputation does not give the
proposed methods a hidden timing advantage.

Outputs
-------
outputs/geneformer/pbmc_p5/main/
├── summary.csv
├── per_class_f1.csv
├── timing_runs.csv
├── quality_by_variant.csv
├── run_config.json
├── native_reference_neighbors10.npy
├── geneformer_native_bucketed/
│   ├── train_embeddings.npy
│   ├── test_embeddings.npy
│   └── quality_seed0.npz
├── matched_fixed_tfidf/
│   └── same
├── adaptive/
│   └── same
└── matched_random_fixed_k/
    ├── seed0/
    ├── seed1/
    ├── seed2/
    ├── seed3/
    └── seed4/

The Native Sequential row reuses Native Bucketed embeddings because the smoke
test established exact numerical equivalence after restoring cell order.
Only their timing and padding behavior differ.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForMaskedLM


PROJECT_DIR = Path(".")

DEFAULT_BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "pbmc_seurat_v4"
    / "p5_holdout"
    / "geneformer_cache"
    / "base"
)

DEFAULT_METHODS_DIR = (
    DEFAULT_BASE_DIR.parent
    / "methods"
)

DEFAULT_CHECKPOINT = Path(
    "./checkpoints/"
    "Geneformer-V1-10M"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "geneformer"
    / "pbmc_p5"
    / "main"
)

NATIVE_DIRNAME = "native_max2048"
ADAPTIVE_DIRNAME = (
    "adaptive_tau0p90_k128_k600"
)
FIXED_DIRNAME = (
    "matched_fixed_tfidf_k599"
)
RANDOM_DIRNAME = (
    "matched_random_fixed_k599"
)

METHOD_NATIVE_SEQUENTIAL = (
    "geneformer_native_sequential"
)
METHOD_NATIVE_BUCKETED = (
    "geneformer_native_bucketed"
)
METHOD_RANDOM = (
    "matched_random_fixed_k"
)
METHOD_FIXED = (
    "matched_fixed_tfidf"
)
METHOD_ADAPTIVE = "adaptive"

METHOD_LABEL = {
    METHOD_NATIVE_SEQUENTIAL: (
        "Geneformer Native Sequential"
    ),
    METHOD_NATIVE_BUCKETED: (
        "Geneformer Native Bucketed"
    ),
    METHOD_RANDOM: (
        "Matched Random"
    ),
    METHOD_FIXED: (
        "Fixed TF-IDF"
    ),
    METHOD_ADAPTIVE: (
        "AdaGeneBudget"
    ),
}

RANDOM_SEEDS = (0, 1, 2, 3, 4)
MATCHED_K = 599
TAU = 0.90
K_MIN = 128
K_MAX = 600
NATIVE_MAX_GENES = 2048

HIDDEN_STATE_INDEX = 5
EXPECTED_HIDDEN_SIZE = 256
NEIGHBOR_K = 10
CLASSIFIER_K = 5

EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957
EXPECTED_GENES = 14_809

SUMMARY_COLUMNS = [
    "method",
    "method_label",
    "cache",
    "tau",
    "fixed_k",
    "batching",
    "sampling",
    "quality_num_seeds",
    "timing_repeats",
    "selection_timing_repeats",
    "selection_in_dataloader",
    "selection_time_s",
    "selection_time_std_s",
    "bucketing_time_s",
    "train_mean_selected_genes",
    "test_mean_selected_genes",
    "train_median_selected_genes",
    "test_median_selected_genes",
    "end_to_end_time_mean_s",
    "end_to_end_time_std_s",
    "embedding_time_mean_s",
    "embedding_time_std_s",
    "cells_per_s_mean",
    "cells_per_s_std",
    "embedding_only_cells_per_s_mean",
    "embedding_only_cells_per_s_std",
    "peak_gpu_memory_gb",
    "padding_overhead_ratio_mean",
    "matched_fixed_k",
    "target_adaptive_train_mean",
    "matched_fixed_expected_mean",
    "reference_method",
    "accuracy",
    "accuracy_std",
    "macro_f1",
    "macro_f1_std",
    "weighted_f1",
    "weighted_f1_std",
    "balanced_accuracy",
    "balanced_accuracy_std",
    "embedding_cosine_mean",
    "embedding_cosine_mean_std",
    "embedding_cosine_median",
    "embedding_cosine_median_std",
    "embedding_cosine_min",
    "embedding_cosine_min_std",
    "neighbor_recall_10",
    "neighbor_recall_10_std",
    "speedup_vs_native_bucketed",
    "memory_reduction_vs_native_bucketed",
    "train_retained_mass_mean",
    "test_retained_mass_mean",
    "train_upper_clipped_fraction",
    "test_upper_clipped_fraction",
    "train_lower_clipped_fraction",
    "test_lower_clipped_fraction",
]


@dataclass(frozen=True)
class Variant:
    """One deterministic embedding/quality variant."""

    variant_id: str
    method: str
    seed: int | None
    cache_dir: Path
    batching: str
    sampling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
    )
    parser.add_argument(
        "--methods-dir",
        type=Path,
        default=DEFAULT_METHODS_DIR,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--selection-timing-repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--knn-query-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--neighbor-query-batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def parse_and_validate_args(
    args: argparse.Namespace,
) -> None:
    positive_fields = {
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "timing_repeats": args.timing_repeats,
        "selection_timing_repeats": (
            args.selection_timing_repeats
        ),
        "knn_query_batch_size": (
            args.knn_query_batch_size
        ),
        "neighbor_query_batch_size": (
            args.neighbor_query_batch_size
        ),
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must "
                "be positive."
            )

    if args.resume and args.overwrite:
        raise ValueError(
            "--resume and --overwrite cannot be "
            "used together."
        )


def prepare_output_dir(
    output_dir: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
            output_dir.mkdir(
                parents=True,
                exist_ok=False,
            )
            return
        if resume:
            return
        raise FileExistsError(
            f"{output_dir} already exists. Use "
            "--resume to continue or --overwrite "
            "to rebuild."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


def load_method_split(
    cache_dir: Path,
    split: str,
) -> dict[str, np.ndarray]:
    split_dir = cache_dir / split
    require_directory(split_dir)

    indptr = np.load(
        split_dir / "indptr.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gene_pos = np.load(
        split_dir / "gene_pos.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if indptr.ndim != 1:
        raise RuntimeError(
            f"{split_dir}: indptr must be 1D."
        )
    if gene_pos.ndim != 1:
        raise RuntimeError(
            f"{split_dir}: gene_pos must be 1D."
        )
    if int(indptr[0]) != 0:
        raise RuntimeError(
            f"{split_dir}: indptr must begin at 0."
        )
    if np.any(np.diff(indptr) < 0):
        raise RuntimeError(
            f"{split_dir}: non-monotonic indptr."
        )
    if int(indptr[-1]) != len(gene_pos):
        raise RuntimeError(
            f"{split_dir}: indptr/gene_pos mismatch."
        )

    return {
        "indptr": indptr,
        "gene_pos": gene_pos,
        "lengths": np.diff(indptr).astype(
            np.int64,
            copy=False,
        ),
    }


def load_inputs(
    base_dir: Path,
    methods_dir: Path,
) -> dict[str, Any]:
    require_directory(base_dir)
    require_directory(methods_dir)

    base_meta = read_json(
        base_dir / "meta.json"
    )
    if base_meta.get("status") != "PASS":
        raise RuntimeError(
            "Base cache status is not PASS."
        )

    token_ids = np.load(
        base_dir / "token_ids.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gene_median = np.load(
        base_dir / "gene_median.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    idf = np.load(
        base_dir / "idf.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if token_ids.shape != (EXPECTED_GENES,):
        raise RuntimeError(
            "Unexpected token_ids shape."
        )
    if gene_median.shape != (
        EXPECTED_GENES,
    ):
        raise RuntimeError(
            "Unexpected gene_median shape."
        )
    if idf.shape != (EXPECTED_GENES,):
        raise RuntimeError(
            "Unexpected IDF shape."
        )

    base_splits = {}

    for split, expected_cells in (
        ("train", EXPECTED_TRAIN_CELLS),
        ("test", EXPECTED_TEST_CELLS),
    ):
        split_dir = base_dir / split
        indptr = np.load(
            split_dir / "indptr.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        gene_pos = np.load(
            split_dir / "gene_pos.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        counts = np.load(
            split_dir / "counts.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        metadata_arrays = {}
        for name in (
            "obs_names",
            "labels_l1",
            "labels_l2",
            "donors",
            "times",
        ):
            metadata_arrays[name] = np.load(
                split_dir / f"{name}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )

        if indptr.shape != (
            expected_cells + 1,
        ):
            raise RuntimeError(
                f"{split}: invalid base indptr shape."
            )
        if int(indptr[-1]) != len(gene_pos):
            raise RuntimeError(
                f"{split}: base indptr/gene_pos "
                "mismatch."
            )
        if len(counts) != len(gene_pos):
            raise RuntimeError(
                f"{split}: base counts/gene_pos "
                "mismatch."
            )
        for name, values in (
            metadata_arrays.items()
        ):
            if len(values) != expected_cells:
                raise RuntimeError(
                    f"{split}: {name} length "
                    "mismatch."
                )

        base_splits[split] = {
            "indptr": indptr,
            "gene_pos": gene_pos,
            "counts": counts,
            "lengths": np.diff(
                indptr
            ).astype(
                np.int64,
                copy=False,
            ),
            **metadata_arrays,
        }

    native_dir = (
        methods_dir / NATIVE_DIRNAME
    )
    fixed_dir = (
        methods_dir / FIXED_DIRNAME
    )
    adaptive_dir = (
        methods_dir / ADAPTIVE_DIRNAME
    )
    random_root = (
        methods_dir / RANDOM_DIRNAME
    )

    for directory in (
        native_dir,
        fixed_dir,
        adaptive_dir,
        random_root,
    ):
        require_directory(directory)

    method_caches = {
        "native": {
            split: load_method_split(
                native_dir,
                split,
            )
            for split in ("train", "test")
        },
        "fixed": {
            split: load_method_split(
                fixed_dir,
                split,
            )
            for split in ("train", "test")
        },
        "adaptive": {
            split: load_method_split(
                adaptive_dir,
                split,
            )
            for split in ("train", "test")
        },
        "random": {
            seed: {
                split: load_method_split(
                    random_root / f"seed{seed}",
                    split,
                )
                for split in ("train", "test")
            }
            for seed in RANDOM_SEEDS
        },
    }

    matched_meta = read_json(
        methods_dir / "matched_budget.json"
    )
    adaptive_meta = read_json(
        adaptive_dir / "meta.json"
    )

    return {
        "base_meta": base_meta,
        "base_splits": base_splits,
        "method_caches": method_caches,
        "token_ids": token_ids,
        "gene_median": gene_median,
        "idf": idf,
        "paths": {
            "native": native_dir,
            "fixed": fixed_dir,
            "adaptive": adaptive_dir,
            "random": random_root,
        },
        "matched_meta": matched_meta,
        "adaptive_meta": adaptive_meta,
    }


def build_variants(
    inputs: dict[str, Any],
) -> list[Variant]:
    variants = [
        Variant(
            variant_id=(
                METHOD_NATIVE_BUCKETED
            ),
            method=(
                METHOD_NATIVE_BUCKETED
            ),
            seed=None,
            cache_dir=inputs["paths"][
                "native"
            ],
            batching="bucketed",
            sampling="native",
        ),
        Variant(
            variant_id=METHOD_FIXED,
            method=METHOD_FIXED,
            seed=None,
            cache_dir=inputs["paths"][
                "fixed"
            ],
            batching="bucketed",
            sampling="tfidf",
        ),
        Variant(
            variant_id=METHOD_ADAPTIVE,
            method=METHOD_ADAPTIVE,
            seed=None,
            cache_dir=inputs["paths"][
                "adaptive"
            ],
            batching="bucketed",
            sampling="adaptive_tfidf",
        ),
    ]

    for seed in RANDOM_SEEDS:
        variants.append(
            Variant(
                variant_id=(
                    f"{METHOD_RANDOM}_seed{seed}"
                ),
                method=METHOD_RANDOM,
                seed=seed,
                cache_dir=(
                    inputs["paths"]["random"]
                    / f"seed{seed}"
                ),
                batching="bucketed",
                sampling="random",
            )
        )

    return variants


def variant_output_dir(
    output_dir: Path,
    variant: Variant,
) -> Path:
    if variant.method == METHOD_RANDOM:
        assert variant.seed is not None
        return (
            output_dir
            / METHOD_RANDOM
            / f"seed{variant.seed}"
        )
    return output_dir / variant.method


def method_cache(
    inputs: dict[str, Any],
    variant: Variant,
    split: str,
) -> dict[str, np.ndarray]:
    if variant.method == (
        METHOD_NATIVE_BUCKETED
    ):
        return inputs[
            "method_caches"
        ]["native"][split]
    if variant.method == METHOD_FIXED:
        return inputs[
            "method_caches"
        ]["fixed"][split]
    if variant.method == (
        METHOD_ADAPTIVE
    ):
        return inputs[
            "method_caches"
        ]["adaptive"][split]
    if variant.method == METHOD_RANDOM:
        assert variant.seed is not None
        return inputs[
            "method_caches"
        ]["random"][variant.seed][split]
    raise KeyError(variant.method)


def row_positions(
    cache: dict[str, np.ndarray],
    row: int,
) -> np.ndarray:
    start = int(cache["indptr"][row])
    stop = int(cache["indptr"][row + 1])
    return np.asarray(
        cache["gene_pos"][start:stop],
        dtype=np.int64,
    )


def row_tokens(
    cache: dict[str, np.ndarray],
    row: int,
    token_ids: np.ndarray,
) -> np.ndarray:
    positions = row_positions(
        cache,
        row,
    )
    tokens = np.asarray(
        token_ids[positions],
        dtype=np.int64,
    )
    if len(tokens) == 0:
        raise RuntimeError(
            f"Empty token sequence at row {row}."
        )
    return tokens


def batching_order(
    lengths: np.ndarray,
    batching: str,
) -> np.ndarray:
    if batching == "sequential":
        return np.arange(
            len(lengths),
            dtype=np.int64,
        )
    if batching == "bucketed":
        return np.argsort(
            lengths,
            kind="stable",
        )
    raise ValueError(
        f"Unknown batching mode {batching!r}."
    )


def padding_overhead_ratio(
    lengths: np.ndarray,
    order: np.ndarray,
    batch_size: int,
) -> float:
    actual = int(
        np.sum(
            lengths,
            dtype=np.int64,
        )
    )
    padded = 0

    for start in range(
        0,
        len(order),
        batch_size,
    ):
        indices = order[
            start : start + batch_size
        ]
        batch_lengths = lengths[indices]
        padded += int(
            np.max(batch_lengths)
            * len(indices)
        )

    if actual <= 0:
        raise RuntimeError(
            "Non-positive actual token count."
        )
    return float(padded / actual)


def collate_rows(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = cache["lengths"][rows]
    maximum_length = int(
        np.max(lengths)
    )

    input_ids = torch.full(
        (
            len(rows),
            maximum_length,
        ),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (
            len(rows),
            maximum_length,
        ),
        dtype=torch.long,
        device=device,
    )

    for batch_index, row in enumerate(rows):
        tokens = row_tokens(
            cache=cache,
            row=int(row),
            token_ids=token_ids,
        )
        length = len(tokens)
        input_ids[
            batch_index,
            :length,
        ] = torch.as_tensor(
            tokens,
            dtype=torch.long,
            device=device,
        )
        attention_mask[
            batch_index,
            :length,
        ] = 1

    return input_ids, attention_mask


@torch.inference_mode()
def model_batch_embeddings(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
) -> np.ndarray:
    (
        input_ids,
        attention_mask,
    ) = collate_rows(
        cache=cache,
        rows=rows,
        token_ids=token_ids,
        pad_token_id=pad_token_id,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model did not return hidden states."
        )
    if len(hidden_states) <= (
        HIDDEN_STATE_INDEX
    ):
        raise RuntimeError(
            "Requested hidden-state index is "
            "not available."
        )

    hidden = hidden_states[
        HIDDEN_STATE_INDEX
    ]
    mask = attention_mask.unsqueeze(
        -1
    ).to(hidden.dtype)

    pooled = (
        (hidden * mask).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    )

    return (
        pooled.detach()
        .cpu()
        .float()
        .numpy()
    )


def warmup_model(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    order: np.ndarray,
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
) -> None:
    number_of_batches = min(
        warmup_batches,
        math.ceil(
            len(order) / batch_size
        ),
    )

    for batch_index in range(
        number_of_batches
    ):
        start = batch_index * batch_size
        rows = order[
            start : start + batch_size
        ]
        _ = model_batch_embeddings(
            model=model,
            cache=cache,
            rows=rows,
            token_ids=token_ids,
            pad_token_id=pad_token_id,
            device=device,
        )

    torch.cuda.synchronize(device)


def time_embedding_pass(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    order: np.ndarray,
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> float:
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    for start in range(
        0,
        len(order),
        batch_size,
    ):
        rows = order[
            start : start + batch_size
        ]
        _ = model_batch_embeddings(
            model=model,
            cache=cache,
            rows=rows,
            token_ids=token_ids,
            pad_token_id=pad_token_id,
            device=device,
        )

    torch.cuda.synchronize(device)
    return float(
        time.perf_counter() - started
    )


def measure_embedding_timing(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    warmup_batches: int,
    timing_repeats: int,
    batching: str,
) -> dict[str, Any]:
    lengths = cache["lengths"]

    bucketing_started = (
        time.perf_counter()
    )
    order = batching_order(
        lengths,
        batching,
    )
    bucketing_time = float(
        time.perf_counter()
        - bucketing_started
    )

    padding_ratio = (
        padding_overhead_ratio(
            lengths=lengths,
            order=order,
            batch_size=batch_size,
        )
    )

    warmup_model(
        model=model,
        cache=cache,
        order=order,
        token_ids=token_ids,
        pad_token_id=pad_token_id,
        device=device,
        batch_size=batch_size,
        warmup_batches=warmup_batches,
    )

    run_times = []
    peak_memory_gb = 0.0

    for repeat in range(
        timing_repeats
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            device
        )

        elapsed = time_embedding_pass(
            model=model,
            cache=cache,
            order=order,
            token_ids=token_ids,
            pad_token_id=pad_token_id,
            device=device,
            batch_size=batch_size,
        )
        run_times.append(elapsed)

        peak_memory_gb = max(
            peak_memory_gb,
            float(
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024**3)
            ),
        )

        print(
            f"    timing repeat "
            f"{repeat + 1}/{timing_repeats}: "
            f"{elapsed:.3f}s",
            flush=True,
        )

    return {
        "order": order,
        "bucketing_time_s": (
            bucketing_time
        ),
        "padding_overhead_ratio": (
            padding_ratio
        ),
        "embedding_times_s": run_times,
        "peak_gpu_memory_gb": (
            peak_memory_gb
        ),
    }


def native_order(
    positions: np.ndarray,
    counts: np.ndarray,
    gene_median: np.ndarray,
) -> np.ndarray:
    scores = (
        counts.astype(
            np.float64,
            copy=False,
        )
        / gene_median[positions]
    )
    return np.lexsort(
        (
            positions,
            -scores,
        )
    )


def tfidf_order(
    positions: np.ndarray,
    counts: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    scores = (
        counts.astype(
            np.float64,
            copy=False,
        )
        * idf[positions]
    )
    return np.lexsort(
        (
            positions,
            -scores,
        )
    )


def base_row(
    base_split: dict[str, np.ndarray],
    row: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = int(
        base_split["indptr"][row]
    )
    stop = int(
        base_split["indptr"][row + 1]
    )

    positions = np.asarray(
        base_split["gene_pos"][
            start:stop
        ],
        dtype=np.int64,
    )
    counts = np.asarray(
        base_split["counts"][
            start:stop
        ],
        dtype=np.float64,
    )
    return positions, counts


def recompute_selection_once(
    method: str,
    base_split: dict[str, np.ndarray],
    gene_median: np.ndarray,
    idf: np.ndarray,
    seed: int | None,
) -> int:
    """Recompute all test selections without disk writes."""
    selected_total = 0

    rng = None
    if method == METHOD_RANDOM:
        if seed is None:
            raise RuntimeError(
                "Random method requires a seed."
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, 1, MATCHED_K]
            )
        )

    for row in range(
        len(base_split["lengths"])
    ):
        positions, counts = base_row(
            base_split,
            row,
        )
        n_expressed = len(positions)

        if method in (
            METHOD_NATIVE_BUCKETED,
            METHOD_NATIVE_SEQUENTIAL,
        ):
            order = native_order(
                positions,
                counts,
                gene_median,
            )
            chosen = positions[
                order[
                    : min(
                        NATIVE_MAX_GENES,
                        n_expressed,
                    )
                ]
            ]

        elif method == METHOD_FIXED:
            k = min(
                MATCHED_K,
                n_expressed,
            )
            select_order = tfidf_order(
                positions,
                counts,
                idf,
            )
            selected_indices = (
                select_order[:k]
            )
            selected_positions = positions[
                selected_indices
            ]
            selected_counts = counts[
                selected_indices
            ]
            final_order = native_order(
                selected_positions,
                selected_counts,
                gene_median,
            )
            chosen = selected_positions[
                final_order
            ]

        elif method == METHOD_ADAPTIVE:
            select_order = tfidf_order(
                positions,
                counts,
                idf,
            )
            scores = (
                counts * idf[positions]
            )
            sorted_scores = scores[
                select_order
            ]
            total_mass = float(
                np.sum(
                    sorted_scores,
                    dtype=np.float64,
                )
            )
            cumulative = np.cumsum(
                sorted_scores,
                dtype=np.float64,
            )
            raw_k = int(
                np.searchsorted(
                    cumulative,
                    TAU * total_mass,
                    side="left",
                )
                + 1
            )
            k = min(
                n_expressed,
                K_MAX,
                max(K_MIN, raw_k),
            )
            selected_indices = (
                select_order[:k]
            )
            selected_positions = positions[
                selected_indices
            ]
            selected_counts = counts[
                selected_indices
            ]
            final_order = native_order(
                selected_positions,
                selected_counts,
                gene_median,
            )
            chosen = selected_positions[
                final_order
            ]

        elif method == METHOD_RANDOM:
            assert rng is not None
            k = min(
                MATCHED_K,
                n_expressed,
            )
            if k == n_expressed:
                selected_indices = np.arange(
                    n_expressed,
                    dtype=np.int64,
                )
            else:
                selected_indices = rng.choice(
                    n_expressed,
                    size=k,
                    replace=False,
                )
            selected_positions = positions[
                selected_indices
            ]
            selected_counts = counts[
                selected_indices
            ]
            final_order = native_order(
                selected_positions,
                selected_counts,
                gene_median,
            )
            chosen = selected_positions[
                final_order
            ]

        else:
            raise ValueError(
                f"Unsupported method {method}."
            )

        selected_total += len(chosen)

    return int(selected_total)


def measure_selection_time(
    method: str,
    base_split: dict[str, np.ndarray],
    gene_median: np.ndarray,
    idf: np.ndarray,
    seed: int | None,
    repeats: int,
    expected_selected_total: int,
) -> list[float]:
    results = []

    for repeat in range(repeats):
        started = time.perf_counter()
        selected_total = (
            recompute_selection_once(
                method=method,
                base_split=base_split,
                gene_median=gene_median,
                idf=idf,
                seed=seed,
            )
        )
        elapsed = float(
            time.perf_counter()
            - started
        )

        if selected_total != (
            expected_selected_total
        ):
            raise RuntimeError(
                f"{method}: recomputed selected "
                f"total={selected_total}, expected "
                f"{expected_selected_total}."
            )

        results.append(elapsed)
        print(
            f"    selection repeat "
            f"{repeat + 1}/{repeats}: "
            f"{elapsed:.3f}s",
            flush=True,
        )

    return results


def valid_embedding_file(
    path: Path,
    expected_rows: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except Exception:
        return False

    return (
        array.shape
        == (
            expected_rows,
            EXPECTED_HIDDEN_SIZE,
        )
        and array.dtype
        == np.dtype(np.float32)
    )


def extract_embeddings_to_file(
    model: torch.nn.Module,
    cache: dict[str, np.ndarray],
    token_ids: np.ndarray,
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    resume: bool,
) -> None:
    expected_rows = len(
        cache["lengths"]
    )

    if (
        resume
        and valid_embedding_file(
            output_path,
            expected_rows,
        )
    ):
        print(
            "    [resume] valid embedding "
            f"file exists: {output_path}",
            flush=True,
        )
        return

    if output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected_rows,
            EXPECTED_HIDDEN_SIZE,
        ),
    )

    order = batching_order(
        cache["lengths"],
        "bucketed",
    )
    started = time.perf_counter()

    for start in range(
        0,
        len(order),
        batch_size,
    ):
        rows = order[
            start : start + batch_size
        ]
        embeddings = (
            model_batch_embeddings(
                model=model,
                cache=cache,
                rows=rows,
                token_ids=token_ids,
                pad_token_id=pad_token_id,
                device=device,
            )
        )

        output[rows] = embeddings

        completed = min(
            start + len(rows),
            len(order),
        )
        if (
            start == 0
            or completed == len(order)
            or completed % 10_000
            < batch_size
        ):
            elapsed = (
                time.perf_counter()
                - started
            )
            print(
                f"    embeddings "
                f"{completed:,}/{len(order):,} "
                f"({100 * completed / len(order):.1f}%); "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    output.flush()
    del output

    loaded = np.load(
        output_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    if loaded.shape != (
        expected_rows,
        EXPECTED_HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"Invalid extracted shape "
            f"{loaded.shape}."
        )

    # Chunked finite-value validation.
    for start in range(
        0,
        expected_rows,
        10_000,
    ):
        stop = min(
            start + 10_000,
            expected_rows,
        )
        if not np.all(
            np.isfinite(
                loaded[start:stop]
            )
        ):
            raise RuntimeError(
                f"Non-finite embeddings in "
                f"{output_path}, rows "
                f"{start}:{stop}."
            )


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    int,
]:
    require_directory(checkpoint)

    model = (
        AutoModelForMaskedLM
        .from_pretrained(
            checkpoint,
            local_files_only=True,
        )
    )
    model.eval()
    model.to(
        device=device,
        dtype=torch.float32,
    )

    hidden_size = int(
        getattr(
            model.config,
            "hidden_size",
            -1,
        )
    )
    if hidden_size != (
        EXPECTED_HIDDEN_SIZE
    ):
        raise RuntimeError(
            f"Unexpected hidden size "
            f"{hidden_size}."
        )

    pad_token_id = getattr(
        model.config,
        "pad_token_id",
        None,
    )
    if pad_token_id is None:
        raise RuntimeError(
            "Checkpoint config has no "
            "pad_token_id."
        )

    return model, int(pad_token_id)


def predict_exact_knn_cosine(
    train_embeddings_path: Path,
    test_embeddings_path: Path,
    train_labels: np.ndarray,
    device: torch.device,
    query_batch_size: int,
    k: int,
    n_classes: int,
) -> np.ndarray:
    train_array = np.load(
        train_embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    test_array = np.load(
        test_embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    train_tensor = torch.as_tensor(
        np.asarray(train_array),
        dtype=torch.float32,
        device=device,
    )
    train_tensor = F.normalize(
        train_tensor,
        p=2,
        dim=1,
    )

    train_label_tensor = (
        torch.as_tensor(
            train_labels,
            dtype=torch.long,
            device=device,
        )
    )

    predictions = np.empty(
        len(test_array),
        dtype=np.int64,
    )

    for start in range(
        0,
        len(test_array),
        query_batch_size,
    ):
        stop = min(
            start + query_batch_size,
            len(test_array),
        )

        query = torch.as_tensor(
            np.asarray(
                test_array[start:stop]
            ),
            dtype=torch.float32,
            device=device,
        )
        query = F.normalize(
            query,
            p=2,
            dim=1,
        )

        similarities = (
            query @ train_tensor.T
        )
        neighbor_indices = torch.topk(
            similarities,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        neighbor_labels = (
            train_label_tensor[
                neighbor_indices
            ]
        )

        votes = torch.zeros(
            (
                stop - start,
                n_classes,
            ),
            dtype=torch.int32,
            device=device,
        )
        votes.scatter_add_(
            dim=1,
            index=neighbor_labels,
            src=torch.ones_like(
                neighbor_labels,
                dtype=torch.int32,
            ),
        )
        batch_predictions = (
            torch.argmax(
                votes,
                dim=1,
            )
        )

        predictions[start:stop] = (
            batch_predictions
            .cpu()
            .numpy()
        )

        del (
            query,
            similarities,
            neighbor_indices,
            neighbor_labels,
            votes,
            batch_predictions,
        )

    del train_tensor
    del train_label_tensor
    torch.cuda.empty_cache()
    return predictions


def exact_neighbors_cosine(
    embeddings_path: Path,
    device: torch.device,
    query_batch_size: int,
    k: int,
) -> np.ndarray:
    array = np.load(
        embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    n_cells = len(array)

    database = torch.as_tensor(
        np.asarray(array),
        dtype=torch.float32,
        device=device,
    )
    database = F.normalize(
        database,
        p=2,
        dim=1,
    )

    neighbors = np.empty(
        (n_cells, k),
        dtype=np.int32,
    )

    for start in range(
        0,
        n_cells,
        query_batch_size,
    ):
        stop = min(
            start + query_batch_size,
            n_cells,
        )
        query = database[start:stop]
        similarities = (
            query @ database.T
        )

        local_rows = torch.arange(
            stop - start,
            device=device,
        )
        global_columns = torch.arange(
            start,
            stop,
            device=device,
        )
        similarities[
            local_rows,
            global_columns,
        ] = -torch.inf

        indices = torch.topk(
            similarities,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        neighbors[start:stop] = (
            indices.cpu()
            .numpy()
            .astype(
                np.int32,
                copy=False,
            )
        )

        del (
            similarities,
            indices,
            local_rows,
            global_columns,
        )

    del database
    torch.cuda.empty_cache()
    return neighbors


def rowwise_embedding_cosine(
    reference_path: Path,
    method_path: Path,
    chunk_size: int = 10_000,
) -> np.ndarray:
    reference = np.load(
        reference_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    method = np.load(
        method_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if reference.shape != method.shape:
        raise RuntimeError(
            "Embedding cosine shape mismatch."
        )

    values = np.empty(
        len(reference),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(reference),
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            len(reference),
        )
        first = np.asarray(
            reference[start:stop],
            dtype=np.float64,
        )
        second = np.asarray(
            method[start:stop],
            dtype=np.float64,
        )

        numerator = np.sum(
            first * second,
            axis=1,
        )
        denominator = (
            np.linalg.norm(
                first,
                axis=1,
            )
            * np.linalg.norm(
                second,
                axis=1,
            )
        )
        values[start:stop] = (
            numerator
            / np.maximum(
                denominator,
                1e-12,
            )
        ).astype(np.float32)

    return values


def neighbor_recall(
    reference: np.ndarray,
    method: np.ndarray,
) -> float:
    if reference.shape != method.shape:
        raise RuntimeError(
            "Neighbor arrays have different "
            "shapes."
        )

    recalls = np.empty(
        len(reference),
        dtype=np.float64,
    )

    for row in range(
        len(reference)
    ):
        recalls[row] = (
            len(
                np.intersect1d(
                    reference[row],
                    method[row],
                    assume_unique=True,
                )
            )
            / reference.shape[1]
        )

    return float(np.mean(recalls))


def quality_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
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


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    variant: Variant,
) -> list[dict[str, Any]]:
    rows = []

    for class_id, class_name in enumerate(
        label_encoder.classes_
    ):
        binary_true = (
            y_true == class_id
        ).astype(np.int64)
        binary_pred = (
            y_pred == class_id
        ).astype(np.int64)

        rows.append(
            {
                "method": variant.method,
                "method_label": (
                    METHOD_LABEL[
                        variant.method
                    ]
                ),
                "seed": variant.seed,
                "cell_type": str(
                    class_name
                ),
                "support": int(
                    np.sum(binary_true)
                ),
                "f1": float(
                    f1_score(
                        binary_true,
                        binary_pred,
                        zero_division=0,
                    )
                ),
            }
        )

    return rows


def save_quality_npz(
    path: Path,
    inputs: dict[str, Any],
    y_pred_text: np.ndarray,
    test_embeddings_path: Path,
) -> None:
    test_embeddings = np.load(
        test_embeddings_path,
        mmap_mode="r",
        allow_pickle=False,
    )
    train_base = inputs[
        "base_splits"
    ]["train"]
    test_base = inputs[
        "base_splits"
    ]["test"]

    np.savez_compressed(
        path,
        train_labels=np.asarray(
            train_base["labels_l1"]
        ).astype(str),
        test_labels=np.asarray(
            test_base["labels_l1"]
        ).astype(str),
        test_predictions=np.asarray(
            y_pred_text
        ).astype(str),
        test_embeddings=np.asarray(
            test_embeddings
        ),
        test_donors=np.asarray(
            test_base["donors"]
        ).astype(str),
        test_times=np.asarray(
            test_base["times"]
        ).astype(str),
        test_obs_names=np.asarray(
            test_base["obs_names"]
        ).astype(str),
    )


def aggregate_mean_std(
    values: Iterable[float],
) -> tuple[float, float]:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )
    if len(array) == 0:
        return float("nan"), float("nan")
    return (
        float(np.mean(array)),
        float(
            np.std(
                array,
                ddof=0,
            )
        ),
    )


def build_summary_rows(
    timing_records: pd.DataFrame,
    quality_records: pd.DataFrame,
    inputs: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    adaptive_meta = inputs[
        "adaptive_meta"
    ]
    matched_meta = inputs[
        "matched_meta"
    ]

    base_method_meta = {
        METHOD_NATIVE_SEQUENTIAL: {
            "cache": str(
                inputs["paths"][
                    "native"
                ].resolve()
            ),
            "tau": np.nan,
            "fixed_k": (
                NATIVE_MAX_GENES
            ),
            "batching": "sequential",
            "sampling": "native",
            "train_lengths": (
                inputs[
                    "method_caches"
                ]["native"]["train"][
                    "lengths"
                ]
            ),
            "test_lengths": (
                inputs[
                    "method_caches"
                ]["native"]["test"][
                    "lengths"
                ]
            ),
        },
        METHOD_NATIVE_BUCKETED: {
            "cache": str(
                inputs["paths"][
                    "native"
                ].resolve()
            ),
            "tau": np.nan,
            "fixed_k": (
                NATIVE_MAX_GENES
            ),
            "batching": "bucketed",
            "sampling": "native",
            "train_lengths": (
                inputs[
                    "method_caches"
                ]["native"]["train"][
                    "lengths"
                ]
            ),
            "test_lengths": (
                inputs[
                    "method_caches"
                ]["native"]["test"][
                    "lengths"
                ]
            ),
        },
        METHOD_RANDOM: {
            "cache": str(
                inputs["paths"][
                    "random"
                ].resolve()
            ),
            "tau": np.nan,
            "fixed_k": MATCHED_K,
            "batching": "bucketed",
            "sampling": "random",
            "train_lengths": (
                inputs[
                    "method_caches"
                ]["random"][0]["train"][
                    "lengths"
                ]
            ),
            "test_lengths": (
                inputs[
                    "method_caches"
                ]["random"][0]["test"][
                    "lengths"
                ]
            ),
        },
        METHOD_FIXED: {
            "cache": str(
                inputs["paths"][
                    "fixed"
                ].resolve()
            ),
            "tau": np.nan,
            "fixed_k": MATCHED_K,
            "batching": "bucketed",
            "sampling": "tfidf",
            "train_lengths": (
                inputs[
                    "method_caches"
                ]["fixed"]["train"][
                    "lengths"
                ]
            ),
            "test_lengths": (
                inputs[
                    "method_caches"
                ]["fixed"]["test"][
                    "lengths"
                ]
            ),
        },
        METHOD_ADAPTIVE: {
            "cache": str(
                inputs["paths"][
                    "adaptive"
                ].resolve()
            ),
            "tau": TAU,
            "fixed_k": np.nan,
            "batching": "bucketed",
            "sampling": "adaptive_tfidf",
            "train_lengths": (
                inputs[
                    "method_caches"
                ]["adaptive"]["train"][
                    "lengths"
                ]
            ),
            "test_lengths": (
                inputs[
                    "method_caches"
                ]["adaptive"]["test"][
                    "lengths"
                ]
            ),
        },
    }

    rows = []

    for method in (
        METHOD_NATIVE_SEQUENTIAL,
        METHOD_NATIVE_BUCKETED,
        METHOD_RANDOM,
        METHOD_FIXED,
        METHOD_ADAPTIVE,
    ):
        timing = timing_records.loc[
            timing_records["method"]
            == method
        ].copy()
        quality = quality_records.loc[
            quality_records["method"]
            == method
        ].copy()

        if timing.empty:
            raise RuntimeError(
                f"Missing timing records for "
                f"{method}."
            )
        if quality.empty:
            raise RuntimeError(
                f"Missing quality records for "
                f"{method}."
            )

        meta = base_method_meta[method]
        train_lengths = meta[
            "train_lengths"
        ]
        test_lengths = meta[
            "test_lengths"
        ]

        selection_mean, selection_std = (
            aggregate_mean_std(
                timing[
                    "selection_time_s"
                ]
            )
        )
        embedding_mean, embedding_std = (
            aggregate_mean_std(
                timing[
                    "embedding_time_s"
                ]
            )
        )
        end_to_end_mean, end_to_end_std = (
            aggregate_mean_std(
                timing[
                    "end_to_end_time_s"
                ]
            )
        )
        cells_per_s_mean, cells_per_s_std = (
            aggregate_mean_std(
                timing[
                    "cells_per_s"
                ]
            )
        )
        (
            embedding_cells_per_s_mean,
            embedding_cells_per_s_std,
        ) = aggregate_mean_std(
            timing[
                "embedding_only_cells_per_s"
            ]
        )

        row = {
            "method": method,
            "method_label": (
                METHOD_LABEL[method]
            ),
            "cache": meta["cache"],
            "tau": meta["tau"],
            "fixed_k": meta[
                "fixed_k"
            ],
            "batching": meta[
                "batching"
            ],
            "sampling": meta[
                "sampling"
            ],
            "quality_num_seeds": int(
                quality["seed"]
                .nunique(
                    dropna=False
                )
            ),
            "timing_repeats": (
                args.timing_repeats
            ),
            "selection_timing_repeats": (
                args.selection_timing_repeats
            ),
            "selection_in_dataloader": False,
            "selection_time_s": (
                selection_mean
            ),
            "selection_time_std_s": (
                selection_std
            ),
            "bucketing_time_s": float(
                timing[
                    "bucketing_time_s"
                ].mean()
            ),
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
            "end_to_end_time_mean_s": (
                end_to_end_mean
            ),
            "end_to_end_time_std_s": (
                end_to_end_std
            ),
            "embedding_time_mean_s": (
                embedding_mean
            ),
            "embedding_time_std_s": (
                embedding_std
            ),
            "cells_per_s_mean": (
                cells_per_s_mean
            ),
            "cells_per_s_std": (
                cells_per_s_std
            ),
            "embedding_only_cells_per_s_mean": (
                embedding_cells_per_s_mean
            ),
            "embedding_only_cells_per_s_std": (
                embedding_cells_per_s_std
            ),
            "peak_gpu_memory_gb": float(
                timing[
                    "peak_gpu_memory_gb"
                ].max()
            ),
            "padding_overhead_ratio_mean": float(
                timing[
                    "padding_overhead_ratio"
                ].mean()
            ),
            "matched_fixed_k": (
                MATCHED_K
            ),
            "target_adaptive_train_mean": float(
                matched_meta[
                    "target_adaptive_train_mean"
                ]
            ),
            "matched_fixed_expected_mean": float(
                matched_meta[
                    "matched_fixed_expected_train_mean"
                ]
            ),
            "reference_method": (
                METHOD_NATIVE_BUCKETED
            ),
        }

        for metric in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "embedding_cosine_mean",
            "embedding_cosine_median",
            "embedding_cosine_min",
            "neighbor_recall_10",
        ):
            mean, std = aggregate_mean_std(
                quality[metric]
            )
            row[metric] = mean
            row[f"{metric}_std"] = std

        if method == METHOD_ADAPTIVE:
            row.update(
                {
                    "train_retained_mass_mean": float(
                        adaptive_meta[
                            "train"
                        ][
                            "retained_mass_mean"
                        ]
                    ),
                    "test_retained_mass_mean": float(
                        adaptive_meta[
                            "test"
                        ][
                            "retained_mass_mean"
                        ]
                    ),
                    "train_upper_clipped_fraction": float(
                        adaptive_meta[
                            "train"
                        ][
                            "upper_clipped_fraction"
                        ]
                    ),
                    "test_upper_clipped_fraction": float(
                        adaptive_meta[
                            "test"
                        ][
                            "upper_clipped_fraction"
                        ]
                    ),
                    "train_lower_clipped_fraction": float(
                        adaptive_meta[
                            "train"
                        ][
                            "lower_clipped_fraction"
                        ]
                    ),
                    "test_lower_clipped_fraction": float(
                        adaptive_meta[
                            "test"
                        ][
                            "lower_clipped_fraction"
                        ]
                    ),
                }
            )
        else:
            row.update(
                {
                    "train_retained_mass_mean": np.nan,
                    "test_retained_mass_mean": np.nan,
                    "train_upper_clipped_fraction": np.nan,
                    "test_upper_clipped_fraction": np.nan,
                    "train_lower_clipped_fraction": np.nan,
                    "test_lower_clipped_fraction": np.nan,
                }
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    reference = summary.loc[
        summary["method"]
        == METHOD_NATIVE_BUCKETED
    ].iloc[0]

    summary[
        "speedup_vs_native_bucketed"
    ] = (
        reference[
            "end_to_end_time_mean_s"
        ]
        / summary[
            "end_to_end_time_mean_s"
        ]
    )
    summary[
        "memory_reduction_vs_native_bucketed"
    ] = (
        1.0
        - summary[
            "peak_gpu_memory_gb"
        ]
        / reference[
            "peak_gpu_memory_gb"
        ]
    )

    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan

    return summary[
        SUMMARY_COLUMNS
    ]


def main() -> int:
    args = parse_args()
    parse_and_validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    prepare_output_dir(
        args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    inputs = load_inputs(
        args.base_dir,
        args.methods_dir,
    )
    variants = build_variants(inputs)

    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    run_config = {
        "status": "RUNNING",
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "base_dir": str(
            args.base_dir.resolve()
        ),
        "methods_dir": str(
            args.methods_dir.resolve()
        ),
        "output_dir": str(
            args.output_dir.resolve()
        ),
        "device": str(device),
        "dtype": "float32",
        "hidden_state_index": (
            HIDDEN_STATE_INDEX
        ),
        "pooling": (
            "mean_non_padding"
        ),
        "batch_size": (
            args.batch_size
        ),
        "warmup_batches": (
            args.warmup_batches
        ),
        "timing_repeats": (
            args.timing_repeats
        ),
        "selection_timing_repeats": (
            args.selection_timing_repeats
        ),
        "classifier": (
            "exact 5-NN cosine, train donors "
            "to held-out P5 test donor"
        ),
        "neighbor_metric": (
            "exact test-test cosine "
            "neighbor recall@10"
        ),
        "random_seeds": list(
            RANDOM_SEEDS
        ),
        "matched_k": MATCHED_K,
        "tau": TAU,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "native_max_genes": (
            NATIVE_MAX_GENES
        ),
        "selection_timing_note": (
            "Selection is recomputed from the "
            "common base cache without disk writes "
            "and added to end-to-end time."
        ),
    }
    write_json(
        args.output_dir
        / "run_config.json",
        run_config,
    )

    print("=" * 118)
    print(
        "GENEFORMER PBMC P5 FULL BENCHMARK"
    )
    print("=" * 118)
    print(
        "Checkpoint:",
        args.checkpoint,
    )
    print(
        "Train/Test:",
        EXPECTED_TRAIN_CELLS,
        "/",
        EXPECTED_TEST_CELLS,
    )
    print(
        "Batch size:",
        args.batch_size,
    )
    print(
        "Timing repeats:",
        args.timing_repeats,
    )
    print(
        "Quality variants:",
        len(variants),
    )

    model, pad_token_id = load_model(
        args.checkpoint,
        device,
    )
    run_config[
        "pad_token_id"
    ] = pad_token_id
    write_json(
        args.output_dir
        / "run_config.json",
        run_config,
    )

    timing_rows = []

    # Native selection is identical for sequential and bucketed.
    timing_targets = [
        (
            METHOD_NATIVE_SEQUENTIAL,
            None,
            inputs["method_caches"][
                "native"
            ]["test"],
            "sequential",
        ),
        (
            METHOD_NATIVE_BUCKETED,
            None,
            inputs["method_caches"][
                "native"
            ]["test"],
            "bucketed",
        ),
        (
            METHOD_FIXED,
            None,
            inputs["method_caches"][
                "fixed"
            ]["test"],
            "bucketed",
        ),
        (
            METHOD_ADAPTIVE,
            None,
            inputs["method_caches"][
                "adaptive"
            ]["test"],
            "bucketed",
        ),
    ]
    for seed in RANDOM_SEEDS:
        timing_targets.append(
            (
                METHOD_RANDOM,
                seed,
                inputs[
                    "method_caches"
                ]["random"][seed][
                    "test"
                ],
                "bucketed",
            )
        )

    for (
        method,
        seed,
        cache,
        batching,
    ) in timing_targets:
        print()
        print(
            f"[TIMING] {method}"
            + (
                f" seed={seed}"
                if seed is not None
                else ""
            ),
            flush=True,
        )

        timing = measure_embedding_timing(
            model=model,
            cache=cache,
            token_ids=inputs[
                "token_ids"
            ],
            pad_token_id=pad_token_id,
            device=device,
            batch_size=args.batch_size,
            warmup_batches=(
                args.warmup_batches
            ),
            timing_repeats=(
                args.timing_repeats
            ),
            batching=batching,
        )

        expected_selected_total = int(
            np.sum(
                cache["lengths"],
                dtype=np.int64,
            )
        )
        selection_times = (
            measure_selection_time(
                method=method,
                base_split=inputs[
                    "base_splits"
                ]["test"],
                gene_median=inputs[
                    "gene_median"
                ],
                idf=inputs["idf"],
                seed=seed,
                repeats=(
                    args.selection_timing_repeats
                ),
                expected_selected_total=(
                    expected_selected_total
                ),
            )
        )

        selection_mean = float(
            np.mean(selection_times)
        )

        for repeat_index, embedding_time in (
            enumerate(
                timing[
                    "embedding_times_s"
                ]
            )
        ):
            end_to_end = (
                selection_mean
                + timing[
                    "bucketing_time_s"
                ]
                + embedding_time
            )
            timing_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "repeat": repeat_index,
                    "batching": batching,
                    "selection_time_s": (
                        selection_mean
                    ),
                    "selection_time_std_s": float(
                        np.std(
                            selection_times,
                            ddof=0,
                        )
                    ),
                    "bucketing_time_s": (
                        timing[
                            "bucketing_time_s"
                        ]
                    ),
                    "embedding_time_s": (
                        embedding_time
                    ),
                    "end_to_end_time_s": (
                        end_to_end
                    ),
                    "cells_per_s": (
                        EXPECTED_TEST_CELLS
                        / end_to_end
                    ),
                    "embedding_only_cells_per_s": (
                        EXPECTED_TEST_CELLS
                        / embedding_time
                    ),
                    "peak_gpu_memory_gb": (
                        timing[
                            "peak_gpu_memory_gb"
                        ]
                    ),
                    "padding_overhead_ratio": (
                        timing[
                            "padding_overhead_ratio"
                        ]
                    ),
                }
            )

    timing_frame = pd.DataFrame(
        timing_rows
    )
    timing_frame.to_csv(
        args.output_dir
        / "timing_runs.csv",
        index=False,
    )

    # Extract train/test embeddings for each deterministic quality variant.
    for variant in variants:
        print()
        print(
            f"[EMBEDDINGS] {variant.variant_id}",
            flush=True,
        )

        variant_dir = variant_output_dir(
            args.output_dir,
            variant,
        )
        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for split in ("train", "test"):
            cache = method_cache(
                inputs,
                variant,
                split,
            )
            output_path = (
                variant_dir
                / f"{split}_embeddings.npy"
            )
            print(
                f"  {split}:",
                flush=True,
            )
            extract_embeddings_to_file(
                model=model,
                cache=cache,
                token_ids=inputs[
                    "token_ids"
                ],
                pad_token_id=pad_token_id,
                device=device,
                batch_size=args.batch_size,
                output_path=output_path,
                resume=args.resume,
            )

    # Model no longer needed. Release memory before exact KNN.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    train_labels_text = np.asarray(
        inputs["base_splits"][
            "train"
        ]["labels_l1"]
    ).astype(str)
    test_labels_text = np.asarray(
        inputs["base_splits"][
            "test"
        ]["labels_l1"]
    ).astype(str)

    label_encoder = LabelEncoder()
    train_labels = (
        label_encoder.fit_transform(
            train_labels_text
        )
    )
    unseen_test = (
        set(test_labels_text)
        - set(label_encoder.classes_)
    )
    if unseen_test:
        raise RuntimeError(
            "Test contains labels absent from "
            f"train: {sorted(unseen_test)}"
        )
    test_labels = (
        label_encoder.transform(
            test_labels_text
        )
    )
    n_classes = len(
        label_encoder.classes_
    )

    native_variant = next(
        variant
        for variant in variants
        if variant.method
        == METHOD_NATIVE_BUCKETED
    )
    native_dir = variant_output_dir(
        args.output_dir,
        native_variant,
    )
    native_test_path = (
        native_dir / "test_embeddings.npy"
    )

    reference_neighbors_path = (
        args.output_dir
        / "native_reference_neighbors10.npy"
    )
    if (
        args.resume
        and reference_neighbors_path.is_file()
    ):
        reference_neighbors = np.load(
            reference_neighbors_path,
            allow_pickle=False,
        )
        if reference_neighbors.shape != (
            EXPECTED_TEST_CELLS,
            NEIGHBOR_K,
        ):
            raise RuntimeError(
                "Invalid resumed native neighbor "
                "array shape."
            )
    else:
        print(
            "\n[QUALITY] Native reference "
            "test-test neighbors@10",
            flush=True,
        )
        reference_neighbors = (
            exact_neighbors_cosine(
                embeddings_path=(
                    native_test_path
                ),
                device=device,
                query_batch_size=(
                    args.neighbor_query_batch_size
                ),
                k=NEIGHBOR_K,
            )
        )
        np.save(
            reference_neighbors_path,
            reference_neighbors,
            allow_pickle=False,
        )

    quality_rows = []
    per_class_rows = []

    for variant in variants:
        print()
        print(
            f"[QUALITY] {variant.variant_id}",
            flush=True,
        )

        variant_dir = variant_output_dir(
            args.output_dir,
            variant,
        )
        train_path = (
            variant_dir
            / "train_embeddings.npy"
        )
        test_path = (
            variant_dir
            / "test_embeddings.npy"
        )

        predictions = (
            predict_exact_knn_cosine(
                train_embeddings_path=(
                    train_path
                ),
                test_embeddings_path=(
                    test_path
                ),
                train_labels=train_labels,
                device=device,
                query_batch_size=(
                    args.knn_query_batch_size
                ),
                k=CLASSIFIER_K,
                n_classes=n_classes,
            )
        )

        metrics = quality_metrics(
            y_true=test_labels,
            y_pred=predictions,
        )

        cosine_values = (
            rowwise_embedding_cosine(
                reference_path=(
                    native_test_path
                ),
                method_path=test_path,
            )
        )

        if (
            variant.method
            == METHOD_NATIVE_BUCKETED
        ):
            method_neighbors = (
                reference_neighbors
            )
            recall = 1.0
        else:
            method_neighbors = (
                exact_neighbors_cosine(
                    embeddings_path=test_path,
                    device=device,
                    query_batch_size=(
                        args.neighbor_query_batch_size
                    ),
                    k=NEIGHBOR_K,
                )
            )
            recall = neighbor_recall(
                reference_neighbors,
                method_neighbors,
            )

        quality_row = {
            "variant_id": (
                variant.variant_id
            ),
            "method": variant.method,
            "method_label": (
                METHOD_LABEL[
                    variant.method
                ]
            ),
            "seed": variant.seed,
            **metrics,
            "embedding_cosine_mean": float(
                np.mean(cosine_values)
            ),
            "embedding_cosine_median": float(
                np.median(cosine_values)
            ),
            "embedding_cosine_min": float(
                np.min(cosine_values)
            ),
            "neighbor_recall_10": float(
                recall
            ),
        }
        quality_rows.append(
            quality_row
        )

        per_class_rows.extend(
            per_class_metrics(
                y_true=test_labels,
                y_pred=predictions,
                label_encoder=(
                    label_encoder
                ),
                variant=variant,
            )
        )

        prediction_text = (
            label_encoder.inverse_transform(
                predictions
            )
        )
        quality_npz_name = (
            f"quality_seed"
            f"{variant.seed or 0}.npz"
        )
        save_quality_npz(
            path=(
                variant_dir
                / quality_npz_name
            ),
            inputs=inputs,
            y_pred_text=prediction_text,
            test_embeddings_path=(
                test_path
            ),
        )

        print(
            "    "
            f"accuracy={metrics['accuracy']:.6f}, "
            f"macro_f1={metrics['macro_f1']:.6f}, "
            f"balanced_acc="
            f"{metrics['balanced_accuracy']:.6f}, "
            f"cosine="
            f"{quality_row['embedding_cosine_mean']:.6f}, "
            f"R@10={recall:.6f}",
            flush=True,
        )

    # Native sequential quality is exactly the same as bucketed.
    native_quality = next(
        row
        for row in quality_rows
        if row["method"]
        == METHOD_NATIVE_BUCKETED
    )
    sequential_quality = dict(
        native_quality
    )
    sequential_quality[
        "variant_id"
    ] = METHOD_NATIVE_SEQUENTIAL
    sequential_quality[
        "method"
    ] = METHOD_NATIVE_SEQUENTIAL
    sequential_quality[
        "method_label"
    ] = METHOD_LABEL[
        METHOD_NATIVE_SEQUENTIAL
    ]
    quality_rows.append(
        sequential_quality
    )

    native_per_class = [
        row
        for row in per_class_rows
        if row["method"]
        == METHOD_NATIVE_BUCKETED
    ]
    for row in native_per_class:
        copied = dict(row)
        copied["method"] = (
            METHOD_NATIVE_SEQUENTIAL
        )
        copied["method_label"] = (
            METHOD_LABEL[
                METHOD_NATIVE_SEQUENTIAL
            ]
        )
        per_class_rows.append(
            copied
        )

    quality_frame = pd.DataFrame(
        quality_rows
    )
    quality_frame.to_csv(
        args.output_dir
        / "quality_by_variant.csv",
        index=False,
    )

    per_class_frame = pd.DataFrame(
        per_class_rows
    )
    per_class_frame.to_csv(
        args.output_dir
        / "per_class_f1.csv",
        index=False,
    )

    summary = build_summary_rows(
        timing_records=timing_frame,
        quality_records=quality_frame,
        inputs=inputs,
        args=args,
    )
    summary.to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )

    run_config["status"] = "PASS"
    run_config[
        "completed_at_unix"
    ] = time.time()
    run_config[
        "output_files"
    ] = {
        "summary": "summary.csv",
        "per_class_f1": (
            "per_class_f1.csv"
        ),
        "timing_runs": (
            "timing_runs.csv"
        ),
        "quality_by_variant": (
            "quality_by_variant.csv"
        ),
    }
    write_json(
        args.output_dir
        / "run_config.json",
        run_config,
    )

    print()
    print("=" * 118)
    print(
        "GENEFORMER PBMC FINAL SUMMARY"
    )
    print("=" * 118)
    display_columns = [
        "method_label",
        "test_mean_selected_genes",
        "cells_per_s_mean",
        "embedding_only_cells_per_s_mean",
        "peak_gpu_memory_gb",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "neighbor_recall_10",
        "speedup_vs_native_bucketed",
    ]
    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )
    print(
        "\nOutput:",
        args.output_dir,
    )
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
