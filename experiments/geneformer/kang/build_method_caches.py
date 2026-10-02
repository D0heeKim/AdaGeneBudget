#!/usr/bin/env python3
"""Build method-specific Geneformer token caches for Kang 2018.

The common base cache must already exist. This script creates token sequences for:

- Geneformer Native Max-2048 (shared by sequential and bucketed inference),
- Matched Fixed TF-IDF,
- AdaGeneBudget (tau=0.90, Kmin=128, Kmax=600),
- Matched Random for seeds 0--4.

Important semantics
-------------------
1. IDF is loaded from the training-only base cache.
2. The matched fixed budget K* is chosen from the training split only.
3. TF-IDF or random sampling decides which genes survive.
4. Every surviving subset is re-ordered by the Geneformer native rank-value
   score before token IDs are saved:

       native_score = (raw_count / original_n_counts * 10,000) / gene_median

5. original_n_counts is never recomputed after vocabulary filtering or gene
   selection.
6. Ties are deterministic: Geneformer token ID ascending is the secondary key.

This script does not load the neural network and does not use a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_DIR = Path(".")
DEFAULT_BASE_CACHE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "base"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "methods"
)

EXPECTED_TRAIN_CELLS = 19_583
EXPECTED_TEST_CELLS = 5_090
EXPECTED_SUPPORTED_GENES = 12_288
EXPECTED_MODEL_INPUT_SIZE = 2_048

DEFAULT_TAU = 0.90
DEFAULT_K_MIN = 128
DEFAULT_K_MAX = 600
DEFAULT_RANDOM_SEEDS = (0, 1, 2, 3, 4)

SplitName = Literal["train", "test"]
MethodName = Literal["native", "fixed_tfidf", "adaptive", "random"]


@dataclass(frozen=True)
class BaseResources:
    base_dir: Path
    train_matrix: sp.csr_matrix
    test_matrix: sp.csr_matrix
    train_n_counts: np.ndarray
    test_n_counts: np.ndarray
    token_ids: np.ndarray
    gene_medians: np.ndarray
    idf: np.ndarray
    ensembl_ids: np.ndarray
    base_meta: dict[str, Any]


@dataclass(frozen=True)
class SelectionResult:
    indptr: np.ndarray
    gene_positions: np.ndarray
    token_ids: np.ndarray
    lengths: np.ndarray
    retained_mass: np.ndarray | None
    raw_adaptive_k: np.ndarray | None
    lower_clipped: np.ndarray | None
    upper_clipped: np.ndarray | None
    compute_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=DEFAULT_BASE_CACHE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=DEFAULT_K_MIN,
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=DEFAULT_K_MAX,
    )
    parser.add_argument(
        "--model-input-size",
        type=int,
        default=EXPECTED_MODEL_INPUT_SIZE,
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_RANDOM_SEEDS),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def stable_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty array.")

    float_values = values.astype(np.float64, copy=False)
    return {
        "count": int(values.size),
        "mean": float(np.mean(float_values)),
        "std": float(np.std(float_values, ddof=0)),
        "min": float(np.min(float_values)),
        "q01": float(np.quantile(float_values, 0.01)),
        "q05": float(np.quantile(float_values, 0.05)),
        "q25": float(np.quantile(float_values, 0.25)),
        "median": float(np.median(float_values)),
        "q75": float(np.quantile(float_values, 0.75)),
        "q95": float(np.quantile(float_values, 0.95)),
        "q99": float(np.quantile(float_values, 0.99)),
        "max": float(np.max(float_values)),
    }


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_base_resources(base_dir: Path) -> BaseResources:
    required = [
        base_dir / "meta.json",
        base_dir / "gene_token_ids.npy",
        base_dir / "gene_medians.npy",
        base_dir / "gene_ensembl_ids.npy",
        base_dir / "idf.npy",
        base_dir / "train" / "mapped_raw_counts_csr.npz",
        base_dir / "train" / "n_counts.npy",
        base_dir / "test" / "mapped_raw_counts_csr.npz",
        base_dir / "test" / "n_counts.npy",
    ]
    for path in required:
        require_file(path)

    base_meta = json.loads(
        (base_dir / "meta.json").read_text(encoding="utf-8")
    )
    if base_meta.get("status") != "PASS":
        raise RuntimeError(
            "Base cache meta.json does not report status=PASS."
        )

    train_matrix = sp.load_npz(
        base_dir / "train" / "mapped_raw_counts_csr.npz"
    ).tocsr()
    test_matrix = sp.load_npz(
        base_dir / "test" / "mapped_raw_counts_csr.npz"
    ).tocsr()

    for matrix in (train_matrix, test_matrix):
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()

    train_n_counts = np.load(
        base_dir / "train" / "n_counts.npy",
        allow_pickle=False,
    ).astype(np.float64, copy=False)
    test_n_counts = np.load(
        base_dir / "test" / "n_counts.npy",
        allow_pickle=False,
    ).astype(np.float64, copy=False)
    token_ids = np.load(
        base_dir / "gene_token_ids.npy",
        allow_pickle=False,
    ).astype(np.int64, copy=False)
    gene_medians = np.load(
        base_dir / "gene_medians.npy",
        allow_pickle=False,
    ).astype(np.float64, copy=False)
    idf = np.load(
        base_dir / "idf.npy",
        allow_pickle=False,
    ).astype(np.float64, copy=False)
    ensembl_ids = np.load(
        base_dir / "gene_ensembl_ids.npy",
        allow_pickle=False,
    ).astype(str)

    expected_train_shape = (
        EXPECTED_TRAIN_CELLS,
        EXPECTED_SUPPORTED_GENES,
    )
    expected_test_shape = (
        EXPECTED_TEST_CELLS,
        EXPECTED_SUPPORTED_GENES,
    )
    if train_matrix.shape != expected_train_shape:
        raise RuntimeError(
            f"Unexpected train matrix shape: {train_matrix.shape}; "
            f"expected {expected_train_shape}."
        )
    if test_matrix.shape != expected_test_shape:
        raise RuntimeError(
            f"Unexpected test matrix shape: {test_matrix.shape}; "
            f"expected {expected_test_shape}."
        )

    gene_count = train_matrix.shape[1]
    for name, array in [
        ("token_ids", token_ids),
        ("gene_medians", gene_medians),
        ("idf", idf),
        ("ensembl_ids", ensembl_ids),
    ]:
        if array.shape != (gene_count,):
            raise RuntimeError(
                f"{name} shape mismatch: {array.shape}; "
                f"expected {(gene_count,)}."
            )

    if train_n_counts.shape != (train_matrix.shape[0],):
        raise RuntimeError("train n_counts shape mismatch.")
    if test_n_counts.shape != (test_matrix.shape[0],):
        raise RuntimeError("test n_counts shape mismatch.")
    if np.any(train_n_counts <= 0) or np.any(test_n_counts <= 0):
        raise RuntimeError("n_counts must be strictly positive.")
    if np.any(gene_medians <= 0) or not np.isfinite(gene_medians).all():
        raise RuntimeError("Gene medians must be finite and positive.")
    if np.any(idf <= 0) or not np.isfinite(idf).all():
        raise RuntimeError("IDF must be finite and positive.")
    if not np.all(np.diff(token_ids) > 0):
        raise RuntimeError(
            "Base cache token IDs must be strictly increasing."
        )

    return BaseResources(
        base_dir=base_dir,
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        train_n_counts=train_n_counts,
        test_n_counts=test_n_counts,
        token_ids=token_ids,
        gene_medians=gene_medians,
        idf=idf,
        ensembl_ids=ensembl_ids,
        base_meta=base_meta,
    )


def deterministic_descending_order(
    scores: np.ndarray,
    token_ids: np.ndarray,
) -> np.ndarray:
    """Sort by score descending, then token ID ascending."""
    scores = np.asarray(scores, dtype=np.float64)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if scores.shape != token_ids.shape:
        raise ValueError("scores and token_ids must have the same shape.")
    return np.lexsort((token_ids, -scores)).astype(np.int64, copy=False)


def native_scores(
    raw_values: np.ndarray,
    gene_positions: np.ndarray,
    n_counts: float,
    gene_medians: np.ndarray,
) -> np.ndarray:
    if n_counts <= 0 or not math.isfinite(n_counts):
        raise RuntimeError(f"Invalid original n_counts: {n_counts}")

    values = np.asarray(raw_values, dtype=np.float64)
    positions = np.asarray(gene_positions, dtype=np.int64)
    scores = (
        values
        / float(n_counts)
        * 10_000.0
        / gene_medians[positions]
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite Geneformer native score detected.")
    return scores


def reorder_subset_by_native_rank(
    selected_local_indices: np.ndarray,
    row_positions: np.ndarray,
    row_values: np.ndarray,
    n_counts: float,
    token_ids: np.ndarray,
    gene_medians: np.ndarray,
) -> np.ndarray:
    selected_local_indices = np.asarray(
        selected_local_indices,
        dtype=np.int64,
    )
    if selected_local_indices.size == 0:
        return selected_local_indices

    selected_positions = row_positions[selected_local_indices]
    selected_values = row_values[selected_local_indices]
    scores = native_scores(
        selected_values,
        selected_positions,
        n_counts,
        gene_medians,
    )
    order = deterministic_descending_order(
        scores,
        token_ids[selected_positions],
    )
    return selected_local_indices[order]


def matched_fixed_budget(
    nonzero_counts: np.ndarray,
    target_mean: float,
) -> tuple[int, float]:
    nonzero_counts = np.asarray(nonzero_counts, dtype=np.int64)
    if nonzero_counts.size == 0:
        raise ValueError("nonzero_counts is empty.")
    if np.any(nonzero_counts < 0):
        raise ValueError("nonzero_counts contains negative values.")

    maximum = int(nonzero_counts.max())
    if maximum <= 0:
        raise RuntimeError("All training cells have zero supported genes.")

    lower = 1
    upper = maximum
    best_k = 1
    best_mean = float(np.minimum(nonzero_counts, 1).mean())
    best_difference = abs(best_mean - target_mean)

    while lower <= upper:
        candidate = (lower + upper) // 2
        realized_mean = float(
            np.minimum(nonzero_counts, candidate).mean()
        )
        difference = abs(realized_mean - target_mean)

        if (
            difference < best_difference
            or (
                math.isclose(difference, best_difference)
                and candidate < best_k
            )
        ):
            best_k = candidate
            best_mean = realized_mean
            best_difference = difference

        if realized_mean < target_mean:
            lower = candidate + 1
        else:
            upper = candidate - 1

    for candidate in range(
        max(1, best_k - 3),
        min(maximum, best_k + 3) + 1,
    ):
        realized_mean = float(
            np.minimum(nonzero_counts, candidate).mean()
        )
        difference = abs(realized_mean - target_mean)
        if (
            difference < best_difference
            or (
                math.isclose(difference, best_difference)
                and candidate < best_k
            )
        ):
            best_k = candidate
            best_mean = realized_mean
            best_difference = difference

    return int(best_k), float(best_mean)


def choose_adaptive_subset(
    row_positions: np.ndarray,
    row_values: np.ndarray,
    idf: np.ndarray,
    token_ids: np.ndarray,
    tau: float,
    k_min: int,
    k_max: int,
) -> tuple[np.ndarray, float, int, bool, bool]:
    n_genes = int(row_positions.size)
    if n_genes == 0:
        return (
            np.empty(0, dtype=np.int64),
            1.0,
            0,
            False,
            False,
        )

    tfidf_scores = (
        row_values.astype(np.float64, copy=False)
        * idf[row_positions]
    )
    if np.any(tfidf_scores < 0) or not np.isfinite(tfidf_scores).all():
        raise RuntimeError("Invalid TF-IDF scores.")

    total_mass = float(tfidf_scores.sum(dtype=np.float64))
    selection_order = deterministic_descending_order(
        tfidf_scores,
        token_ids[row_positions],
    )

    if total_mass <= 0:
        raw_k = n_genes
    else:
        cumulative = np.cumsum(
            tfidf_scores[selection_order],
            dtype=np.float64,
        )
        raw_k = int(
            np.searchsorted(
                cumulative,
                tau * total_mass,
                side="left",
            )
            + 1
        )

    lower_clipped = bool(n_genes >= k_min and raw_k < k_min)
    upper_clipped = bool(raw_k > k_max)
    selected_k = min(
        n_genes,
        k_max,
        max(k_min, raw_k),
    )
    selected_local = selection_order[:selected_k]
    retained_mass = float(
        tfidf_scores[selected_local].sum(dtype=np.float64)
        / max(total_mass, 1e-12)
    )

    return (
        selected_local.astype(np.int64, copy=False),
        retained_mass,
        raw_k,
        lower_clipped,
        upper_clipped,
    )


def choose_fixed_tfidf_subset(
    row_positions: np.ndarray,
    row_values: np.ndarray,
    idf: np.ndarray,
    token_ids: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    n_genes = int(row_positions.size)
    if n_genes == 0:
        return np.empty(0, dtype=np.int64), 1.0

    tfidf_scores = (
        row_values.astype(np.float64, copy=False)
        * idf[row_positions]
    )
    total_mass = float(tfidf_scores.sum(dtype=np.float64))
    order = deterministic_descending_order(
        tfidf_scores,
        token_ids[row_positions],
    )
    selected_local = order[: min(n_genes, budget)]
    retained_mass = float(
        tfidf_scores[selected_local].sum(dtype=np.float64)
        / max(total_mass, 1e-12)
    )
    return selected_local, retained_mass


def build_selection(
    matrix: sp.csr_matrix,
    n_counts: np.ndarray,
    token_ids: np.ndarray,
    gene_medians: np.ndarray,
    idf: np.ndarray,
    method: MethodName,
    split_name: SplitName,
    model_input_size: int,
    tau: float,
    k_min: int,
    k_max: int,
    fixed_k: int | None,
    random_seed: int | None,
    progress_every: int,
) -> SelectionResult:
    if matrix.shape[0] != n_counts.size:
        raise RuntimeError("matrix and n_counts row count mismatch.")

    if method in {"fixed_tfidf", "random"} and fixed_k is None:
        raise ValueError(f"fixed_k is required for method={method}.")
    if method == "random" and random_seed is None:
        raise ValueError("random_seed is required for random selection.")

    rng = (
        np.random.default_rng(random_seed)
        if method == "random"
        else None
    )

    n_rows = matrix.shape[0]
    lengths = np.empty(n_rows, dtype=np.int32)
    retained_mass = (
        np.empty(n_rows, dtype=np.float32)
        if method in {"fixed_tfidf", "adaptive"}
        else None
    )
    raw_adaptive_k = (
        np.empty(n_rows, dtype=np.int32)
        if method == "adaptive"
        else None
    )
    lower_clipped = (
        np.empty(n_rows, dtype=np.bool_)
        if method == "adaptive"
        else None
    )
    upper_clipped = (
        np.empty(n_rows, dtype=np.bool_)
        if method == "adaptive"
        else None
    )

    position_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []

    start_time = time.perf_counter()

    for row_index in range(n_rows):
        start = int(matrix.indptr[row_index])
        stop = int(matrix.indptr[row_index + 1])
        row_positions = matrix.indices[start:stop].astype(
            np.int64,
            copy=False,
        )
        row_values = matrix.data[start:stop].astype(
            np.float64,
            copy=False,
        )
        n_genes = int(row_positions.size)

        if method == "native":
            if n_genes == 0:
                selected_local = np.empty(0, dtype=np.int64)
            else:
                all_local = np.arange(n_genes, dtype=np.int64)
                ranked_all = reorder_subset_by_native_rank(
                    all_local,
                    row_positions,
                    row_values,
                    float(n_counts[row_index]),
                    token_ids,
                    gene_medians,
                )
                selected_local = ranked_all[:model_input_size]

        elif method == "fixed_tfidf":
            assert fixed_k is not None
            selected_local, retained = choose_fixed_tfidf_subset(
                row_positions,
                row_values,
                idf,
                token_ids,
                fixed_k,
            )
            retained_mass[row_index] = retained
            selected_local = reorder_subset_by_native_rank(
                selected_local,
                row_positions,
                row_values,
                float(n_counts[row_index]),
                token_ids,
                gene_medians,
            )

        elif method == "adaptive":
            (
                selected_local,
                retained,
                raw_k,
                was_lower_clipped,
                was_upper_clipped,
            ) = choose_adaptive_subset(
                row_positions,
                row_values,
                idf,
                token_ids,
                tau,
                k_min,
                k_max,
            )
            retained_mass[row_index] = retained
            raw_adaptive_k[row_index] = raw_k
            lower_clipped[row_index] = was_lower_clipped
            upper_clipped[row_index] = was_upper_clipped
            selected_local = reorder_subset_by_native_rank(
                selected_local,
                row_positions,
                row_values,
                float(n_counts[row_index]),
                token_ids,
                gene_medians,
            )

        elif method == "random":
            assert fixed_k is not None
            assert rng is not None
            selected_k = min(n_genes, fixed_k)
            if selected_k == 0:
                selected_local = np.empty(0, dtype=np.int64)
            elif selected_k == n_genes:
                selected_local = np.arange(n_genes, dtype=np.int64)
            else:
                selected_local = rng.choice(
                    n_genes,
                    size=selected_k,
                    replace=False,
                ).astype(np.int64, copy=False)
            selected_local = reorder_subset_by_native_rank(
                selected_local,
                row_positions,
                row_values,
                float(n_counts[row_index]),
                token_ids,
                gene_medians,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        selected_positions = row_positions[selected_local].astype(
            np.int32,
            copy=False,
        )
        selected_token_ids = token_ids[selected_positions].astype(
            np.int32,
            copy=False,
        )

        lengths[row_index] = int(selected_positions.size)
        position_parts.append(selected_positions)
        token_parts.append(selected_token_ids)

        completed = row_index + 1
        if (
            completed == 1
            or completed == n_rows
            or completed % progress_every == 0
        ):
            suffix = (
                f", seed={random_seed}"
                if random_seed is not None
                else ""
            )
            print(
                f"[select] {method}/{split_name}{suffix}: "
                f"{completed:,}/{n_rows:,} "
                f"({100.0 * completed / n_rows:.1f}%)",
                flush=True,
            )

    compute_time_s = time.perf_counter() - start_time

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths.astype(np.int64), out=indptr[1:])
    gene_positions = (
        np.concatenate(position_parts).astype(np.int32, copy=False)
        if position_parts
        else np.empty(0, dtype=np.int32)
    )
    flattened_token_ids = (
        np.concatenate(token_parts).astype(np.int32, copy=False)
        if token_parts
        else np.empty(0, dtype=np.int32)
    )

    if int(indptr[-1]) != gene_positions.size:
        raise RuntimeError("indptr and gene_positions length mismatch.")
    if gene_positions.size != flattened_token_ids.size:
        raise RuntimeError("gene_positions and token_ids length mismatch.")

    return SelectionResult(
        indptr=indptr,
        gene_positions=gene_positions,
        token_ids=flattened_token_ids,
        lengths=lengths,
        retained_mass=retained_mass,
        raw_adaptive_k=raw_adaptive_k,
        lower_clipped=lower_clipped,
        upper_clipped=upper_clipped,
        compute_time_s=float(compute_time_s),
    )


def method_directory_name(
    method: MethodName,
    fixed_k: int | None,
    random_seed: int | None,
) -> str:
    if method == "native":
        return "native_max2048"
    if method == "fixed_tfidf":
        assert fixed_k is not None
        return f"matched_fixed_tfidf_k{fixed_k}"
    if method == "adaptive":
        return "adaptive_tau0p90_k128_600"
    if method == "random":
        assert fixed_k is not None and random_seed is not None
        return f"matched_random_k{fixed_k}_seed{random_seed}"
    raise ValueError(method)


def save_selection(
    split_dir: Path,
    result: SelectionResult,
    method: MethodName,
    split_name: SplitName,
    settings: dict[str, Any],
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "indptr.npy", result.indptr, allow_pickle=False)
    np.save(
        split_dir / "gene_positions.npy",
        result.gene_positions,
        allow_pickle=False,
    )
    np.save(
        split_dir / "token_ids.npy",
        result.token_ids,
        allow_pickle=False,
    )
    np.save(
        split_dir / "lengths.npy",
        result.lengths,
        allow_pickle=False,
    )

    if result.retained_mass is not None:
        np.save(
            split_dir / "retained_mass.npy",
            result.retained_mass,
            allow_pickle=False,
        )
    if result.raw_adaptive_k is not None:
        np.save(
            split_dir / "raw_adaptive_k.npy",
            result.raw_adaptive_k,
            allow_pickle=False,
        )
    if result.lower_clipped is not None:
        np.save(
            split_dir / "lower_clipped.npy",
            result.lower_clipped,
            allow_pickle=False,
        )
    if result.upper_clipped is not None:
        np.save(
            split_dir / "upper_clipped.npy",
            result.upper_clipped,
            allow_pickle=False,
        )

    meta: dict[str, Any] = {
        "status": "PASS",
        "method": method,
        "split": split_name,
        "cells": int(result.lengths.size),
        "total_saved_tokens": int(result.token_ids.size),
        "length_statistics": summarize(result.lengths),
        "empty_sequences": int(np.sum(result.lengths == 0)),
        "selection_compute_time_s": float(result.compute_time_s),
        "settings": settings,
        "ordering": (
            "Geneformer native score descending; token ID ascending for ties"
        ),
    }

    if result.retained_mass is not None:
        meta["retained_mass_statistics"] = summarize(
            result.retained_mass
        )
    if result.raw_adaptive_k is not None:
        meta["raw_adaptive_k_statistics"] = summarize(
            result.raw_adaptive_k
        )
        meta["lower_clipped_count"] = int(
            np.sum(result.lower_clipped)
        )
        meta["lower_clipped_fraction"] = float(
            np.mean(result.lower_clipped)
        )
        meta["upper_clipped_count"] = int(
            np.sum(result.upper_clipped)
        )
        meta["upper_clipped_fraction"] = float(
            np.mean(result.upper_clipped)
        )

    stable_json_dump(meta, split_dir / "meta.json")
    return meta


def validate_saved_method(
    method_dir: Path,
    expected_train_cells: int,
    expected_test_cells: int,
    maximum_length: int,
) -> None:
    for split_name, expected_cells in [
        ("train", expected_train_cells),
        ("test", expected_test_cells),
    ]:
        split_dir = method_dir / split_name
        indptr = np.load(split_dir / "indptr.npy", allow_pickle=False)
        positions = np.load(
            split_dir / "gene_positions.npy",
            allow_pickle=False,
        )
        token_ids = np.load(
            split_dir / "token_ids.npy",
            allow_pickle=False,
        )
        lengths = np.load(
            split_dir / "lengths.npy",
            allow_pickle=False,
        )

        if indptr.shape != (expected_cells + 1,):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: indptr shape mismatch."
            )
        if lengths.shape != (expected_cells,):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: lengths shape mismatch."
            )
        if not np.array_equal(np.diff(indptr), lengths):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: lengths != diff(indptr)."
            )
        if int(indptr[-1]) != positions.size:
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: flattened size mismatch."
            )
        if positions.size != token_ids.size:
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: token size mismatch."
            )
        if np.any(lengths < 0) or np.any(lengths > maximum_length):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: invalid sequence length."
            )
        if np.any(positions < 0) or np.any(
            positions >= EXPECTED_SUPPORTED_GENES
        ):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: invalid gene position."
            )
        if np.any(token_ids < 2) or np.any(token_ids >= 25_426):
            raise RuntimeError(
                f"{method_dir.name}/{split_name}: invalid token ID."
            )

    print(f"[validate] {method_dir.name}: PASS", flush=True)


def build_and_save_method(
    resources: BaseResources,
    output_root: Path,
    method: MethodName,
    model_input_size: int,
    tau: float,
    k_min: int,
    k_max: int,
    fixed_k: int | None,
    random_seed_train: int | None,
    random_seed_test: int | None,
    progress_every: int,
) -> dict[str, Any]:
    directory_name = method_directory_name(
        method,
        fixed_k,
        random_seed_train,
    )
    method_dir = output_root / directory_name
    method_dir.mkdir(parents=True, exist_ok=False)

    settings = {
        "model_input_size": model_input_size,
        "tau": tau if method == "adaptive" else None,
        "k_min": k_min if method == "adaptive" else None,
        "k_max": k_max if method == "adaptive" else None,
        "fixed_k": fixed_k,
        "random_seed_train": random_seed_train,
        "random_seed_test": random_seed_test,
        "idf_source": "Kang training split only",
        "n_counts_source": (
            "original uncompressed .X row sum before vocabulary filtering"
        ),
    }

    train_result = build_selection(
        resources.train_matrix,
        resources.train_n_counts,
        resources.token_ids,
        resources.gene_medians,
        resources.idf,
        method,
        "train",
        model_input_size,
        tau,
        k_min,
        k_max,
        fixed_k,
        random_seed_train,
        progress_every,
    )
    test_result = build_selection(
        resources.test_matrix,
        resources.test_n_counts,
        resources.token_ids,
        resources.gene_medians,
        resources.idf,
        method,
        "test",
        model_input_size,
        tau,
        k_min,
        k_max,
        fixed_k,
        random_seed_test,
        progress_every,
    )

    train_meta = save_selection(
        method_dir / "train",
        train_result,
        method,
        "train",
        settings,
    )
    test_meta = save_selection(
        method_dir / "test",
        test_result,
        method,
        "test",
        settings,
    )

    maximum_length = (
        model_input_size
        if method == "native"
        else (k_max if method == "adaptive" else int(fixed_k))
    )
    validate_saved_method(
        method_dir,
        EXPECTED_TRAIN_CELLS,
        EXPECTED_TEST_CELLS,
        maximum_length,
    )

    root_meta = {
        "status": "PASS",
        "method": method,
        "directory_name": directory_name,
        "directory": str(method_dir.resolve()),
        "settings": settings,
        "train": train_meta,
        "test": test_meta,
        "combined_selection_compute_time_s": float(
            train_result.compute_time_s + test_result.compute_time_s
        ),
    }
    stable_json_dump(root_meta, method_dir / "meta.json")
    return root_meta


def main() -> int:
    args = parse_args()

    if not 0.0 < args.tau <= 1.0:
        raise ValueError("--tau must be in (0, 1].")
    if args.k_min <= 0:
        raise ValueError("--k-min must be positive.")
    if args.k_max < args.k_min:
        raise ValueError("--k-max must be >= --k-min.")
    if args.model_input_size <= 0:
        raise ValueError("--model-input-size must be positive.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if len(args.random_seeds) == 0:
        raise ValueError("At least one random seed is required.")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("Random seeds must be unique.")

    resources = load_base_resources(args.base_cache)

    if args.output_dir.exists():
        entries = list(args.output_dir.iterdir())
        if entries and not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. "
                "Use --overwrite only after confirming replacement is safe."
            )
        if entries and args.overwrite:
            shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96, flush=True)
    print("KANG GENEFORMER METHOD-SPECIFIC TOKEN CACHE BUILD", flush=True)
    print("=" * 96, flush=True)
    print("BASE CACHE:", args.base_cache, flush=True)
    print("OUTPUT:", args.output_dir, flush=True)
    print(
        f"SETTINGS: tau={args.tau}, Kmin={args.k_min}, "
        f"Kmax={args.k_max}, native_max={args.model_input_size}",
        flush=True,
    )

    all_meta: dict[str, dict[str, Any]] = {}

    native_meta = build_and_save_method(
        resources,
        args.output_dir,
        "native",
        args.model_input_size,
        args.tau,
        args.k_min,
        args.k_max,
        fixed_k=None,
        random_seed_train=None,
        random_seed_test=None,
        progress_every=args.progress_every,
    )
    all_meta[native_meta["directory_name"]] = native_meta

    adaptive_meta = build_and_save_method(
        resources,
        args.output_dir,
        "adaptive",
        args.model_input_size,
        args.tau,
        args.k_min,
        args.k_max,
        fixed_k=None,
        random_seed_train=None,
        random_seed_test=None,
        progress_every=args.progress_every,
    )
    all_meta[adaptive_meta["directory_name"]] = adaptive_meta

    adaptive_train_mean = float(
        adaptive_meta["train"]["length_statistics"]["mean"]
    )
    train_nnz = np.diff(resources.train_matrix.indptr).astype(np.int64)
    fixed_k, matched_expected_mean = matched_fixed_budget(
        train_nnz,
        adaptive_train_mean,
    )

    print(
        "[matched budget] "
        f"Ada train mean={adaptive_train_mean:.6f}; "
        f"K*={fixed_k}; "
        f"expected fixed train mean={matched_expected_mean:.6f}",
        flush=True,
    )

    fixed_meta = build_and_save_method(
        resources,
        args.output_dir,
        "fixed_tfidf",
        args.model_input_size,
        args.tau,
        args.k_min,
        args.k_max,
        fixed_k=fixed_k,
        random_seed_train=None,
        random_seed_test=None,
        progress_every=args.progress_every,
    )
    all_meta[fixed_meta["directory_name"]] = fixed_meta

    for seed in args.random_seeds:
        random_meta = build_and_save_method(
            resources,
            args.output_dir,
            "random",
            args.model_input_size,
            args.tau,
            args.k_min,
            args.k_max,
            fixed_k=fixed_k,
            random_seed_train=int(seed),
            random_seed_test=int(seed) + 100_000,
            progress_every=args.progress_every,
        )
        all_meta[random_meta["directory_name"]] = random_meta

    summary_rows: list[dict[str, Any]] = []
    for directory_name, meta in all_meta.items():
        row: dict[str, Any] = {
            "cache": directory_name,
            "method": meta["method"],
            "fixed_k": meta["settings"].get("fixed_k"),
            "random_seed_train": meta["settings"].get(
                "random_seed_train"
            ),
            "random_seed_test": meta["settings"].get(
                "random_seed_test"
            ),
            "train_mean_genes": meta["train"][
                "length_statistics"
            ]["mean"],
            "test_mean_genes": meta["test"][
                "length_statistics"
            ]["mean"],
            "train_median_genes": meta["train"][
                "length_statistics"
            ]["median"],
            "test_median_genes": meta["test"][
                "length_statistics"
            ]["median"],
            "selection_compute_time_s": meta[
                "combined_selection_compute_time_s"
            ],
        }
        if meta["method"] in {"adaptive", "fixed_tfidf"}:
            row["train_retained_mass_mean"] = meta["train"][
                "retained_mass_statistics"
            ]["mean"]
            row["test_retained_mass_mean"] = meta["test"][
                "retained_mass_statistics"
            ]["mean"]
        if meta["method"] == "adaptive":
            row["train_upper_clipped_fraction"] = meta["train"][
                "upper_clipped_fraction"
            ]
            row["test_upper_clipped_fraction"] = meta["test"][
                "upper_clipped_fraction"
            ]
            row["train_lower_clipped_fraction"] = meta["train"][
                "lower_clipped_fraction"
            ]
            row["test_lower_clipped_fraction"] = meta["test"][
                "lower_clipped_fraction"
            ]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )

    root_meta = {
        "status": "PASS",
        "dataset": "kang_2018_patient_1015_holdout",
        "base_cache": str(args.base_cache.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "native_reference": "Geneformer Native Max-2048",
        "tau": float(args.tau),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "model_input_size": int(args.model_input_size),
        "adaptive_train_mean_selected_genes": adaptive_train_mean,
        "matched_fixed_k": int(fixed_k),
        "matched_fixed_expected_train_mean": matched_expected_mean,
        "matched_budget_uses_test_statistics": False,
        "random_seeds": [int(seed) for seed in args.random_seeds],
        "random_test_seed_offset": 100_000,
        "method_caches": all_meta,
    }
    stable_json_dump(root_meta, args.output_dir / "meta.json")

    print("\n" + "=" * 96, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 96, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("MATCHED FIXED K:", fixed_k, flush=True)
    print("IDF SOURCE: TRAIN ONLY", flush=True)
    print("ALL SAVED SUBSETS REORDERED BY GENEFORMER NATIVE RANK", flush=True)
    print("FINAL STATUS: PASS", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

