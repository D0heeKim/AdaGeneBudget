#!/usr/bin/env python3
"""Build PBMC Geneformer Native and AdaGeneBudget method caches.

Input
-----
Validated PBMC Geneformer base cache:
    data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base

Methods
-------
1. Native Max-2048
   - Rank every expressed supported gene by the official Geneformer
     median-scaled expression score.
   - Because 10,000 / n_counts is constant within a cell, the ranking is
       count_g / gene_median_g
   - Keep the first 2,048 genes.
   - Geneformer V1 has no CLS/EOS tokens, so all 2,048 positions are usable.

2. AdaGeneBudget
   - Selection score:
       raw_count_g * train_only_IDF_g
   - Find the smallest raw K reaching cumulative mass tau=0.90.
   - Clamp to Kmin=128 and Kmax=600.
   - IMPORTANT: TF-IDF determines only which genes remain.
   - Reorder the selected genes by the official Geneformer native ranking
     before writing the final token sequence.

The script also computes the train-only matched fixed budget K*, but does
not build Fixed TF-IDF or Random caches yet.

Output
------
.../geneformer_cache/methods/
├── native_max2048/
│   ├── meta.json
│   ├── train/
│   │   ├── indptr.npy
│   │   ├── gene_pos.npy
│   │   └── meta.json
│   └── test/
│       └── same
├── adaptive_tau0p90_k128_k600/
│   ├── meta.json
│   ├── train/
│   │   ├── indptr.npy
│   │   ├── gene_pos.npy
│   │   ├── raw_k.npy
│   │   ├── selected_k.npy
│   │   ├── retained_mass.npy
│   │   ├── upper_clipped.npy
│   │   ├── lower_clipped.npy
│   │   └── meta.json
│   └── test/
│       └── same
├── adaptive_budget_by_celltype_l1.csv
└── matched_budget.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

DEFAULT_OUTPUT_DIR = (
    DEFAULT_BASE_DIR.parent
    / "methods"
)

NATIVE_METHOD_NAME = "native_max2048"
ADAPTIVE_METHOD_NAME = (
    "adaptive_tau0p90_k128_k600"
)

NATIVE_MAX_GENES = 2_048
TAU = 0.90
K_MIN = 128
K_MAX = 600

GENE_POS_DTYPE = np.uint16
INDPTR_DTYPE = np.int64
RAW_K_DTYPE = np.uint16
SELECTED_K_DTYPE = np.uint16
RETAINED_MASS_DTYPE = np.float32
FLAG_DTYPE = np.uint8

EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957
EXPECTED_GENES = 14_809


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=TAU,
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=K_MIN,
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=K_MAX,
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def load_json(path: Path) -> dict[str, Any]:
    require_file(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def stable_descending_order(
    scores: np.ndarray,
) -> np.ndarray:
    """Descending score; ties preserve current order."""
    return np.argsort(
        -scores,
        kind="stable",
    )


def load_base(
    base_dir: Path,
) -> dict[str, Any]:
    require_directory(base_dir)

    root_meta = load_json(
        base_dir / "meta.json"
    )
    if root_meta.get("status") != "PASS":
        raise RuntimeError(
            "Base-cache root status is not PASS."
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

    for name, array in {
        "token_ids": token_ids,
        "gene_median": gene_median,
        "idf": idf,
    }.items():
        if array.ndim != 1:
            raise RuntimeError(
                f"{name} must be one-dimensional."
            )
        if len(array) != EXPECTED_GENES:
            raise RuntimeError(
                f"{name} length={len(array)}; "
                f"expected {EXPECTED_GENES}."
            )

    if not np.all(
        np.isfinite(gene_median)
    ):
        raise RuntimeError(
            "Non-finite Geneformer medians."
        )
    if np.any(gene_median <= 0):
        raise RuntimeError(
            "Geneformer medians must be positive."
        )
    if not np.all(np.isfinite(idf)):
        raise RuntimeError(
            "Non-finite IDF."
        )
    if np.any(idf <= 0):
        raise RuntimeError(
            "IDF must be positive."
        )
    if len(np.unique(token_ids)) != len(
        token_ids
    ):
        raise RuntimeError(
            "Duplicate Geneformer token IDs."
        )

    splits = {}
    for split_name, expected_cells in [
        ("train", EXPECTED_TRAIN_CELLS),
        ("test", EXPECTED_TEST_CELLS),
    ]:
        split_dir = base_dir / split_name
        require_directory(split_dir)

        split_meta = load_json(
            split_dir / "meta.json"
        )
        if split_meta.get("status") != "PASS":
            raise RuntimeError(
                f"{split_name} base status "
                "is not PASS."
            )

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
        labels_l1 = np.load(
            split_dir / "labels_l1.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        obs_names = np.load(
            split_dir / "obs_names.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        if indptr.shape != (
            expected_cells + 1,
        ):
            raise RuntimeError(
                f"{split_name} indptr shape "
                f"{indptr.shape} is invalid."
            )
        if int(indptr[0]) != 0:
            raise RuntimeError(
                f"{split_name} indptr must begin "
                "at zero."
            )
        if np.any(np.diff(indptr) < 0):
            raise RuntimeError(
                f"{split_name} indptr is not "
                "monotonic."
            )
        if int(indptr[-1]) != len(gene_pos):
            raise RuntimeError(
                f"{split_name} indptr/gene_pos "
                "length mismatch."
            )
        if len(counts) != len(gene_pos):
            raise RuntimeError(
                f"{split_name} counts/gene_pos "
                "length mismatch."
            )
        if len(labels_l1) != expected_cells:
            raise RuntimeError(
                f"{split_name} labels length "
                "mismatch."
            )
        if len(obs_names) != expected_cells:
            raise RuntimeError(
                f"{split_name} obs_names length "
                "mismatch."
            )
        if (
            len(gene_pos)
            and int(np.max(gene_pos))
            >= EXPECTED_GENES
        ):
            raise RuntimeError(
                f"{split_name} gene position "
                "out of range."
            )
        if (
            len(gene_pos)
            and int(np.min(gene_pos)) < 0
        ):
            raise RuntimeError(
                f"{split_name} negative gene "
                "position."
            )
        if (
            len(counts)
            and int(np.min(counts)) <= 0
        ):
            raise RuntimeError(
                f"{split_name} cache must contain "
                "positive nonzero counts only."
            )

        lengths = np.diff(
            indptr
        ).astype(
            np.int64,
            copy=False,
        )
        if np.any(lengths == 0):
            bad = np.flatnonzero(
                lengths == 0
            )[:20]
            raise RuntimeError(
                f"{split_name} contains cells with "
                "zero supported expressed genes: "
                f"{bad.tolist()}"
            )

        splits[split_name] = {
            "dir": split_dir,
            "meta": split_meta,
            "indptr": indptr,
            "gene_pos": gene_pos,
            "counts": counts,
            "labels_l1": labels_l1,
            "obs_names": obs_names,
            "lengths": lengths,
        }

    return {
        "meta": root_meta,
        "token_ids": token_ids,
        "gene_median": gene_median,
        "idf": idf,
        "splits": splits,
    }


def prepare_method_dir(
    final_dir: Path,
    overwrite: bool,
) -> Path:
    building_dir = final_dir.with_name(
        final_dir.name + ".building"
    )

    for path in [final_dir, building_dir]:
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. "
                    "Inspect it first, or rerun "
                    "with --overwrite."
                )
            shutil.rmtree(path)

    building_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    building_dir.mkdir(
        parents=False,
        exist_ok=False,
    )
    return building_dir


def write_indptr(
    lengths: np.ndarray,
    path: Path,
) -> np.ndarray:
    indptr = np.empty(
        len(lengths) + 1,
        dtype=INDPTR_DTYPE,
    )
    indptr[0] = 0
    np.cumsum(
        lengths.astype(
            np.int64,
            copy=False,
        ),
        out=indptr[1:],
    )
    np.save(
        path,
        indptr,
        allow_pickle=False,
    )
    return indptr


def native_order_for_row(
    positions: np.ndarray,
    counts: np.ndarray,
    gene_median: np.ndarray,
) -> np.ndarray:
    native_scores = (
        counts.astype(
            np.float64,
            copy=False,
        )
        / gene_median[positions]
    )
    return stable_descending_order(
        native_scores
    )


def build_native_split(
    split_name: str,
    split: dict[str, Any],
    gene_median: np.ndarray,
    output_dir: Path,
    progress_every: int,
) -> dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    base_indptr = split["indptr"]
    base_gene_pos = split["gene_pos"]
    base_counts = split["counts"]
    base_lengths = split["lengths"]

    selected_lengths = np.minimum(
        base_lengths,
        NATIVE_MAX_GENES,
    ).astype(
        SELECTED_K_DTYPE,
        copy=False,
    )

    output_indptr = write_indptr(
        selected_lengths,
        output_dir / "indptr.npy",
    )
    total_selected = int(
        output_indptr[-1]
    )

    output_gene_pos = (
        np.lib.format.open_memmap(
            output_dir / "gene_pos.npy",
            mode="w+",
            dtype=GENE_POS_DTYPE,
            shape=(total_selected,),
        )
    )

    started = time.perf_counter()
    truncated_cells = 0

    for row in range(
        len(base_lengths)
    ):
        start = int(base_indptr[row])
        stop = int(base_indptr[row + 1])

        positions = np.asarray(
            base_gene_pos[start:stop],
            dtype=np.int64,
        )
        counts = np.asarray(
            base_counts[start:stop],
            dtype=np.float64,
        )

        order = native_order_for_row(
            positions,
            counts,
            gene_median,
        )

        k = int(selected_lengths[row])
        chosen_positions = positions[
            order[:k]
        ]

        out_start = int(
            output_indptr[row]
        )
        out_stop = int(
            output_indptr[row + 1]
        )
        output_gene_pos[
            out_start:out_stop
        ] = chosen_positions.astype(
            GENE_POS_DTYPE,
            copy=False,
        )

        if len(positions) > (
            NATIVE_MAX_GENES
        ):
            truncated_cells += 1

        completed = row + 1
        if (
            completed == 1
            or completed
            % progress_every == 0
            or completed
            == len(base_lengths)
        ):
            elapsed = (
                time.perf_counter()
                - started
            )
            print(
                f"[native/{split_name}] "
                f"{completed:,}/"
                f"{len(base_lengths):,} cells "
                f"({100 * completed / len(base_lengths):.1f}%); "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    output_gene_pos.flush()
    del output_gene_pos

    np.save(
        output_dir / "selected_k.npy",
        selected_lengths,
        allow_pickle=False,
    )

    selected_fraction = (
        selected_lengths.astype(
            np.float64
        )
        / base_lengths.astype(
            np.float64
        )
    )

    metadata = {
        "status": "PASS",
        "method": "native",
        "split": split_name,
        "native_max_genes": (
            NATIVE_MAX_GENES
        ),
        "cells": int(
            len(base_lengths)
        ),
        "total_selected_genes": int(
            total_selected
        ),
        "mean_selected_genes": float(
            np.mean(selected_lengths)
        ),
        "median_selected_genes": float(
            np.median(selected_lengths)
        ),
        "minimum_selected_genes": int(
            np.min(selected_lengths)
        ),
        "maximum_selected_genes": int(
            np.max(selected_lengths)
        ),
        "truncated_cells": int(
            truncated_cells
        ),
        "truncated_fraction": float(
            truncated_cells
            / len(base_lengths)
        ),
        "mean_fraction_expressed_genes_retained": float(
            np.mean(selected_fraction)
        ),
        "selection_time_s": float(
            time.perf_counter()
            - started
        ),
        "final_sequence_order": (
            "Official Geneformer median-scaled "
            "expression rank."
        ),
        "native_rank_equivalent_score": (
            "raw_count / official_gene_median; "
            "10,000 / n_counts is omitted because "
            "it is constant within each cell."
        ),
    }

    (
        output_dir / "meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    validate_method_split(
        output_dir=output_dir,
        expected_cells=len(base_lengths),
        maximum_length=NATIVE_MAX_GENES,
        expected_lengths=(
            selected_lengths
        ),
    )
    return metadata


def adaptive_first_pass(
    split_name: str,
    split: dict[str, Any],
    idf: np.ndarray,
    progress_every: int,
) -> dict[str, np.ndarray | float]:
    base_indptr = split["indptr"]
    base_gene_pos = split["gene_pos"]
    base_counts = split["counts"]
    base_lengths = split["lengths"]

    n_cells = len(base_lengths)

    raw_k = np.empty(
        n_cells,
        dtype=RAW_K_DTYPE,
    )
    selected_k = np.empty(
        n_cells,
        dtype=SELECTED_K_DTYPE,
    )
    retained_mass = np.empty(
        n_cells,
        dtype=RETAINED_MASS_DTYPE,
    )
    upper_clipped = np.zeros(
        n_cells,
        dtype=FLAG_DTYPE,
    )
    lower_clipped = np.zeros(
        n_cells,
        dtype=FLAG_DTYPE,
    )

    started = time.perf_counter()

    for row in range(n_cells):
        start = int(base_indptr[row])
        stop = int(base_indptr[row + 1])

        positions = np.asarray(
            base_gene_pos[start:stop],
            dtype=np.int64,
        )
        counts = np.asarray(
            base_counts[start:stop],
            dtype=np.float64,
        )
        n_expressed = len(positions)

        scores = (
            counts * idf[positions]
        )
        total_mass = float(
            np.sum(
                scores,
                dtype=np.float64,
            )
        )
        if (
            not np.isfinite(total_mass)
            or total_mass <= 0
        ):
            raise RuntimeError(
                f"{split_name} row {row}: "
                "invalid TF-IDF total mass."
            )

        tfidf_order = (
            stable_descending_order(
                scores
            )
        )
        cumulative = np.cumsum(
            scores[tfidf_order],
            dtype=np.float64,
        )

        raw = int(
            np.searchsorted(
                cumulative,
                TAU * total_mass,
                side="left",
            )
            + 1
        )

        if raw < 1 or raw > n_expressed:
            raise RuntimeError(
                f"{split_name} row {row}: "
                f"invalid raw K={raw}, "
                f"n={n_expressed}."
            )

        selected = min(
            n_expressed,
            K_MAX,
            max(K_MIN, raw),
        )

        is_upper = int(
            raw > K_MAX
        )
        is_lower = int(
            n_expressed >= K_MIN
            and raw < K_MIN
        )

        actual_mass = float(
            np.sum(
                scores[
                    tfidf_order[:selected]
                ],
                dtype=np.float64,
            )
            / total_mass
        )

        raw_k[row] = raw
        selected_k[row] = selected
        retained_mass[row] = (
            actual_mass
        )
        upper_clipped[row] = is_upper
        lower_clipped[row] = is_lower

        completed = row + 1
        if (
            completed == 1
            or completed
            % progress_every == 0
            or completed == n_cells
        ):
            elapsed = (
                time.perf_counter()
                - started
            )
            print(
                f"[adaptive/{split_name} pass 1] "
                f"{completed:,}/{n_cells:,} cells "
                f"({100 * completed / n_cells:.1f}%); "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    return {
        "raw_k": raw_k,
        "selected_k": selected_k,
        "retained_mass": retained_mass,
        "upper_clipped": upper_clipped,
        "lower_clipped": lower_clipped,
        "elapsed_seconds": float(
            time.perf_counter()
            - started
        ),
    }


def build_adaptive_split(
    split_name: str,
    split: dict[str, Any],
    idf: np.ndarray,
    gene_median: np.ndarray,
    output_dir: Path,
    progress_every: int,
) -> dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    pass_one = adaptive_first_pass(
        split_name=split_name,
        split=split,
        idf=idf,
        progress_every=progress_every,
    )

    raw_k = pass_one["raw_k"]
    selected_k = pass_one[
        "selected_k"
    ]
    retained_mass = pass_one[
        "retained_mass"
    ]
    upper_clipped = pass_one[
        "upper_clipped"
    ]
    lower_clipped = pass_one[
        "lower_clipped"
    ]

    assert isinstance(
        raw_k,
        np.ndarray,
    )
    assert isinstance(
        selected_k,
        np.ndarray,
    )
    assert isinstance(
        retained_mass,
        np.ndarray,
    )
    assert isinstance(
        upper_clipped,
        np.ndarray,
    )
    assert isinstance(
        lower_clipped,
        np.ndarray,
    )

    output_indptr = write_indptr(
        selected_k,
        output_dir / "indptr.npy",
    )
    total_selected = int(
        output_indptr[-1]
    )

    output_gene_pos = (
        np.lib.format.open_memmap(
            output_dir / "gene_pos.npy",
            mode="w+",
            dtype=GENE_POS_DTYPE,
            shape=(total_selected,),
        )
    )

    base_indptr = split["indptr"]
    base_gene_pos = split["gene_pos"]
    base_counts = split["counts"]
    n_cells = len(split["lengths"])

    started_second = time.perf_counter()

    for row in range(n_cells):
        start = int(base_indptr[row])
        stop = int(base_indptr[row + 1])

        positions = np.asarray(
            base_gene_pos[start:stop],
            dtype=np.int64,
        )
        counts = np.asarray(
            base_counts[start:stop],
            dtype=np.float64,
        )

        tfidf_scores = (
            counts * idf[positions]
        )
        tfidf_order = (
            stable_descending_order(
                tfidf_scores
            )
        )

        k = int(selected_k[row])
        selected_indices = (
            tfidf_order[:k]
        )
        selected_positions = positions[
            selected_indices
        ]
        selected_counts = counts[
            selected_indices
        ]

        native_scores = (
            selected_counts
            / gene_median[
                selected_positions
            ]
        )

        # TF-IDF defines the retained SET.
        # Geneformer native ranking defines final ORDER.
        # Tie break by ascending local gene position.
        native_order = np.lexsort(
            (
                selected_positions,
                -native_scores,
            )
        )
        final_positions = (
            selected_positions[
                native_order
            ]
        )

        out_start = int(
            output_indptr[row]
        )
        out_stop = int(
            output_indptr[row + 1]
        )
        if out_stop - out_start != k:
            raise RuntimeError(
                f"{split_name} row {row}: "
                "adaptive output-length mismatch."
            )

        output_gene_pos[
            out_start:out_stop
        ] = final_positions.astype(
            GENE_POS_DTYPE,
            copy=False,
        )

        completed = row + 1
        if (
            completed == 1
            or completed
            % progress_every == 0
            or completed == n_cells
        ):
            elapsed = (
                time.perf_counter()
                - started_second
            )
            print(
                f"[adaptive/{split_name} pass 2] "
                f"{completed:,}/{n_cells:,} cells "
                f"({100 * completed / n_cells:.1f}%); "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    output_gene_pos.flush()
    del output_gene_pos

    for filename, array in [
        ("raw_k.npy", raw_k),
        ("selected_k.npy", selected_k),
        (
            "retained_mass.npy",
            retained_mass,
        ),
        (
            "upper_clipped.npy",
            upper_clipped,
        ),
        (
            "lower_clipped.npy",
            lower_clipped,
        ),
    ]:
        np.save(
            output_dir / filename,
            array,
            allow_pickle=False,
        )

    metadata = {
        "status": "PASS",
        "method": "adaptive",
        "split": split_name,
        "tau": TAU,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "cells": int(n_cells),
        "total_selected_genes": int(
            total_selected
        ),
        "mean_raw_k": float(
            np.mean(raw_k)
        ),
        "median_raw_k": float(
            np.median(raw_k)
        ),
        "minimum_raw_k": int(
            np.min(raw_k)
        ),
        "maximum_raw_k": int(
            np.max(raw_k)
        ),
        "mean_selected_genes": float(
            np.mean(selected_k)
        ),
        "median_selected_genes": float(
            np.median(selected_k)
        ),
        "minimum_selected_genes": int(
            np.min(selected_k)
        ),
        "maximum_selected_genes": int(
            np.max(selected_k)
        ),
        "retained_mass_mean": float(
            np.mean(retained_mass)
        ),
        "retained_mass_median": float(
            np.median(retained_mass)
        ),
        "retained_mass_minimum": float(
            np.min(retained_mass)
        ),
        "upper_clipped_cells": int(
            np.sum(upper_clipped)
        ),
        "upper_clipped_fraction": float(
            np.mean(upper_clipped)
        ),
        "lower_clipped_cells": int(
            np.sum(lower_clipped)
        ),
        "lower_clipped_fraction": float(
            np.mean(lower_clipped)
        ),
        "fraction_at_kmax": float(
            np.mean(selected_k == K_MAX)
        ),
        "fraction_at_kmin": float(
            np.mean(selected_k == K_MIN)
        ),
        "selection_pass_1_time_s": float(
            pass_one["elapsed_seconds"]
        ),
        "selection_pass_2_time_s": float(
            time.perf_counter()
            - started_second
        ),
        "selection_time_s": float(
            float(
                pass_one[
                    "elapsed_seconds"
                ]
            )
            + (
                time.perf_counter()
                - started_second
            )
        ),
        "selection_score": (
            "raw_count * train_only_IDF"
        ),
        "final_sequence_order": (
            "Official Geneformer median-scaled "
            "expression rank after TF-IDF subset "
            "selection."
        ),
        "native_rank_equivalent_score": (
            "raw_count / official_gene_median; "
            "10,000 / n_counts is omitted because "
            "it is constant within each cell."
        ),
    }

    (
        output_dir / "meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    validate_method_split(
        output_dir=output_dir,
        expected_cells=n_cells,
        maximum_length=K_MAX,
        expected_lengths=selected_k,
    )
    return metadata


def validate_method_split(
    output_dir: Path,
    expected_cells: int,
    maximum_length: int,
    expected_lengths: np.ndarray,
) -> None:
    indptr = np.load(
        output_dir / "indptr.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gene_pos = np.load(
        output_dir / "gene_pos.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if indptr.shape != (
        expected_cells + 1,
    ):
        raise RuntimeError(
            f"{output_dir}: invalid indptr shape."
        )
    if int(indptr[0]) != 0:
        raise RuntimeError(
            f"{output_dir}: indptr does not "
            "start at zero."
        )
    if np.any(np.diff(indptr) < 0):
        raise RuntimeError(
            f"{output_dir}: non-monotonic indptr."
        )
    if int(indptr[-1]) != len(gene_pos):
        raise RuntimeError(
            f"{output_dir}: indptr/gene_pos "
            "mismatch."
        )

    lengths = np.diff(
        indptr
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        lengths,
        expected_lengths.astype(
            np.int64,
            copy=False,
        ),
    ):
        raise RuntimeError(
            f"{output_dir}: stored lengths differ "
            "from expected lengths."
        )
    if np.any(lengths <= 0):
        raise RuntimeError(
            f"{output_dir}: non-positive sequence "
            "length."
        )
    if np.any(lengths > maximum_length):
        raise RuntimeError(
            f"{output_dir}: sequence exceeds "
            f"maximum {maximum_length}."
        )
    if (
        len(gene_pos)
        and int(np.max(gene_pos))
        >= EXPECTED_GENES
    ):
        raise RuntimeError(
            f"{output_dir}: gene position "
            "out of range."
        )


def matched_fixed_budget(
    nonzero_genes: np.ndarray,
    target_mean: float,
) -> tuple[int, float, float]:
    """Choose K using train cells only."""
    nonzero_genes = np.asarray(
        nonzero_genes,
        dtype=np.int64,
    )

    if np.any(nonzero_genes <= 0):
        raise RuntimeError(
            "Matched-budget input contains "
            "non-positive gene counts."
        )

    low = 1
    high = int(
        np.max(nonzero_genes)
    )

    best_k = 1
    best_mean = float(
        np.mean(
            np.minimum(
                nonzero_genes,
                1,
            )
        )
    )
    best_difference = abs(
        best_mean - target_mean
    )

    while low <= high:
        middle = (
            low + high
        ) // 2
        realized_mean = float(
            np.mean(
                np.minimum(
                    nonzero_genes,
                    middle,
                )
            )
        )
        difference = abs(
            realized_mean
            - target_mean
        )

        if (
            difference
            < best_difference
            or (
                difference
                == best_difference
                and middle < best_k
            )
        ):
            best_k = middle
            best_mean = realized_mean
            best_difference = difference

        if realized_mean < target_mean:
            low = middle + 1
        else:
            high = middle - 1

    for candidate in range(
        max(1, best_k - 3),
        min(
            int(
                np.max(
                    nonzero_genes
                )
            ),
            best_k + 3,
        )
        + 1,
    ):
        realized_mean = float(
            np.mean(
                np.minimum(
                    nonzero_genes,
                    candidate,
                )
            )
        )
        difference = abs(
            realized_mean
            - target_mean
        )
        if (
            difference
            < best_difference
            or (
                difference
                == best_difference
                and candidate < best_k
            )
        ):
            best_k = candidate
            best_mean = realized_mean
            best_difference = difference

    return (
        int(best_k),
        float(best_mean),
        float(best_difference),
    )


def make_budget_by_celltype(
    base: dict[str, Any],
    adaptive_building_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    rows = []

    for split_name in [
        "train",
        "test",
    ]:
        labels = np.asarray(
            base["splits"][split_name][
                "labels_l1"
            ]
        ).astype(str)

        selected_k = np.load(
            adaptive_building_dir
            / split_name
            / "selected_k.npy",
            allow_pickle=False,
        )
        raw_k = np.load(
            adaptive_building_dir
            / split_name
            / "raw_k.npy",
            allow_pickle=False,
        )
        retained_mass = np.load(
            adaptive_building_dir
            / split_name
            / "retained_mass.npy",
            allow_pickle=False,
        )
        upper = np.load(
            adaptive_building_dir
            / split_name
            / "upper_clipped.npy",
            allow_pickle=False,
        )
        lower = np.load(
            adaptive_building_dir
            / split_name
            / "lower_clipped.npy",
            allow_pickle=False,
        )

        frame = pd.DataFrame(
            {
                "cell_type": labels,
                "selected_k": (
                    selected_k
                ),
                "raw_k": raw_k,
                "retained_mass": (
                    retained_mass
                ),
                "upper_clipped": upper,
                "lower_clipped": lower,
            }
        )

        grouped = (
            frame.groupby(
                "cell_type",
                sort=True,
            )
            .agg(
                cells=(
                    "selected_k",
                    "size",
                ),
                mean_selected_k=(
                    "selected_k",
                    "mean",
                ),
                median_selected_k=(
                    "selected_k",
                    "median",
                ),
                minimum_selected_k=(
                    "selected_k",
                    "min",
                ),
                maximum_selected_k=(
                    "selected_k",
                    "max",
                ),
                mean_raw_k=(
                    "raw_k",
                    "mean",
                ),
                mean_retained_mass=(
                    "retained_mass",
                    "mean",
                ),
                fraction_kmax=(
                    "selected_k",
                    lambda value: float(
                        np.mean(
                            np.asarray(value)
                            == K_MAX
                        )
                    ),
                ),
                upper_clipped_fraction=(
                    "upper_clipped",
                    "mean",
                ),
                lower_clipped_fraction=(
                    "lower_clipped",
                    "mean",
                ),
            )
            .reset_index()
        )
        grouped.insert(
            0,
            "split",
            split_name,
        )
        rows.append(grouped)

    output = pd.concat(
        rows,
        ignore_index=True,
    )
    output.to_csv(
        output_path,
        index=False,
    )
    return output


def main() -> int:
    global TAU, K_MIN, K_MAX, ADAPTIVE_METHOD_NAME

    args = parse_args()

    if not 0.0 < float(args.tau) <= 1.0:
        raise ValueError(
            f"--tau must be in (0, 1], got {args.tau}"
        )
    if int(args.k_min) <= 0:
        raise ValueError(
            f"--k-min must be positive, got {args.k_min}"
        )
    if int(args.k_max) < int(args.k_min):
        raise ValueError(
            "--k-max must be greater than or equal to --k-min"
        )
    if int(args.k_max) > NATIVE_MAX_GENES:
        raise ValueError(
            f"--k-max cannot exceed {NATIVE_MAX_GENES}"
        )

    TAU = float(args.tau)
    K_MIN = int(args.k_min)
    K_MAX = int(args.k_max)

    tau_tag = f"{TAU:.2f}".replace(".", "p")
    ADAPTIVE_METHOD_NAME = (
        f"adaptive_tau{tau_tag}_k{K_MIN}_k{K_MAX}"
    )

    print(
        "Resolved adaptive configuration:",
        f"tau={TAU}, Kmin={K_MIN}, Kmax={K_MAX}, "
        f"method={ADAPTIVE_METHOD_NAME}",
        flush=True,
    )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    base = load_base(
        args.base_dir
    )

    native_final = (
        args.output_dir
        / NATIVE_METHOD_NAME
    )
    adaptive_final = (
        args.output_dir
        / ADAPTIVE_METHOD_NAME
    )

    native_building = (
        prepare_method_dir(
            native_final,
            args.overwrite,
        )
    )
    adaptive_building = (
        prepare_method_dir(
            adaptive_final,
            args.overwrite,
        )
    )

    total_started = time.perf_counter()

    print("=" * 110)
    print(
        "PBMC GENEFORMER NATIVE + "
        "ADAPTIVE METHOD CACHES"
    )
    print("=" * 110)
    print(
        "Base cache:",
        args.base_dir,
    )
    print(
        "Native:",
        f"Max-{NATIVE_MAX_GENES}",
    )
    print(
        "Adaptive:",
        f"tau={TAU}, Kmin={K_MIN}, "
        f"Kmax={K_MAX}",
    )
    print(
        "Critical ordering rule:",
        "TF-IDF selects the subset; "
        "Geneformer native rank determines "
        "the final sequence order.",
    )

    try:
        native_split_meta = {}
        adaptive_split_meta = {}

        for split_name in [
            "train",
            "test",
        ]:
            native_split_meta[
                split_name
            ] = build_native_split(
                split_name=split_name,
                split=base["splits"][
                    split_name
                ],
                gene_median=base[
                    "gene_median"
                ],
                output_dir=(
                    native_building
                    / split_name
                ),
                progress_every=(
                    args.progress_every
                ),
            )

        native_root_meta = {
            "status": "PASS",
            "method": "native",
            "method_name": (
                NATIVE_METHOD_NAME
            ),
            "native_max_genes": (
                NATIVE_MAX_GENES
            ),
            "special_tokens": False,
            "base_cache": str(
                args.base_dir.resolve()
            ),
            "selection_score": (
                "raw_count / "
                "official_gene_median"
            ),
            "official_normalization_note": (
                "The official score is "
                "raw_count / n_counts * 10,000 "
                "/ gene_median. The per-cell "
                "10,000/n_counts factor does not "
                "affect within-cell ranking."
            ),
            "train": (
                native_split_meta[
                    "train"
                ]
            ),
            "test": (
                native_split_meta[
                    "test"
                ]
            ),
        }
        (
            native_building
            / "meta.json"
        ).write_text(
            json.dumps(
                native_root_meta,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        for split_name in [
            "train",
            "test",
        ]:
            adaptive_split_meta[
                split_name
            ] = build_adaptive_split(
                split_name=split_name,
                split=base["splits"][
                    split_name
                ],
                idf=base["idf"],
                gene_median=base[
                    "gene_median"
                ],
                output_dir=(
                    adaptive_building
                    / split_name
                ),
                progress_every=(
                    args.progress_every
                ),
            )

        adaptive_root_meta = {
            "status": "PASS",
            "method": "adaptive",
            "method_name": (
                ADAPTIVE_METHOD_NAME
            ),
            "tau": TAU,
            "k_min": K_MIN,
            "k_max": K_MAX,
            "special_tokens": False,
            "base_cache": str(
                args.base_dir.resolve()
            ),
            "idf_fit_split": "train_only",
            "selection_score": (
                "raw_count * train_only_IDF"
            ),
            "final_sequence_order": (
                "Official Geneformer "
                "median-scaled expression rank."
            ),
            "train": (
                adaptive_split_meta[
                    "train"
                ]
            ),
            "test": (
                adaptive_split_meta[
                    "test"
                ]
            ),
        }
        (
            adaptive_building
            / "meta.json"
        ).write_text(
            json.dumps(
                adaptive_root_meta,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        budget_by_celltype = (
            make_budget_by_celltype(
                base=base,
                adaptive_building_dir=(
                    adaptive_building
                ),
                output_path=(
                    args.output_dir
                    / (
                        "adaptive_budget_by_"
                        "celltype_l1.csv"
                    )
                ),
            )
        )

        adaptive_train_mean = float(
            adaptive_split_meta[
                "train"
            ][
                "mean_selected_genes"
            ]
        )

        (
            matched_k,
            matched_expected_mean,
            matched_difference,
        ) = matched_fixed_budget(
            nonzero_genes=base[
                "splits"
            ]["train"]["lengths"],
            target_mean=(
                adaptive_train_mean
            ),
        )

        matched_meta = {
            "status": "PASS",
            "fit_split": "train_only",
            "target_method": (
                ADAPTIVE_METHOD_NAME
            ),
            "target_adaptive_train_mean": (
                adaptive_train_mean
            ),
            "matched_fixed_k": (
                matched_k
            ),
            "matched_fixed_expected_train_mean": (
                matched_expected_mean
            ),
            "absolute_mean_difference": (
                matched_difference
            ),
            "matching_rule": (
                "Choose integer K minimizing "
                "abs(mean_train(min(n_nonzero_i, K)) "
                "- adaptive_train_mean)."
            ),
            "test_statistics_used": False,
        }

        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        (
            args.output_dir
            / "matched_budget.json"
        ).write_text(
            json.dumps(
                matched_meta,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        os.replace(
            native_building,
            native_final,
        )
        os.replace(
            adaptive_building,
            adaptive_final,
        )

    except Exception:
        print(
            "\n[ERROR] Build failed. "
            "Incomplete data remain only in "
            "*.building directories.",
            flush=True,
        )
        raise

    elapsed = (
        time.perf_counter()
        - total_started
    )

    print()
    print("=" * 110)
    print(
        "PBMC GENEFORMER METHOD CACHE "
        "SUMMARY"
    )
    print("=" * 110)

    print("\nNATIVE MAX-2048")
    for split_name in [
        "train",
        "test",
    ]:
        meta = native_split_meta[
            split_name
        ]
        print(
            f"{split_name}: "
            f"mean={meta['mean_selected_genes']:.3f}, "
            f"median={meta['median_selected_genes']:.1f}, "
            f"truncated="
            f"{meta['truncated_fraction']:.4f}"
        )

    print("\nADAPTIVE")
    for split_name in [
        "train",
        "test",
    ]:
        meta = adaptive_split_meta[
            split_name
        ]
        print(
            f"{split_name}: "
            f"mean={meta['mean_selected_genes']:.3f}, "
            f"median={meta['median_selected_genes']:.1f}, "
            f"retained_mass="
            f"{meta['retained_mass_mean']:.6f}, "
            f"upper_clip="
            f"{meta['upper_clipped_fraction']:.6f}, "
            f"lower_clip="
            f"{meta['lower_clipped_fraction']:.6f}, "
            f"fraction_kmax="
            f"{meta['fraction_at_kmax']:.6f}"
        )

    print("\nMATCHED FIXED BUDGET")
    print(
        "Adaptive train mean:",
        f"{adaptive_train_mean:.6f}",
    )
    print(
        "Matched K:",
        matched_k,
    )
    print(
        "Expected train mean:",
        f"{matched_expected_mean:.6f}",
    )
    print(
        "Absolute difference:",
        f"{matched_difference:.9f}",
    )

    print(
        "\nCell-type budget rows:",
        len(budget_by_celltype),
    )
    print(
        "Output:",
        args.output_dir,
    )
    print(
        "Elapsed:",
        f"{elapsed / 60:.1f} min",
    )
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
