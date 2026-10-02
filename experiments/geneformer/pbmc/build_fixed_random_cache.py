#!/usr/bin/env python3
"""Build PBMC Geneformer matched Fixed TF-IDF and Random caches.

The matched fixed budget K is read from the train-only result produced by
16_build_geneformer_pbmc_native_adaptive_cache.py.

Methods
-------
1. Matched Fixed TF-IDF
   - Select the top min(K, n_expressed) genes using
       raw_count * train_only_IDF
   - Reorder the selected genes using the official Geneformer native rank:
       raw_count / official_gene_median

2. Matched Random, seeds 0--4
   - Uniformly sample min(K, n_expressed) expressed genes without replacement.
   - Reorder the sampled genes using the same Geneformer native rank.

The final sequence order is therefore identical in definition across Native,
Fixed TF-IDF, Random, and AdaGeneBudget. Only the retained gene set differs.

Outputs
-------
.../geneformer_cache/methods/
├── matched_fixed_tfidf_k599/
│   ├── meta.json
│   ├── train/
│   │   ├── indptr.npy
│   │   ├── gene_pos.npy
│   │   ├── selected_k.npy
│   │   └── meta.json
│   └── test/
│       └── same
├── matched_random_fixed_k599/
│   ├── meta.json
│   ├── seed0/
│   │   ├── train/
│   │   └── test/
│   ├── seed1/
│   ├── seed2/
│   ├── seed3/
│   └── seed4/
└── matched_cache_summary.csv
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

DEFAULT_METHODS_DIR = (
    DEFAULT_BASE_DIR.parent
    / "methods"
)

DEFAULT_MATCHED_BUDGET = (
    DEFAULT_METHODS_DIR
    / "matched_budget.json"
)

EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957
EXPECTED_GENES = 14_809

RANDOM_SEEDS = (0, 1, 2, 3, 4)

GENE_POS_DTYPE = np.uint16
INDPTR_DTYPE = np.int64
SELECTED_K_DTYPE = np.uint16


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
        "--matched-budget-json",
        type=Path,
        default=DEFAULT_MATCHED_BUDGET,
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


def prepare_building_dir(
    final_dir: Path,
    overwrite: bool,
) -> Path:
    building_dir = final_dir.with_name(
        final_dir.name + ".building"
    )

    for path in (final_dir, building_dir):
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists. "
                    "Inspect it or rerun with --overwrite."
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


def load_base(
    base_dir: Path,
) -> dict[str, Any]:
    require_directory(base_dir)

    root_meta = load_json(
        base_dir / "meta.json"
    )
    if root_meta.get("status") != "PASS":
        raise RuntimeError(
            "Base cache status is not PASS."
        )

    idf = np.load(
        base_dir / "idf.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gene_median = np.load(
        base_dir / "gene_median.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    for name, array in {
        "idf": idf,
        "gene_median": gene_median,
    }.items():
        if array.shape != (
            EXPECTED_GENES,
        ):
            raise RuntimeError(
                f"{name} shape={array.shape}; "
                f"expected {(EXPECTED_GENES,)}."
            )
        if not np.all(
            np.isfinite(array)
        ):
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

    if np.any(idf <= 0):
        raise RuntimeError(
            "IDF must be positive."
        )
    if np.any(gene_median <= 0):
        raise RuntimeError(
            "Gene medians must be positive."
        )

    splits = {}

    for split_name, expected_cells in (
        ("train", EXPECTED_TRAIN_CELLS),
        ("test", EXPECTED_TEST_CELLS),
    ):
        split_dir = base_dir / split_name
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
        counts = np.load(
            split_dir / "counts.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

        if indptr.shape != (
            expected_cells + 1,
        ):
            raise RuntimeError(
                f"{split_name}: invalid indptr shape."
            )
        if int(indptr[0]) != 0:
            raise RuntimeError(
                f"{split_name}: indptr must start at zero."
            )
        if np.any(
            np.diff(indptr) < 0
        ):
            raise RuntimeError(
                f"{split_name}: indptr is not monotonic."
            )
        if int(indptr[-1]) != len(
            gene_pos
        ):
            raise RuntimeError(
                f"{split_name}: indptr/gene_pos mismatch."
            )
        if len(counts) != len(
            gene_pos
        ):
            raise RuntimeError(
                f"{split_name}: counts/gene_pos mismatch."
            )

        lengths = np.diff(
            indptr
        ).astype(
            np.int64,
            copy=False,
        )

        if np.any(lengths <= 0):
            raise RuntimeError(
                f"{split_name}: non-positive row length."
            )

        splits[split_name] = {
            "indptr": indptr,
            "gene_pos": gene_pos,
            "counts": counts,
            "lengths": lengths,
        }

    return {
        "meta": root_meta,
        "idf": idf,
        "gene_median": gene_median,
        "splits": splits,
    }


def load_matched_k(
    path: Path,
) -> tuple[int, dict[str, Any]]:
    metadata = load_json(path)

    if metadata.get("status") != "PASS":
        raise RuntimeError(
            "Matched-budget status is not PASS."
        )
    if metadata.get("fit_split") != "train_only":
        raise RuntimeError(
            "Matched budget was not fit on train only."
        )
    if metadata.get(
        "test_statistics_used"
    ) is not False:
        raise RuntimeError(
            "Matched budget reports test-statistic use."
        )

    matched_k = int(
        metadata["matched_fixed_k"]
    )

    if matched_k <= 0:
        raise RuntimeError(
            f"Invalid matched K={matched_k}."
        )
    if matched_k >= 65_536:
        raise RuntimeError(
            "Matched K exceeds uint16 capacity."
        )

    return matched_k, metadata


def write_indptr(
    selected_lengths: np.ndarray,
    path: Path,
) -> np.ndarray:
    indptr = np.empty(
        len(selected_lengths) + 1,
        dtype=INDPTR_DTYPE,
    )
    indptr[0] = 0
    np.cumsum(
        selected_lengths.astype(
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


def native_reorder(
    selected_positions: np.ndarray,
    selected_counts: np.ndarray,
    gene_median: np.ndarray,
) -> np.ndarray:
    native_scores = (
        selected_counts.astype(
            np.float64,
            copy=False,
        )
        / gene_median[
            selected_positions
        ]
    )

    # Primary: descending native score.
    # Tie break: ascending local gene position.
    order = np.lexsort(
        (
            selected_positions,
            -native_scores,
        )
    )
    return selected_positions[order]


def validate_method_split(
    split_dir: Path,
    expected_cells: int,
    expected_lengths: np.ndarray,
    matched_k: int,
) -> None:
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
    selected_k = np.load(
        split_dir / "selected_k.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if indptr.shape != (
        expected_cells + 1,
    ):
        raise RuntimeError(
            f"{split_dir}: invalid indptr shape."
        )
    if int(indptr[-1]) != len(
        gene_pos
    ):
        raise RuntimeError(
            f"{split_dir}: indptr/gene_pos mismatch."
        )
    if len(selected_k) != expected_cells:
        raise RuntimeError(
            f"{split_dir}: selected_k length mismatch."
        )

    stored_lengths = np.diff(
        indptr
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        stored_lengths,
        expected_lengths.astype(
            np.int64,
            copy=False,
        ),
    ):
        raise RuntimeError(
            f"{split_dir}: stored lengths mismatch."
        )
    if not np.array_equal(
        selected_k.astype(
            np.int64,
            copy=False,
        ),
        expected_lengths.astype(
            np.int64,
            copy=False,
        ),
    ):
        raise RuntimeError(
            f"{split_dir}: selected_k values mismatch."
        )
    if np.any(
        stored_lengths > matched_k
    ):
        raise RuntimeError(
            f"{split_dir}: row exceeds matched K."
        )
    if (
        len(gene_pos)
        and int(
            np.max(gene_pos)
        )
        >= EXPECTED_GENES
    ):
        raise RuntimeError(
            f"{split_dir}: gene position out of range."
        )


def create_split_outputs(
    root_dir: Path,
    split_name: str,
    selected_lengths: np.ndarray,
    random_seeds: tuple[int, ...],
) -> tuple[
    Path,
    np.ndarray,
    np.memmap,
    dict[int, tuple[Path, np.memmap]],
]:
    fixed_split_dir = (
        root_dir
        / "fixed"
        / split_name
    )
    fixed_split_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_indptr = write_indptr(
        selected_lengths,
        fixed_split_dir / "indptr.npy",
    )
    np.save(
        fixed_split_dir / "selected_k.npy",
        selected_lengths,
        allow_pickle=False,
    )

    total_selected = int(
        output_indptr[-1]
    )

    fixed_output = (
        np.lib.format.open_memmap(
            fixed_split_dir / "gene_pos.npy",
            mode="w+",
            dtype=GENE_POS_DTYPE,
            shape=(total_selected,),
        )
    )

    random_outputs = {}

    for seed in random_seeds:
        random_split_dir = (
            root_dir
            / "random"
            / f"seed{seed}"
            / split_name
        )
        random_split_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        np.save(
            random_split_dir / "indptr.npy",
            output_indptr,
            allow_pickle=False,
        )
        np.save(
            random_split_dir / "selected_k.npy",
            selected_lengths,
            allow_pickle=False,
        )

        random_output = (
            np.lib.format.open_memmap(
                random_split_dir / "gene_pos.npy",
                mode="w+",
                dtype=GENE_POS_DTYPE,
                shape=(total_selected,),
            )
        )
        random_outputs[seed] = (
            random_split_dir,
            random_output,
        )

    return (
        fixed_split_dir,
        output_indptr,
        fixed_output,
        random_outputs,
    )


def build_split(
    split_name: str,
    split: dict[str, Any],
    idf: np.ndarray,
    gene_median: np.ndarray,
    matched_k: int,
    root_building_dir: Path,
    progress_every: int,
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
]:
    base_indptr = split["indptr"]
    base_gene_pos = split["gene_pos"]
    base_counts = split["counts"]
    base_lengths = split["lengths"]

    selected_lengths = np.minimum(
        base_lengths,
        matched_k,
    ).astype(
        SELECTED_K_DTYPE,
        copy=False,
    )

    (
        fixed_split_dir,
        output_indptr,
        fixed_output,
        random_outputs,
    ) = create_split_outputs(
        root_building_dir,
        split_name,
        selected_lengths,
        RANDOM_SEEDS,
    )

    # Independent deterministic RNG stream per split and seed.
    split_code = (
        0 if split_name == "train" else 1
    )
    random_generators = {
        seed: np.random.default_rng(
            np.random.SeedSequence(
                [seed, split_code, matched_k]
            )
        )
        for seed in RANDOM_SEEDS
    }

    n_cells = len(base_lengths)
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
        k = int(
            selected_lengths[row]
        )

        if k <= 0 or k > n_expressed:
            raise RuntimeError(
                f"{split_name} row {row}: invalid "
                f"k={k}, expressed={n_expressed}."
            )

        # Fixed TF-IDF subset.
        tfidf_scores = (
            counts * idf[positions]
        )
        tfidf_order = np.lexsort(
            (
                positions,
                -tfidf_scores,
            )
        )
        fixed_indices = tfidf_order[:k]

        fixed_positions = (
            native_reorder(
                positions[
                    fixed_indices
                ],
                counts[
                    fixed_indices
                ],
                gene_median,
            )
        )

        out_start = int(
            output_indptr[row]
        )
        out_stop = int(
            output_indptr[row + 1]
        )

        fixed_output[
            out_start:out_stop
        ] = fixed_positions.astype(
            GENE_POS_DTYPE,
            copy=False,
        )

        # Random subsets, independently sampled for each seed.
        for seed in RANDOM_SEEDS:
            rng = random_generators[seed]

            if k == n_expressed:
                random_indices = np.arange(
                    n_expressed,
                    dtype=np.int64,
                )
            else:
                random_indices = rng.choice(
                    n_expressed,
                    size=k,
                    replace=False,
                )

            random_positions = (
                native_reorder(
                    positions[
                        random_indices
                    ],
                    counts[
                        random_indices
                    ],
                    gene_median,
                )
            )

            random_outputs[seed][1][
                out_start:out_stop
            ] = random_positions.astype(
                GENE_POS_DTYPE,
                copy=False,
            )

        completed = row + 1
        if (
            completed == 1
            or completed % progress_every == 0
            or completed == n_cells
        ):
            elapsed = (
                time.perf_counter()
                - started
            )
            print(
                f"[fixed+random/{split_name}] "
                f"{completed:,}/{n_cells:,} cells "
                f"({100 * completed / n_cells:.1f}%); "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    fixed_output.flush()
    del fixed_output

    for seed in RANDOM_SEEDS:
        random_outputs[seed][1].flush()

    # Release memmaps explicitly.
    random_split_dirs = {
        seed: random_outputs[seed][0]
        for seed in RANDOM_SEEDS
    }
    random_outputs.clear()

    elapsed = (
        time.perf_counter()
        - started
    )

    mean_selected = float(
        np.mean(selected_lengths)
    )
    median_selected = float(
        np.median(selected_lengths)
    )
    minimum_selected = int(
        np.min(selected_lengths)
    )
    maximum_selected = int(
        np.max(selected_lengths)
    )
    total_selected = int(
        output_indptr[-1]
    )
    under_k_fraction = float(
        np.mean(
            base_lengths < matched_k
        )
    )

    common_metadata = {
        "status": "PASS",
        "split": split_name,
        "matched_fixed_k": int(
            matched_k
        ),
        "cells": int(n_cells),
        "total_selected_genes": (
            total_selected
        ),
        "mean_selected_genes": (
            mean_selected
        ),
        "median_selected_genes": (
            median_selected
        ),
        "minimum_selected_genes": (
            minimum_selected
        ),
        "maximum_selected_genes": (
            maximum_selected
        ),
        "fraction_cells_with_fewer_than_k_expressed_genes": (
            under_k_fraction
        ),
        "cache_generation_time_s_shared": float(
            elapsed
        ),
        "final_sequence_order": (
            "Official Geneformer median-scaled "
            "expression rank after subset selection."
        ),
    }

    fixed_metadata = {
        **common_metadata,
        "method": "matched_fixed_tfidf",
        "selection_score": (
            "raw_count * train_only_IDF"
        ),
        "idf_fit_split": "train_only",
    }
    (
        fixed_split_dir / "meta.json"
    ).write_text(
        json.dumps(
            fixed_metadata,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    random_metadata_by_seed = {}

    for seed in RANDOM_SEEDS:
        random_metadata = {
            **common_metadata,
            "method": (
                "matched_random_fixed_k"
            ),
            "random_seed": int(seed),
            "selection_rule": (
                "Uniform sampling without replacement "
                "among expressed supported genes."
            ),
            "random_stream": (
                "NumPy Generator seeded from "
                "[seed, split_code, matched_k]."
            ),
        }
        random_metadata_by_seed[
            seed
        ] = random_metadata

        (
            random_split_dirs[seed]
            / "meta.json"
        ).write_text(
            json.dumps(
                random_metadata,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    validate_method_split(
        fixed_split_dir,
        expected_cells=n_cells,
        expected_lengths=selected_lengths,
        matched_k=matched_k,
    )

    for seed in RANDOM_SEEDS:
        validate_method_split(
            random_split_dirs[seed],
            expected_cells=n_cells,
            expected_lengths=selected_lengths,
            matched_k=matched_k,
        )

    return (
        fixed_metadata,
        random_metadata_by_seed,
    )


def main() -> int:
    args = parse_args()

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    base = load_base(
        args.base_dir
    )
    matched_k, matched_metadata = (
        load_matched_k(
            args.matched_budget_json
        )
    )

    fixed_final_dir = (
        args.methods_dir
        / f"matched_fixed_tfidf_k{matched_k}"
    )
    random_final_dir = (
        args.methods_dir
        / f"matched_random_fixed_k{matched_k}"
    )

    fixed_building_dir = (
        prepare_building_dir(
            fixed_final_dir,
            args.overwrite,
        )
    )
    random_building_dir = (
        prepare_building_dir(
            random_final_dir,
            args.overwrite,
        )
    )

    # Temporary common root lets us process each base row once.
    temporary_root = (
        args.methods_dir
        / f".fixed_random_k{matched_k}.building"
    )
    if temporary_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{temporary_root} already exists."
            )
        shutil.rmtree(
            temporary_root
        )
    temporary_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    started = time.perf_counter()

    print("=" * 110)
    print(
        "PBMC GENEFORMER MATCHED "
        "FIXED TF-IDF + RANDOM CACHES"
    )
    print("=" * 110)
    print(
        "Base cache:",
        args.base_dir,
    )
    print(
        "Matched-budget file:",
        args.matched_budget_json,
    )
    print(
        "Matched K:",
        matched_k,
    )
    print(
        "Random seeds:",
        list(RANDOM_SEEDS),
    )

    try:
        fixed_split_meta = {}
        random_seed_split_meta = {
            seed: {}
            for seed in RANDOM_SEEDS
        }

        for split_name in (
            "train",
            "test",
        ):
            (
                fixed_meta,
                random_meta,
            ) = build_split(
                split_name=split_name,
                split=base["splits"][
                    split_name
                ],
                idf=base["idf"],
                gene_median=base[
                    "gene_median"
                ],
                matched_k=matched_k,
                root_building_dir=(
                    temporary_root
                ),
                progress_every=(
                    args.progress_every
                ),
            )

            fixed_split_meta[
                split_name
            ] = fixed_meta

            for seed in RANDOM_SEEDS:
                random_seed_split_meta[
                    seed
                ][split_name] = (
                    random_meta[seed]
                )

        # Move fixed split directories into its final building root.
        for split_name in (
            "train",
            "test",
        ):
            shutil.move(
                str(
                    temporary_root
                    / "fixed"
                    / split_name
                ),
                str(
                    fixed_building_dir
                    / split_name
                ),
            )

        fixed_root_meta = {
            "status": "PASS",
            "method": (
                "matched_fixed_tfidf"
            ),
            "method_name": (
                f"matched_fixed_tfidf_k"
                f"{matched_k}"
            ),
            "matched_fixed_k": (
                matched_k
            ),
            "matched_budget_json": str(
                args.matched_budget_json.resolve()
            ),
            "matched_budget_fit_split": (
                "train_only"
            ),
            "test_statistics_used_for_k": False,
            "base_cache": str(
                args.base_dir.resolve()
            ),
            "selection_score": (
                "raw_count * train_only_IDF"
            ),
            "final_sequence_order": (
                "Official Geneformer "
                "median-scaled expression rank."
            ),
            "train": (
                fixed_split_meta["train"]
            ),
            "test": (
                fixed_split_meta["test"]
            ),
        }
        (
            fixed_building_dir
            / "meta.json"
        ).write_text(
            json.dumps(
                fixed_root_meta,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # Move random seed roots into its final building root.
        for seed in RANDOM_SEEDS:
            shutil.move(
                str(
                    temporary_root
                    / "random"
                    / f"seed{seed}"
                ),
                str(
                    random_building_dir
                    / f"seed{seed}"
                ),
            )

        random_root_meta = {
            "status": "PASS",
            "method": (
                "matched_random_fixed_k"
            ),
            "method_name": (
                f"matched_random_fixed_k"
                f"{matched_k}"
            ),
            "matched_fixed_k": (
                matched_k
            ),
            "matched_budget_json": str(
                args.matched_budget_json.resolve()
            ),
            "matched_budget_fit_split": (
                "train_only"
            ),
            "test_statistics_used_for_k": False,
            "base_cache": str(
                args.base_dir.resolve()
            ),
            "random_seeds": list(
                RANDOM_SEEDS
            ),
            "selection_rule": (
                "Uniform sampling without replacement "
                "among expressed supported genes."
            ),
            "final_sequence_order": (
                "Official Geneformer "
                "median-scaled expression rank."
            ),
            "seeds": {
                str(seed): (
                    random_seed_split_meta[
                        seed
                    ]
                )
                for seed in RANDOM_SEEDS
            },
        }
        (
            random_building_dir
            / "meta.json"
        ).write_text(
            json.dumps(
                random_root_meta,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        summary_rows = []

        for method, seed, split_name, metadata in (
            [
                (
                    "matched_fixed_tfidf",
                    None,
                    split_name,
                    fixed_split_meta[
                        split_name
                    ],
                )
                for split_name in (
                    "train",
                    "test",
                )
            ]
            + [
                (
                    "matched_random_fixed_k",
                    seed,
                    split_name,
                    random_seed_split_meta[
                        seed
                    ][split_name],
                )
                for seed in RANDOM_SEEDS
                for split_name in (
                    "train",
                    "test",
                )
            ]
        ):
            summary_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "split": split_name,
                    "matched_fixed_k": (
                        matched_k
                    ),
                    "cells": metadata[
                        "cells"
                    ],
                    "mean_selected_genes": (
                        metadata[
                            "mean_selected_genes"
                        ]
                    ),
                    "median_selected_genes": (
                        metadata[
                            "median_selected_genes"
                        ]
                    ),
                    "minimum_selected_genes": (
                        metadata[
                            "minimum_selected_genes"
                        ]
                    ),
                    "maximum_selected_genes": (
                        metadata[
                            "maximum_selected_genes"
                        ]
                    ),
                    "fraction_cells_with_fewer_than_k_expressed_genes": (
                        metadata[
                            "fraction_cells_with_fewer_than_k_expressed_genes"
                        ]
                    ),
                }
            )

        pd.DataFrame(
            summary_rows
        ).to_csv(
            args.methods_dir
            / "matched_cache_summary.csv",
            index=False,
        )

        shutil.rmtree(
            temporary_root
        )

        os.replace(
            fixed_building_dir,
            fixed_final_dir,
        )
        os.replace(
            random_building_dir,
            random_final_dir,
        )

    except Exception:
        print(
            "\n[ERROR] Build failed. "
            "Incomplete outputs remain only in "
            "*.building directories.",
            flush=True,
        )
        raise

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print("=" * 110)
    print(
        "PBMC MATCHED CACHE SUMMARY"
    )
    print("=" * 110)
    print(
        "Matched K:",
        matched_k,
    )
    print(
        "Adaptive train target mean:",
        f"{float(matched_metadata['target_adaptive_train_mean']):.6f}",
    )
    print(
        "Fixed expected train mean:",
        f"{float(matched_metadata['matched_fixed_expected_train_mean']):.6f}",
    )

    for split_name in (
        "train",
        "test",
    ):
        metadata = fixed_split_meta[
            split_name
        ]
        print(
            f"{split_name}: "
            f"mean={metadata['mean_selected_genes']:.3f}, "
            f"median={metadata['median_selected_genes']:.1f}, "
            f"under_K_fraction="
            f"{metadata['fraction_cells_with_fewer_than_k_expressed_genes']:.6f}"
        )

    print(
        "Random seeds built:",
        list(RANDOM_SEEDS),
    )
    print(
        "Fixed output:",
        fixed_final_dir,
    )
    print(
        "Random output:",
        random_final_dir,
    )
    print(
        "Elapsed:",
        f"{elapsed / 60:.1f} min",
    )
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
