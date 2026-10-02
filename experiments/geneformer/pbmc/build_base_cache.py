#!/usr/bin/env python3
"""Build the PBMC P5-held-out Geneformer base cache.

This is the first materialized Geneformer cache for PBMC. It uses the
validated audit outputs and does not reuse any scGPT cache.

Scientific invariants
---------------------
- Frozen split: P1/P2/P3/P4/P6/P7/P8 train; P5 test.
- Gene mapping: validated Geneformer mapping from the PBMC audit.
- Raw counts: nonnegative integer-like H5AD X.
- Normalization denominator: original full-gene row sum, before dropping
  genes unsupported by Geneformer. This matches the intended Geneformer
  tokenization logic.
- IDF: computed from TRAIN cells only with smooth IDF
      log((N_train + 1) / (df + 1)) + 1
  and then reused for test cells.
- Geneformer native-ranking denominator: official Geneformer gene median
  aligned to the supported Ensembl genes.

Storage format
--------------
The sparse matrix is stored split-wise as aligned NPY arrays:
    indptr.npy      int64, shape [n_cells + 1]
    gene_pos.npy    uint16, local column positions in [0, 14809)
    counts.npy      uint32, raw counts
    n_counts.npy    float64, original all-gene library size per cell

The use of uint16 for gene positions is safe because 14,809 < 65,536.
The use of uint32 for counts is validated during the first pass.

Outputs
-------
data/processed/pbmc_seurat_v4/p5_holdout/geneformer_cache/base/
├── meta.json
├── gene_table.csv
├── source_indices.npy
├── token_ids.npy
├── gene_median.npy
├── idf.npy
├── train_df.npy
├── train/
│   ├── meta.json
│   ├── indptr.npy
│   ├── gene_pos.npy
│   ├── counts.npy
│   ├── n_counts.npy
│   ├── obs_names.npy
│   ├── labels_l1.npy
│   ├── labels_l2.npy
│   ├── donors.npy
│   └── times.npy
└── test/
    └── same split-level files
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_DIR = Path(".")

DEFAULT_H5AD = Path(
    "./data/raw/"
    "pbmc_seurat_v4/pbmc_seurat_v4_rna_only.h5ad"
)

DEFAULT_SPLIT = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "pbmc_seurat_v4"
    / "p5_holdout"
    / "split_indices.npz"
)

DEFAULT_AUDIT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "geneformer_pbmc_audit"
)

DEFAULT_DICTIONARY_DIR = (
    PROJECT_DIR
    / "Geneformer"
    / "geneformer"
    / "gene_dictionaries_30m"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "pbmc_seurat_v4"
    / "p5_holdout"
    / "geneformer_cache"
    / "base"
)

EXPECTED_TOTAL_CELLS = 152_094
EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957
EXPECTED_TOTAL_GENES = 20_729
EXPECTED_SUPPORTED_GENES = 14_809
EXPECTED_TEST_DONOR = "P5"

MAIN_LABEL = "celltype.l1"
SECONDARY_LABEL = "celltype.l2"
DONOR_COLUMN = "donor"
TIME_COLUMN = "time"

GENE_POS_DTYPE = np.uint16
COUNT_DTYPE = np.uint32
INDPTR_DTYPE = np.int64
N_COUNTS_DTYPE = np.float64

SMOOTH_IDF_FORMULA = (
    "log((n_train_cells + 1) / (document_frequency + 1)) + 1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=DEFAULT_H5AD,
    )
    parser.add_argument(
        "--split-npz",
        type=Path,
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
    )
    parser.add_argument(
        "--dictionary-dir",
        type=Path,
        default=DEFAULT_DICTIONARY_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
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


def load_pickle(path: Path) -> Any:
    require_file(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def find_unique_file(
    directory: Path,
    pattern: str,
) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} "
            f"in {directory}; found {len(matches)}: {matches}"
        )
    return matches[0]


def resolve_split_keys(
    split: np.lib.npyio.NpzFile,
) -> tuple[str, str]:
    candidate_map = {
        "train": [
            "train_idx",
            "train_indices",
        ],
        "test": [
            "test_idx",
            "test_indices",
        ],
    }

    resolved: dict[str, str] = {}

    for split_name, candidates in (
        candidate_map.items()
    ):
        matches = [
            key
            for key in candidates
            if key in split.files
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve {split_name} key. "
                f"Candidates={candidates}; available={split.files}; "
                f"matches={matches}"
            )
        resolved[split_name] = matches[0]

    return (
        resolved["train"],
        resolved["test"],
    )


def load_and_validate_audit(
    audit_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_directory(audit_dir)

    summary_path = (
        audit_dir / "audit_summary.json"
    )
    mapping_path = (
        audit_dir / "gene_mapping.csv"
    )

    require_file(summary_path)
    require_file(mapping_path)

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if summary.get("status") != "PASS":
        raise RuntimeError(
            "PBMC Geneformer audit status is not PASS."
        )

    mapping = pd.read_csv(mapping_path)

    required_columns = {
        "source_index",
        "source_feature",
        "ensembl_id",
        "token_id",
        "mapping_method",
        "geneformer_supported",
    }
    missing = required_columns - set(
        mapping.columns
    )
    if missing:
        raise RuntimeError(
            "Audit gene_mapping.csv missing columns: "
            f"{sorted(missing)}"
        )

    supported_mask = (
        mapping["geneformer_supported"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    supported = (
        mapping.loc[supported_mask]
        .copy()
        .sort_values("source_index")
        .reset_index(drop=True)
    )

    if len(supported) != EXPECTED_SUPPORTED_GENES:
        raise RuntimeError(
            f"Supported genes={len(supported)}; expected "
            f"{EXPECTED_SUPPORTED_GENES}."
        )

    supported["source_index"] = (
        pd.to_numeric(
            supported["source_index"],
            errors="raise",
        ).astype(np.int64)
    )
    supported["token_id"] = (
        pd.to_numeric(
            supported["token_id"],
            errors="raise",
        ).astype(np.int64)
    )

    if supported["source_index"].duplicated().any():
        raise RuntimeError(
            "Duplicate supported source_index values."
        )
    if supported["token_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate supported Geneformer token IDs."
        )
    if not np.all(
        np.diff(
            supported["source_index"]
            .to_numpy()
        )
        > 0
    ):
        raise RuntimeError(
            "Supported source indices must be strictly increasing."
        )

    return supported, summary


def load_gene_medians(
    dictionary_dir: Path,
    ensembl_ids: np.ndarray,
) -> tuple[np.ndarray, Path]:
    require_directory(dictionary_dir)

    median_path = find_unique_file(
        dictionary_dir,
        "gene_median_dictionary*gc30M*.pkl",
    )
    median_raw = load_pickle(
        median_path
    )
    median_dictionary = {
        str(key).upper(): float(value)
        for key, value in median_raw.items()
    }

    medians = np.empty(
        len(ensembl_ids),
        dtype=np.float64,
    )
    missing = []

    for index, ensembl_id in enumerate(
        ensembl_ids.astype(str)
    ):
        value = median_dictionary.get(
            ensembl_id.upper()
        )
        if value is None:
            missing.append(ensembl_id)
            continue
        medians[index] = value

    if missing:
        raise RuntimeError(
            "Official Geneformer gene medians are missing "
            f"for {len(missing)} supported genes. "
            f"Examples={missing[:20]}"
        )

    if not np.all(np.isfinite(medians)):
        raise RuntimeError(
            "Non-finite official Geneformer gene medians."
        )
    if np.any(medians <= 0):
        bad = np.flatnonzero(
            medians <= 0
        )[:20]
        raise RuntimeError(
            "Non-positive Geneformer gene medians at "
            f"positions {bad.tolist()}."
        )

    return medians, median_path


def unicode_array(
    values: pd.Series | pd.Index | np.ndarray,
) -> np.ndarray:
    return np.asarray(
        values.astype(str),
        dtype=str,
    )


def load_backed_chunk(
    adata: ad.AnnData,
    global_rows: np.ndarray,
    source_indices: np.ndarray,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
]:
    """Read rows from backed AnnData while preserving requested order."""
    global_rows = np.asarray(
        global_rows,
        dtype=np.int64,
    )

    sort_order = np.argsort(
        global_rows,
        kind="stable",
    )
    sorted_rows = global_rows[
        sort_order
    ]

    if (
        len(sorted_rows) > 1
        and np.any(
            np.diff(sorted_rows) == 0
        )
    ):
        raise RuntimeError(
            "Duplicate global rows in a chunk."
        )

    inverse_order = np.empty_like(
        sort_order
    )
    inverse_order[sort_order] = np.arange(
        len(sort_order)
    )

    full_matrix = adata[
        sorted_rows,
        :,
    ].X

    if sparse.issparse(full_matrix):
        full_matrix = full_matrix.tocsr(
            copy=True
        )
    else:
        full_matrix = sparse.csr_matrix(
            np.asarray(full_matrix)
        )

    full_matrix.sum_duplicates()
    full_matrix.eliminate_zeros()
    full_matrix.sort_indices()

    if not np.array_equal(
        sort_order,
        np.arange(len(sort_order)),
    ):
        full_matrix = full_matrix[
            inverse_order,
            :,
        ].tocsr()

    original_row_sums = np.asarray(
        full_matrix.sum(axis=1)
    ).ravel().astype(
        N_COUNTS_DTYPE,
        copy=False,
    )

    mapped_matrix = full_matrix[
        :,
        source_indices,
    ].tocsr()
    mapped_matrix.sum_duplicates()
    mapped_matrix.eliminate_zeros()
    mapped_matrix.sort_indices()

    return (
        mapped_matrix,
        original_row_sums,
    )


def first_pass(
    adata: ad.AnnData,
    split_indices: np.ndarray,
    source_indices: np.ndarray,
    chunk_size: int,
    split_name: str,
    collect_df: bool,
) -> dict[str, Any]:
    n_cells = len(split_indices)
    row_nnz = np.empty(
        n_cells,
        dtype=np.int64,
    )
    n_counts = np.empty(
        n_cells,
        dtype=N_COUNTS_DTYPE,
    )

    document_frequency = (
        np.zeros(
            len(source_indices),
            dtype=np.int64,
        )
        if collect_df
        else None
    )

    maximum_count = 0
    minimum_positive_count = None
    total_nnz = 0

    started = time.time()

    for start in range(
        0,
        n_cells,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_cells,
        )

        matrix, row_sums = (
            load_backed_chunk(
                adata,
                split_indices[start:stop],
                source_indices,
            )
        )

        if matrix.nnz:
            values = np.asarray(
                matrix.data
            )
            if np.any(values < 0):
                raise RuntimeError(
                    f"Negative counts in {split_name}."
                )
            residual = np.max(
                np.abs(
                    values
                    - np.rint(values)
                )
            )
            if residual > 1e-6:
                raise RuntimeError(
                    f"Non-integer-like counts in {split_name}; "
                    f"max residual={residual}."
                )

            chunk_max = int(
                np.max(values)
            )
            chunk_min = int(
                np.min(values)
            )
            maximum_count = max(
                maximum_count,
                chunk_max,
            )
            minimum_positive_count = (
                chunk_min
                if minimum_positive_count is None
                else min(
                    minimum_positive_count,
                    chunk_min,
                )
            )

        chunk_row_nnz = np.diff(
            matrix.indptr
        ).astype(
            np.int64,
            copy=False,
        )
        row_nnz[start:stop] = (
            chunk_row_nnz
        )
        n_counts[start:stop] = row_sums
        total_nnz += int(matrix.nnz)

        if document_frequency is not None:
            document_frequency += np.asarray(
                matrix.getnnz(axis=0)
            ).ravel().astype(
                np.int64,
                copy=False,
            )

        if (
            start == 0
            or stop == n_cells
            or stop % (chunk_size * 25)
            == 0
        ):
            elapsed = time.time() - started
            print(
                f"[{split_name} pass 1] "
                f"{stop:,}/{n_cells:,} cells; "
                f"nnz={total_nnz:,}; "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    if np.any(n_counts <= 0):
        bad = np.flatnonzero(
            n_counts <= 0
        )[:20]
        raise RuntimeError(
            f"Non-positive original n_counts in {split_name}: "
            f"{bad.tolist()}"
        )

    if maximum_count > np.iinfo(
        COUNT_DTYPE
    ).max:
        raise RuntimeError(
            f"Count {maximum_count} exceeds "
            f"{COUNT_DTYPE} capacity."
        )

    if np.max(row_nnz) > len(
        source_indices
    ):
        raise RuntimeError(
            f"Invalid row nnz in {split_name}."
        )

    return {
        "row_nnz": row_nnz,
        "n_counts": n_counts,
        "document_frequency": (
            document_frequency
        ),
        "total_nnz": int(total_nnz),
        "maximum_count": int(
            maximum_count
        ),
        "minimum_positive_count": (
            int(minimum_positive_count)
            if minimum_positive_count
            is not None
            else None
        ),
        "elapsed_seconds": float(
            time.time() - started
        ),
    }


def write_split_cache(
    adata: ad.AnnData,
    split_indices: np.ndarray,
    source_indices: np.ndarray,
    first_pass_result: dict[str, Any],
    output_dir: Path,
    chunk_size: int,
    split_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    n_cells = len(split_indices)
    row_nnz = first_pass_result[
        "row_nnz"
    ]
    expected_total_nnz = int(
        first_pass_result["total_nnz"]
    )

    indptr = np.empty(
        n_cells + 1,
        dtype=INDPTR_DTYPE,
    )
    indptr[0] = 0
    np.cumsum(
        row_nnz,
        out=indptr[1:],
    )

    if int(indptr[-1]) != (
        expected_total_nnz
    ):
        raise RuntimeError(
            f"{split_name}: indptr total does not "
            "match first-pass nnz."
        )

    np.save(
        output_dir / "indptr.npy",
        indptr,
        allow_pickle=False,
    )
    np.save(
        output_dir / "n_counts.npy",
        first_pass_result[
            "n_counts"
        ],
        allow_pickle=False,
    )

    gene_pos = np.lib.format.open_memmap(
        output_dir / "gene_pos.npy",
        mode="w+",
        dtype=GENE_POS_DTYPE,
        shape=(expected_total_nnz,),
    )
    counts = np.lib.format.open_memmap(
        output_dir / "counts.npy",
        mode="w+",
        dtype=COUNT_DTYPE,
        shape=(expected_total_nnz,),
    )

    started = time.time()
    write_offset = 0

    for start in range(
        0,
        n_cells,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_cells,
        )

        matrix, row_sums = (
            load_backed_chunk(
                adata,
                split_indices[start:stop],
                source_indices,
            )
        )

        expected_row_sums = (
            first_pass_result[
                "n_counts"
            ][start:stop]
        )
        if not np.allclose(
            row_sums,
            expected_row_sums,
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"{split_name}: row sums changed "
                "between passes."
            )

        chunk_nnz = int(matrix.nnz)
        next_offset = (
            write_offset + chunk_nnz
        )

        gene_pos[
            write_offset:next_offset
        ] = matrix.indices.astype(
            GENE_POS_DTYPE,
            copy=False,
        )
        counts[
            write_offset:next_offset
        ] = np.rint(
            matrix.data
        ).astype(
            COUNT_DTYPE,
            copy=False,
        )

        write_offset = next_offset

        if (
            start == 0
            or stop == n_cells
            or stop % (chunk_size * 25)
            == 0
        ):
            elapsed = time.time() - started
            print(
                f"[{split_name} pass 2] "
                f"{stop:,}/{n_cells:,} cells; "
                f"written nnz={write_offset:,}; "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    gene_pos.flush()
    counts.flush()
    del gene_pos
    del counts

    if write_offset != expected_total_nnz:
        raise RuntimeError(
            f"{split_name}: wrote {write_offset} nnz; "
            f"expected {expected_total_nnz}."
        )

    split_obs = adata.obs.iloc[
        split_indices
    ]

    np.save(
        output_dir / "obs_names.npy",
        unicode_array(
            adata.obs_names[
                split_indices
            ]
        ),
        allow_pickle=False,
    )
    np.save(
        output_dir / "labels_l1.npy",
        unicode_array(
            split_obs[MAIN_LABEL]
        ),
        allow_pickle=False,
    )
    np.save(
        output_dir / "labels_l2.npy",
        unicode_array(
            split_obs[
                SECONDARY_LABEL
            ]
        ),
        allow_pickle=False,
    )
    np.save(
        output_dir / "donors.npy",
        unicode_array(
            split_obs[DONOR_COLUMN]
        ),
        allow_pickle=False,
    )
    np.save(
        output_dir / "times.npy",
        unicode_array(
            split_obs[TIME_COLUMN]
        ),
        allow_pickle=False,
    )

    stored_indptr = np.load(
        output_dir / "indptr.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    stored_gene_pos = np.load(
        output_dir / "gene_pos.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    stored_counts = np.load(
        output_dir / "counts.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    if stored_indptr.shape != (
        n_cells + 1,
    ):
        raise RuntimeError(
            f"{split_name}: stored indptr shape error."
        )
    if int(stored_indptr[-1]) != len(
        stored_gene_pos
    ):
        raise RuntimeError(
            f"{split_name}: stored indptr/gene_pos mismatch."
        )
    if len(stored_counts) != len(
        stored_gene_pos
    ):
        raise RuntimeError(
            f"{split_name}: stored count/gene_pos mismatch."
        )
    if (
        len(stored_gene_pos)
        and int(
            np.max(
                stored_gene_pos
            )
        )
        >= len(source_indices)
    ):
        raise RuntimeError(
            f"{split_name}: stored gene position "
            "out of bounds."
        )

    metadata = {
        "status": "PASS",
        "split": split_name,
        "cells": int(n_cells),
        "genes": int(
            len(source_indices)
        ),
        "nnz": int(
            expected_total_nnz
        ),
        "mean_nonzero_supported_genes": float(
            np.mean(row_nnz)
        ),
        "median_nonzero_supported_genes": float(
            np.median(row_nnz)
        ),
        "minimum_nonzero_supported_genes": int(
            np.min(row_nnz)
        ),
        "maximum_nonzero_supported_genes": int(
            np.max(row_nnz)
        ),
        "mean_original_n_counts": float(
            np.mean(
                first_pass_result[
                    "n_counts"
                ]
            )
        ),
        "median_original_n_counts": float(
            np.median(
                first_pass_result[
                    "n_counts"
                ]
            )
        ),
        "minimum_positive_count": (
            first_pass_result[
                "minimum_positive_count"
            ]
        ),
        "maximum_count": int(
            first_pass_result[
                "maximum_count"
            ]
        ),
        "dtypes": {
            "indptr": str(
                np.dtype(INDPTR_DTYPE)
            ),
            "gene_pos": str(
                np.dtype(GENE_POS_DTYPE)
            ),
            "counts": str(
                np.dtype(COUNT_DTYPE)
            ),
            "n_counts": str(
                np.dtype(N_COUNTS_DTYPE)
            ),
        },
        "first_pass_seconds": float(
            first_pass_result[
                "elapsed_seconds"
            ]
        ),
        "second_pass_seconds": float(
            time.time() - started
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

    return metadata


def main() -> int:
    args = parse_args()

    require_file(args.h5ad)
    require_file(args.split_npz)
    require_directory(args.audit_dir)
    require_directory(
        args.dictionary_dir
    )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be positive."
        )

    final_dir = args.output_dir
    building_dir = final_dir.with_name(
        final_dir.name + ".building"
    )

    for path in [
        final_dir,
        building_dir,
    ]:
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{path} already exists. "
                    "Inspect it first, or rerun with "
                    "--overwrite to rebuild."
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

    start_total = time.time()

    supported, audit_summary = (
        load_and_validate_audit(
            args.audit_dir
        )
    )

    source_indices = supported[
        "source_index"
    ].to_numpy(
        dtype=np.int64
    )
    token_ids = supported[
        "token_id"
    ].to_numpy(
        dtype=np.int64
    )
    ensembl_ids = supported[
        "ensembl_id"
    ].astype(str).to_numpy()

    gene_median, median_path = (
        load_gene_medians(
            args.dictionary_dir,
            ensembl_ids,
        )
    )

    split_npz = np.load(
        args.split_npz,
        allow_pickle=False,
    )
    train_key, test_key = (
        resolve_split_keys(
            split_npz
        )
    )
    train_indices = np.asarray(
        split_npz[train_key],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        split_npz[test_key],
        dtype=np.int64,
    )

    if len(train_indices) != (
        EXPECTED_TRAIN_CELLS
    ):
        raise RuntimeError(
            "Unexpected PBMC train-cell count."
        )
    if len(test_indices) != (
        EXPECTED_TEST_CELLS
    ):
        raise RuntimeError(
            "Unexpected PBMC test-cell count."
        )

    np.save(
        building_dir
        / "source_indices.npy",
        source_indices,
        allow_pickle=False,
    )
    np.save(
        building_dir / "token_ids.npy",
        token_ids,
        allow_pickle=False,
    )
    np.save(
        building_dir / "gene_median.npy",
        gene_median,
        allow_pickle=False,
    )

    gene_table = supported[
        [
            "source_index",
            "source_feature",
            "ensembl_id",
            "token_id",
            "mapping_method",
        ]
    ].copy()
    gene_table.insert(
        0,
        "gene_pos",
        np.arange(
            len(gene_table),
            dtype=np.int64,
        ),
    )
    gene_table[
        "geneformer_median"
    ] = gene_median
    gene_table.to_csv(
        building_dir
        / "gene_table.csv",
        index=False,
    )

    adata = ad.read_h5ad(
        args.h5ad,
        backed="r",
    )

    try:
        if adata.n_obs != (
            EXPECTED_TOTAL_CELLS
        ):
            raise RuntimeError(
                "Unexpected total PBMC cells."
            )
        if adata.n_vars != (
            EXPECTED_TOTAL_GENES
        ):
            raise RuntimeError(
                "Unexpected total PBMC genes."
            )

        if not np.array_equal(
            np.asarray(
                adata.var_names[
                    source_indices
                ].astype(str)
            ),
            supported[
                "source_feature"
            ].astype(str).to_numpy(),
        ):
            raise RuntimeError(
                "Audit mapping source features no longer "
                "match H5AD var_names."
            )

        test_donors = sorted(
            adata.obs.iloc[
                test_indices
            ][DONOR_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )
        if test_donors != [
            EXPECTED_TEST_DONOR
        ]:
            raise RuntimeError(
                f"Expected P5 test donor; observed "
                f"{test_donors}."
            )

        print("=" * 100)
        print(
            "PBMC GENEFORMER BASE CACHE"
        )
        print("=" * 100)
        print(
            "H5AD:",
            args.h5ad,
        )
        print(
            "Train/Test:",
            len(train_indices),
            "/",
            len(test_indices),
        )
        print(
            "Supported genes:",
            len(source_indices),
        )
        print(
            "Chunk size:",
            args.chunk_size,
        )
        print(
            "Output building directory:",
            building_dir,
        )

        train_first = first_pass(
            adata=adata,
            split_indices=train_indices,
            source_indices=source_indices,
            chunk_size=args.chunk_size,
            split_name="train",
            collect_df=True,
        )

        train_df = train_first[
            "document_frequency"
        ]
        if train_df is None:
            raise RuntimeError(
                "Train document frequency missing."
            )
        if np.any(train_df < 0):
            raise RuntimeError(
                "Negative train document frequency."
            )
        if np.any(
            train_df > len(train_indices)
        ):
            raise RuntimeError(
                "Train document frequency exceeds "
                "train-cell count."
            )

        idf = (
            np.log(
                (
                    len(train_indices) + 1.0
                )
                / (
                    train_df.astype(
                        np.float64
                    )
                    + 1.0
                )
            )
            + 1.0
        )

        if not np.all(np.isfinite(idf)):
            raise RuntimeError(
                "Non-finite train-only IDF."
            )

        np.save(
            building_dir
            / "train_df.npy",
            train_df.astype(
                np.int64,
                copy=False,
            ),
            allow_pickle=False,
        )
        np.save(
            building_dir / "idf.npy",
            idf.astype(
                np.float64,
                copy=False,
            ),
            allow_pickle=False,
        )

        test_first = first_pass(
            adata=adata,
            split_indices=test_indices,
            source_indices=source_indices,
            chunk_size=args.chunk_size,
            split_name="test",
            collect_df=False,
        )

        train_meta = write_split_cache(
            adata=adata,
            split_indices=train_indices,
            source_indices=source_indices,
            first_pass_result=train_first,
            output_dir=(
                building_dir / "train"
            ),
            chunk_size=args.chunk_size,
            split_name="train",
        )

        test_meta = write_split_cache(
            adata=adata,
            split_indices=test_indices,
            source_indices=source_indices,
            first_pass_result=test_first,
            output_dir=(
                building_dir / "test"
            ),
            chunk_size=args.chunk_size,
            split_name="test",
        )

    finally:
        if getattr(
            adata,
            "file",
            None,
        ) is not None:
            adata.file.close()

    root_meta = {
        "status": "PASS",
        "dataset": "pbmc_seurat_v4",
        "split_name": "p5_holdout",
        "h5ad": str(
            args.h5ad.resolve()
        ),
        "split_npz": str(
            args.split_npz.resolve()
        ),
        "resolved_split_keys": {
            "train": train_key,
            "test": test_key,
        },
        "audit_summary": str(
            (
                args.audit_dir
                / "audit_summary.json"
            ).resolve()
        ),
        "official_gene_median_dictionary": str(
            median_path.resolve()
        ),
        "cells": {
            "total": (
                EXPECTED_TOTAL_CELLS
            ),
            "train": (
                EXPECTED_TRAIN_CELLS
            ),
            "test": (
                EXPECTED_TEST_CELLS
            ),
        },
        "genes": {
            "source_total": (
                EXPECTED_TOTAL_GENES
            ),
            "geneformer_supported": (
                EXPECTED_SUPPORTED_GENES
            ),
        },
        "labels": {
            "main": MAIN_LABEL,
            "secondary": SECONDARY_LABEL,
        },
        "held_out_donor": (
            EXPECTED_TEST_DONOR
        ),
        "normalization": {
            "n_counts_definition": (
                "Original H5AD X row sum over all "
                "20,729 source genes before "
                "Geneformer filtering."
            ),
            "gene_median_source": (
                "Official Geneformer gc30M "
                "gene median dictionary."
            ),
        },
        "idf": {
            "fit_split": "train_only",
            "formula": (
                SMOOTH_IDF_FORMULA
            ),
            "minimum": float(
                np.min(idf)
            ),
            "maximum": float(
                np.max(idf)
            ),
            "mean": float(
                np.mean(idf)
            ),
        },
        "storage": {
            "gene_pos_dtype": str(
                np.dtype(
                    GENE_POS_DTYPE
                )
            ),
            "counts_dtype": str(
                np.dtype(
                    COUNT_DTYPE
                )
            ),
            "indptr_dtype": str(
                np.dtype(
                    INDPTR_DTYPE
                )
            ),
            "n_counts_dtype": str(
                np.dtype(
                    N_COUNTS_DTYPE
                )
            ),
        },
        "train": train_meta,
        "test": test_meta,
        "elapsed_seconds": float(
            time.time() - start_total
        ),
    }

    (
        building_dir / "meta.json"
    ).write_text(
        json.dumps(
            root_meta,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        building_dir,
        final_dir,
    )

    print()
    print("=" * 100)
    print(
        "PBMC GENEFORMER BASE CACHE READY"
    )
    print("=" * 100)
    print(
        "Output:",
        final_dir,
    )
    print(
        "Train nnz:",
        f"{train_meta['nnz']:,}",
    )
    print(
        "Test nnz:",
        f"{test_meta['nnz']:,}",
    )
    print(
        "Train mean supported genes:",
        f"{train_meta['mean_nonzero_supported_genes']:.3f}",
    )
    print(
        "Test mean supported genes:",
        f"{test_meta['mean_nonzero_supported_genes']:.3f}",
    )
    print(
        "IDF range:",
        f"{root_meta['idf']['minimum']:.6f}",
        "to",
        f"{root_meta['idf']['maximum']:.6f}",
    )
    print(
        "Elapsed:",
        f"{root_meta['elapsed_seconds'] / 60:.1f} min",
    )
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
