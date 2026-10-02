#!/usr/bin/env python3
"""Build the Geneformer-compatible base cache for Kang 2018.

This step performs only common preprocessing shared by all methods:

- maps source gene symbols to the local Geneformer V1 GC30M vocabulary,
- materializes row-efficient CSR matrices in deterministic token-ID order,
- preserves original uncompressed per-cell total counts,
- verifies those totals against obs["nCount_RNA"],
- computes IDF from the training split only,
- saves cell order, labels, donor/condition metadata, gene metadata, and hashes.

It does not perform Random, Fixed TF-IDF, AdaGeneBudget, or Geneformer
native ranking/tokenization. Those method-specific caches are built later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


PROJECT_DIR = Path(".")
GENEFORMER_REPO = PROJECT_DIR / "Geneformer"
DICTIONARY_DIR = (
    GENEFORMER_REPO
    / "geneformer"
    / "gene_dictionaries_30m"
)

TRAIN_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "train.h5ad"
)
TEST_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "test.h5ad"
)
DEFAULT_CACHE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "base"
)

EXPECTED_TRAIN_CELLS = 19_583
EXPECTED_TEST_CELLS = 5_090
EXPECTED_SOURCE_GENES = 15_706
EXPECTED_SUPPORTED_GENES = 12_288

LABEL_COLUMN = "cell_type"
CONDITION_COLUMN = "label"
DONOR_COLUMN = "replicate"
COUNT_COLUMN = "nCount_RNA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_PATH,
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_PATH,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    bytes_read = 0
    next_report = 1024**3

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)
            bytes_read += len(block)

            if bytes_read >= next_report:
                print(
                    f"[hash] {path.name}: "
                    f"{bytes_read / 1024**3:.1f} GiB read",
                    flush=True,
                )
                next_report += 1024**3

    return digest.hexdigest()


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

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def validate_source_schema(
    adata: ad.AnnData,
    expected_cells: int,
    split_name: str,
) -> None:
    if adata.n_obs != expected_cells:
        raise RuntimeError(
            f"{split_name}: expected {expected_cells:,} cells, "
            f"found {adata.n_obs:,}."
        )

    if adata.n_vars != EXPECTED_SOURCE_GENES:
        raise RuntimeError(
            f"{split_name}: expected {EXPECTED_SOURCE_GENES:,} genes, "
            f"found {adata.n_vars:,}."
        )

    if not adata.var_names.is_unique:
        raise RuntimeError(
            f"{split_name}: source var_names are not unique."
        )

    required_obs_columns = [
        LABEL_COLUMN,
        CONDITION_COLUMN,
        DONOR_COLUMN,
        COUNT_COLUMN,
    ]

    missing = [
        column
        for column in required_obs_columns
        if column not in adata.obs.columns
    ]

    if missing:
        raise KeyError(
            f"{split_name}: missing obs columns: {missing}"
        )


def build_gene_mapping(
    source_features: np.ndarray,
) -> pd.DataFrame:
    name_to_id = load_pickle(
        DICTIONARY_DIR / "gene_name_id_dict_gc30M.pkl"
    )
    ensembl_mapping = load_pickle(
        DICTIONARY_DIR / "ensembl_mapping_dict_gc30M.pkl"
    )
    token_dictionary = load_pickle(
        DICTIONARY_DIR / "token_dictionary_gc30M.pkl"
    )
    gene_medians = load_pickle(
        DICTIONARY_DIR / "gene_median_dictionary_gc30M.pkl"
    )

    uppercase_name_to_id: dict[str, str] = {}

    for symbol, ensembl_id in name_to_id.items():
        uppercase_symbol = str(symbol).upper()
        previous = uppercase_name_to_id.get(uppercase_symbol)

        if previous is not None and previous != ensembl_id:
            raise RuntimeError(
                "Ambiguous uppercase symbol mapping: "
                f"{uppercase_symbol}: {previous} vs {ensembl_id}"
            )

        uppercase_name_to_id[uppercase_symbol] = str(ensembl_id)

    rows: list[dict[str, Any]] = []

    for source_index, raw_feature in enumerate(source_features):
        feature = str(raw_feature)
        upper_feature = feature.upper()
        ensembl_without_version = upper_feature.split(".", maxsplit=1)[0]

        is_direct_ensembl_gene_id = (
            ensembl_without_version.startswith("ENSG")
            and ensembl_without_version[4:].isdigit()
        )

        if is_direct_ensembl_gene_id:
            ensembl_id = ensembl_without_version
            method = "direct_ensembl"
        else:
            exact_id = name_to_id.get(feature)

            if exact_id is not None:
                ensembl_id = str(exact_id).upper()
                method = "symbol_exact"
            else:
                uppercase_id = uppercase_name_to_id.get(upper_feature)

                if uppercase_id is None:
                    continue

                ensembl_id = str(uppercase_id).upper()
                method = "symbol_uppercase"

        canonical_id = str(
            ensembl_mapping.get(ensembl_id, ensembl_id)
        ).upper()

        if canonical_id not in token_dictionary:
            continue

        if canonical_id not in gene_medians:
            raise RuntimeError(
                "Token dictionary gene lacks a median: "
                f"{canonical_id}"
            )

        rows.append(
            {
                "source_index": int(source_index),
                "source_feature": feature,
                "ensembl_id": ensembl_id,
                "canonical_ensembl_id": canonical_id,
                "token_id": int(token_dictionary[canonical_id]),
                "gene_median": float(gene_medians[canonical_id]),
                "mapping_method": method,
            }
        )

    mapping = pd.DataFrame(rows)

    if mapping.empty:
        raise RuntimeError(
            "No Kang genes mapped to the Geneformer vocabulary."
        )

    duplicate_canonical = mapping[
        mapping.duplicated(
            subset=["canonical_ensembl_id"],
            keep=False,
        )
    ]

    if not duplicate_canonical.empty:
        examples = (
            duplicate_canonical[
                [
                    "source_feature",
                    "canonical_ensembl_id",
                ]
            ]
            .head(20)
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "Kang unexpectedly contains duplicate mappings. "
            "The audit reported zero duplicates, so stop rather than "
            f"silently changing behavior. Examples: {examples}"
        )

    if mapping["token_id"].duplicated().any():
        raise RuntimeError(
            "Different mapped genes share a token ID."
        )

    mapping = (
        mapping
        .sort_values(
            ["token_id", "source_index"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    mapping.insert(
        0,
        "cache_column",
        np.arange(len(mapping), dtype=np.int64),
    )

    if len(mapping) != EXPECTED_SUPPORTED_GENES:
        raise RuntimeError(
            f"Expected {EXPECTED_SUPPORTED_GENES:,} supported genes, "
            f"found {len(mapping):,}."
        )

    return mapping


def to_clean_csr(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        output = matrix.tocsr()
    else:
        output = sp.csr_matrix(np.asarray(matrix))

    output.sum_duplicates()
    output.eliminate_zeros()
    output.sort_indices()

    return output


def materialize_split(
    adata: ad.AnnData,
    source_columns: np.ndarray,
    chunk_rows: int,
    split_name: str,
) -> tuple[sp.csr_matrix, np.ndarray]:
    matrix_blocks: list[sp.csr_matrix] = []
    n_counts_parts: list[np.ndarray] = []

    for start in range(0, adata.n_obs, chunk_rows):
        stop = min(start + chunk_rows, adata.n_obs)

        full_block = to_clean_csr(
            adata.X[start:stop, :]
        )

        full_values = np.asarray(
            full_block.data,
            dtype=np.float64,
        )

        if not np.isfinite(full_values).all():
            raise RuntimeError(
                f"{split_name}: non-finite matrix values detected."
            )

        if np.any(full_values < 0):
            raise RuntimeError(
                f"{split_name}: negative matrix values detected."
            )

        if not np.allclose(
            full_values,
            np.rint(full_values),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{split_name}: matrix is not raw-count-like."
            )

        original_n_counts = np.asarray(
            full_block.sum(axis=1)
        ).reshape(-1).astype(np.float64)

        mapped_block = (
            full_block[:, source_columns]
            .tocsr()
            .astype(np.float32)
        )
        mapped_block.sum_duplicates()
        mapped_block.eliminate_zeros()
        mapped_block.sort_indices()

        matrix_blocks.append(mapped_block)
        n_counts_parts.append(original_n_counts)

        print(
            f"[matrix] {split_name}: "
            f"{stop:,}/{adata.n_obs:,} "
            f"({100.0 * stop / adata.n_obs:.1f}%)",
            flush=True,
        )

    matrix = sp.vstack(
        matrix_blocks,
        format="csr",
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()

    n_counts = np.concatenate(n_counts_parts)

    return matrix, n_counts


def validate_count_column(
    adata: ad.AnnData,
    n_counts: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    obs_counts = np.asarray(
        adata.obs[COUNT_COLUMN],
        dtype=np.float64,
    )

    absolute_difference = np.abs(
        obs_counts - n_counts
    )

    exact = np.isclose(
        obs_counts,
        n_counts,
        rtol=0.0,
        atol=1e-6,
    )

    report = {
        "column": COUNT_COLUMN,
        "mean_absolute_difference": float(
            np.mean(absolute_difference)
        ),
        "median_absolute_difference": float(
            np.median(absolute_difference)
        ),
        "max_absolute_difference": float(
            np.max(absolute_difference)
        ),
        "fraction_close_atol_1e-6": float(
            np.mean(exact)
        ),
    }

    if not np.all(exact):
        mismatch_indices = np.flatnonzero(~exact)[:20]

        raise RuntimeError(
            f"{split_name}: original .X row sums do not match "
            f"obs[{COUNT_COLUMN!r}]. First mismatch indices: "
            f"{mismatch_indices.tolist()}"
        )

    return report


def save_string_array(path: Path, values: Any) -> None:
    array = np.asarray(values).astype(str)
    np.save(path, array, allow_pickle=False)


def save_split_cache(
    split_dir: Path,
    matrix: sp.csr_matrix,
    n_counts: np.ndarray,
    adata: ad.AnnData,
    split_name: str,
) -> dict[str, Any]:
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_path = split_dir / "mapped_raw_counts_csr.npz"
    sp.save_npz(
        matrix_path,
        matrix,
        compressed=False,
    )

    np.save(
        split_dir / "n_counts.npy",
        n_counts.astype(np.float64),
        allow_pickle=False,
    )
    np.save(
        split_dir / "row_nnz.npy",
        np.diff(matrix.indptr).astype(np.int32),
        allow_pickle=False,
    )
    np.save(
        split_dir / "source_row_indices.npy",
        np.arange(adata.n_obs, dtype=np.int64),
        allow_pickle=False,
    )

    save_string_array(
        split_dir / "obs_names.npy",
        adata.obs_names,
    )
    save_string_array(
        split_dir / "labels.npy",
        adata.obs[LABEL_COLUMN],
    )
    save_string_array(
        split_dir / "conditions.npy",
        adata.obs[CONDITION_COLUMN],
    )
    save_string_array(
        split_dir / "donors.npy",
        adata.obs[DONOR_COLUMN],
    )

    row_nnz = np.diff(matrix.indptr)

    split_meta = {
        "split": split_name,
        "cells": int(matrix.shape[0]),
        "supported_genes": int(matrix.shape[1]),
        "matrix_shape": [
            int(matrix.shape[0]),
            int(matrix.shape[1]),
        ],
        "matrix_nnz": int(matrix.nnz),
        "matrix_dtype": str(matrix.dtype),
        "indices_dtype": str(matrix.indices.dtype),
        "indptr_dtype": str(matrix.indptr.dtype),
        "n_counts_statistics": summarize(n_counts),
        "supported_gene_nnz_statistics": summarize(row_nnz),
        "empty_supported_cells": int(
            np.sum(row_nnz == 0)
        ),
        "cells_exceeding_context_2048": int(
            np.sum(row_nnz > 2048)
        ),
        "fraction_exceeding_context_2048": float(
            np.mean(row_nnz > 2048)
        ),
        "paths": {
            "matrix": str(matrix_path.resolve()),
            "n_counts": str(
                (split_dir / "n_counts.npy").resolve()
            ),
            "obs_names": str(
                (split_dir / "obs_names.npy").resolve()
            ),
            "labels": str(
                (split_dir / "labels.npy").resolve()
            ),
            "conditions": str(
                (split_dir / "conditions.npy").resolve()
            ),
            "donors": str(
                (split_dir / "donors.npy").resolve()
            ),
        },
    }

    stable_json_dump(
        split_meta,
        split_dir / "meta.json",
    )

    return split_meta


def compute_train_only_idf(
    train_matrix: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix.sum_duplicates()
    train_matrix.eliminate_zeros()
    train_matrix.sort_indices()

    document_frequency = np.bincount(
        train_matrix.indices,
        minlength=train_matrix.shape[1],
    ).astype(np.int64)

    if np.any(document_frequency > train_matrix.shape[0]):
        raise RuntimeError(
            "Document frequency exceeds the number of training cells. "
            "This indicates duplicate column indices within a CSR row."
        )

    idf = (
        np.log(
            (train_matrix.shape[0] + 1)
            / (document_frequency + 1)
        )
        + 1.0
    ).astype(np.float32)

    if not np.isfinite(idf).all():
        raise RuntimeError(
            "IDF contains NaN or Inf."
        )

    return document_frequency, idf


def validate_saved_cache(
    cache_dir: Path,
    expected_train_shape: tuple[int, int],
    expected_test_shape: tuple[int, int],
) -> None:
    train = sp.load_npz(
        cache_dir
        / "train"
        / "mapped_raw_counts_csr.npz"
    ).tocsr()
    test = sp.load_npz(
        cache_dir
        / "test"
        / "mapped_raw_counts_csr.npz"
    ).tocsr()

    if train.shape != expected_train_shape:
        raise RuntimeError(
            f"Saved train shape mismatch: {train.shape}"
        )

    if test.shape != expected_test_shape:
        raise RuntimeError(
            f"Saved test shape mismatch: {test.shape}"
        )

    token_ids = np.load(
        cache_dir / "gene_token_ids.npy",
        allow_pickle=False,
    )
    medians = np.load(
        cache_dir / "gene_medians.npy",
        allow_pickle=False,
    )
    idf = np.load(
        cache_dir / "idf.npy",
        allow_pickle=False,
    )
    document_frequency = np.load(
        cache_dir / "document_frequency.npy",
        allow_pickle=False,
    )

    expected_genes = expected_train_shape[1]

    for name, array in [
        ("token_ids", token_ids),
        ("medians", medians),
        ("idf", idf),
        ("document_frequency", document_frequency),
    ]:
        if array.shape != (expected_genes,):
            raise RuntimeError(
                f"Saved {name} shape mismatch: {array.shape}"
            )

    if not np.all(np.diff(token_ids.astype(np.int64)) > 0):
        raise RuntimeError(
            "Gene token IDs are not strictly increasing."
        )

    if np.any(medians <= 0) or not np.isfinite(medians).all():
        raise RuntimeError(
            "Invalid Geneformer gene medians."
        )

    if np.any(document_frequency < 0):
        raise RuntimeError(
            "Negative document frequency detected."
        )

    if np.any(
        document_frequency > expected_train_shape[0]
    ):
        raise RuntimeError(
            "Document frequency exceeds train cell count."
        )

    print("[validate] Saved Kang base cache: PASS", flush=True)


def main() -> int:
    args = parse_args()

    if args.chunk_rows <= 0:
        raise ValueError(
            "--chunk-rows must be positive."
        )

    required_paths = [
        args.train_path,
        args.test_path,
        DICTIONARY_DIR
        / "gene_name_id_dict_gc30M.pkl",
        DICTIONARY_DIR
        / "ensembl_mapping_dict_gc30M.pkl",
        DICTIONARY_DIR
        / "gene_median_dictionary_gc30M.pkl",
        DICTIONARY_DIR
        / "token_dictionary_gc30M.pkl",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.cache_dir.exists():
        existing_entries = list(
            args.cache_dir.iterdir()
        )

        if existing_entries and not args.overwrite:
            raise FileExistsError(
                f"Cache directory is not empty: {args.cache_dir}. "
                "Use --overwrite only after confirming it is safe "
                "to replace."
            )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    train = ad.read_h5ad(
        args.train_path,
        backed="r",
    )
    test = ad.read_h5ad(
        args.test_path,
        backed="r",
    )

    try:
        validate_source_schema(
            train,
            EXPECTED_TRAIN_CELLS,
            "train",
        )
        validate_source_schema(
            test,
            EXPECTED_TEST_CELLS,
            "test",
        )

        train_features = np.asarray(
            train.var_names.astype(str),
        )
        test_features = np.asarray(
            test.var_names.astype(str),
        )

        if not np.array_equal(
            train_features,
            test_features,
        ):
            raise RuntimeError(
                "Kang train/test feature order differs."
            )

        mapping = build_gene_mapping(
            train_features
        )
        source_columns = mapping[
            "source_index"
        ].to_numpy(dtype=np.int64)

        print(
            "Gene mapping:",
            f"{len(mapping):,}/{len(train_features):,}",
            "source genes supported;",
            "0 duplicate mappings.",
            flush=True,
        )

        train_matrix, train_n_counts = materialize_split(
            train,
            source_columns,
            args.chunk_rows,
            "train",
        )
        test_matrix, test_n_counts = materialize_split(
            test,
            source_columns,
            args.chunk_rows,
            "test",
        )

        train_count_check = validate_count_column(
            train,
            train_n_counts,
            "train",
        )
        test_count_check = validate_count_column(
            test,
            test_n_counts,
            "test",
        )

        mapping.to_csv(
            args.cache_dir / "gene_mapping.csv",
            index=False,
        )
        save_string_array(
            args.cache_dir / "gene_ensembl_ids.npy",
            mapping["canonical_ensembl_id"],
        )
        np.save(
            args.cache_dir / "gene_token_ids.npy",
            mapping["token_id"].to_numpy(
                dtype=np.int32
            ),
            allow_pickle=False,
        )
        np.save(
            args.cache_dir / "gene_medians.npy",
            mapping["gene_median"].to_numpy(
                dtype=np.float64
            ),
            allow_pickle=False,
        )

        train_meta = save_split_cache(
            args.cache_dir / "train",
            train_matrix,
            train_n_counts,
            train,
            "train",
        )
        test_meta = save_split_cache(
            args.cache_dir / "test",
            test_matrix,
            test_n_counts,
            test,
            "test",
        )

        document_frequency, idf = (
            compute_train_only_idf(
                train_matrix
            )
        )
        np.save(
            args.cache_dir
            / "document_frequency.npy",
            document_frequency,
            allow_pickle=False,
        )
        np.save(
            args.cache_dir / "idf.npy",
            idf,
            allow_pickle=False,
        )

        hash_paths = {
            "train_h5ad": args.train_path,
            "test_h5ad": args.test_path,
            "gene_name_id_dict": (
                DICTIONARY_DIR
                / "gene_name_id_dict_gc30M.pkl"
            ),
            "ensembl_mapping_dict": (
                DICTIONARY_DIR
                / "ensembl_mapping_dict_gc30M.pkl"
            ),
            "gene_median_dictionary": (
                DICTIONARY_DIR
                / "gene_median_dictionary_gc30M.pkl"
            ),
            "token_dictionary": (
                DICTIONARY_DIR
                / "token_dictionary_gc30M.pkl"
            ),
            "builder_script": Path(__file__).resolve(),
        }

        hashes = {}

        for name, path in hash_paths.items():
            print(
                f"[hash] Computing SHA-256: {path}",
                flush=True,
            )
            hashes[name] = {
                "path": str(path.resolve()),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "mtime_ns": int(
                    path.stat().st_mtime_ns
                ),
                "sha256": sha256_file(path),
            }

        elapsed = time.perf_counter() - start_time

        root_meta = {
            "status": "PASS",
            "cache_type": (
                "Geneformer V1 common base cache"
            ),
            "dataset": "kang_2018_patient_1015_holdout",
            "cache_directory": str(
                args.cache_dir.resolve()
            ),
            "source_gene_count": int(
                len(train_features)
            ),
            "supported_gene_count": int(
                len(mapping)
            ),
            "gene_column_order": (
                "strictly increasing Geneformer token ID"
            ),
            "duplicate_mapping_rule": (
                "sum raw counts for source columns mapping "
                "to the same canonical Ensembl ID"
            ),
            "duplicate_mapping_groups_observed": 0,
            "n_counts_semantics": (
                "sum of all genes in the original, uncompressed "
                ".X row before Geneformer vocabulary filtering "
                "or method-specific selection"
            ),
            "idf_semantics": {
                "formula": (
                    "log((N_train + 1) / (df_g + 1)) + 1"
                ),
                "computed_from": (
                    "Kang training split only"
                ),
                "test_statistics_used": False,
            },
            "train": train_meta,
            "test": test_meta,
            "count_column_validation": {
                "train": train_count_check,
                "test": test_count_check,
            },
            "idf_statistics": summarize(idf),
            "document_frequency_statistics": (
                summarize(document_frequency)
            ),
            "hashes": hashes,
            "build_elapsed_seconds": float(
                elapsed
            ),
        }

        stable_json_dump(
            root_meta,
            args.cache_dir / "meta.json",
        )

    finally:
        if getattr(train, "isbacked", False):
            train.file.close()

        if getattr(test, "isbacked", False):
            test.file.close()

    validate_saved_cache(
        args.cache_dir,
        (
            EXPECTED_TRAIN_CELLS,
            EXPECTED_SUPPORTED_GENES,
        ),
        (
            EXPECTED_TEST_CELLS,
            EXPECTED_SUPPORTED_GENES,
        ),
    )

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(
        "CACHE:",
        args.cache_dir,
    )
    print(
        "TRAIN SHAPE:",
        (
            EXPECTED_TRAIN_CELLS,
            EXPECTED_SUPPORTED_GENES,
        ),
    )
    print(
        "TEST SHAPE:",
        (
            EXPECTED_TEST_CELLS,
            EXPECTED_SUPPORTED_GENES,
        ),
    )
    print(
        "IDF SOURCE: TRAIN ONLY",
    )
    print(
        "N_COUNTS: ORIGINAL UNCOMPRESSED .X ROW SUM",
    )
    print(
        "FINAL STATUS: PASS",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
