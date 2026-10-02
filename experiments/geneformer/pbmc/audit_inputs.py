#!/usr/bin/env python3
"""Audit PBMC Seurat v4 for the Geneformer AdaGeneBudget benchmark.

This script does not build embeddings or method caches. It validates the
frozen PBMC P5-held-out split and the exact Geneformer-compatible gene space
before any preprocessing is written.

Expected project setting
------------------------
- Source:
  ./data/raw/pbmc_seurat_v4/pbmc_seurat_v4_rna_only.h5ad
- Split:
  ./data/processed/pbmc_seurat_v4/p5_holdout/split_indices.npz
- Main label:
  celltype.l1
- Secondary label:
  celltype.l2
- Held-out donor:
  P5
- Expected split:
  train = 132,137 cells
  test  = 19,957 cells
- Expected Geneformer-compatible genes:
  14,809 / 20,729

Outputs
-------
outputs/reports/geneformer_pbmc_audit/
├── audit_summary.json
├── gene_mapping.csv
├── train_celltype_l1_counts.csv
├── test_celltype_l1_counts.csv
├── train_celltype_l2_counts.csv
└── test_celltype_l2_counts.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
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

DEFAULT_DICTIONARY_DIR = (
    PROJECT_DIR
    / "Geneformer"
    / "geneformer"
    / "gene_dictionaries_30m"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "reports"
    / "geneformer_pbmc_audit"
)

EXPECTED_TOTAL_CELLS = 152_094
EXPECTED_TRAIN_CELLS = 132_137
EXPECTED_TEST_CELLS = 19_957
EXPECTED_TOTAL_GENES = 20_729
EXPECTED_MAPPED_GENES = 14_809
EXPECTED_TEST_DONOR = "P5"

MAIN_LABEL = "celltype.l1"
SECONDARY_LABEL = "celltype.l2"
DONOR_COLUMN = "donor"
TIME_COLUMN = "time"


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
        "--sample-cells-per-split",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
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


def load_geneformer_dictionaries(
    dictionary_dir: Path,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, str],
    dict[str, str],
]:
    require_directory(dictionary_dir)

    name_to_id_path = (
        dictionary_dir
        / "gene_name_id_dict_gc30M.pkl"
    )
    canonical_path = (
        dictionary_dir
        / "ensembl_mapping_dict_gc30M.pkl"
    )
    token_path = find_unique_file(
        dictionary_dir,
        "token_dictionary*gc30M*.pkl",
    )

    name_to_id_raw = load_pickle(name_to_id_path)
    canonical_raw = load_pickle(canonical_path)
    token_raw = load_pickle(token_path)

    name_to_id = {
        str(key): str(value).upper()
        for key, value in name_to_id_raw.items()
    }
    canonical = {
        str(key).upper(): str(value).upper()
        for key, value in canonical_raw.items()
    }
    token_dictionary = {
        str(key): int(value)
        for key, value in token_raw.items()
    }

    uppercase_buckets: dict[str, set[str]] = {}
    for name, ensembl_id in name_to_id.items():
        uppercase_buckets.setdefault(
            name.upper(),
            set(),
        ).add(ensembl_id)

    collisions = {
        name: ids
        for name, ids in uppercase_buckets.items()
        if len(ids) > 1
    }
    if collisions:
        examples = list(collisions.items())[:20]
        raise RuntimeError(
            "Uppercase gene-name collisions detected: "
            f"{examples}"
        )

    uppercase_name_to_id = {
        name: next(iter(ids))
        for name, ids in uppercase_buckets.items()
    }

    paths = {
        "gene_name_id_dict": str(
            name_to_id_path.resolve()
        ),
        "ensembl_mapping_dict": str(
            canonical_path.resolve()
        ),
        "token_dictionary": str(
            token_path.resolve()
        ),
    }

    return (
        name_to_id,
        canonical,
        token_dictionary,
        uppercase_name_to_id,
        paths,
    )


def map_feature(
    feature: str,
    name_to_id: dict[str, str],
    canonical: dict[str, str],
    token_dictionary: dict[str, int],
    uppercase_name_to_id: dict[str, str],
) -> tuple[
    str | None,
    int | None,
    str,
]:
    raw = str(feature).strip()
    upper = raw.upper()
    no_version = upper.split(".", 1)[0]

    if re.fullmatch(r"ENSG[0-9]+", no_version):
        ensembl_id = no_version
        method = "direct_ensembl"
    elif raw in name_to_id:
        ensembl_id = name_to_id[raw]
        method = "symbol_exact"
    elif upper in uppercase_name_to_id:
        ensembl_id = uppercase_name_to_id[upper]
        method = "symbol_uppercase"
    else:
        return None, None, "unmapped"

    canonical_id = canonical.get(
        ensembl_id,
        ensembl_id,
    )
    token_id = token_dictionary.get(
        canonical_id
    )

    if token_id is None:
        return (
            canonical_id,
            None,
            f"{method}_not_in_vocab",
        )

    return (
        canonical_id,
        int(token_id),
        method,
    )


def inspect_matrix_sample(
    adata: ad.AnnData,
    indices: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    if len(indices) > sample_size:
        selected = np.sort(
            rng.choice(
                indices,
                size=sample_size,
                replace=False,
            )
        )
    else:
        selected = np.sort(indices)

    matrix = adata[selected, :].X

    if sparse.issparse(matrix):
        matrix = matrix.tocsr(copy=False)
        values = np.asarray(
            matrix.data,
            dtype=np.float64,
        )
        nonzero_per_cell = np.diff(
            matrix.indptr
        ).astype(np.int64)
        row_sums = np.asarray(
            matrix.sum(axis=1)
        ).ravel().astype(np.float64)
        storage = "sparse"
    else:
        dense = np.asarray(
            matrix,
            dtype=np.float64,
        )
        values = dense.ravel()
        values = values[values != 0]
        nonzero_per_cell = np.count_nonzero(
            dense,
            axis=1,
        ).astype(np.int64)
        row_sums = dense.sum(
            axis=1
        ).astype(np.float64)
        storage = "dense"

    if values.size == 0:
        raise RuntimeError(
            "Sampled expression matrix contains no "
            "nonzero values."
        )

    minimum = float(values.min())
    maximum = float(values.max())
    integer_residual = float(
        np.max(
            np.abs(
                values
                - np.rint(values)
            )
        )
    )

    return {
        "sampled_cells": int(len(selected)),
        "storage": storage,
        "nonzero_values": int(values.size),
        "minimum_nonzero_value": minimum,
        "maximum_nonzero_value": maximum,
        "maximum_integer_residual": (
            integer_residual
        ),
        "nonzero_genes_mean": float(
            np.mean(nonzero_per_cell)
        ),
        "nonzero_genes_median": float(
            np.median(nonzero_per_cell)
        ),
        "nonzero_genes_min": int(
            np.min(nonzero_per_cell)
        ),
        "nonzero_genes_max": int(
            np.max(nonzero_per_cell)
        ),
        "row_sum_mean": float(
            np.mean(row_sums)
        ),
        "row_sum_median": float(
            np.median(row_sums)
        ),
        "row_sum_min": float(
            np.min(row_sums)
        ),
        "row_sum_max": float(
            np.max(row_sums)
        ),
        "nonnegative": bool(
            minimum >= 0.0
        ),
        "integer_like": bool(
            integer_residual <= 1e-6
        ),
    }


def save_label_counts(
    labels: pd.Series,
    path: Path,
) -> None:
    (
        labels.astype(str)
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
        .to_csv(
            path,
            index=False,
        )
    )


def main() -> int:
    args = parse_args()

    require_file(args.h5ad)
    require_file(args.split_npz)
    require_directory(args.dictionary_dir)

    if args.sample_cells_per_split <= 0:
        raise ValueError(
            "--sample-cells-per-split must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = np.load(
        args.split_npz,
        allow_pickle=False,
    )

    split_key_candidates = {
        "train": [
            "train_idx",
            "train_indices",
        ],
        "test": [
            "test_idx",
            "test_indices",
        ],
    }

    resolved_split_keys = {}

    for split_name, candidates in (
        split_key_candidates.items()
    ):
        matches = [
            key
            for key in candidates
            if key in split.files
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve "
                f"{split_name} split key. "
                f"Candidates={candidates}; "
                f"available={split.files}; "
                f"matches={matches}"
            )

        resolved_split_keys[
            split_name
        ] = matches[0]

    train_idx = np.asarray(
        split[
            resolved_split_keys["train"]
        ],
        dtype=np.int64,
    )
    test_idx = np.asarray(
        split[
            resolved_split_keys["test"]
        ],
        dtype=np.int64,
    )

    if train_idx.ndim != 1 or test_idx.ndim != 1:
        raise RuntimeError(
            "train_idx and test_idx must be 1D."
        )
    if len(np.unique(train_idx)) != len(train_idx):
        raise RuntimeError(
            "Duplicate values in train_idx."
        )
    if len(np.unique(test_idx)) != len(test_idx):
        raise RuntimeError(
            "Duplicate values in test_idx."
        )
    if np.intersect1d(
        train_idx,
        test_idx,
    ).size:
        raise RuntimeError(
            "Train/test split overlap detected."
        )

    adata = ad.read_h5ad(
        args.h5ad,
        backed="r",
    )

    try:
        if adata.n_obs != EXPECTED_TOTAL_CELLS:
            raise RuntimeError(
                f"Unexpected total cells: {adata.n_obs}; "
                f"expected {EXPECTED_TOTAL_CELLS}."
            )
        if adata.n_vars != EXPECTED_TOTAL_GENES:
            raise RuntimeError(
                f"Unexpected total genes: {adata.n_vars}; "
                f"expected {EXPECTED_TOTAL_GENES}."
            )
        if len(train_idx) != EXPECTED_TRAIN_CELLS:
            raise RuntimeError(
                f"Unexpected train cells: {len(train_idx)}; "
                f"expected {EXPECTED_TRAIN_CELLS}."
            )
        if len(test_idx) != EXPECTED_TEST_CELLS:
            raise RuntimeError(
                f"Unexpected test cells: {len(test_idx)}; "
                f"expected {EXPECTED_TEST_CELLS}."
            )

        all_indices = np.concatenate(
            [train_idx, test_idx]
        )
        if np.min(all_indices) < 0:
            raise RuntimeError(
                "Negative split index detected."
            )
        if np.max(all_indices) >= adata.n_obs:
            raise RuntimeError(
                "Split index exceeds H5AD bounds."
            )
        if len(np.unique(all_indices)) != adata.n_obs:
            raise RuntimeError(
                "Train and test indices do not cover "
                "every H5AD cell exactly once."
            )

        required_obs_columns = {
            MAIN_LABEL,
            SECONDARY_LABEL,
            DONOR_COLUMN,
            TIME_COLUMN,
        }
        missing_obs_columns = (
            required_obs_columns
            - set(adata.obs.columns)
        )
        if missing_obs_columns:
            raise RuntimeError(
                "H5AD missing obs columns: "
                f"{sorted(missing_obs_columns)}"
            )

        train_obs = adata.obs.iloc[
            train_idx
        ].copy()
        test_obs = adata.obs.iloc[
            test_idx
        ].copy()

        train_donors = sorted(
            train_obs[DONOR_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )
        test_donors = sorted(
            test_obs[DONOR_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )

        if test_donors != [
            EXPECTED_TEST_DONOR
        ]:
            raise RuntimeError(
                f"Expected held-out donor "
                f"{EXPECTED_TEST_DONOR!r}; "
                f"observed test donors={test_donors}"
            )
        if EXPECTED_TEST_DONOR in train_donors:
            raise RuntimeError(
                "Held-out donor appears in train split."
            )

        save_label_counts(
            train_obs[MAIN_LABEL],
            args.output_dir
            / "train_celltype_l1_counts.csv",
        )
        save_label_counts(
            test_obs[MAIN_LABEL],
            args.output_dir
            / "test_celltype_l1_counts.csv",
        )
        save_label_counts(
            train_obs[SECONDARY_LABEL],
            args.output_dir
            / "train_celltype_l2_counts.csv",
        )
        save_label_counts(
            test_obs[SECONDARY_LABEL],
            args.output_dir
            / "test_celltype_l2_counts.csv",
        )

        train_matrix_audit = (
            inspect_matrix_sample(
                adata,
                train_idx,
                args.sample_cells_per_split,
                args.seed,
            )
        )
        test_matrix_audit = (
            inspect_matrix_sample(
                adata,
                test_idx,
                args.sample_cells_per_split,
                args.seed + 1,
            )
        )

        if not train_matrix_audit[
            "nonnegative"
        ]:
            raise RuntimeError(
                "Negative train expression values "
                "detected."
            )
        if not test_matrix_audit[
            "nonnegative"
        ]:
            raise RuntimeError(
                "Negative test expression values "
                "detected."
            )
        if not train_matrix_audit[
            "integer_like"
        ]:
            raise RuntimeError(
                "Train expression values are not "
                "integer-like raw counts."
            )
        if not test_matrix_audit[
            "integer_like"
        ]:
            raise RuntimeError(
                "Test expression values are not "
                "integer-like raw counts."
            )

        source_features = np.asarray(
            adata.var_names.astype(str)
        )

    finally:
        if getattr(
            adata,
            "file",
            None,
        ) is not None:
            adata.file.close()

    (
        name_to_id,
        canonical,
        token_dictionary,
        uppercase_name_to_id,
        dictionary_paths,
    ) = load_geneformer_dictionaries(
        args.dictionary_dir
    )

    mapping_rows: list[
        dict[str, Any]
    ] = []

    for source_index, feature in enumerate(
        source_features
    ):
        (
            ensembl_id,
            token_id,
            mapping_method,
        ) = map_feature(
            feature,
            name_to_id,
            canonical,
            token_dictionary,
            uppercase_name_to_id,
        )

        mapping_rows.append(
            {
                "source_index": int(
                    source_index
                ),
                "source_feature": str(
                    feature
                ),
                "ensembl_id": (
                    ensembl_id
                ),
                "token_id": (
                    token_id
                ),
                "mapping_method": (
                    mapping_method
                ),
                "geneformer_supported": bool(
                    token_id is not None
                ),
            }
        )

    mapping = pd.DataFrame(
        mapping_rows
    )
    mapping.to_csv(
        args.output_dir
        / "gene_mapping.csv",
        index=False,
    )

    supported = mapping.loc[
        mapping["geneformer_supported"]
    ].copy()

    if len(supported) != EXPECTED_MAPPED_GENES:
        raise RuntimeError(
            "Unexpected Geneformer-supported gene "
            f"count: {len(supported)}; expected "
            f"{EXPECTED_MAPPED_GENES}."
        )

    if supported[
        "source_index"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate source indices among "
            "supported genes."
        )

    if supported[
        "token_id"
    ].duplicated().any():
        duplicates = (
            supported.loc[
                supported["token_id"]
                .duplicated(
                    keep=False
                )
            ]
            .sort_values("token_id")
            .head(30)
        )
        raise RuntimeError(
            "Multiple PBMC features map to the same "
            "Geneformer token:\n"
            + duplicates.to_string(
                index=False
            )
        )

    mapping_method_counts = (
        mapping["mapping_method"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary = {
        "status": "PASS",
        "h5ad": str(
            args.h5ad.resolve()
        ),
        "split_npz": str(
            args.split_npz.resolve()
        ),
        "dictionary_paths": (
            dictionary_paths
        ),
        "shape": {
            "cells": int(
                EXPECTED_TOTAL_CELLS
            ),
            "genes": int(
                EXPECTED_TOTAL_GENES
            ),
        },
        "split": {
            "resolved_train_key": (
                resolved_split_keys["train"]
            ),
            "resolved_test_key": (
                resolved_split_keys["test"]
            ),
            "train_cells": int(
                len(train_idx)
            ),
            "test_cells": int(
                len(test_idx)
            ),
            "train_donors": (
                train_donors
            ),
            "test_donors": (
                test_donors
            ),
            "held_out_donor": (
                EXPECTED_TEST_DONOR
            ),
            "disjoint": True,
            "complete_coverage": True,
        },
        "columns": {
            "main_label": MAIN_LABEL,
            "secondary_label": (
                SECONDARY_LABEL
            ),
            "donor": DONOR_COLUMN,
            "time": TIME_COLUMN,
        },
        "expression_audit": {
            "train": (
                train_matrix_audit
            ),
            "test": (
                test_matrix_audit
            ),
        },
        "gene_mapping": {
            "total_source_genes": int(
                len(mapping)
            ),
            "geneformer_supported_genes": int(
                len(supported)
            ),
            "unsupported_genes": int(
                len(mapping)
                - len(supported)
            ),
            "supported_fraction": float(
                len(supported)
                / len(mapping)
            ),
            "mapping_method_counts": {
                str(key): int(value)
                for key, value in (
                    mapping_method_counts.items()
                )
            },
            "unique_supported_token_ids": int(
                supported[
                    "token_id"
                ].nunique()
            ),
        },
    }

    (
        args.output_dir
        / "audit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print(
        "GENEFORMER PBMC P5-HOLDOUT AUDIT"
    )
    print("=" * 100)
    print(
        "H5AD:",
        args.h5ad,
    )
    print(
        "Shape:",
        (
            EXPECTED_TOTAL_CELLS,
            EXPECTED_TOTAL_GENES,
        ),
    )
    print(
        "Train/Test:",
        len(train_idx),
        "/",
        len(test_idx),
    )
    print(
        "Train donors:",
        train_donors,
    )
    print(
        "Test donors:",
        test_donors,
    )
    print(
        "Geneformer supported genes:",
        f"{len(supported)} / {len(mapping)}",
    )

    print("\nTRAIN MATRIX SAMPLE")
    print(
        json.dumps(
            train_matrix_audit,
            indent=2,
        )
    )

    print("\nTEST MATRIX SAMPLE")
    print(
        json.dumps(
            test_matrix_audit,
            indent=2,
        )
    )

    print("\nMAPPING METHOD COUNTS")
    print(
        pd.Series(
            mapping_method_counts
        ).to_string()
    )

    print(
        "\nOUTPUT:",
        args.output_dir,
    )
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
