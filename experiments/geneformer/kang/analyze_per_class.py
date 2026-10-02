#!/usr/bin/env python3
"""Analyze per-cell-type Geneformer Kang benchmark results.

This CPU-only script verifies whether AdaGeneBudget's large Macro-F1 and
balanced-accuracy gains are broadly distributed across cell types or driven
mainly by the extremely rare Megakaryocyte class in the held-out donor.

Inputs are the prediction CSV files produced by:
    07_run_geneformer_kang_benchmark.py

Outputs:
    per_class_long.csv
    per_class_wide.csv
    macro_sensitivity.csv
    confusion_<method>.csv
    summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_DIR = Path(".")
DEFAULT_INPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "geneformer"
    / "kang_patient1015"
    / "main"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_INPUT_DIR
    / "analysis"
    / "per_class"
)

METHOD_ORDER = [
    "geneformer_native_bucketed",
    "native_rank_fixed_k",
    "matched_random_fixed_k",
    "matched_fixed_tfidf",
    "adaptive",
]

METHOD_LABEL = {
    "geneformer_native_bucketed": "Native Bucketed",
    "native_rank_fixed_k": "Native Rank",
    "matched_random_fixed_k": "Matched Random (seed 0)",
    "matched_fixed_tfidf": "Fixed TF-IDF",
    "adaptive": "AdaGeneBudget",
}

SUPPORT_THRESHOLDS = [0, 5, 20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_predictions(
    input_dir: Path,
    method: str,
) -> pd.DataFrame:
    path = (
        input_dir
        / method
        / "predictions_seed0.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "obs_name",
        "cell_type",
        "condition",
        "donor",
        "prediction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    if frame["obs_name"].duplicated().any():
        raise RuntimeError(
            f"Duplicate obs_name values in {path}"
        )

    frame = frame.copy()
    for column in [
        "obs_name",
        "cell_type",
        "condition",
        "donor",
        "prediction",
    ]:
        frame[column] = frame[column].astype(str)

    return frame


def validate_identical_cells(
    frames: dict[str, pd.DataFrame],
) -> None:
    reference_method = METHOD_ORDER[0]
    reference = frames[reference_method]

    for method in METHOD_ORDER[1:]:
        current = frames[method]

        if len(current) != len(reference):
            raise RuntimeError(
                f"Cell-count mismatch: {method}"
            )

        for column in [
            "obs_name",
            "cell_type",
            "condition",
            "donor",
        ]:
            if not np.array_equal(
                reference[column].to_numpy(),
                current[column].to_numpy(),
            ):
                raise RuntimeError(
                    f"Cell-order or metadata mismatch "
                    f"for {method}, column={column}"
                )


def compute_per_class(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    return pd.DataFrame(
        {
            "cell_type": labels,
            "support": support.astype(np.int64),
            "precision": precision.astype(np.float64),
            "recall": recall.astype(np.float64),
            "f1": f1.astype(np.float64),
        }
    )


def compute_macro_sensitivity(
    per_class_frame: pd.DataFrame,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for threshold in SUPPORT_THRESHOLDS:
        included = per_class_frame[
            per_class_frame["support"] >= threshold
        ]

        if included.empty:
            continue

        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "minimum_test_support": int(threshold),
                "included_class_count": int(len(included)),
                "included_classes": "|".join(
                    included["cell_type"].tolist()
                ),
                "macro_f1_over_included_classes": float(
                    included["f1"].mean()
                ),
                "mean_recall_over_included_classes": float(
                    included["recall"].mean()
                ),
            }
        )

    excluded_megakaryocytes = per_class_frame[
        per_class_frame["cell_type"]
        != "Megakaryocytes"
    ]

    rows.append(
        {
            "method": method,
            "method_label": METHOD_LABEL[method],
            "minimum_test_support": -1,
            "included_class_count": int(
                len(excluded_megakaryocytes)
            ),
            "included_classes": "|".join(
                excluded_megakaryocytes[
                    "cell_type"
                ].tolist()
            ),
            "macro_f1_over_included_classes": float(
                excluded_megakaryocytes["f1"].mean()
            ),
            "mean_recall_over_included_classes": float(
                excluded_megakaryocytes[
                    "recall"
                ].mean()
            ),
            "definition": "All classes except Megakaryocytes",
        }
    )

    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = {
        method: load_predictions(
            args.input_dir,
            method,
        )
        for method in METHOD_ORDER
    }
    validate_identical_cells(frames)

    reference = frames[METHOD_ORDER[0]]
    true_labels = reference[
        "cell_type"
    ].to_numpy()

    labels = sorted(
        np.unique(true_labels).tolist()
    )

    long_parts: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []

    for method in METHOD_ORDER:
        predictions = frames[method][
            "prediction"
        ].to_numpy()

        per_class = compute_per_class(
            true_labels,
            predictions,
            labels,
        )
        per_class.insert(
            0,
            "method_label",
            METHOD_LABEL[method],
        )
        per_class.insert(
            0,
            "method",
            method,
        )
        long_parts.append(per_class)

        sensitivity_rows.extend(
            compute_macro_sensitivity(
                per_class,
                method,
            )
        )

        global_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABEL[method],
                "accuracy": float(
                    accuracy_score(
                        true_labels,
                        predictions,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        true_labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(
                        true_labels,
                        predictions,
                    )
                ),
            }
        )

        matrix = confusion_matrix(
            true_labels,
            predictions,
            labels=labels,
        )
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        ).to_csv(
            args.output_dir
            / f"confusion_{method}.csv"
        )

    long_frame = pd.concat(
        long_parts,
        ignore_index=True,
    )
    long_frame.to_csv(
        args.output_dir / "per_class_long.csv",
        index=False,
    )

    wide = long_frame.pivot(
        index=["cell_type", "support"],
        columns="method",
        values=["precision", "recall", "f1"],
    )
    wide.columns = [
        f"{metric}__{method}"
        for metric, method in wide.columns
    ]
    wide = wide.reset_index()

    for metric in ["f1", "recall", "precision"]:
        wide[
            f"{metric}__adaptive_minus_fixed"
        ] = (
            wide[f"{metric}__adaptive"]
            - wide[
                f"{metric}__matched_fixed_tfidf"
            ]
        )
        wide[
            f"{metric}__adaptive_minus_native"
        ] = (
            wide[f"{metric}__adaptive"]
            - wide[
                f"{metric}__geneformer_native_bucketed"
            ]
        )

    wide = wide.sort_values(
        ["support", "cell_type"],
        ascending=[False, True],
    )
    wide.to_csv(
        args.output_dir / "per_class_wide.csv",
        index=False,
    )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )
    sensitivity.to_csv(
        args.output_dir / "macro_sensitivity.csv",
        index=False,
    )

    global_frame = pd.DataFrame(
        global_rows
    )
    global_frame.to_csv(
        args.output_dir / "global_metrics_check.csv",
        index=False,
    )

    native = global_frame[
        global_frame["method"]
        == "geneformer_native_bucketed"
    ].iloc[0]
    fixed = global_frame[
        global_frame["method"]
        == "matched_fixed_tfidf"
    ].iloc[0]
    adaptive = global_frame[
        global_frame["method"]
        == "adaptive"
    ].iloc[0]

    nonrare = sensitivity[
        sensitivity["minimum_test_support"] == 20
    ].set_index("method")

    no_mega = sensitivity[
        sensitivity["minimum_test_support"] == -1
    ].set_index("method")

    summary = {
        "status": "PASS",
        "input_dir": str(
            args.input_dir.resolve()
        ),
        "output_dir": str(
            args.output_dir.resolve()
        ),
        "test_cells": int(len(reference)),
        "cell_types": labels,
        "support": {
            label: int(
                np.sum(true_labels == label)
            )
            for label in labels
        },
        "global_deltas": {
            "adaptive_minus_native": {
                "accuracy": float(
                    adaptive["accuracy"]
                    - native["accuracy"]
                ),
                "macro_f1": float(
                    adaptive["macro_f1"]
                    - native["macro_f1"]
                ),
                "balanced_accuracy": float(
                    adaptive["balanced_accuracy"]
                    - native["balanced_accuracy"]
                ),
            },
            "adaptive_minus_fixed": {
                "accuracy": float(
                    adaptive["accuracy"]
                    - fixed["accuracy"]
                ),
                "macro_f1": float(
                    adaptive["macro_f1"]
                    - fixed["macro_f1"]
                ),
                "balanced_accuracy": float(
                    adaptive["balanced_accuracy"]
                    - fixed["balanced_accuracy"]
                ),
            },
        },
        "support_at_least_20": {
            method: {
                "macro_f1": float(
                    nonrare.loc[
                        method,
                        "macro_f1_over_included_classes",
                    ]
                ),
                "mean_recall": float(
                    nonrare.loc[
                        method,
                        "mean_recall_over_included_classes",
                    ]
                ),
                "included_classes": str(
                    nonrare.loc[
                        method,
                        "included_classes",
                    ]
                ),
            }
            for method in METHOD_ORDER
        },
        "excluding_megakaryocytes": {
            method: {
                "macro_f1": float(
                    no_mega.loc[
                        method,
                        "macro_f1_over_included_classes",
                    ]
                ),
                "mean_recall": float(
                    no_mega.loc[
                        method,
                        "mean_recall_over_included_classes",
                    ]
                ),
            }
            for method in METHOD_ORDER
        },
    }

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(
            json_safe(summary),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("GENEFORMER KANG PER-CLASS ANALYSIS")
    print("=" * 100)
    print("\nTEST SUPPORT")
    print(
        wide[
            ["cell_type", "support"]
        ].to_string(index=False)
    )

    display_columns = [
        "cell_type",
        "support",
        "f1__geneformer_native_bucketed",
        "f1__matched_fixed_tfidf",
        "f1__adaptive",
        "f1__adaptive_minus_fixed",
        "f1__adaptive_minus_native",
    ]
    print("\nPER-CLASS F1")
    print(
        wide[display_columns].to_string(
            index=False
        )
    )

    print("\nMACRO SENSITIVITY")
    print(
        sensitivity[
            [
                "method_label",
                "minimum_test_support",
                "included_class_count",
                "macro_f1_over_included_classes",
                "mean_recall_over_included_classes",
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUT:", args.output_dir)
    print("FINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
