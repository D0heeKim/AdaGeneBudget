#!/usr/bin/env python3
"""Add a Native-Rank Top-K baseline to the completed Kang Geneformer benchmark.

Dataset/protocol
----------------
- Kang 2018
- patient 1015 held out
- Geneformer V1-10M
- matched fixed budget K=398
- Native ranking followed by deterministic top-K truncation
- 5-NN cosine annotation
- quality: one deterministic evaluation
- timing: five repeats over train + test, matching the original Kang benchmark
- batch size 64, four workers, three warm-up steps
- FP32, hidden_states[5], non-padding mean pooling

This script creates only the new method and then appends it to the validated
Kang benchmark. Existing outputs are backed up before CSV modification.

New cache
---------
data/processed/kang_2018/patient_1015_holdout/geneformer_cache/
    methods/native_rank_fixed_k398

New benchmark output
--------------------
outputs/geneformer/kang_patient1015/main/native_rank_fixed_k

Files modified at the very end, after successful evaluation
------------------------------------------------------------
- method-cache-root/summary.csv
- outputs/geneformer/kang_patient1015/main/summary.csv
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(".")

CACHE_BUILDER_SCRIPT = (
    PROJECT / "experiments/geneformer/kang/build_method_caches.py"
)
BENCHMARK_SCRIPT = (
    PROJECT / "experiments/geneformer/kang/run_main.py"
)

BASE_CACHE = (
    PROJECT
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "base"
)
METHOD_ROOT = BASE_CACHE.parent / "methods"
METHOD_CACHE_SUMMARY = METHOD_ROOT / "summary.csv"

CHECKPOINT = Path(
    "./checkpoints/Geneformer-V1-10M"
)

MAIN_OUTPUT = (
    PROJECT
    / "outputs"
    / "geneformer"
    / "kang_patient1015"
    / "main"
)

METHOD = "native_rank_fixed_k"
METHOD_LABEL = "Native-Rank Top-K"
K = 398

NEW_CACHE = METHOD_ROOT / f"native_rank_fixed_k{K}"
NEW_OUTPUT = MAIN_OUTPUT / METHOD

EXPECTED_TRAIN_CELLS = 19_583
EXPECTED_TEST_CELLS = 5_090

BATCH_SIZE = 64
NUM_WORKERS = 4
WARMUP_STEPS = 3
TIMING_REPEATS = 5
KNN_K = 5
NEIGHBOR_K = 10
SIMILARITY_CHUNK = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--overwrite-new-method",
        action="store_true",
        help=(
            "Delete only the new Native-Rank Top-K cache/output "
            "before rerunning. Existing benchmark methods are untouched."
        ),
    )
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_module(name: str, path: Path) -> Any:
    require(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def stable_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def backup(path: Path, tag: str) -> Path:
    directory = path.parent / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = (
        directory
        / f"{path.stem}_before_{tag}_{timestamp}{path.suffix}"
    )
    shutil.copy2(path, destination)
    return destination


def update_split_meta(
    split_directory: Path,
    split: str,
) -> dict[str, Any]:
    meta_path = split_directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["method"] = METHOD
    meta["method_label"] = METHOD_LABEL
    meta["fixed_k"] = K
    meta["selection_rule"] = (
        "Geneformer native rank followed by deterministic top-K truncation"
    )
    meta["split"] = split
    stable_json(meta, meta_path)
    return meta


def build_new_cache(
    builder: Any,
    progress_every: int,
) -> dict[str, Any]:
    if NEW_CACHE.exists():
        raise FileExistsError(NEW_CACHE)

    temporary = NEW_CACHE.with_name(NEW_CACHE.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)

    try:
        resources = builder.load_base_resources(BASE_CACHE)

        settings = {
            "model_input_size": K,
            "fixed_k": K,
            "selection_rule": (
                "Geneformer native rank followed by deterministic top-K "
                "truncation"
            ),
            "n_counts_source": (
                "original uncompressed .X row sum before vocabulary "
                "filtering and selection"
            ),
            "ordering": (
                "Geneformer native score descending; token ID ascending "
                "for ties"
            ),
        }

        train_result = builder.build_selection(
            matrix=resources.train_matrix,
            n_counts=resources.train_n_counts,
            token_ids=resources.token_ids,
            gene_medians=resources.gene_medians,
            idf=resources.idf,
            method="native",
            split_name="train",
            model_input_size=K,
            tau=0.90,
            k_min=128,
            k_max=600,
            fixed_k=None,
            random_seed=None,
            progress_every=progress_every,
        )
        test_result = builder.build_selection(
            matrix=resources.test_matrix,
            n_counts=resources.test_n_counts,
            token_ids=resources.token_ids,
            gene_medians=resources.gene_medians,
            idf=resources.idf,
            method="native",
            split_name="test",
            model_input_size=K,
            tau=0.90,
            k_min=128,
            k_max=600,
            fixed_k=None,
            random_seed=None,
            progress_every=progress_every,
        )

        builder.save_selection(
            temporary / "train",
            train_result,
            "native",
            "train",
            settings,
        )
        builder.save_selection(
            temporary / "test",
            test_result,
            "native",
            "test",
            settings,
        )

        train_meta = update_split_meta(
            temporary / "train",
            "train",
        )
        test_meta = update_split_meta(
            temporary / "test",
            "test",
        )

        builder.validate_saved_method(
            method_dir=temporary,
            expected_train_cells=EXPECTED_TRAIN_CELLS,
            expected_test_cells=EXPECTED_TEST_CELLS,
            maximum_length=K,
        )

        root_meta = {
            "status": "PASS",
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "directory_name": NEW_CACHE.name,
            "directory": str(NEW_CACHE.resolve()),
            "fixed_k": K,
            "settings": settings,
            "train": train_meta,
            "test": test_meta,
            "combined_selection_compute_time_s": float(
                train_result.compute_time_s
                + test_result.compute_time_s
            ),
        }
        stable_json(root_meta, temporary / "meta.json")

        os.replace(temporary, NEW_CACHE)
        return root_meta

    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def append_cache_summary(
    old: pd.DataFrame,
    cache_meta: dict[str, Any],
) -> pd.DataFrame:
    if METHOD in set(old["method"].astype(str)):
        raise RuntimeError(
            f"{METHOD_CACHE_SUMMARY} already contains {METHOD}."
        )

    row = {column: np.nan for column in old.columns}
    values = {
        "cache": NEW_CACHE.name,
        "method": METHOD,
        "fixed_k": K,
        "random_seed_train": np.nan,
        "random_seed_test": np.nan,
        "train_mean_genes": cache_meta["train"][
            "length_statistics"
        ]["mean"],
        "test_mean_genes": cache_meta["test"][
            "length_statistics"
        ]["mean"],
        "train_median_genes": cache_meta["train"][
            "length_statistics"
        ]["median"],
        "test_median_genes": cache_meta["test"][
            "length_statistics"
        ]["median"],
        "selection_compute_time_s": cache_meta[
            "combined_selection_compute_time_s"
        ],
    }
    for key, value in values.items():
        if key in row:
            row[key] = value

    return pd.concat(
        [
            old,
            pd.DataFrame([row], columns=old.columns),
        ],
        ignore_index=True,
    )


def build_main_summary(
    old_summary: pd.DataFrame,
    cache_meta: dict[str, Any],
    metrics: dict[str, float],
    fidelity: dict[str, float],
    timing_rows: list[dict[str, Any]],
    train_lengths: np.ndarray,
    test_lengths: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if METHOD in set(old_summary["method"].astype(str)):
        raise RuntimeError(
            f"{MAIN_OUTPUT / 'summary.csv'} already contains {METHOD}."
        )

    timing = pd.DataFrame(timing_rows)
    native_rows = old_summary.loc[
        old_summary["method"]
        == "geneformer_native_bucketed"
    ]
    if len(native_rows) != 1:
        raise RuntimeError(
            "Expected exactly one Native Bucketed summary row."
        )
    native = native_rows.iloc[0]

    adaptive_rows = old_summary.loc[
        old_summary["method"] == "adaptive"
    ]
    fixed_rows = old_summary.loc[
        old_summary["method"]
        == "matched_fixed_tfidf"
    ]
    if len(adaptive_rows) != 1 or len(fixed_rows) != 1:
        raise RuntimeError(
            "Could not resolve adaptive/fixed reference rows."
        )

    summary: dict[str, Any] = {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "cache": NEW_CACHE.name,
        "tau": None,
        "fixed_k": K,
        "batching": "length_sorted",
        "sampling": "native_rank_top_k",
        "quality_num_seeds": 1,
        "timing_repeats": TIMING_REPEATS,
        "selection_in_dataloader": False,
        "selection_time_s": float(
            cache_meta[
                "combined_selection_compute_time_s"
            ]
        ),
        "bucketing_time_s": float(
            timing["bucketing_time_s"].iloc[0]
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
        "end_to_end_time_mean_s": float(
            timing["end_to_end_time_s"].mean()
        ),
        "end_to_end_time_std_s": float(
            timing["end_to_end_time_s"].std(ddof=1)
        ),
        "embedding_time_mean_s": float(
            timing["embedding_time_s"].mean()
        ),
        "embedding_time_std_s": float(
            timing["embedding_time_s"].std(ddof=1)
        ),
        "cells_per_s_mean": float(
            timing["cells_per_s"].mean()
        ),
        "cells_per_s_std": float(
            timing["cells_per_s"].std(ddof=1)
        ),
        "embedding_only_cells_per_s_mean": float(
            timing[
                "embedding_only_cells_per_s"
            ].mean()
        ),
        "embedding_only_cells_per_s_std": float(
            timing[
                "embedding_only_cells_per_s"
            ].std(ddof=1)
        ),
        "peak_gpu_memory_gb": float(
            timing["peak_gpu_memory_gb"].max()
        ),
        "padding_overhead_ratio_mean": float(
            timing[
                "padding_overhead_ratio"
            ].mean()
        ),
        "matched_fixed_k": K,
        "target_adaptive_train_mean": float(
            adaptive_rows.iloc[0][
                "target_adaptive_train_mean"
            ]
        ),
        "matched_fixed_expected_mean": float(
            fixed_rows.iloc[0][
                "matched_fixed_expected_mean"
            ]
        ),
        "reference_method": (
            "geneformer_native_bucketed"
        ),
        **metrics,
        "accuracy_std": 0.0,
        "macro_f1_std": 0.0,
        "weighted_f1_std": 0.0,
        "balanced_accuracy_std": 0.0,
        **fidelity,
        "embedding_cosine_mean_std": 0.0,
        "embedding_cosine_median_std": 0.0,
        "embedding_cosine_min_std": 0.0,
        "neighbor_recall_10_std": 0.0,
    }

    summary["speedup_vs_native_bucketed"] = float(
        summary["cells_per_s_mean"]
        / float(native["cells_per_s_mean"])
    )
    summary[
        "memory_reduction_vs_native_bucketed"
    ] = float(
        1.0
        - summary["peak_gpu_memory_gb"]
        / float(native["peak_gpu_memory_gb"])
    )

    row = {
        column: summary.get(column, np.nan)
        for column in old_summary.columns
    }
    combined = pd.concat(
        [
            old_summary,
            pd.DataFrame(
                [row],
                columns=old_summary.columns,
            ),
        ],
        ignore_index=True,
    )

    order = {
        "geneformer_native_sequential": 0,
        "geneformer_native_bucketed": 1,
        METHOD: 2,
        "matched_random_fixed_k": 3,
        "matched_fixed_tfidf": 4,
        "adaptive": 5,
    }
    combined["_order"] = (
        combined["method"]
        .map(order)
        .fillna(999)
    )
    combined = (
        combined
        .sort_values("_order", kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return combined, summary


def main() -> int:
    args = parse_args()

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    for path in (
        CACHE_BUILDER_SCRIPT,
        BENCHMARK_SCRIPT,
        BASE_CACHE,
        METHOD_ROOT,
        METHOD_CACHE_SUMMARY,
        CHECKPOINT,
        MAIN_OUTPUT,
        MAIN_OUTPUT / "summary.csv",
        MAIN_OUTPUT
        / "geneformer_native_bucketed"
        / "quality_seed0.npz",
        MAIN_OUTPUT
        / "geneformer_native_bucketed"
        / "native_neighbors_10.npy",
    ):
        require(path)

    old_cache_summary = pd.read_csv(
        METHOD_CACHE_SUMMARY
    )
    old_main_summary = pd.read_csv(
        MAIN_OUTPUT / "summary.csv"
    )

    cache_has_method = (
        METHOD
        in set(
            old_cache_summary["method"].astype(str)
        )
    )
    main_has_method = (
        METHOD
        in set(
            old_main_summary["method"].astype(str)
        )
    )

    if (
        NEW_CACHE.exists()
        or NEW_OUTPUT.exists()
        or cache_has_method
        or main_has_method
    ):
        if not args.overwrite_new_method:
            raise FileExistsError(
                "Native-Rank Top-K artifacts already exist. "
                "Use --overwrite-new-method only after inspecting them."
            )

        if NEW_CACHE.exists():
            shutil.rmtree(NEW_CACHE)
        if NEW_OUTPUT.exists():
            shutil.rmtree(NEW_OUTPUT)

        old_cache_summary = old_cache_summary.loc[
            old_cache_summary["method"].astype(str)
            != METHOD
        ].reset_index(drop=True)
        old_main_summary = old_main_summary.loc[
            old_main_summary["method"].astype(str)
            != METHOD
        ].reset_index(drop=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    builder = load_module(
        "kang_cache_builder_native_rank",
        CACHE_BUILDER_SCRIPT,
    )
    benchmark = load_module(
        "kang_benchmark_native_rank",
        BENCHMARK_SCRIPT,
    )

    print("=" * 112, flush=True)
    print(
        "KANG GENEFORMER NATIVE-RANK TOP-398 BASELINE",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(f"CACHE: {NEW_CACHE}", flush=True)
    print(f"OUTPUT: {NEW_OUTPUT}", flush=True)
    print(
        "QUALITY: deterministic single evaluation",
        flush=True,
    )
    print(
        "TIMING: train+test, five repeats",
        flush=True,
    )

    cache_meta = build_new_cache(
        builder,
        args.progress_every,
    )

    metadata = benchmark.load_base_metadata(
        BASE_CACHE
    )
    train_labels = metadata["train_labels"]
    test_labels = metadata["test_labels"]

    train_cache = benchmark.load_token_cache(
        NEW_CACHE,
        "train",
    )
    test_cache = benchmark.load_token_cache(
        NEW_CACHE,
        "test",
    )

    train_lengths = np.asarray(
        train_cache.lengths,
        dtype=np.int64,
    )
    test_lengths = np.asarray(
        test_cache.lengths,
        dtype=np.int64,
    )

    device = torch.device(args.device)
    benchmark.seed_all(0)
    model, hidden_state_index = benchmark.load_model(
        CHECKPOINT,
        device,
    )

    train_dataset = benchmark.CachedTokenDataset(
        train_cache
    )
    test_dataset = benchmark.CachedTokenDataset(
        test_cache
    )
    train_loader, _ = benchmark.make_loader(
        train_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        "length_sorted",
    )
    test_loader, _ = benchmark.make_loader(
        test_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        "length_sorted",
    )

    train_embeddings, _ = benchmark.extract_embeddings(
        model,
        hidden_state_index,
        train_loader,
        len(train_dataset),
        device,
        measure=False,
    )
    test_embeddings, _ = benchmark.extract_embeddings(
        model,
        hidden_state_index,
        test_loader,
        len(test_dataset),
        device,
        measure=False,
    )

    predictions = benchmark.exact_cosine_knn_predict(
        train_embeddings,
        train_labels,
        test_embeddings,
        KNN_K,
        device,
        SIMILARITY_CHUNK,
    )
    metrics = benchmark.classification_metrics(
        test_labels,
        predictions,
    )

    native_artifact = np.load(
        MAIN_OUTPUT
        / "geneformer_native_bucketed"
        / "quality_seed0.npz",
        allow_pickle=False,
    )
    native_embeddings = (
        native_artifact["test_embeddings"]
        .astype(np.float32)
    )
    native_neighbors = np.load(
        MAIN_OUTPUT
        / "geneformer_native_bucketed"
        / "native_neighbors_10.npy",
        allow_pickle=False,
    )
    fidelity = benchmark.fidelity_metrics(
        native_embeddings,
        test_embeddings,
        native_neighbors,
        NEIGHBOR_K,
        device,
        SIMILARITY_CHUNK,
    )

    NEW_OUTPUT.mkdir(
        parents=True,
        exist_ok=False,
    )
    np.savez_compressed(
        NEW_OUTPUT / "quality_seed0.npz",
        train_labels=train_labels,
        test_labels=test_labels,
        test_predictions=predictions,
        test_embeddings=test_embeddings.astype(
            np.float32,
            copy=False,
        ),
        test_conditions=metadata[
            "test_conditions"
        ],
        test_donors=metadata[
            "test_donors"
        ],
        test_obs_names=metadata[
            "test_obs_names"
        ],
    )

    prediction_frame = pd.DataFrame(
        {
            "obs_name": metadata[
                "test_obs_names"
            ],
            "cell_type": test_labels,
            "condition": metadata[
                "test_conditions"
            ],
            "donor": metadata[
                "test_donors"
            ],
            "prediction": predictions,
        }
    )
    prediction_frame.to_csv(
        NEW_OUTPUT / "predictions_seed0.csv",
        index=False,
    )

    seed_rows = [
        {
            "method": METHOD,
            "seed": 0,
            "cache": NEW_CACHE.name,
            **metrics,
            **fidelity,
        }
    ]

    print(
        "[quality] "
        f"accuracy={metrics['accuracy']:.6f} "
        f"macro_f1={metrics['macro_f1']:.6f} "
        f"balanced={metrics['balanced_accuracy']:.6f} "
        f"cosine={fidelity['embedding_cosine_mean']:.6f} "
        f"neighbor@10={fidelity['neighbor_recall_10']:.6f}",
        flush=True,
    )

    del (
        train_embeddings,
        test_embeddings,
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
    )
    gc.collect()
    torch.cuda.empty_cache()

    train_timing_dataset = (
        benchmark.CachedTokenDataset(
            train_cache
        )
    )
    test_timing_dataset = (
        benchmark.CachedTokenDataset(
            test_cache
        )
    )
    (
        train_timing_loader,
        train_bucketing_time,
    ) = benchmark.make_loader(
        train_timing_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        "length_sorted",
    )
    (
        test_timing_loader,
        test_bucketing_time,
    ) = benchmark.make_loader(
        test_timing_dataset,
        BATCH_SIZE,
        NUM_WORKERS,
        "length_sorted",
    )
    bucketing_time_s = float(
        train_bucketing_time
        + test_bucketing_time
    )

    benchmark.warmup(
        model,
        hidden_state_index,
        train_timing_loader,
        device,
        WARMUP_STEPS,
    )
    benchmark.warmup(
        model,
        hidden_state_index,
        test_timing_loader,
        device,
        WARMUP_STEPS,
    )

    selection_time_s = float(
        cache_meta[
            "combined_selection_compute_time_s"
        ]
    )
    total_cells = (
        EXPECTED_TRAIN_CELLS
        + EXPECTED_TEST_CELLS
    )
    timing_rows: list[dict[str, Any]] = []

    for repeat in range(TIMING_REPEATS):
        train_embeddings, train_timing = (
            benchmark.extract_embeddings(
                model,
                hidden_state_index,
                train_timing_loader,
                len(train_timing_dataset),
                device,
                measure=True,
            )
        )
        test_embeddings, test_timing = (
            benchmark.extract_embeddings(
                model,
                hidden_state_index,
                test_timing_loader,
                len(test_timing_dataset),
                device,
                measure=True,
            )
        )

        embedding_time_s = float(
            train_timing["embedding_time_s"]
            + test_timing["embedding_time_s"]
        )
        end_to_end_time_s = float(
            selection_time_s
            + bucketing_time_s
            + embedding_time_s
        )

        actual_tokens = float(
            train_timing[
                "actual_sequence_tokens"
            ]
            + test_timing[
                "actual_sequence_tokens"
            ]
        )
        padded_tokens = float(
            train_timing[
                "padded_tokens_processed"
            ]
            + test_timing[
                "padded_tokens_processed"
            ]
        )

        row = {
            "method": METHOD,
            "repeat": repeat,
            "cache": NEW_CACHE.name,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
            "embedding_time_s": embedding_time_s,
            "end_to_end_time_s": (
                end_to_end_time_s
            ),
            "cells_per_s": float(
                total_cells
                / end_to_end_time_s
            ),
            "embedding_only_cells_per_s": float(
                total_cells
                / embedding_time_s
            ),
            "peak_gpu_memory_gb": float(
                max(
                    train_timing[
                        "peak_gpu_memory_gb"
                    ],
                    test_timing[
                        "peak_gpu_memory_gb"
                    ],
                )
            ),
            "padding_overhead_ratio": float(
                padded_tokens
                / actual_tokens
            ),
        }
        timing_rows.append(row)

        print(
            "[timing] "
            f"repeat={repeat + 1}/{TIMING_REPEATS} "
            f"E2E={row['end_to_end_time_s']:.4f}s "
            f"throughput={row['cells_per_s']:.2f} cells/s "
            f"memory={row['peak_gpu_memory_gb']:.3f} GB",
            flush=True,
        )

        del train_embeddings, test_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    combined_main_summary, summary = (
        build_main_summary(
            old_summary=old_main_summary,
            cache_meta=cache_meta,
            metrics=metrics,
            fidelity=fidelity,
            timing_rows=timing_rows,
            train_lengths=train_lengths,
            test_lengths=test_lengths,
        )
    )

    benchmark.save_method_results(
        NEW_OUTPUT,
        summary,
        seed_rows,
        timing_rows,
    )

    combined_cache_summary = append_cache_summary(
        old_cache_summary,
        cache_meta,
    )

    cache_backup = backup(
        METHOD_CACHE_SUMMARY,
        METHOD,
    )
    main_backup = backup(
        MAIN_OUTPUT / "summary.csv",
        METHOD,
    )

    atomic_csv(
        combined_cache_summary,
        METHOD_CACHE_SUMMARY,
    )
    atomic_csv(
        combined_main_summary,
        MAIN_OUTPUT / "summary.csv",
    )

    manifest = {
        "status": "PASS",
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "fixed_k": K,
        "deterministic_quality": True,
        "quality_num_seeds": 1,
        "timing_repeats": TIMING_REPEATS,
        "timing_scope": "train plus test",
        "cache": str(NEW_CACHE.resolve()),
        "output": str(NEW_OUTPUT.resolve()),
        "cache_summary_backup": str(
            cache_backup.resolve()
        ),
        "main_summary_backup": str(
            main_backup.resolve()
        ),
        "summary": summary,
    }
    stable_json(
        manifest,
        NEW_OUTPUT / "result_manifest.json",
    )

    print("\n" + "=" * 112, flush=True)
    print(
        "KANG NATIVE-RANK TOP-398 RESULT",
        flush=True,
    )
    print("=" * 112, flush=True)
    fields = [
        "test_mean_selected_genes",
        "end_to_end_time_mean_s",
        "end_to_end_time_std_s",
        "cells_per_s_mean",
        "cells_per_s_std",
        "peak_gpu_memory_gb",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "neighbor_recall_10",
        "speedup_vs_native_bucketed",
        "memory_reduction_vs_native_bucketed",
    ]
    for field in fields:
        print(
            f"{field}: {summary[field]}",
            flush=True,
        )
    print(f"Output: {NEW_OUTPUT}", flush=True)
    print("FINAL STATUS: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
