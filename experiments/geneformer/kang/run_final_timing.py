#!/usr/bin/env python3
"""Recheck Kang Geneformer timing after batch-output scope correction.

Purpose
-------
Measure only the four methods used in the paper efficiency table:

1. Geneformer Native Bucketed
2. Native-Rank Top-398
3. Matched Random, timing seed 0
4. AdaGeneBudget

Scientific protocol
-------------------
- Kang 2018, patient 1015 held out
- reference/train: 19,583 cells
- query/test: 5,090 cells
- timing scope: train + test
- batch size: 64
- DataLoader workers: 4
- warm-up: 3 batches for train and 3 batches for test
- timing repeats: 5
- FP32 inference
- dynamic padding with length sorting
- hidden_states[5]
- non-padding mean pooling
- selection time reused from the validated main summary
- bucketing and embedding timing measured fresh

This script:
- imports the corrected 07_run_geneformer_kang_benchmark.py
- reuses all existing validated method caches
- does not recompute quality
- does not modify outputs/geneformer/kang_patient1015/main
- writes results to a separate directory
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(".")

BENCHMARK_SCRIPT = (
    PROJECT / "experiments/geneformer/kang/run_main.py"
)

METHOD_ROOT = (
    PROJECT
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "methods"
)

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

DEFAULT_OUTPUT = (
    PROJECT
    / "outputs"
    / "geneformer"
    / "kang_patient1015"
    / "timing_train_test_b64_r5_scopefix"
)

TRAIN_CELLS = 19_583
TEST_CELLS = 5_090
TOTAL_CELLS = TRAIN_CELLS + TEST_CELLS

METHODS = [
    (
        "geneformer_native_bucketed",
        "Native Bucketed",
    ),
    (
        "native_rank_fixed_k",
        "Native-Rank Top-398",
    ),
    (
        "matched_random_fixed_k",
        "Matched Random",
    ),
    (
        "adaptive",
        "AdaGeneBudget",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)


def load_module(
    name: str,
    path: Path,
) -> Any:
    require_file(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    # Required for modules containing dataclasses.
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    temporary = path.with_name(
        path.name + ".tmp"
    )
    frame.to_csv(
        temporary,
        index=False,
    )
    temporary.replace(path)


def sample_std(
    values: list[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return float(
        np.std(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            ddof=1,
        )
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def resolve_cache(
    cache_value: Any,
) -> Path:
    text = str(cache_value)

    if text in {"", "nan", "None"}:
        raise RuntimeError(
            f"Invalid cache value: {cache_value}"
        )

    path = Path(text)

    if not path.is_absolute():
        path = METHOD_ROOT / path

    require_dir(path)
    return path


def require_finite_float(
    row: pd.Series,
    column: str,
    method: str,
) -> float:
    if column not in row.index:
        raise RuntimeError(
            f"{method}: missing column {column}"
        )

    value = float(row[column])

    if not np.isfinite(value):
        raise RuntimeError(
            f"{method}: non-finite {column}: {value}"
        )

    return value


def validate_args(
    args: argparse.Namespace,
) -> None:
    expected = {
        "batch_size": 64,
        "num_workers": 4,
        "warmup_steps": 3,
        "timing_repeats": 5,
    }

    observed = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "timing_repeats": args.timing_repeats,
    }

    if observed != expected:
        raise RuntimeError(
            "This audit uses the frozen protocol "
            f"{expected}; observed {observed}."
        )



def measure_split_once(
    benchmark: Any,
    model: torch.nn.Module,
    hidden_state_index: int,
    loader: Any,
    device: torch.device,
) -> dict[str, float]:
    """Time one split while discarding every batch embedding.

    This matches the PBMC timing semantics:
    - full MLM forward
    - CPU output transfer included
    - no full embedding matrix retained
    - batch-local model outputs released before the next forward
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    started = perf_counter()
    actual_tokens = 0
    padded_tokens = 0

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            lengths = batch["lengths"].to(
                device,
                non_blocking=True,
            )

            batch_embeddings = (
                benchmark.model_batch_embeddings(
                    model=model,
                    hidden_state_index=hidden_state_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                )
            )

            actual_tokens += int(
                lengths.sum().item()
            )
            padded_tokens += int(
                input_ids.numel()
            )

            del (
                batch_embeddings,
                input_ids,
                attention_mask,
                lengths,
            )

    torch.cuda.synchronize(device)

    elapsed = float(
        perf_counter() - started
    )
    peak_memory_gb = float(
        torch.cuda.max_memory_allocated(device)
        / (1024**3)
    )

    return {
        "embedding_time_s": elapsed,
        "peak_gpu_memory_gb": peak_memory_gb,
        "actual_sequence_tokens": float(
            actual_tokens
        ),
        "padded_tokens_processed": float(
            padded_tokens
        ),
        "padding_overhead_ratio": float(
            padded_tokens / actual_tokens
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    require_file(BENCHMARK_SCRIPT)
    require_file(
        MAIN_OUTPUT / "summary.csv"
    )
    require_dir(METHOD_ROOT)
    require_dir(CHECKPOINT)

    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    benchmark = load_module(
        "kang_geneformer_benchmark_scopefix",
        BENCHMARK_SCRIPT,
    )

    # Verify that the intended correction is actually present.
    if not hasattr(
        benchmark,
        "model_batch_embeddings",
    ):
        raise RuntimeError(
            "Corrected 07 script does not expose "
            "model_batch_embeddings()."
        )

    for required_name in [
        "load_model",
        "load_token_cache",
        "CachedTokenDataset",
        "make_loader",
        "warmup",
        "extract_embeddings",
    ]:
        if not hasattr(
            benchmark,
            required_name,
        ):
            raise RuntimeError(
                f"Corrected benchmark is missing "
                f"{required_name}."
            )

    old_summary = pd.read_csv(
        MAIN_OUTPUT / "summary.csv"
    )

    records: list[dict[str, Any]] = []

    native_rank_summary_path = (
        MAIN_OUTPUT
        / "native_rank_fixed_k"
        / "summary.json"
    )

    for method, expected_label in METHODS:
        rows = old_summary.loc[
            old_summary["method"].astype(str)
            == method
        ]

        if len(rows) == 1:
            row = rows.iloc[0]
            summary_source = (
                MAIN_OUTPUT / "summary.csv"
            )

        elif (
            method == "native_rank_fixed_k"
            and len(rows) == 0
        ):
            require_file(
                native_rank_summary_path
            )
            payload = json.loads(
                native_rank_summary_path.read_text(
                    encoding="utf-8"
                )
            )
            row = pd.Series(payload)
            summary_source = (
                native_rank_summary_path
            )

        else:
            raise RuntimeError(
                f"Expected exactly one summary source "
                f"for {method}; found {len(rows)} rows "
                f"in main summary."
            )

        cache_path = resolve_cache(
            row["cache"]
        )

        records.append(
            {
                "method": method,
                "method_label": expected_label,
                "cache_path": cache_path,
                "summary_source": str(
                    summary_source.resolve()
                ),
                "selection_time_s": (
                    require_finite_float(
                        row,
                        "selection_time_s",
                        method,
                    )
                ),
                "old_cells_per_s_mean": (
                    require_finite_float(
                        row,
                        "cells_per_s_mean",
                        method,
                    )
                ),
                "old_peak_gpu_memory_gb": (
                    require_finite_float(
                        row,
                        "peak_gpu_memory_gb",
                        method,
                    )
                ),
            }
        )

        print(
            f"[provenance] {method}: "
            f"{summary_source}",
            flush=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device(
        args.device
    )

    torch.set_grad_enabled(False)
    benchmark.seed_all(0)

    model, hidden_state_index = (
        benchmark.load_model(
            CHECKPOINT,
            device,
        )
    )

    all_run_rows: list[
        dict[str, Any]
    ] = []
    all_summary_rows: list[
        dict[str, Any]
    ] = []

    print("=" * 112, flush=True)
    print(
        "KANG GENEFORER TIMING RECHECK "
        "AFTER BATCH-SCOPE CORRECTION",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(
        f"GPU={torch.cuda.get_device_name(device)}",
        flush=True,
    )
    print(
        f"batch_size={args.batch_size}",
        flush=True,
    )
    print(
        f"num_workers={args.num_workers}",
        flush=True,
    )
    print(
        f"warmup_steps={args.warmup_steps}",
        flush=True,
    )
    print(
        f"timing_repeats={args.timing_repeats}",
        flush=True,
    )
    print(
        f"timing_scope=train+test "
        f"({TOTAL_CELLS:,} cells)",
        flush=True,
    )
    print(
        "existing main results will not be modified",
        flush=True,
    )

    for method_index, record in enumerate(
        records,
        start=1,
    ):
        method = record["method"]
        label = record["method_label"]
        cache_path = record["cache_path"]

        print(
            "\n" + "=" * 112,
            flush=True,
        )
        print(
            f"[{method_index}/{len(records)}] "
            f"{method} ({label})",
            flush=True,
        )
        print(
            f"cache={cache_path}",
            flush=True,
        )
        print(
            "=" * 112,
            flush=True,
        )

        train_cache = (
            benchmark.load_token_cache(
                cache_path,
                "train",
            )
        )
        test_cache = (
            benchmark.load_token_cache(
                cache_path,
                "test",
            )
        )

        train_dataset = (
            benchmark.CachedTokenDataset(
                train_cache
            )
        )
        test_dataset = (
            benchmark.CachedTokenDataset(
                test_cache
            )
        )

        if len(train_dataset) != TRAIN_CELLS:
            raise RuntimeError(
                f"{method}: expected "
                f"{TRAIN_CELLS} train cells, "
                f"found {len(train_dataset)}."
            )

        if len(test_dataset) != TEST_CELLS:
            raise RuntimeError(
                f"{method}: expected "
                f"{TEST_CELLS} test cells, "
                f"found {len(test_dataset)}."
            )

        (
            train_loader,
            train_bucketing_time,
        ) = benchmark.make_loader(
            train_dataset,
            args.batch_size,
            args.num_workers,
            "length_sorted",
        )

        (
            test_loader,
            test_bucketing_time,
        ) = benchmark.make_loader(
            test_dataset,
            args.batch_size,
            args.num_workers,
            "length_sorted",
        )

        bucketing_time_s = float(
            train_bucketing_time
            + test_bucketing_time
        )

        benchmark.warmup(
            model,
            hidden_state_index,
            train_loader,
            device,
            args.warmup_steps,
        )

        benchmark.warmup(
            model,
            hidden_state_index,
            test_loader,
            device,
            args.warmup_steps,
        )

        selection_time_s = float(
            record["selection_time_s"]
        )

        method_rows: list[
            dict[str, Any]
        ] = []

        for repeat in range(
            args.timing_repeats
        ):
            train_timing = measure_split_once(
                benchmark=benchmark,
                model=model,
                hidden_state_index=hidden_state_index,
                loader=train_loader,
                device=device,
            )

            test_timing = measure_split_once(
                benchmark=benchmark,
                model=model,
                hidden_state_index=hidden_state_index,
                loader=test_loader,
                device=device,
            )

            embedding_time_s = float(
                train_timing[
                    "embedding_time_s"
                ]
                + test_timing[
                    "embedding_time_s"
                ]
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

            peak_memory_gb = float(
                max(
                    train_timing[
                        "peak_gpu_memory_gb"
                    ],
                    test_timing[
                        "peak_gpu_memory_gb"
                    ],
                )
            )

            row = {
                "method": method,
                "method_label": label,
                "repeat": int(repeat),
                "batch_size": int(
                    args.batch_size
                ),
                "num_workers": int(
                    args.num_workers
                ),
                "warmup_steps": int(
                    args.warmup_steps
                ),
                "timing_scope": (
                    "train_plus_test"
                ),
                "train_cells": TRAIN_CELLS,
                "test_cells": TEST_CELLS,
                "total_cells": TOTAL_CELLS,
                "selection_time_s": (
                    selection_time_s
                ),
                "bucketing_time_s": (
                    bucketing_time_s
                ),
                "embedding_time_s": (
                    embedding_time_s
                ),
                "end_to_end_time_s": (
                    end_to_end_time_s
                ),
                "cells_per_s": float(
                    TOTAL_CELLS
                    / end_to_end_time_s
                ),
                "embedding_only_cells_per_s": (
                    float(
                        TOTAL_CELLS
                        / embedding_time_s
                    )
                ),
                "peak_gpu_memory_gb": (
                    peak_memory_gb
                ),
                "actual_sequence_tokens": (
                    actual_tokens
                ),
                "padded_tokens_processed": (
                    padded_tokens
                ),
                "padding_overhead_ratio": (
                    float(
                        padded_tokens
                        / actual_tokens
                    )
                ),
            }

            method_rows.append(row)
            all_run_rows.append(row)

            print(
                "[timing] "
                f"repeat={repeat + 1}/"
                f"{args.timing_repeats} "
                f"E2E={end_to_end_time_s:.4f}s "
                f"throughput="
                f"{row['cells_per_s']:.2f} cells/s "
                f"peak={peak_memory_gb:.4f} GB "
                f"padding="
                f"{row['padding_overhead_ratio']:.4f}",
                flush=True,
            )

            gc.collect()
            torch.cuda.empty_cache()

        timing = pd.DataFrame(
            method_rows
        )

        train_lengths = np.asarray(
            train_cache.lengths,
            dtype=np.int64,
        )
        test_lengths = np.asarray(
            test_cache.lengths,
            dtype=np.int64,
        )

        summary_row = {
            "method": method,
            "method_label": label,
            "cache": str(
                cache_path.resolve()
            ),
            "batch_size": int(
                args.batch_size
            ),
            "num_workers": int(
                args.num_workers
            ),
            "warmup_steps": int(
                args.warmup_steps
            ),
            "timing_repeats": int(
                args.timing_repeats
            ),
            "timing_scope": (
                "train_plus_test"
            ),
            "train_cells": TRAIN_CELLS,
            "test_cells": TEST_CELLS,
            "total_cells": TOTAL_CELLS,
            "train_mean_selected_genes": (
                float(
                    np.mean(
                        train_lengths
                    )
                )
            ),
            "test_mean_selected_genes": (
                float(
                    np.mean(
                        test_lengths
                    )
                )
            ),
            "selection_time_s": (
                selection_time_s
            ),
            "bucketing_time_s": (
                bucketing_time_s
            ),
            "embedding_time_mean_s": (
                float(
                    timing[
                        "embedding_time_s"
                    ].mean()
                )
            ),
            "embedding_time_std_s": (
                float(
                    timing[
                        "embedding_time_s"
                    ].std(ddof=1)
                )
            ),
            "end_to_end_time_mean_s": (
                float(
                    timing[
                        "end_to_end_time_s"
                    ].mean()
                )
            ),
            "end_to_end_time_std_s": (
                float(
                    timing[
                        "end_to_end_time_s"
                    ].std(ddof=1)
                )
            ),
            "cells_per_s_mean": (
                float(
                    timing[
                        "cells_per_s"
                    ].mean()
                )
            ),
            "cells_per_s_std": (
                float(
                    timing[
                        "cells_per_s"
                    ].std(ddof=1)
                )
            ),
            "embedding_only_cells_per_s_mean": (
                float(
                    timing[
                        "embedding_only_cells_per_s"
                    ].mean()
                )
            ),
            "embedding_only_cells_per_s_std": (
                float(
                    timing[
                        "embedding_only_cells_per_s"
                    ].std(ddof=1)
                )
            ),
            "peak_gpu_memory_gb": (
                float(
                    timing[
                        "peak_gpu_memory_gb"
                    ].max()
                )
            ),
            "padding_overhead_ratio_mean": (
                float(
                    timing[
                        "padding_overhead_ratio"
                    ].mean()
                )
            ),
            "old_cells_per_s_mean": float(
                record[
                    "old_cells_per_s_mean"
                ]
            ),
            "old_peak_gpu_memory_gb": float(
                record[
                    "old_peak_gpu_memory_gb"
                ]
            ),
        }

        summary_row[
            "throughput_change_vs_old"
        ] = float(
            summary_row[
                "cells_per_s_mean"
            ]
            / summary_row[
                "old_cells_per_s_mean"
            ]
        )

        summary_row[
            "memory_change_vs_old"
        ] = float(
            summary_row[
                "peak_gpu_memory_gb"
            ]
            / summary_row[
                "old_peak_gpu_memory_gb"
            ]
        )

        all_summary_rows.append(
            summary_row
        )

        atomic_csv(
            pd.DataFrame(
                all_run_rows
            ),
            output_dir
            / "timing_runs_partial.csv",
        )

        atomic_csv(
            pd.DataFrame(
                all_summary_rows
            ),
            output_dir
            / "timing_summary_partial.csv",
        )

        print(
            "[method summary] "
            f"throughput="
            f"{summary_row['cells_per_s_mean']:.2f}"
            f" ± "
            f"{summary_row['cells_per_s_std']:.2f} "
            f"cells/s; "
            f"peak="
            f"{summary_row['peak_gpu_memory_gb']:.4f} GB; "
            f"old_peak="
            f"{summary_row['old_peak_gpu_memory_gb']:.4f} GB",
            flush=True,
        )

        del (
            train_loader,
            test_loader,
            train_dataset,
            test_dataset,
            train_cache,
            test_cache,
        )

        gc.collect()
        torch.cuda.empty_cache()

    run_frame = pd.DataFrame(
        all_run_rows
    )
    summary_frame = pd.DataFrame(
        all_summary_rows
    )

    native_rows = summary_frame.loc[
        summary_frame["method"]
        == "geneformer_native_bucketed"
    ]

    if len(native_rows) != 1:
        raise RuntimeError(
            "Native Bucketed result missing."
        )

    native = native_rows.iloc[0]
    native_speed = float(
        native["cells_per_s_mean"]
    )
    native_memory = float(
        native["peak_gpu_memory_gb"]
    )

    summary_frame[
        "speedup_vs_native_bucketed"
    ] = (
        summary_frame[
            "cells_per_s_mean"
        ]
        / native_speed
    )

    summary_frame[
        "memory_reduction_vs_native_bucketed"
    ] = (
        1.0
        - summary_frame[
            "peak_gpu_memory_gb"
        ]
        / native_memory
    )

    run_path = (
        output_dir
        / "timing_runs.csv"
    )
    summary_path = (
        output_dir
        / "timing_summary.csv"
    )

    atomic_csv(
        run_frame,
        run_path,
    )
    atomic_csv(
        summary_frame,
        summary_path,
    )

    manifest = {
        "status": "PASS",
        "purpose": (
            "Kang Geneformer timing-only "
            "recheck after isolating each "
            "batch forward scope"
        ),
        "main_results_modified": False,
        "quality_recomputed": False,
        "selection_recomputed": False,
        "timing_scope": (
            "train plus test"
        ),
        "train_cells": TRAIN_CELLS,
        "test_cells": TEST_CELLS,
        "total_cells": TOTAL_CELLS,
        "batch_size": args.batch_size,
        "num_workers": (
            args.num_workers
        ),
        "warmup_steps": (
            args.warmup_steps
        ),
        "timing_repeats": (
            args.timing_repeats
        ),
        "precision": "FP32",
        "batching": (
            "length_sorted dynamic padding"
        ),
        "selection_time_policy": (
            "reused from existing "
            "main/summary.csv"
        ),
        "forward_policy": (
            "full BertForMaskedLM forward; "
            "batch outputs isolated in "
            "model_batch_embeddings scope"
        ),
        "benchmark_script": str(
            BENCHMARK_SCRIPT.resolve()
        ),
        "benchmark_script_sha256": (
            sha256_file(
                BENCHMARK_SCRIPT
            )
        ),
        "methods": [
            item["method"]
            for item in records
        ],
        "timing_runs_csv": str(
            run_path.resolve()
        ),
        "timing_summary_csv": str(
            summary_path.resolve()
        ),
    }

    write_json(
        output_dir / "manifest.json",
        manifest,
    )

    print(
        "\n" + "=" * 132,
        flush=True,
    )
    print(
        "FINAL KANG TIMING SUMMARY",
        flush=True,
    )
    print(
        "=" * 132,
        flush=True,
    )

    columns = [
        "method",
        "test_mean_selected_genes",
        "cells_per_s_mean",
        "cells_per_s_std",
        "peak_gpu_memory_gb",
        "old_peak_gpu_memory_gb",
        "memory_change_vs_old",
        "speedup_vs_native_bucketed",
    ]

    print(
        summary_frame[
            columns
        ].to_string(
            index=False
        ),
        flush=True,
    )

    print(
        f"\nSaved: {summary_path}",
        flush=True,
    )
    print(
        "Existing main outputs modified: NO",
        flush=True,
    )
    print(
        "FINAL STATUS: PASS",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
