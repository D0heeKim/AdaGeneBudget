#!/usr/bin/env python3
"""Run the full Kang Geneformer V1-10M benchmark.

Methods
-------
1. Geneformer Native Max-2048, sequential batching
2. Geneformer Native Max-2048, length-bucketed batching
3. Matched Random fixed-K, seeds 0--4
4. Matched Fixed TF-IDF
5. AdaGeneBudget

Frozen scientific protocol
--------------------------
- Dataset: Kang 2018, patient 1015 held out
- Main label: cell_type
- Geneformer V1-10M
- FP32 inference
- Penultimate transformer block: hidden_states[5]
- Cell embedding: mean over all non-padding gene-token hidden states
- 5-NN cosine classification
- IDF and matched K determined from train only
- AdaGeneBudget: tau=0.90, Kmin=128, Kmax=600
- Matched Random: seeds 0--4
- Systems timing: warmup, then five repeats
- Gene mapping and one-time base-cache construction are excluded from timing
- Method selection compute time and bucketing time are included in end-to-end timing

The method caches are assumed to have already reordered every selected subset
by Geneformer's native rank-value score.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import BertForMaskedLM


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
DEFAULT_METHOD_CACHE_ROOT = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "kang_2018"
    / "patient_1015_holdout"
    / "geneformer_cache"
    / "methods"
)
DEFAULT_CHECKPOINT = Path(
    "./checkpoints/Geneformer-V1-10M"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "geneformer"
    / "kang_patient1015"
    / "main"
)

EXPECTED_TRAIN_CELLS = 19_583
EXPECTED_TEST_CELLS = 5_090
EXPECTED_HIDDEN_SIZE = 256
EXPECTED_NUM_LAYERS = 6
EXPECTED_MAX_POSITION_EMBEDDINGS = 2_048
EXPECTED_PAD_TOKEN_ID = 0
EXPECTED_VOCAB_SIZE = 25_426
EXPECTED_FIXED_K = 398
EXPECTED_TAU = 0.90
EXPECTED_K_MIN = 128
EXPECTED_K_MAX = 600


@dataclass(frozen=True)
class TokenCache:
    directory: Path
    indptr: np.ndarray
    token_ids: np.ndarray
    lengths: np.ndarray
    meta: dict[str, Any]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    cache_directories: dict[int, Path]
    batching: str
    quality_seeds: tuple[int, ...]
    timing_seed: int
    sampling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-cache",
        type=Path,
        default=DEFAULT_BASE_CACHE,
    )
    parser.add_argument(
        "--method-cache-root",
        type=Path,
        default=DEFAULT_METHOD_CACHE_ROOT,
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
        "--quality-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--similarity-chunk",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--overwrite-methods",
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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_string_array(path: Path) -> np.ndarray:
    require_file(path)
    return np.load(
        path,
        allow_pickle=False,
    ).astype(str)


def load_base_metadata(
    base_cache: Path,
) -> dict[str, np.ndarray]:
    required = [
        base_cache / "meta.json",
        base_cache / "train" / "labels.npy",
        base_cache / "test" / "labels.npy",
        base_cache / "train" / "conditions.npy",
        base_cache / "test" / "conditions.npy",
        base_cache / "train" / "donors.npy",
        base_cache / "test" / "donors.npy",
        base_cache / "train" / "obs_names.npy",
        base_cache / "test" / "obs_names.npy",
    ]
    for path in required:
        require_file(path)

    meta = json.loads(
        (base_cache / "meta.json").read_text(encoding="utf-8")
    )
    if meta.get("status") != "PASS":
        raise RuntimeError("Base-cache status is not PASS.")

    result = {
        "train_labels": load_string_array(
            base_cache / "train" / "labels.npy"
        ),
        "test_labels": load_string_array(
            base_cache / "test" / "labels.npy"
        ),
        "train_conditions": load_string_array(
            base_cache / "train" / "conditions.npy"
        ),
        "test_conditions": load_string_array(
            base_cache / "test" / "conditions.npy"
        ),
        "train_donors": load_string_array(
            base_cache / "train" / "donors.npy"
        ),
        "test_donors": load_string_array(
            base_cache / "test" / "donors.npy"
        ),
        "train_obs_names": load_string_array(
            base_cache / "train" / "obs_names.npy"
        ),
        "test_obs_names": load_string_array(
            base_cache / "test" / "obs_names.npy"
        ),
    }

    if result["train_labels"].shape != (EXPECTED_TRAIN_CELLS,):
        raise RuntimeError("Unexpected train-label shape.")
    if result["test_labels"].shape != (EXPECTED_TEST_CELLS,):
        raise RuntimeError("Unexpected test-label shape.")

    for prefix, expected in [
        ("train", EXPECTED_TRAIN_CELLS),
        ("test", EXPECTED_TEST_CELLS),
    ]:
        for field in [
            "labels",
            "conditions",
            "donors",
            "obs_names",
        ]:
            key = f"{prefix}_{field}"
            if result[key].shape != (expected,):
                raise RuntimeError(
                    f"Unexpected shape for {key}: {result[key].shape}"
                )

    return result


def read_method_cache_summary(
    method_cache_root: Path,
    quality_seeds: tuple[int, ...],
) -> tuple[
    list[MethodSpec],
    pd.DataFrame,
    int,
    str,
    str,
]:
    summary_path = method_cache_root / "summary.csv"
    require_file(summary_path)

    summary = pd.read_csv(summary_path)

    required_columns = {
        "cache",
        "method",
        "fixed_k",
        "random_seed_train",
        "train_mean_genes",
        "test_mean_genes",
        "selection_compute_time_s",
    }
    missing = required_columns - set(summary.columns)
    if missing:
        raise RuntimeError(
            f"Method-cache summary is missing columns: {sorted(missing)}"
        )

    adaptive_rows = summary[summary["method"] == "adaptive"]
    fixed_rows = summary[summary["method"] == "fixed_tfidf"]
    native_rows = summary[summary["method"] == "native"]

    if len(adaptive_rows) != 1:
        raise RuntimeError("Expected exactly one adaptive cache.")
    if len(fixed_rows) != 1:
        raise RuntimeError("Expected exactly one fixed-TF-IDF cache.")
    if len(native_rows) != 1:
        raise RuntimeError("Expected exactly one native cache.")

    fixed_k = int(round(float(fixed_rows.iloc[0]["fixed_k"])))
    if fixed_k != EXPECTED_FIXED_K:
        raise RuntimeError(
            f"Expected matched fixed K={EXPECTED_FIXED_K}, got {fixed_k}."
        )

    native_directory_name = str(native_rows.iloc[0]["cache"])
    adaptive_directory_name = str(adaptive_rows.iloc[0]["cache"])
    fixed_directory_name = str(fixed_rows.iloc[0]["cache"])

    random_rows = summary[summary["method"] == "random"].copy()
    random_directories: dict[int, Path] = {}

    for seed in quality_seeds:
        candidates = random_rows[
            random_rows["random_seed_train"].round().astype("Int64")
            == int(seed)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one random cache for seed {seed}; "
                f"found {len(candidates)}."
            )
        directory_name = str(candidates.iloc[0]["cache"])
        random_directories[int(seed)] = (
            method_cache_root / directory_name
        )

    native_directory = (
        method_cache_root / native_directory_name
    )
    adaptive_directory = (
        method_cache_root / adaptive_directory_name
    )
    fixed_directory = (
        method_cache_root / fixed_directory_name
    )

    specs = [
        MethodSpec(
            name="geneformer_native_bucketed",
            display_name="Native Bucketed",
            cache_directories={0: native_directory},
            batching="length_sorted",
            quality_seeds=(0,),
            timing_seed=0,
            sampling="native_rank_max2048",
        ),
        MethodSpec(
            name="geneformer_native_sequential",
            display_name="Native Sequential",
            cache_directories={0: native_directory},
            batching="sequential",
            quality_seeds=(0,),
            timing_seed=0,
            sampling="native_rank_max2048",
        ),
        MethodSpec(
            name="matched_random_fixed_k",
            display_name="Matched Random",
            cache_directories=random_directories,
            batching="length_sorted",
            quality_seeds=quality_seeds,
            timing_seed=int(quality_seeds[0]),
            sampling="random_preselected_then_native_rank",
        ),
        MethodSpec(
            name="matched_fixed_tfidf",
            display_name="Fixed TF-IDF",
            cache_directories={0: fixed_directory},
            batching="length_sorted",
            quality_seeds=(0,),
            timing_seed=0,
            sampling="fixed_tfidf_then_native_rank",
        ),
        MethodSpec(
            name="adaptive",
            display_name="AdaGeneBudget",
            cache_directories={0: adaptive_directory},
            batching="length_sorted",
            quality_seeds=(0,),
            timing_seed=0,
            sampling="adaptive_tfidf_mass_then_native_rank",
        ),
    ]

    return (
        specs,
        summary,
        fixed_k,
        adaptive_directory_name,
        fixed_directory_name,
    )


def load_token_cache(
    method_directory: Path,
    split: str,
) -> TokenCache:
    split_directory = method_directory / split

    required = [
        method_directory / "meta.json",
        split_directory / "meta.json",
        split_directory / "indptr.npy",
        split_directory / "token_ids.npy",
        split_directory / "lengths.npy",
    ]
    for path in required:
        require_file(path)

    root_meta = json.loads(
        (method_directory / "meta.json").read_text(encoding="utf-8")
    )
    split_meta = json.loads(
        (split_directory / "meta.json").read_text(encoding="utf-8")
    )
    if root_meta.get("status") != "PASS":
        raise RuntimeError(
            f"Method cache is not PASS: {method_directory}"
        )
    if split_meta.get("status") != "PASS":
        raise RuntimeError(
            f"Split cache is not PASS: {split_directory}"
        )

    indptr = np.load(
        split_directory / "indptr.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    token_ids = np.load(
        split_directory / "token_ids.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    lengths = np.load(
        split_directory / "lengths.npy",
        mmap_mode="r",
        allow_pickle=False,
    )

    expected_cells = (
        EXPECTED_TRAIN_CELLS
        if split == "train"
        else EXPECTED_TEST_CELLS
    )

    if indptr.shape != (expected_cells + 1,):
        raise RuntimeError(
            f"Invalid indptr shape: {split_directory}"
        )
    if lengths.shape != (expected_cells,):
        raise RuntimeError(
            f"Invalid lengths shape: {split_directory}"
        )
    if int(indptr[0]) != 0:
        raise RuntimeError("indptr must start at zero.")
    if int(indptr[-1]) != int(token_ids.size):
        raise RuntimeError("indptr[-1] != token_ids.size.")
    if not np.array_equal(
        np.diff(np.asarray(indptr, dtype=np.int64)),
        np.asarray(lengths, dtype=np.int64),
    ):
        raise RuntimeError("lengths != diff(indptr).")
    if np.any(np.asarray(lengths) <= 0):
        raise RuntimeError("Empty Geneformer sequence detected.")
    if int(np.max(lengths)) > EXPECTED_MAX_POSITION_EMBEDDINGS:
        raise RuntimeError("Sequence exceeds 2048 positions.")
    if int(np.min(token_ids)) < 2:
        raise RuntimeError("Gene sequence contains special/pad token.")
    if int(np.max(token_ids)) >= EXPECTED_VOCAB_SIZE:
        raise RuntimeError("Token ID exceeds model vocabulary.")

    return TokenCache(
        directory=method_directory,
        indptr=indptr,
        token_ids=token_ids,
        lengths=lengths,
        meta=root_meta,
    )


class CachedTokenDataset(Dataset):
    def __init__(self, cache: TokenCache) -> None:
        self.cache = cache
        self.sequence_lengths = np.asarray(
            cache.lengths,
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self.sequence_lengths.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        start = int(self.cache.indptr[index])
        stop = int(self.cache.indptr[index + 1])
        tokens = np.asarray(
            self.cache.token_ids[start:stop],
            dtype=np.int64,
        ).copy()

        return {
            "id": index,
            "input_ids": torch.from_numpy(tokens),
            "length": int(tokens.size),
        }


class FixedOrderSampler(Sampler[int]):
    def __init__(self, order: np.ndarray) -> None:
        self.order = np.asarray(order, dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order.tolist())

    def __len__(self) -> int:
        return int(self.order.size)


class DynamicPadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(
        self,
        examples: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise RuntimeError("Received an empty minibatch.")

        lengths = torch.as_tensor(
            [int(example["length"]) for example in examples],
            dtype=torch.long,
        )
        ids = torch.as_tensor(
            [int(example["id"]) for example in examples],
            dtype=torch.long,
        )
        maximum_length = int(lengths.max().item())

        input_ids = torch.full(
            (len(examples), maximum_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(examples), maximum_length),
            dtype=torch.long,
        )

        for row_index, example in enumerate(examples):
            tokens = example["input_ids"]
            length = int(tokens.numel())
            input_ids[row_index, :length] = tokens
            attention_mask[row_index, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
            "ids": ids,
        }


def make_loader(
    dataset: CachedTokenDataset,
    batch_size: int,
    num_workers: int,
    batching: str,
) -> tuple[DataLoader, float]:
    start_time = perf_counter()
    lengths = np.asarray(
        dataset.sequence_lengths,
        dtype=np.int64,
    )

    if batching == "sequential":
        order = np.arange(len(dataset), dtype=np.int64)
    elif batching == "length_sorted":
        order = np.argsort(
            -lengths,
            kind="stable",
        )
    else:
        raise ValueError(f"Unknown batching mode: {batching}")

    bucketing_time = perf_counter() - start_time

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": FixedOrderSampler(order),
        "collate_fn": DynamicPadCollator(
            EXPECTED_PAD_TOKEN_ID
        ),
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs), float(bucketing_time)


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[BertForMaskedLM, int]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    model = BertForMaskedLM.from_pretrained(
        checkpoint,
    )
    model.config.output_hidden_states = True
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = model.config
    checks = {
        "hidden_size": int(config.hidden_size)
        == EXPECTED_HIDDEN_SIZE,
        "num_hidden_layers": int(config.num_hidden_layers)
        == EXPECTED_NUM_LAYERS,
        "max_position_embeddings": int(
            config.max_position_embeddings
        )
        == EXPECTED_MAX_POSITION_EMBEDDINGS,
        "pad_token_id": int(config.pad_token_id)
        == EXPECTED_PAD_TOKEN_ID,
        "vocab_size": int(config.vocab_size)
        == EXPECTED_VOCAB_SIZE,
        "fp32": all(
            parameter.dtype == torch.float32
            for parameter in model.parameters()
        ),
        "eval": not model.training,
    }
    failed = [
        name for name, passed in checks.items() if not passed
    ]
    if failed:
        raise RuntimeError(
            "Model checks failed: " + ", ".join(failed)
        )

    hidden_state_index = int(
        config.num_hidden_layers - 1
    )
    if hidden_state_index != 5:
        raise RuntimeError(
            f"Expected hidden_states[5], got index "
            f"{hidden_state_index}."
        )

    return model, hidden_state_index


def mean_nonpadding(
    hidden_states: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(
        hidden_states.shape[1],
        device=hidden_states.device,
    ).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    summed = hidden_states.masked_fill(
        ~valid.unsqueeze(-1),
        0.0,
    ).sum(dim=1)
    return summed / lengths.unsqueeze(1).to(
        hidden_states.dtype
    )


@torch.inference_mode()
def model_batch_embeddings(
    model: BertForMaskedLM,
    hidden_state_index: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> np.ndarray:
    """Run one batch in an isolated scope.

    Keeping model outputs local to this function ensures that MLM logits,
    hidden states, and pooled GPU tensors are released before the next
    minibatch forward begins.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    if outputs.hidden_states is None:
        raise RuntimeError(
            "Model did not return hidden states."
        )

    hidden = outputs.hidden_states[
        hidden_state_index
    ]
    pooled = mean_nonpadding(
        hidden,
        lengths,
    )

    return (
        pooled.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def extract_embeddings(
    model: BertForMaskedLM,
    hidden_state_index: int,
    loader: DataLoader,
    dataset_size: int,
    device: torch.device,
    measure: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    embeddings = np.empty(
        (dataset_size, EXPECTED_HIDDEN_SIZE),
        dtype=np.float32,
    )
    seen = np.zeros(dataset_size, dtype=bool)

    actual_tokens = 0
    padded_tokens = 0

    if measure:
        # Match the PBMC timing implementation: clear releasable cached
        # blocks before resetting peak-allocation statistics.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_time = perf_counter()
    else:
        start_time = 0.0

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
            ids = batch["ids"].numpy().astype(
                np.int64,
                copy=False,
            )

            batch_embeddings = model_batch_embeddings(
                model=model,
                hidden_state_index=hidden_state_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths,
            )

            embeddings[ids] = batch_embeddings
            seen[ids] = True

            actual_tokens += int(
                lengths.sum().item()
            )
            padded_tokens += int(
                input_ids.numel()
            )

            # Do not call empty_cache() per batch; that would contaminate
            # timing. Explicitly remove only this batch's remaining refs.
            del (
                batch_embeddings,
                input_ids,
                attention_mask,
                lengths,
            )

    if measure:
        torch.cuda.synchronize(device)
        elapsed = perf_counter() - start_time
        peak_memory_gb = float(
            torch.cuda.max_memory_allocated(device)
            / (1024**3)
        )
    else:
        elapsed = 0.0
        peak_memory_gb = 0.0

    if not np.all(seen):
        missing = np.flatnonzero(~seen)
        raise RuntimeError(
            f"Missing embeddings for {missing.size} cells: "
            f"{missing[:20].tolist()}"
        )

    return embeddings, {
        "embedding_time_s": float(elapsed),
        "peak_gpu_memory_gb": peak_memory_gb,
        "actual_sequence_tokens": float(actual_tokens),
        "padded_tokens_processed": float(padded_tokens),
        "padding_overhead_ratio": float(
            padded_tokens / actual_tokens
        ),
    }


def warmup(
    model: BertForMaskedLM,
    hidden_state_index: int,
    loader: DataLoader,
    device: torch.device,
    steps: int,
) -> None:
    if steps <= 0:
        return

    model.eval()

    with torch.inference_mode():
        for step, batch in enumerate(loader):
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

            batch_embeddings = model_batch_embeddings(
                model=model,
                hidden_state_index=hidden_state_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                lengths=lengths,
            )

            del (
                batch_embeddings,
                input_ids,
                attention_mask,
                lengths,
            )

            if step + 1 >= steps:
                break

    torch.cuda.synchronize(device)


def exact_cosine_knn_predict(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k: int,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    classes, encoded_labels = np.unique(
        reference_labels.astype(str),
        return_inverse=True,
    )

    reference = F.normalize(
        torch.from_numpy(
            reference_embeddings.astype(
                np.float32,
                copy=False,
            )
        ).to(device),
        dim=1,
    )
    reference_y = torch.from_numpy(
        encoded_labels.astype(np.int64)
    ).to(device)

    predictions: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(
            0,
            len(query_embeddings),
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                len(query_embeddings),
            )
            query = F.normalize(
                torch.from_numpy(
                    query_embeddings[start:stop].astype(
                        np.float32,
                        copy=False,
                    )
                ).to(device),
                dim=1,
            )
            similarity = query @ reference.T
            neighbor_indices = similarity.topk(
                k=k,
                dim=1,
            ).indices
            neighbor_labels = reference_y[
                neighbor_indices
            ]
            votes = F.one_hot(
                neighbor_labels,
                num_classes=len(classes),
            ).sum(dim=1)
            predicted = votes.argmax(
                dim=1
            ).cpu().numpy()
            predictions.append(predicted)

            del (
                query,
                similarity,
                neighbor_indices,
                neighbor_labels,
                votes,
            )

    return classes[np.concatenate(predictions)]


def exact_self_neighbors(
    embeddings: np.ndarray,
    k: int,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    normalized = F.normalize(
        torch.from_numpy(
            embeddings.astype(
                np.float32,
                copy=False,
            )
        ).to(device),
        dim=1,
    )

    output = np.empty(
        (len(embeddings), k),
        dtype=np.int32,
    )

    with torch.inference_mode():
        for start in range(
            0,
            len(embeddings),
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                len(embeddings),
            )
            similarity = (
                normalized[start:stop]
                @ normalized.T
            )

            local_indices = torch.arange(
                stop - start,
                device=device,
            )
            global_indices = torch.arange(
                start,
                stop,
                device=device,
            )
            similarity[
                local_indices,
                global_indices,
            ] = -torch.inf

            output[start:stop] = (
                similarity.topk(
                    k=k,
                    dim=1,
                )
                .indices.cpu()
                .numpy()
                .astype(np.int32)
            )
            del similarity

    return output


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
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
        "weighted_f1": float(
            f1_score(
                true_labels,
                predictions,
                average="weighted",
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


def fidelity_metrics(
    reference_embeddings: np.ndarray,
    compressed_embeddings: np.ndarray,
    reference_neighbors: np.ndarray,
    neighbor_k: int,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float]:
    reference_norm = (
        reference_embeddings
        / np.clip(
            np.linalg.norm(
                reference_embeddings,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )
    )
    compressed_norm = (
        compressed_embeddings
        / np.clip(
            np.linalg.norm(
                compressed_embeddings,
                axis=1,
                keepdims=True,
            ),
            1e-12,
            None,
        )
    )
    cosine = np.sum(
        reference_norm * compressed_norm,
        axis=1,
    )

    compressed_neighbors = exact_self_neighbors(
        compressed_embeddings,
        neighbor_k,
        device,
        chunk_size,
    )
    overlap = (
        compressed_neighbors[:, :, None]
        == reference_neighbors[:, None, :]
    ).any(axis=2).sum(axis=1) / neighbor_k

    return {
        "embedding_cosine_mean": float(
            np.mean(cosine)
        ),
        "embedding_cosine_median": float(
            np.median(cosine)
        ),
        "embedding_cosine_min": float(
            np.min(cosine)
        ),
        "neighbor_recall_10": float(
            np.mean(overlap)
        ),
    }


def get_cache_row(
    cache_summary: pd.DataFrame,
    method_directory: Path,
) -> pd.Series:
    matches = cache_summary[
        cache_summary["cache"]
        == method_directory.name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Could not resolve cache summary row for "
            f"{method_directory.name}."
        )
    return matches.iloc[0]


def save_method_results(
    method_directory: Path,
    summary: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> None:
    method_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    stable_json_dump(
        summary,
        method_directory / "summary.json",
    )
    pd.DataFrame(seed_rows).to_csv(
        method_directory / "seed_metrics.csv",
        index=False,
    )
    pd.DataFrame(timing_rows).to_csv(
        method_directory / "timing_repeats.csv",
        index=False,
    )


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative.")
    if args.timing_repeats <= 0:
        raise ValueError("--timing-repeats must be positive.")
    if len(args.quality_seeds) == 0:
        raise ValueError("At least one quality seed is required.")
    if len(set(args.quality_seeds)) != len(args.quality_seeds):
        raise ValueError("Quality seeds must be unique.")
    if args.knn_k <= 0:
        raise ValueError("--knn-k must be positive.")
    if args.neighbor_k <= 0:
        raise ValueError("--neighbor-k must be positive.")
    if args.similarity_chunk <= 0:
        raise ValueError("--similarity-chunk must be positive.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_seeds = tuple(
        int(seed) for seed in args.quality_seeds
    )

    metadata = load_base_metadata(
        args.base_cache
    )
    (
        specs,
        cache_summary,
        fixed_k,
        adaptive_cache_name,
        fixed_cache_name,
    ) = read_method_cache_summary(
        args.method_cache_root,
        quality_seeds,
    )

    train_labels = metadata["train_labels"]
    test_labels = metadata["test_labels"]

    test_obs = pd.DataFrame(
        {
            "obs_name": metadata["test_obs_names"],
            "cell_type": test_labels,
            "condition": metadata["test_conditions"],
            "donor": metadata["test_donors"],
        }
    )
    test_obs.to_csv(
        output_dir / "test_obs.csv",
        index=False,
    )

    print("=" * 112, flush=True)
    print(
        "KANG PATIENT-1015 HELD-OUT GENEFORMER "
        "FIVE-METHOD BENCHMARK",
        flush=True,
    )
    print("=" * 112, flush=True)
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        flush=True,
    )
    print("DEVICE:", device, flush=True)
    print(
        "GPU:",
        torch.cuda.get_device_name(device),
        flush=True,
    )
    print("CHECKPOINT:", args.checkpoint, flush=True)
    print("BASE CACHE:", args.base_cache, flush=True)
    print(
        "METHOD CACHE ROOT:",
        args.method_cache_root,
        flush=True,
    )
    print("OUTPUT:", output_dir, flush=True)
    print(
        "PROTOCOL:",
        f"batch={args.batch_size}, workers={args.num_workers}, "
        f"warmup={args.warmup_steps}, "
        f"timing_repeats={args.timing_repeats}, "
        f"KNN={args.knn_k}, neighbor@{args.neighbor_k}",
        flush=True,
    )

    seed_all(0)
    model, hidden_state_index = load_model(
        args.checkpoint,
        device,
    )

    summaries: dict[str, dict[str, Any]] = {}
    native_reference_embeddings: np.ndarray | None = None
    native_reference_neighbors: np.ndarray | None = None

    metric_names = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "embedding_cosine_median",
        "embedding_cosine_min",
        "neighbor_recall_10",
    ]

    for spec in specs:
        benchmark_method_dir = (
            output_dir / spec.name
        )
        summary_path = (
            benchmark_method_dir / "summary.json"
        )

        if (
            summary_path.is_file()
            and not args.overwrite_methods
        ):
            print(
                f"[resume] {spec.name}",
                flush=True,
            )
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )
            summaries[spec.name] = summary

            if spec.name == "geneformer_native_bucketed":
                artifact = np.load(
                    benchmark_method_dir
                    / "quality_seed0.npz",
                    allow_pickle=False,
                )
                native_reference_embeddings = (
                    artifact["test_embeddings"]
                    .astype(np.float32)
                )
                native_reference_neighbors = np.load(
                    benchmark_method_dir
                    / "native_neighbors_10.npy",
                    allow_pickle=False,
                )
            continue

        print("\n" + "=" * 112, flush=True)
        print(
            f"METHOD: {spec.name} "
            f"({spec.display_name})",
            flush=True,
        )
        print("=" * 112, flush=True)

        seed_rows: list[dict[str, Any]] = []
        metric_store: dict[str, list[float]] = {
            name: [] for name in metric_names
        }

        for quality_seed in spec.quality_seeds:
            seed_all(quality_seed)

            cache_directory = (
                spec.cache_directories[
                    quality_seed
                    if quality_seed
                    in spec.cache_directories
                    else 0
                ]
            )
            train_cache = load_token_cache(
                cache_directory,
                "train",
            )
            test_cache = load_token_cache(
                cache_directory,
                "test",
            )
            train_dataset = CachedTokenDataset(
                train_cache
            )
            test_dataset = CachedTokenDataset(
                test_cache
            )
            train_loader, _ = make_loader(
                train_dataset,
                args.batch_size,
                args.num_workers,
                spec.batching,
            )
            test_loader, _ = make_loader(
                test_dataset,
                args.batch_size,
                args.num_workers,
                spec.batching,
            )

            train_embeddings, _ = extract_embeddings(
                model,
                hidden_state_index,
                train_loader,
                len(train_dataset),
                device,
                measure=False,
            )
            test_embeddings, _ = extract_embeddings(
                model,
                hidden_state_index,
                test_loader,
                len(test_dataset),
                device,
                measure=False,
            )

            predictions = exact_cosine_knn_predict(
                train_embeddings,
                train_labels,
                test_embeddings,
                args.knn_k,
                device,
                args.similarity_chunk,
            )
            metrics = classification_metrics(
                test_labels,
                predictions,
            )

            if spec.name == "geneformer_native_bucketed":
                native_reference_embeddings = (
                    test_embeddings.astype(
                        np.float32,
                        copy=True,
                    )
                )
                native_reference_neighbors = (
                    exact_self_neighbors(
                        native_reference_embeddings,
                        args.neighbor_k,
                        device,
                        args.similarity_chunk,
                    )
                )
                fidelity = {
                    "embedding_cosine_mean": 1.0,
                    "embedding_cosine_median": 1.0,
                    "embedding_cosine_min": 1.0,
                    "neighbor_recall_10": 1.0,
                }
            else:
                if (
                    native_reference_embeddings is None
                    or native_reference_neighbors is None
                ):
                    raise RuntimeError(
                        "Native bucketed reference must run first."
                    )
                fidelity = fidelity_metrics(
                    native_reference_embeddings,
                    test_embeddings,
                    native_reference_neighbors,
                    args.neighbor_k,
                    device,
                    args.similarity_chunk,
                )

            row = {
                "method": spec.name,
                "seed": int(quality_seed),
                "cache": cache_directory.name,
                **metrics,
                **fidelity,
            }
            seed_rows.append(row)
            for metric_name in metric_store:
                metric_store[metric_name].append(
                    float(row[metric_name])
                )

            benchmark_method_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Save quality artifacts for every quality seed.
            np.savez_compressed(
                benchmark_method_dir
                / f"quality_seed{quality_seed}.npz",
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

            prediction_frame = test_obs.copy()
            prediction_frame[
                "prediction"
            ] = predictions

            prediction_frame.to_csv(
                benchmark_method_dir
                / f"predictions_seed{quality_seed}.csv",
                index=False,
            )

            # Native Bucketed has only one deterministic quality seed.
            if (
                spec.name == "geneformer_native_bucketed"
                and quality_seed == spec.quality_seeds[0]
            ):
                np.save(
                    benchmark_method_dir
                    / "native_neighbors_10.npy",
                    native_reference_neighbors,
                    allow_pickle=False,
                )

            print(
                "[quality] "
                f"seed={quality_seed} "
                f"acc={metrics['accuracy']:.6f} "
                f"macro={metrics['macro_f1']:.6f} "
                f"balanced={metrics['balanced_accuracy']:.6f} "
                f"cos={fidelity['embedding_cosine_mean']:.6f} "
                f"neighbor={fidelity['neighbor_recall_10']:.6f}",
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

        timing_cache_directory = (
            spec.cache_directories[
                spec.timing_seed
                if spec.timing_seed
                in spec.cache_directories
                else 0
            ]
        )
        timing_cache_row = get_cache_row(
            cache_summary,
            timing_cache_directory,
        )
        selection_time_s = float(
            timing_cache_row[
                "selection_compute_time_s"
            ]
        )

        train_timing_cache = load_token_cache(
            timing_cache_directory,
            "train",
        )
        test_timing_cache = load_token_cache(
            timing_cache_directory,
            "test",
        )
        train_timing_dataset = CachedTokenDataset(
            train_timing_cache
        )
        test_timing_dataset = CachedTokenDataset(
            test_timing_cache
        )

        (
            train_timing_loader,
            train_bucketing_time,
        ) = make_loader(
            train_timing_dataset,
            args.batch_size,
            args.num_workers,
            spec.batching,
        )
        (
            test_timing_loader,
            test_bucketing_time,
        ) = make_loader(
            test_timing_dataset,
            args.batch_size,
            args.num_workers,
            spec.batching,
        )
        bucketing_time_s = float(
            train_bucketing_time
            + test_bucketing_time
        )

        warmup(
            model,
            hidden_state_index,
            train_timing_loader,
            device,
            args.warmup_steps,
        )
        warmup(
            model,
            hidden_state_index,
            test_timing_loader,
            device,
            args.warmup_steps,
        )

        timing_rows: list[dict[str, Any]] = []
        end_to_end_times: list[float] = []
        embedding_times: list[float] = []
        throughputs: list[float] = []
        embedding_throughputs: list[float] = []
        peak_memories: list[float] = []
        padding_ratios: list[float] = []

        total_cells = (
            len(train_timing_dataset)
            + len(test_timing_dataset)
        )

        for repeat in range(args.timing_repeats):
            train_embeddings, train_timing = (
                extract_embeddings(
                    model,
                    hidden_state_index,
                    train_timing_loader,
                    len(train_timing_dataset),
                    device,
                    measure=True,
                )
            )
            test_embeddings, test_timing = (
                extract_embeddings(
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
            cells_per_s = float(
                total_cells / end_to_end_time_s
            )
            embedding_only_cells_per_s = float(
                total_cells / embedding_time_s
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
            padding_ratio = float(
                padded_tokens / actual_tokens
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
                "method": spec.name,
                "repeat": int(repeat),
                "cache": timing_cache_directory.name,
                "selection_time_s": selection_time_s,
                "bucketing_time_s": bucketing_time_s,
                "embedding_time_s": embedding_time_s,
                "end_to_end_time_s": end_to_end_time_s,
                "cells_per_s": cells_per_s,
                "embedding_only_cells_per_s": (
                    embedding_only_cells_per_s
                ),
                "peak_gpu_memory_gb": peak_memory_gb,
                "padding_overhead_ratio": padding_ratio,
            }
            timing_rows.append(row)
            end_to_end_times.append(
                end_to_end_time_s
            )
            embedding_times.append(
                embedding_time_s
            )
            throughputs.append(cells_per_s)
            embedding_throughputs.append(
                embedding_only_cells_per_s
            )
            peak_memories.append(
                peak_memory_gb
            )
            padding_ratios.append(
                padding_ratio
            )

            print(
                "[timing] "
                f"repeat={repeat + 1} "
                f"e2e={cells_per_s:.2f} cells/s "
                f"embedding-only="
                f"{embedding_only_cells_per_s:.2f} cells/s "
                f"memory={peak_memory_gb:.3f} GB "
                f"padding={padding_ratio:.4f}",
                flush=True,
            )

            del train_embeddings, test_embeddings
            gc.collect()
            torch.cuda.empty_cache()

        train_lengths = np.asarray(
            train_timing_cache.lengths,
            dtype=np.int64,
        )
        test_lengths = np.asarray(
            test_timing_cache.lengths,
            dtype=np.int64,
        )

        adaptive_row = cache_summary[
            cache_summary["cache"]
            == adaptive_cache_name
        ].iloc[0]
        fixed_row = cache_summary[
            cache_summary["cache"]
            == fixed_cache_name
        ].iloc[0]

        summary: dict[str, Any] = {
            "method": spec.name,
            "method_label": spec.display_name,
            "cache": timing_cache_directory.name,
            "tau": (
                EXPECTED_TAU
                if spec.name == "adaptive"
                else None
            ),
            "fixed_k": (
                fixed_k
                if spec.name
                in {
                    "matched_random_fixed_k",
                    "matched_fixed_tfidf",
                }
                else None
            ),
            "batching": spec.batching,
            "sampling": spec.sampling,
            "quality_num_seeds": int(
                len(spec.quality_seeds)
            ),
            "timing_repeats": int(
                args.timing_repeats
            ),
            "selection_in_dataloader": False,
            "selection_time_s": selection_time_s,
            "bucketing_time_s": bucketing_time_s,
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
                np.mean(end_to_end_times)
            ),
            "end_to_end_time_std_s": sample_std(
                end_to_end_times
            ),
            "embedding_time_mean_s": float(
                np.mean(embedding_times)
            ),
            "embedding_time_std_s": sample_std(
                embedding_times
            ),
            "cells_per_s_mean": float(
                np.mean(throughputs)
            ),
            "cells_per_s_std": sample_std(
                throughputs
            ),
            "embedding_only_cells_per_s_mean": float(
                np.mean(embedding_throughputs)
            ),
            "embedding_only_cells_per_s_std": (
                sample_std(
                    embedding_throughputs
                )
            ),
            "peak_gpu_memory_gb": float(
                np.max(peak_memories)
            ),
            "padding_overhead_ratio_mean": float(
                np.mean(padding_ratios)
            ),
            "matched_fixed_k": int(fixed_k),
            "target_adaptive_train_mean": float(
                adaptive_row["train_mean_genes"]
            ),
            "matched_fixed_expected_mean": float(
                fixed_row["train_mean_genes"]
            ),
            "reference_method": (
                "geneformer_native_bucketed"
            ),
        }

        for metric_name, values in metric_store.items():
            summary[metric_name] = float(
                np.mean(values)
            )
            summary[
                f"{metric_name}_std"
            ] = sample_std(values)

        if spec.name == "adaptive":
            for source, destination in [
                (
                    "train_retained_mass_mean",
                    "train_retained_mass_mean",
                ),
                (
                    "test_retained_mass_mean",
                    "test_retained_mass_mean",
                ),
                (
                    "train_upper_clipped_fraction",
                    "train_upper_clipped_fraction",
                ),
                (
                    "test_upper_clipped_fraction",
                    "test_upper_clipped_fraction",
                ),
                (
                    "train_lower_clipped_fraction",
                    "train_lower_clipped_fraction",
                ),
                (
                    "test_lower_clipped_fraction",
                    "test_lower_clipped_fraction",
                ),
            ]:
                summary[destination] = float(
                    timing_cache_row[source]
                )

        save_method_results(
            benchmark_method_dir,
            summary,
            seed_rows,
            timing_rows,
        )
        summaries[spec.name] = summary

        pd.DataFrame(
            list(summaries.values())
        ).to_csv(
            output_dir / "summary_partial.csv",
            index=False,
        )

        del (
            train_timing_loader,
            test_timing_loader,
            train_timing_dataset,
            test_timing_dataset,
        )
        gc.collect()
        torch.cuda.empty_cache()

    reference_summary = summaries[
        "geneformer_native_bucketed"
    ]
    reference_speed = float(
        reference_summary["cells_per_s_mean"]
    )
    reference_memory = float(
        reference_summary["peak_gpu_memory_gb"]
    )

    for summary in summaries.values():
        summary["speedup_vs_native_bucketed"] = float(
            summary["cells_per_s_mean"]
            / reference_speed
        )
        summary[
            "memory_reduction_vs_native_bucketed"
        ] = float(
            1.0
            - summary["peak_gpu_memory_gb"]
            / reference_memory
        )
        stable_json_dump(
            summary,
            output_dir
            / summary["method"]
            / "summary.json",
        )

    final_order = [
        "geneformer_native_sequential",
        "geneformer_native_bucketed",
        "matched_random_fixed_k",
        "matched_fixed_tfidf",
        "adaptive",
    ]
    final_frame = pd.DataFrame(
        [summaries[name] for name in final_order]
    )
    final_frame.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    config = {
        **{
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "checkpoint_model": "Geneformer V1-10M",
        "precision": "FP32",
        "hidden_state_index": int(
            hidden_state_index
        ),
        "hidden_state_semantics": (
            "output of penultimate transformer block"
        ),
        "pooling": (
            "mean over all non-padding gene-token hidden states"
        ),
        "train_cells": EXPECTED_TRAIN_CELLS,
        "test_cells": EXPECTED_TEST_CELLS,
        "fixed_k": int(fixed_k),
        "tau": EXPECTED_TAU,
        "k_min": EXPECTED_K_MIN,
        "k_max": EXPECTED_K_MAX,
        "native_max_length": (
            EXPECTED_MAX_POSITION_EMBEDDINGS
        ),
        "idf_source": "Kang training split only",
        "n_counts_source": (
            "original uncompressed .X row sum before "
            "vocabulary filtering and selection"
        ),
        "quality_reference": (
            "Geneformer Native Max-2048, length-bucketed"
        ),
    }
    stable_json_dump(
        config,
        output_dir / "config.json",
    )

    display_columns = [
        "method",
        "test_mean_selected_genes",
        "cells_per_s_mean",
        "embedding_only_cells_per_s_mean",
        "speedup_vs_native_bucketed",
        "peak_gpu_memory_gb",
        "padding_overhead_ratio_mean",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "embedding_cosine_mean",
        "neighbor_recall_10",
    ]

    print("\n" + "=" * 150, flush=True)
    print("FINAL FIVE-METHOD SUMMARY", flush=True)
    print("=" * 150, flush=True)
    print(
        final_frame[
            display_columns
        ].to_string(index=False),
        flush=True,
    )
    print("SUMMARY:", output_dir / "summary.csv", flush=True)
    print("FINAL STATUS: PASS", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
