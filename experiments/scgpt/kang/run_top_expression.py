#!/usr/bin/env python3
"""Run the fixed-budget Top-Expression baseline for scGPT-Kang.

This is a thin, provenance-preserving wrapper around the already validated
scGPT fixed-budget implementation (`05_scgpt_pancreas_fixed_grid.py`). It reads
the matched fixed K from the canonical Kang run instead of hard-coding it.

Run from the AdaGeneBudget_release repository root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main-output", type=Path,
                   default=Path("outputs/scgpt/kang_patient1015/main_csr"))
    p.add_argument("--fixed-script", type=Path,
                   default=Path("experiments/scgpt/_shared/fixed_selection.py"))
    p.add_argument("--repo", type=Path, default=Path("scGPT"))
    p.add_argument("--model-dir", type=Path, default=Path("checkpoints/scgpt"))
    p.add_argument("--train", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/train.h5ad"))
    p.add_argument("--test", type=Path,
                   default=Path("data/processed/kang_2018/patient_1015_holdout/test.h5ad"))
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/top_expression/scgpt_kang"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--timing-repeats", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.main_output / "config.json"
    full_reference = args.main_output / "full_bucketed.npz"
    for path in (config_path, full_reference, args.fixed_script, args.repo,
                 args.model_dir, args.train, args.test):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "matched_fixed_k" in config:
        fixed_k = int(config["matched_fixed_k"])
    elif "fixed_k" in config:
        fixed_k = int(config["fixed_k"])
    else:
        raise KeyError(f"No matched fixed K in {config_path}")

    print(f"Resolved scGPT-Kang matched fixed K = {fixed_k}", flush=True)
    cmd = [
        sys.executable, "-u", str(args.fixed_script),
        "--repo", str(args.repo),
        "--model-dir", str(args.model_dir),
        "--train", str(args.train),
        "--test", str(args.test),
        "--full-reference", str(full_reference),
        "--label-col", args.label_col,
        "--methods", "expression",
        "--budgets", str(fixed_k),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--timing-repeats", str(args.timing_repeats),
        "--output-dir", str(args.output_dir),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    print("COMMAND:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
