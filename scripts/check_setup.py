#!/usr/bin/env python3
"""Check the local, non-versioned resources expected by the paper drivers."""
from pathlib import Path

REQUIRED = [
    (Path("scGPT"), "dir/symlink", "official scGPT source tree"),
    (Path("Geneformer"), "dir/symlink", "official Geneformer source tree"),
    (Path("checkpoints/scgpt/args.json"), "file", "scGPT checkpoint args"),
    (Path("checkpoints/scgpt/vocab.json"), "file", "scGPT vocabulary"),
    (Path("checkpoints/scgpt/best_model.pt"), "file", "scGPT whole-human weights"),
    (Path("checkpoints/Geneformer-V1-10M"), "dir", "Geneformer V1-10M checkpoint"),
    (Path("data/processed/kang_2018/patient_1015_holdout/train.h5ad"), "file", "Kang reference split"),
    (Path("data/processed/kang_2018/patient_1015_holdout/test.h5ad"), "file", "Kang held-out donor 1015 split"),
    (Path("data/raw/pbmc_seurat_v4/pbmc_seurat_v4_rna_only.h5ad"), "file", "PBMC RNA source"),
    (Path("data/processed/pbmc_seurat_v4/p5_holdout/split_indices.npz"), "file", "PBMC P5 split indices"),
]


def ok(path: Path, kind: str) -> bool:
    if kind == "file":
        return path.is_file()
    if kind == "dir":
        return path.is_dir()
    return path.exists() or path.is_symlink()


def main() -> int:
    missing = 0
    print("AdaGeneBudget local setup check\n")
    for path, kind, description in REQUIRED:
        exists = ok(path, kind)
        print(f"[{'OK' if exists else 'MISSING'}] {path}  — {description}")
        missing += int(not exists)
    print(f"\n{len(REQUIRED)-missing}/{len(REQUIRED)} required paths available.")
    if missing:
        print("See data/README.md, checkpoints/README.md, and third_party/README.md.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
