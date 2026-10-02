# Data

Raw and processed datasets are intentionally not distributed in this repository.

Expected paths used by the reproduction scripts:

```text
data/
├── raw/
│   └── pbmc_seurat_v4/
│       └── pbmc_seurat_v4_rna_only.h5ad
└── processed/
    ├── kang_2018/
    │   └── patient_1015_holdout/
    │       ├── train.h5ad
    │       └── test.h5ad
    └── pbmc_seurat_v4/
        └── p5_holdout/
            └── split_indices.npz
```

The paper uses donor 1015 as the held-out Kang query split and donor P5 as the held-out PBMC query split. Add dataset download/preprocessing instructions here before public release if redistribution is not permitted.
