# AdaGeneBudget

Code release candidate for **AdaGeneBudget: Cell-Adaptive Gene-Token Allocation for Efficient Single-Cell Foundation Models**.

AdaGeneBudget is a training-free gene-token selection method for frozen single-cell foundation models. It combines per-cell expression with reference-derived inverse detection-frequency (IDF) weights, ranks expressed genes by the resulting expression-specificity score, and keeps the shortest prefix whose cumulative score mass reaches a target threshold subject to minimum and maximum token budgets.

Paper defaults:

```text
tau   = 0.90
K_min = 128
K_max = 600
```

This repository currently focuses on the experiments needed for the camera-ready **Table 1 (efficiency)** and **Table 2 (reference-mapping annotation)** for scGPT and Geneformer on Kang and PBMC.

## Repository layout

```text
AdaGeneBudget/
├── src/adagenebudget/          # reusable, backbone-independent method code
│   ├── core.py
│   ├── idf.py
│   ├── scoring.py
│   ├── budget.py
│   ├── selection.py
│   ├── baselines.py
│   └── backends/
│       ├── scgpt.py
│       └── geneformer.py
├── experiments/
│   ├── scgpt/
│   │   ├── _shared/            # validated scGPT experiment utilities
│   │   ├── kang/
│   │   └── pbmc/
│   └── geneformer/
│       ├── kang/
│       └── pbmc/
├── scripts/                    # reproduction order / setup checks
├── data/README.md              # expected local data layout
├── checkpoints/README.md       # expected checkpoint layout
├── third_party/README.md       # external scGPT/Geneformer repositories
├── results/                    # camera-ready Table 1/2 reference values
├── tests/                      # lightweight method-level tests
└── PROVENANCE.md               # mapping from original research scripts
```

Large datasets, model checkpoints, generated caches, and upstream scGPT/Geneformer source trees are intentionally **not committed**.

## Core method

Install the reusable package from the repository root:

```bash
pip install -e .
pytest -q
```

Minimal use:

```python
from adagenebudget import AdaGeneBudget

selector = AdaGeneBudget(tau=0.90, k_min=128, k_max=600)
selector.fit(reference_matrix)

selection = selector.select_row(
    gene_positions,
    expression_values,
)
selected_local_indices = selection.local_indices
```

`AdaGeneBudget` selects a **gene set**. The final sequence order is then restored according to the target backbone:

- **scGPT:** selected genes are restored to the native/input gene order.
- **Geneformer:** selected genes are reordered using the Geneformer native median-scaled expression rank.

Helpers for these steps are in `src/adagenebudget/backends/`.

### Core modules

- `idf.py`: reference detection counts and smoothed IDF
- `scoring.py`: expression-specificity score and deterministic ranking
- `budget.py`: score-mass cutoff and matched fixed-budget utility
- `selection.py`: AdaGeneBudget and Fixed TF-IDF selection
- `baselines.py`: Top-Expression and generic random selection helpers
- `backends/`: backbone-specific post-selection ordering

## External repositories and checkpoints

The official scGPT and Geneformer repositories and pretrained checkpoints are not redistributed here. See:

- `third_party/README.md`
- `checkpoints/README.md`
- `data/README.md`

A convenient local layout is:

```text
AdaGeneBudget/
├── scGPT -> /path/to/scGPT
├── Geneformer -> /path/to/Geneformer
├── checkpoints/
└── data/
```

For example:

```bash
ln -s /path/to/scGPT ./scGPT
ln -s /path/to/Geneformer ./Geneformer
```

The root `.gitignore` excludes these local clones/symlinks, model weights, raw datasets, generated caches, and experiment outputs.

Before running experiments, check the local layout:

```bash
python scripts/check_setup.py
```

## Table 1 / Table 2 reproduction

Run commands from the repository root. Detailed per-backbone notes are in each experiment directory.

```bash
bash scripts/reproduce_scgpt_kang.sh
bash scripts/reproduce_scgpt_pbmc.sh
bash scripts/reproduce_geneformer_kang.sh
bash scripts/reproduce_geneformer_pbmc.sh
```

To run all four sequentially:

```bash
bash scripts/reproduce_tables_1_2.sh
```

The camera-ready reference values are provided in:

```text
results/paper_table1.csv
results/paper_table2.csv
```

## Why the paper drivers still contain some duplicated selection logic

The reusable implementation in `src/adagenebudget/` extracts the final method semantics into a clean package. The paper reproduction drivers, however, are intentionally kept close to the validated scripts that produced the reported results.

A full rewrite of every timing/cache driver to call the package directly could silently change RNG streams, boundary-tie behavior, cache order, batching, or timing scope. Therefore this release uses two layers:

1. **Clean reusable method implementation:** `src/adagenebudget/`
2. **Provenance-preserving paper drivers:** `experiments/`

Once the reorganized repository has reproduced the published results end-to-end on the original data/checkpoints, the drivers can be migrated further to the shared package under explicit equivalence tests.

See `PROVENANCE.md` for the original-to-release mapping.

## Current release-candidate checks

The repository can be syntax-compiled without the external model repositories, and the core unit tests exercise the IDF formula, adaptive cutoff/bounds, matched fixed budget, scGPT native-order restoration, and Geneformer native reordering.

Full end-to-end reproduction still requires the original datasets, processed split artifacts, pretrained checkpoints, external model repositories, and CUDA environment described in the experiment scripts.
