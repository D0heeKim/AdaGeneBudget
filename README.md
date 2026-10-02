# AdaGeneBudget

Code release candidate for **AdaGeneBudget: Cell-Adaptive Gene-Token Allocation for Efficient Single-Cell Foundation Models**.

AdaGeneBudget is a training-free gene-token selection method for frozen single-cell foundation models. It combines per-cell expression with reference-derived inverse detection-frequency (IDF) weights, ranks expressed genes by the resulting expression-specificity score, and keeps the shortest prefix whose cumulative score mass reaches a target threshold subject to minimum and maximum token budgets.

Paper defaults:

```text
tau   = 0.90
K_min = 128
K_max = 600
```


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


Before running experiments, check the local layout:

```bash
python scripts/check_setup.py
```
