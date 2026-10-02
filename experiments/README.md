# Paper experiment drivers

The experiment directories contain the reorganized scripts used for the camera-ready scGPT and Geneformer Kang/PBMC results.

They intentionally remain more verbose than the reusable package under `src/adagenebudget/`. Their purpose is **result provenance**, including the original cache formats, RNG conventions, batching, and timing scope.

For normal use of AdaGeneBudget itself, import `adagenebudget` from `src/` instead of importing these experiment files.

For reproduction order, use the shell scripts under `scripts/` and see `PROVENANCE.md`.
