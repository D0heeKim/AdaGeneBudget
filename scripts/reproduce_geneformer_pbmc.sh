#!/usr/bin/env bash
set -euo pipefail
python -u experiments/geneformer/pbmc/audit_inputs.py
python -u experiments/geneformer/pbmc/build_base_cache.py
python -u experiments/geneformer/pbmc/build_native_adaptive_cache.py
python -u experiments/geneformer/pbmc/build_fixed_random_cache.py
python -u experiments/geneformer/pbmc/build_top_expression_cache.py
python -u experiments/geneformer/pbmc/run_main.py
python -u experiments/geneformer/pbmc/run_top_expression.py
python -u experiments/geneformer/pbmc/add_native_rank.py
python -u experiments/geneformer/pbmc/run_final_timing.py
python -u experiments/geneformer/pbmc/run_top_expression_timing.py
