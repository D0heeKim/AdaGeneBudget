#!/usr/bin/env bash
set -euo pipefail
python -u experiments/geneformer/kang/build_base_cache.py
python -u experiments/geneformer/kang/build_method_caches.py
python -u experiments/geneformer/kang/build_expression_ablation_caches.py --overwrite-existing
python -u experiments/geneformer/kang/run_main.py
python -u experiments/geneformer/kang/run_expression_ablation.py --overwrite
python -u experiments/geneformer/kang/add_native_rank.py
python -u experiments/geneformer/kang/run_final_timing.py
python -u experiments/geneformer/kang/run_top_expression_timing.py
