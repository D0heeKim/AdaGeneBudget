#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scgpt/pbmc/run_reference_pipeline.py
python -u experiments/scgpt/pbmc/run_adagenebudget.py
python -u experiments/scgpt/pbmc/run_top_expression.py
python -u experiments/scgpt/pbmc/run_final_timing.py
python -u experiments/scgpt/pbmc/run_top_expression_timing.py
