#!/usr/bin/env bash
set -euo pipefail
python -u experiments/scgpt/kang/run_main.py
python -u experiments/scgpt/kang/run_native_bucketed.py
python -u experiments/scgpt/kang/run_matched_random.py
python -u experiments/scgpt/kang/run_top_expression.py
python -u experiments/scgpt/kang/run_top_expression_timing.py
