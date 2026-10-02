#!/usr/bin/env bash
set -euo pipefail
bash scripts/reproduce_scgpt_kang.sh
bash scripts/reproduce_scgpt_pbmc.sh
bash scripts/reproduce_geneformer_kang.sh
bash scripts/reproduce_geneformer_pbmc.sh
