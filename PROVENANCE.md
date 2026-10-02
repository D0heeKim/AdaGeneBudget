# Experiment provenance

This file records how the original research scripts supplied for the camera-ready experiments were reorganized for the public release candidate.

The goal is to remove chronological names such as `04e`, `11b`, `36`, and `37` from the public workflow while retaining the exact validated implementations behind the reported results.

## scGPT — Kang

| Original research script | Release location | Role |
|---|---|---|
| `11_scgpt_kang_main.py` | `experiments/scgpt/kang/run_main.py` | Main Full / native / Fixed TF-IDF / AdaGeneBudget experiment |
| `11b_scgpt_kang_native_bucketed.py` | `experiments/scgpt/kang/run_native_bucketed.py` | Native Bucketed comparator |
| `11c_scgpt_kang_matched_random.py` | `experiments/scgpt/kang/run_matched_random.py` | Matched Random comparator |
| `01a_scgpt_kang_top_expression.py` | `experiments/scgpt/kang/run_top_expression.py` | Top-Expression annotation result |
| `01f_scgpt_kang_top_expression_timing.py` | `experiments/scgpt/kang/run_top_expression_timing.py` | Top-Expression Table-1 timing |

## scGPT — PBMC

| Original research script | Release location | Role |
|---|---|---|
| `16_scgpt_pbmc_p5_all_methods.py` | `experiments/scgpt/pbmc/run_reference_pipeline.py` | Original common PBMC benchmark artifacts / comparators |
| `04e_scgpt_pbmc_corrected_ada_validation.py` | `experiments/scgpt/pbmc/run_adagenebudget.py` | Canonical final AdaGeneBudget PBMC selection/quality semantics |
| `29_scgpt_pbmc_p5_test_only_timing.py` | `experiments/scgpt/pbmc/run_final_timing.py` | Final timing recheck |
| `01b_scgpt_pbmc_top_expression.py` | `experiments/scgpt/pbmc/run_top_expression.py` | Top-Expression annotation result |
| `01g_scgpt_pbmc_top_expression_timing_train_plus_test.py` | `experiments/scgpt/pbmc/run_top_expression_timing.py` | Top-Expression Table-1 timing |
| `16_scgpt_pbmc_p5_all_methods_corrected.py` | `experiments/scgpt/pbmc/run_all_methods.py` | Final-semantics cross-check driver |

For scGPT, score-based selection determines the retained set; the final selected subset is restored to the native/input gene order before model input construction.

## scGPT shared implementation dependencies

The original Kang/PBMC drivers imported reusable classes/functions from earlier scGPT experiment modules. Those validated helpers are retained under `experiments/scgpt/_shared/` with personal absolute paths removed:

- `full_input.py`
- `fixed_selection.py`
- `adaptive_selection.py`
- `runtime_utils.py`

They are implementation dependencies, not additional datasets claimed in the camera-ready evaluation.

## Geneformer — Kang

| Original research script | Release location | Role |
|---|---|---|
| `04_build_geneformer_kang_base_cache.py` | `experiments/geneformer/kang/build_base_cache.py` | Base Geneformer-compatible cache |
| `05_build_geneformer_kang_method_caches.py` | `experiments/geneformer/kang/build_method_caches.py` | Native / Random / Fixed TF-IDF / AdaGeneBudget caches |
| `05b_build_geneformer_kang_expression_ablation_caches.py` | `experiments/geneformer/kang/build_expression_ablation_caches.py` | Fixed Expression (= Top-Expression) and Adaptive Expression caches |
| `07_run_geneformer_kang_benchmark.py` | `experiments/geneformer/kang/run_main.py` | Main annotation benchmark |
| `06b_run_geneformer_kang_expression_ablation.py` | `experiments/geneformer/kang/run_expression_ablation.py` | Top-Expression / expression-allocation ablation quality |
| `25_add_geneformer_kang_native_rank_topk.py` | `experiments/geneformer/kang/add_native_rank.py` | Native Rank comparator |
| `37_recheck_geneformer_kang_train_test_b64_r5_scopefix.py` | `experiments/geneformer/kang/run_final_timing.py` | Final timing for main methods |
| `01h_geneformer_kang_top_expression_timing.py` | `experiments/geneformer/kang/run_top_expression_timing.py` | Top-Expression final timing |
| `08_analyze_geneformer_kang_per_class.py` | `experiments/geneformer/kang/analyze_per_class.py` | Per-class analysis used beyond Tables 1/2 |

The `fixed_expression` row in the original expression-ablation code is the paper's **Top-Expression** policy: fixed-K raw-expression selection followed by Geneformer's native post-selection ordering.

## Geneformer — PBMC

| Original research script | Release location | Role |
|---|---|---|
| `14_audit_geneformer_pbmc.py` | `experiments/geneformer/pbmc/audit_inputs.py` | PBMC/Geneformer mapping audit |
| `15_build_geneformer_pbmc_base_cache.py` | `experiments/geneformer/pbmc/build_base_cache.py` | Base cache |
| `16_build_geneformer_pbmc_native_adaptive_cache.py` | `experiments/geneformer/pbmc/build_native_adaptive_cache.py` | Native and AdaGeneBudget caches |
| `17_build_geneformer_pbmc_fixed_random_cache.py` | `experiments/geneformer/pbmc/build_fixed_random_cache.py` | Fixed TF-IDF and Matched Random caches |
| `01c_geneformer_pbmc_build_top_expression_cache.py` | `experiments/geneformer/pbmc/build_top_expression_cache.py` | Top-Expression cache |
| `19_run_geneformer_pbmc_main.py` | `experiments/geneformer/pbmc/run_main.py` | Main annotation benchmark |
| `01d_geneformer_pbmc_top_expression_eval.py` | `experiments/geneformer/pbmc/run_top_expression.py` | Top-Expression annotation result |
| `22_add_geneformer_pbmc_native_rank_topk.py` | `experiments/geneformer/pbmc/add_native_rank.py` | Native Rank comparator |
| `36_recheck_geneformer_pbmc_train_test_b64_r5.py` | `experiments/geneformer/pbmc/run_final_timing.py` | Final timing for main methods |
| `01i_geneformer_pbmc_top_expression_timing.py` | `experiments/geneformer/pbmc/run_top_expression_timing.py` | Top-Expression final timing |

For Geneformer, gene-selection score determines the retained set and the surviving genes are then reordered according to the validated Geneformer native median-scaled expression ranking.

## Core extraction

The repeated method logic was additionally extracted to `src/adagenebudget/`:

- reference detection frequency / IDF
- expression-specificity scoring
- deterministic ranking
- adaptive cumulative-mass cutoff with `tau`, `K_min`, and `K_max`
- matched fixed-budget search
- Fixed TF-IDF and Top-Expression helpers
- scGPT and Geneformer post-selection ordering helpers

The exact paper drivers remain close to the validated originals until end-to-end equivalence is confirmed on the original compute environment.
