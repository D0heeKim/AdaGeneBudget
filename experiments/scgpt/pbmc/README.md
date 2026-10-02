# scGPT — PBMC

Paper split: donor `P5` held out as query.

The final PBMC result was assembled from the common benchmark artifacts plus the final AdaGeneBudget ordering correction/validation and timing recheck. The public-facing canonical Ada script is named `run_adagenebudget.py`.

Recommended provenance-preserving order:

```bash
python -u experiments/scgpt/pbmc/run_reference_pipeline.py
python -u experiments/scgpt/pbmc/run_adagenebudget.py
python -u experiments/scgpt/pbmc/run_top_expression.py
python -u experiments/scgpt/pbmc/run_final_timing.py
python -u experiments/scgpt/pbmc/run_top_expression_timing.py
```

`run_all_methods.py` is retained as a final-semantics cross-check driver but is not required by the provenance-preserving sequence above.
