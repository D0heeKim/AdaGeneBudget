# scGPT — Kang

Paper split: donor `1015` held out as query.

Recommended reproduction order:

```bash
python -u experiments/scgpt/kang/run_main.py
python -u experiments/scgpt/kang/run_native_bucketed.py
python -u experiments/scgpt/kang/run_matched_random.py
python -u experiments/scgpt/kang/run_top_expression.py
python -u experiments/scgpt/kang/run_top_expression_timing.py
```

`run_main.py` contains the validated AdaGeneBudget and Fixed TF-IDF path; the additional files supply the final Native Bucketed, Matched Random, and Top-Expression results used for Tables 1/2.
