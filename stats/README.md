# `stats/` — Statistical artifacts

> **Superseded.** These are the original outputs. Two evaluation-harness defects were found after release; the corrected analysis is in [`../reanalysis/`](../reanalysis/) and [`../reanalysis/VERDICT.md`](../reanalysis/VERDICT.md) documents them.

Curated statistical tables that back the manuscript's quantitative claims. All numbers are **single-seed (3407)** and derive from the per-instance outputs in [`../results/eval_details/`](../results/eval_details/). CSV files carry the data; `.tex` files are LaTeX renderings used in the manuscript build.

| File | Contents |
|------|----------|
| `accuracy_with_wilson_ci.csv` | Per (model, condition, training domain, benchmark) accuracy with **Wilson 95% confidence intervals** (`wilson_low`, `wilson_high`) and `n`/`correct` counts. |
| `key_paired_bootstrap_comparisons.csv` | **Curated main-paper** paired bootstrap comparisons (the clean subset of `../tables/paired_bootstrap_comparisons.csv`, with only well-defined comparisons that have shared `example_id`s). Columns include `n_shared`, `mean_diff`, `ci_low`/`ci_high`, and percentage-point variants. |
| `humaneval_floor_summary.csv` | **Artefact, not a result.** Records `accuracy=0.0` for every CodeAlpaca-trained condition. The HumanEval metric never executed, so the resulting `NaN` was written as zero. Real `pass@1` is 0.213–0.341; see `../reanalysis/humaneval_rescore_summary.json`. |
| `sample_size_summary.csv` | Evaluation subset sizes per (model, training domain, benchmark), documenting the **Qwen3.5 GSM8K 128 vs 256** exception. |

## Interpreting the bootstrap comparisons

- 10,000 resamples, percentile 95% intervals, bootstrap seed 3407.
- Resampling is over **shared `example_id`s only** between the two compared systems.
- Intervals reflect **evaluation-subset uncertainty for fixed trained models**, not training-run variance (single seed).
- `condition_b = __base__` denotes the un-fine-tuned base model.

See [`../tables/README.md`](../tables/README.md) for the off-target accuracy-change convention and the full caveat list.
