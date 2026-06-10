# `tables/` — Aggregate result tables

Machine-readable aggregate tables. All numbers are **single-seed (3407)** point estimates from the two tested models. Each table is provided as CSV with a matching LaTeX rendering (`*.tex`) used in the manuscript.

## Files

| File | Main / supp. | Contents |
|------|--------------|----------|
| `all_results_flat.csv` | main | One row per **benchmark × condition × model × training domain** (164 rows). Columns include `accuracy`, `n_eval`, `delta_vs_base`, `delta_vs_full`, `trainable_params`, `pct_trainable`, `efficiency_ratio`, `estimated_cu`, plus relative `detail_path`/`adapter_dir` pointers. **Local Colab/RunPod absolute paths were stripped to relative paths** (see Caveats). |
| `result_matrix_accuracy.csv` | main | Pivoted accuracy: one row per (model, training domain, condition); columns are the five benchmarks (`arc_challenge`, `gsm8k`, `hellaswag`, `humaneval`, `mmlu`). Empty `humaneval` cells = not evaluated; zeros where evaluated reflect the floor effect. |
| `efficiency_table.csv` | main | Per-condition efficiency: `mean_accuracy`, `mean_efficiency_ratio`, `trainable_params`, `pct_trainable`, `estimated_cu`, and off-target deltas. |
| `forgetting_summary.csv` | main | Off-target accuracy change per condition/domain (`non_target_mean_delta_vs_base`) and the legacy `forgetting_score` (see Caveats). |
| `paired_bootstrap_comparisons.csv` | supplementary | Full paired bootstrap output for many condition pairs. **Contains rows with `n_shared=0` and empty CIs** where no shared `example_id`s exist (cross-domain / HumanEval). See Caveats; for the curated main-paper subset use [`../stats/key_paired_bootstrap_comparisons.csv`](../stats/key_paired_bootstrap_comparisons.csv). |
| `practitioner_recipe.csv` | main | Source for the "Setting-specific low-parameter summary" in the README: smallest condition within 95% of the broad/full-LoRA mean accuracy across the four non-code benchmarks, per (model, domain). |
| `<model>__condition_param_summary.csv` | supplementary | Per-condition target-module counts and parameter coverage for each model. |
| `<model>__verification_summary.csv` | supplementary | Sanity check that each condition's expected vs. actual LoRA host modules match (`status=complete`). |

## How to interpret bootstrap CIs

Paired bootstrap CIs are computed over **shared `example_id`s** between the two compared systems: 10,000 resamples, percentile 95% intervals, bootstrap seed 3407. They quantify **evaluation-subset uncertainty for the fixed trained models**, **not** training-run variance (the study is single-seed). `mean_diff` is in accuracy fraction; `*_pp` columns are the same values in percentage points.

## Why HumanEval is a floor-effect result

All released CodeAlpaca-trained conditions solved **zero** HumanEval examples on both models. HumanEval is therefore reported as a **scale-limited negative (floor-effect) result** and is **excluded from the main quantitative comparisons** (see `../stats/humaneval_floor_summary.csv`). HumanEval/`humaneval` columns and any code "recipe" rows must **not** be read as evidence of functional code-generation ability.

## Sample-size caveats

Evaluation subsets are fixed and released, but sample sizes are not identical across all combinations (see `../stats/sample_size_summary.csv`):

- MMLU = 512, HellaSwag = 512, ARC-Challenge = 299.
- **GSM8K = 256, except Qwen3.5** baseline and GSM8K-trained runs, which use **128** (CodeAlpaca/UltraChat use 256). Cross-domain GSM8K comparisons for Qwen mix 128- and 256-example subsets and are therefore **exploratory point estimates**, not fully powered measurements. Paired CIs are computed only over shared identifiers to avoid overstating precision.
- HumanEval = 164, except Falcon `all_eligible`/CodeAlpaca = 157.

## Caveats

- **Local paths removed.** In `all_results_flat.csv`, the `detail_path` and `adapter_dir` columns originally contained machine-specific absolute paths (`/content/drive/MyDrive/...`, `/workspace/...`). These were rewritten to repository-relative paths (`results/eval_details/...`, `models/.../final_adapter`). No numeric values were changed. Adapter directories are pointers only; adapter weights are not redistributed (see [../DATA_AVAILABILITY.md](../DATA_AVAILABILITY.md)).
- **`forgetting_score` sign convention (legacy).** The preferred quantity is `non_target_mean_delta_vs_base` = **off-target accuracy change** = (accuracy_after − accuracy_base): **negative = degradation/forgetting, positive = transfer**. The legacy `forgetting_score` column equals `−non_target_mean_delta_vs_base` (positive = more forgetting) and is retained only for backward compatibility. The manuscript (v4) uses the off-target accuracy-change convention.
- **NaN `efficiency_ratio`/`estimated_cu`** appear where the off-target delta or compute estimate is undefined for that row; they are not errors.
