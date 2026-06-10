# `results/` — Discovery manifests, per-instance outputs, and summary

Raw and discovery-level outputs of the pipeline. All numbers are **single-seed (3407)**.

## `discovery/`

Target-module discovery for each model:

| File | Contents |
|------|----------|
| `<model>__manifest.json` | Full module inventory: per-module class, layer index, layer type, component label, direct parameter counts, LoRA eligibility, and candidate target parameters. Records the topology (e.g. Qwen3.5 layer types, GDN:attention ratio). |
| `<model>__condition_specs.json` | The exact target-module set for each placement condition (`attention_only`, `ssm_only`, `all_layers`, …). This is the specification needed to **regenerate the LoRA adapters**. |

Models: `qwen3_5_0_8b_base` (sequential GDN/attention), `falcon_h1_0_5b_base` (parallel Mamba-2/attention).

## `eval_details/`

Per-instance evaluation outputs (JSONL), one file per **model × condition × benchmark**. Each line contains the fixed `example_id`, the model output, the extracted answer, and the correctness label. File naming:

```
<model>__base__<benchmark>.jsonl                                  # base (un-fine-tuned) model
train__<model>__<condition>__<domain>__seed3407__inline_eval__<benchmark>.jsonl   # fine-tuned
```

These per-instance files are the source from which all aggregate tables and **all bootstrap confidence intervals** are recomputed. Because they include the `example_id`s, paired comparisons can be reproduced exactly (shared-identifier matching).

> HumanEval JSONL files exist only for CodeAlpaca-trained conditions and record the **floor-effect** result (zero solved); see [`../stats/humaneval_floor_summary.csv`](../stats/humaneval_floor_summary.csv).

## `summary/paper3_summary.json`

Machine-readable summary of the whole experiment: preset, models/datasets used, row counts, and the highest-mean-accuracy rows with their parameter/efficiency metadata. Some efficiency/CU fields are `NaN` where undefined (not errors).

## Sample sizes

Evaluation subset sizes are documented in [`../stats/sample_size_summary.csv`](../stats/sample_size_summary.csv). Note the **Qwen3.5 GSM8K 128 vs 256** exception described in [`../tables/README.md`](../tables/README.md).
