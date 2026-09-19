# Data Availability

This document describes what is and is not redistributed in this repository, and how to reconstruct anything that is not included.

## Raw training/evaluation datasets

Raw datasets (GSM8K, CodeAlpaca, UltraChat, HumanEval, MMLU, ARC-Challenge, HellaSwag) are **not redistributed** unless their licenses explicitly permit it. They are obtained from their original public sources. The notebook and the released sample identifiers are sufficient to **reconstruct the exact subsets** used for training and evaluation.

## Base model weights

Base model weights for `Qwen/Qwen3.5-0.8B-Base` and `tiiuae/Falcon-H1-0.5B-Base` are **not redistributed**. They are downloaded from Hugging Face at the revisions documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Trained LoRA adapters

Trained LoRA adapter weights are included **only where license-compatible**. Where redistribution is not permitted (e.g., due to base-model license terms), the repository instead provides:

- the per-condition **target-module specifications** (`results/discovery/<model>__condition_specs.json`);
- the discovery **manifests** (`results/discovery/<model>__manifest.json`);
- the full **training pipeline** (`notebook/paper3_lora_placement.ipynb`) with the exact hyperparameters,

which together allow the adapters to be **regenerated** deterministically (single seed 3407).

## What IS released

The following derived artifacts are released in full:

- **Per-instance predictions**, extracted answers, and correctness labels for every benchmark × condition × model (`results/eval_details/*.jsonl`). Note that `example_id` in these files is the **position** within that run's fixed subset, not the index of the item in the benchmark; runs that used subsets of different sizes share positions that point at different questions. The position-to-index maps are released in `results/eval_indices/`, and `reanalysis/` uses them so that paired comparisons are over shared questions.
- **Aggregate metrics**: accuracy tables, efficiency/parameter summaries, off-target accuracy-change summaries, paired bootstrap comparisons (`tables/`).
- **Statistical artifacts**: Wilson confidence intervals, key paired bootstrap comparisons and sample-size tables (`stats/`). These are the original outputs and are superseded; see `reanalysis/`.
- **Figures** used in the manuscript (`figures/`, PDF + PNG).
- **Machine-readable experiment summary** (`results/summary/paper3_summary.json`).

## Corrected re-analysis

Two defects in the evaluation harness were found after the first release: GSM8K
answers were extracted as the last `####` match from generations run without a stop
sequence, and the HumanEval metric never executed. `reanalysis/` contains the
corrected per-instance labels, the scripts that derive them from the per-instance
outputs already released here, and the corrected tables. No model was retrained.
`reanalysis/VERDICT.md` documents what changed. Aggregates under `tables/`, `stats/`
and `figures/` are the original outputs and are superseded.

## Archival identifiers

The code and data in this repository are publicly available at
<https://github.com/hecboar/lora-placement-hybrid> and the preprint at
<https://arxiv.org/abs/2604.22127> (cite version 2).
