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

- **Per-instance predictions**, extracted answers, and correctness labels for every benchmark × condition × model (`results/eval_details/*.jsonl`), including the fixed evaluation `example_id`s.
- **Aggregate metrics**: accuracy tables, efficiency/parameter summaries, off-target accuracy-change summaries, paired bootstrap comparisons (`tables/`).
- **Statistical artifacts**: Wilson confidence intervals, key paired bootstrap comparisons, the HumanEval floor-effect summary, and sample-size tables (`stats/`).
- **Figures** used in the manuscript (`figures/`, PDF + PNG).
- **Machine-readable experiment summary** (`results/summary/paper3_summary.json`).

## Archival identifiers

The following placeholders **must be filled before submission**:

```
Zenodo DOI: [to be added before submission]
Git commit: [to be added before submission]
```

Once the Zenodo archive is created, the DOI and the exact Git commit hash will be inserted here, in `README.md`, in `CITATION.cff`, and in the manuscript.
