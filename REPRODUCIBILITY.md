# Reproducibility

This document describes the hardware, software, model identifiers, datasets, and exact protocol used in *"Where Should LoRA Go? An Exploratory Study of Component-Type Placement for Parameter-Efficient Adaptation of Small Hybrid Language Models."*

The study is **single-seed (3407)**, two-model, and sub-billion-parameter. Results are reported as tested-setting observations. The procedures below are sufficient to regenerate all released artifacts (adapters, per-instance outputs, tables, figures) up to evaluation-subset sampling, which is fixed by the released sample identifiers.

## Hardware

| Platform | GPU | Memory | Provider |
|----------|-----|--------|----------|
| Google Colab Pro | NVIDIA L4 | 24 GB | Google |
| RunPod | NVIDIA RTX 4090 | 24 GB | RunPod |

A single training run takes approximately **15–40 minutes** depending on model and hardware. The full sweep (2 models × 6 placement conditions × 3 training domains, plus baselines and evaluation) is on the order of **~25–30 GPU-hours** on a single 24 GB GPU.

## Base models

| Key | Hugging Face identifier | Revision / commit |
|-----|-------------------------|-------------------|
| `qwen3_5_0_8b_base` | `Qwen/Qwen3.5-0.8B-Base` | run-time default branch revision; pin the exact commit hash at archival |
| `falcon_h1_0_5b_base` | `tiiuae/Falcon-H1-0.5B-Base` | run-time default branch revision; pin the exact commit hash at archival |

> Base model weights are **not** redistributed in this repository; they are obtained from Hugging Face. Pin the exact revision shown above when reproducing.

## Datasets

| Domain | Dataset | Train samples | Notes |
|--------|---------|---------------|-------|
| Mathematics | GSM8K (train split) | 2,000 | |
| Code-instruction | CodeAlpaca | 2,000 | domain-shift adaptation source |
| General instruction | UltraChat | 2,000 | |

Raw datasets are not redistributed unless their licenses permit; see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The fixed evaluation subsets are released as per-instance outputs under `results/eval_details/`.

## LoRA configuration

| Parameter | Value |
|-----------|-------|
| Rank `r` | 16 |
| `alpha` | 32 (scaling factor 2.0) |
| Dropout | 0.05 |

Per-condition target-module sets are recorded in `results/discovery/<model>__condition_specs.json` and `results/discovery/<model>__manifest.json`.

## Training configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 |
| LR schedule | cosine |
| Epochs | 3 |
| Effective batch size | 16 (batch size 4 × gradient accumulation 4) |
| Max sequence length | 1,024 |
| Optimizer | 8-bit Adam |
| Precision | bf16 mixed precision |
| Gradient checkpointing | enabled |
| Random seed | **3407** (single seed) |

## Evaluation protocol

- **Decoding:** greedy (deterministic) for all benchmarks.
- **Benchmarks and sample sizes** (fixed subsets, released with per-instance outputs):

| Benchmark | Sample size | Notes |
|-----------|-------------|-------|
| MMLU | 512 | |
| ARC-Challenge | 299 | |
| HellaSwag | 512 | |
| GSM8K | 256 | **Exception:** Qwen3.5 baseline and GSM8K-trained runs use **128**; CodeAlpaca/UltraChat runs use 256 |
| HumanEval | 164 (157 for Falcon `all_eligible`/CodeAlpaca) | floor-effect negative result; excluded from main comparisons |

Exact per-benchmark sample sizes are in [`stats/sample_size_summary.csv`](stats/sample_size_summary.csv).

## Statistical / bootstrap settings

- **Resamples:** 10,000
- **Interval type:** percentile 95% CI
- **Bootstrap seed:** 3407
- Paired comparisons resample **only over shared `example_id`** between the two compared systems.
- Intervals quantify **evaluation-subset uncertainty for fixed trained models** — they do **not** estimate training-run variance (single seed).

## Single-seed limitation

All results derive from one random seed (3407). Effect sizes are large enough to be practically informative, but multi-seed replication is required to quantify training-run variance and to strengthen statistical claims. The bootstrap analysis addresses evaluation-subset uncertainty only.

## Execution instructions

```bash
git clone https://github.com/hecboar/lora-placement-hybrid.git
cd lora-placement-hybrid
pip install -r requirements.txt        # or: conda env create -f environment.yml

jupyter notebook notebook/paper3_lora_placement.ipynb
```

In the notebook:

1. Set `ROOT_DIR` to your output path.
2. Run the setup cells (environment, model/dataset discovery).
3. Run the experiment driver cell in the order indicated in the final cell. The pipeline is checkpoint-safe: completed experiments auto-skip, so a disconnected runtime can resume by re-running the setup cells.
4. Aggregation cells regenerate the tables in `tables/`, the statistics in `stats/`, and the figures in `figures/`.

> **Note on exact package versions.** The notebook records library versions at runtime via `*.__version__`. Pin the versions observed in your environment when archiving for full bit-level reproducibility (Python 3.10 was used; see `requirements.txt` for the dependency list).
