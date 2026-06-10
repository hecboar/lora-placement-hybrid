# Where Should LoRA Go? An Exploratory Study of Component-Type Placement for Parameter-Efficient Adaptation of Small Hybrid Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: under review](https://img.shields.io/badge/status-under%20review%20(PeerJ%20CS)-blue.svg)](#citation)

> **TL;DR** — Across two small representative hybrid LMs, attention-only LoRA is the strongest low-parameter default candidate among the tested placement conditions. Recurrent-backbone adaptation behaves differently in the tested sequential and parallel architectures, and broader multi-seed/larger-scale validation remains necessary.

This repository accompanies an **exploratory, controlled study** of *where* LoRA should be placed in hybrid language models that combine softmax attention with a recurrent sequence-mixing component (state-space models or gated linear attention). The study is **single-seed (3407), two-model, and sub-billion-parameter**; findings are reported as tested-setting observations, not universal laws.

## Key Findings

All numbers below are single-seed point estimates from the two tested models. Wording is intentionally cautious; see [Statistical Notes](#statistical-notes) and the manuscript for full context.

| # | Finding (tested settings) | Qwen3.5-0.8B (sequential GDN/attn) | Falcon-H1-0.5B (parallel Mamba-2/attn) |
|---|---------------------------|------------------------------------|----------------------------------------|
| 1 | **Attention-only is the strongest low-parameter target candidate** | +10.2 pp on GSM8K with 1.08M params (0.14%) | +17.2 pp on GSM8K with 2.21M params (0.42%) |
| 2 | **Recurrent-backbone adaptation behaves differently across the tested topologies** | GDN-only **substantially degrades GSM8K under this setup** (−14.8 pp) | SSM-only **remains constructive on GSM8K** (+8.6 pp) |
| 3 | **Different off-target transfer/degradation patterns are observed** | Up to −16.0 pp off-target degradation after UltraChat | Up to +10.9 pp positive transfer after UltraChat |
| 4 | **Attention-only shows comparatively smaller off-target degradation in several settings** | Smallest off-target degradation across domains | Near-zero off-target change on cross-domain eval |
| 5 | **Higher accuracy-per-parameter in the tested settings** | 9.4 pp/M vs 0.7 pp/M (`all_layers`) | 7.8 pp/M vs 1.1 pp/M (`all_eligible`) |

> These observations suggest that **component type and hybrid topology should be considered when selecting LoRA targets**. They are not universal rules: topology is confounded with model family, recurrent mechanism, and pre-training in this two-model design.

## Repository Structure

```
├── README.md                          ← You are here
├── LICENSE                            ← MIT License
├── REPRODUCIBILITY.md                 ← Hardware, configs, exact protocol
├── DATA_AVAILABILITY.md               ← What is / is not redistributed
├── CITATION.cff                       ← How to cite this work
├── ARCHIVE_CHECKLIST.md               ← Pre-submission / Zenodo checklist
├── requirements.txt                   ← Python dependencies
├── environment.yml                    ← Conda environment
├── notebook/
│   └── paper3_lora_placement.ipynb    ← Full reproducible pipeline
├── figures/                           ← Publication figures (PDF + PNG, both models)
├── results/
│   ├── discovery/                     ← Target-module manifests + condition specs
│   ├── eval_details/                  ← Per-instance evaluation outputs (JSONL)
│   └── summary/paper3_summary.json    ← Machine-readable experiment summary
├── tables/                            ← Aggregate result tables (CSV) + README
└── stats/                             ← Wilson CIs, key bootstrap, HumanEval floor, sample sizes
```

See [`tables/README.md`](tables/README.md) and [`results/README.md`](results/README.md) for a description of every file and how to interpret it.

## Quick Start

```bash
# Clone
git clone https://github.com/hecboar/lora-placement-hybrid.git
cd lora-placement-hybrid

# Inspect results only (no GPU needed)
python -c "import pandas as pd; print(pd.read_csv('tables/all_results_flat.csv').head())"

# Reproduce from scratch — GPU machine (≥24GB VRAM)
pip install -r requirements.txt
jupyter notebook notebook/paper3_lora_placement.ipynb
```

In the notebook, set `ROOT_DIR` to your preferred output path and follow the execution order in the final cell. The pipeline is checkpoint-safe — if your runtime disconnects, re-run the setup cells and resume; completed experiments auto-skip.

**Inspect results only** — load the main table directly:

```python
import pandas as pd

df = pd.read_csv("tables/all_results_flat.csv")
# Filter to GSM8K-trained Falcon, compare attention-only vs all_eligible
falcon_gsm = df[(df.model_key == "falcon_h1_0_5b_base") & (df.train_dataset == "gsm8k_train")]
print(falcon_gsm[["condition", "benchmark", "accuracy", "delta_vs_base"]].to_string())
```

## Models and Conditions

| Model | Topology | Conditions | Params range |
|-------|----------|------------|-------------|
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | Sequential (18 GDN + 6 softmax attn, 3:1) | `all_layers`, `softmax_only`, `gdn_only`, `mlp_only`, `softmax_plus_mlp`, `gdn_plus_mlp` | 1.08M – 10.82M |
| [Falcon-H1-0.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) | Parallel (attn ∥ Mamba-2 per block) | `all_eligible`, `attention_only`, `ssm_only`, `mlp_only`, `attention_plus_mlp`, `ssm_plus_mlp` | 2.21M – 11.47M |

All conditions use: LoRA rank=16, α=32, dropout=0.05, lr=2e-4 (cosine), 3 epochs, effective batch=16, seq_len=1024, 8-bit Adam, bf16, gradient checkpointing. Full protocol in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Training Domains

Each model is fine-tuned independently on three adaptation domains (2,000 examples each) and evaluated on a fixed benchmark suite.

| Domain | Dataset | Train samples | Role in this study |
|--------|---------|---------------|--------------------|
| Mathematics | GSM8K (train split) | 2,000 | Mathematics adaptation; primary eval on GSM8K |
| Code-instruction | CodeAlpaca | 2,000 | Code-instruction / **domain-shift adaptation source** (not evidence of functional code-generation improvement) |
| General instruction | UltraChat | 2,000 | General instruction adaptation; eval on MMLU, ARC-C, HellaSwag |
| *(probe)* | HumanEval | — | **Floor-effect negative result** at sub-1B scale; excluded from main quantitative comparisons (see below) |

## Setting-specific low-parameter summary

> **Caption.** Smallest-parameter placement condition achieving ≥95% of the corresponding broad/full-LoRA mean accuracy across the four primary non-code benchmarks (MMLU, ARC-C, HellaSwag, GSM8K) after each training domain. **This is not a universal recipe**, and the CodeAlpaca row should **not** be interpreted as evidence of functional code-generation improvement (it reflects a domain-shift adaptation source; see [HumanEval Floor Effect](#humaneval-floor-effect)).

| Model | Training domain | Smallest condition within 95% | Trainable params |
|-------|-----------------|-------------------------------|------------------|
| Falcon-H1-0.5B | GSM8K | `attention_only` | 2.21M |
| Falcon-H1-0.5B | UltraChat | `attention_only` | 2.21M |
| Falcon-H1-0.5B | CodeAlpaca† | `attention_only` | 2.21M |
| Qwen3.5-0.8B | GSM8K | `softmax_only` | 1.08M |
| Qwen3.5-0.8B | UltraChat | `softmax_only` | 1.08M |
| Qwen3.5-0.8B | CodeAlpaca† | `all_layers` | 10.82M |

†CodeAlpaca rows describe non-code benchmark retention under a code-instruction domain shift, **not** code-generation ability. Source data: [`tables/practitioner_recipe.csv`](tables/practitioner_recipe.csv).

## Statistical Notes

- **Single seed:** all training runs use random seed **3407**. The study does **not** estimate training-run variance; multi-seed replication is future work.
- **Paired bootstrap CIs** are computed over **matched `example_id`** between the two compared systems (shared evaluation instances only).
- **10,000 bootstrap resamples**, **percentile 95%** intervals, **bootstrap seed 3407**.
- These intervals quantify **evaluation-subset uncertainty for fixed trained models** — they are *not* training-run variance.
- Evaluation subsets are fixed and released with per-instance outputs. Sample sizes vary by benchmark; the main exception is **GSM8K for Qwen3.5**, where the baseline and GSM8K-trained runs use **128** examples while CodeAlpaca/UltraChat runs use **256** (see [`stats/sample_size_summary.csv`](stats/sample_size_summary.csv)). Cross-domain comparisons with differing sample sizes are interpreted as exploratory.

## HumanEval Floor Effect

All released CodeAlpaca-trained conditions solved **zero** HumanEval examples on both models. HumanEval is therefore reported as a **scale-limited negative (floor-effect) result** and is **excluded from the main quantitative comparisons**. It does not support any code-generation claim. See [`stats/humaneval_floor_summary.csv`](stats/humaneval_floor_summary.csv).

## Reproducibility

This repository includes:

- the full notebook/pipeline (`notebook/paper3_lora_placement.ipynb`);
- target-module discovery manifests (`results/discovery/*_manifest.json`);
- condition specifications (`results/discovery/*_condition_specs.json`);
- fixed evaluation sample identifiers (embedded in the per-instance JSONL outputs);
- per-instance evaluation outputs (`results/eval_details/*.jsonl`);
- aggregate tables (`tables/`) and statistical artifacts (`stats/`);
- publication figures (`figures/`);
- scripts/configs to regenerate trained LoRA adapters (the adapter weights themselves are not redistributed — see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md));
- environment files (`requirements.txt`, `environment.yml`).

Full hardware, hyperparameters, and the exact protocol are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```
Zenodo DOI: [to be added before submission]
Git commit: [to be added before submission]
```

## Related Papers

This work is the third in a series studying hybrid language model internals:

1. **Paper 1** — *How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models*
2. **Paper 2** — *Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures*
3. **Paper 3** — This work: *Where Should LoRA Go?*

Paper 2 reported that the recurrent backbone acts as a "functional backbone" while attention behaves as "refinement." This study explores whether that hierarchy is reflected in adaptation: in the tested settings, the strongest low-parameter LoRA target is the minority attention pathway rather than the dominant recurrent backbone — though this is a tested-setting observation, not a universal claim.

## Citation

This manuscript is under review at **PeerJ Computer Science**. A Zenodo archive will be created at acceptance/submission; the DOI and Git commit hash will be inserted here and in the manuscript.

```bibtex
@misc{borobia2026lora,
  title  = {Where Should LoRA Go? An Exploratory Study of Component-Type
            Placement for Parameter-Efficient Adaptation of Small Hybrid
            Language Models},
  author = {Borobia, H{\'e}ctor and Segu{\'i}-Mas, Elies and Tormo-Carb{\'o}, Guillermina},
  year   = {2026},
  note   = {Under review at PeerJ Computer Science},
  howpublished = {\url{https://github.com/hecboar/lora-placement-hybrid}},
  doi    = {[to be added after Zenodo archive]}
}
```

See also [CITATION.cff](CITATION.cff).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
