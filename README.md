# Where Should LoRA Go? Component-Type Placement for Parameter-Efficient Adaptation of Hybrid Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2504.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2504.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **TL;DR** — In hybrid LMs (attention + SSM/GDN), target LoRA at the attention pathway only. It matches or beats full-model adaptation with 5–10× fewer parameters, while adapting the recurrent backbone is destructive in sequential hybrids but safe in parallel ones.

## Key Findings

| # | Finding | Qwen3.5-0.8B (sequential) | Falcon-H1-0.5B (parallel) |
|---|---------|---------------------------|---------------------------|
| 1 | **Attention-only is the best LoRA target** | +10.2 pp on GSM8K with 1.08M params (0.14%) | +17.2 pp on GSM8K with 2.21M params (0.42%) |
| 2 | **Backbone adaptation depends on topology** | GDN-only **destroys** GSM8K (−14.8 pp) | SSM-only **improves** GSM8K (+8.6 pp) |
| 3 | **Parallel hybrids transfer; sequential hybrids forget** | Up to −16.0 pp forgetting after UltraChat | Up to +10.9 pp positive transfer after UltraChat |
| 4 | **Attention-only protects against forgetting** | Least destructive across all domains | Near-zero forgetting on cross-domain eval |
| 5 | **9–13× more efficient per parameter** | 9.4 pp/M vs 0.7 pp/M (all_layers) | 7.8 pp/M vs 1.1 pp/M (all_eligible) |

## Repository Structure

```
├── README.md                          ← You are here
├── LICENSE                            ← MIT License
├── notebook/
│   └── paper3_lora_placement.ipynb    ← Full reproducible pipeline
├── figures/                           ← Publication figures (PDF + PNG, both models)
├── results/
│   ├── discovery/
│   │   ├── falcon_h1_0_5b_base__condition_specs.json
│   │   ├── falcon_h1_0_5b_base__manifest.json
│   │   ├── qwen3_5_0_8b_base__condition_specs.json
│   │   └── qwen3_5_0_8b_base__manifest.json
│   ├── eval_details/                  ← Per-example evaluation results (JSONL)
│   └── summary/
│       └── paper3_summary.json        ← Machine-readable experiment summary
└── tables/
    ├── all_results_flat.csv           ← 164 rows: every benchmark × condition × model
    ├── result_matrix_accuracy.csv     ← Pivot: condition × benchmark accuracy
    ├── efficiency_table.csv           ← Efficiency ratios, forgetting, params
    ├── forgetting_summary.csv         ← Cross-task forgetting scores
    ├── paired_bootstrap_comparisons.csv ← Statistical comparisons (CIs)
    ├── practitioner_recipe.csv        ← Recommended conditions per domain
    ├── falcon_h1_0_5b_base__condition_param_summary.csv
    ├── falcon_h1_0_5b_base__verification_summary.csv
    ├── qwen3_5_0_8b_base__condition_param_summary.csv
    └── qwen3_5_0_8b_base__verification_summary.csv
```

## Quick Start

```bash
# Clone
git clone https://github.com/hecboar/lora-placement-hybrid.git
cd lora-placement-hybrid

# Reproduce from scratch — GPU machine (≥24GB VRAM)
jupyter notebook notebook/paper3_lora_placement.ipynb
```

In the notebook, set `ROOT_DIR` to your preferred output path and follow the execution order in the final cell. The pipeline is checkpoint-safe — if your runtime disconnects, re-run cells 0–4 and resume. Completed experiments auto-skip.

**Estimated compute:** ~30h GPU on NVIDIA L4 or RTX 4090.

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
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Sequential (18 GDN + 6 softmax attn) | `all_layers`, `softmax_only`, `gdn_only`, `mlp_only`, `softmax_plus_mlp`, `gdn_plus_mlp` | 1.08M – 10.82M |
| [Falcon-H1-0.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-0_5B-Base) | Parallel (attn ∥ Mamba-2 per block) | `all_eligible`, `attention_only`, `ssm_only`, `mlp_only`, `attention_plus_mlp`, `ssm_plus_mlp` | 2.21M – 11.47M |

All conditions use: LoRA rank=16, α=32, dropout=0.05, lr=2e-4, 3 epochs, batch=16, seq_len=1024.

## Training Domains

| Domain | Dataset | Samples | Target benchmark |
|--------|---------|---------|-----------------|
| Mathematics | GSM8K (train split) | 2,000 | GSM8K |
| Code | CodeAlpaca | 2,000 | HumanEval |
| General instruction | UltraChat | 2,000 | MMLU, ARC-C, HellaSwag |

## Practitioner Recipe

| Model | Domain | Recommended | Params | vs. full-model |
|-------|--------|-------------|--------|----------------|
| Falcon-H1 | Any | `attention_only` | 2.21M | ≥95% accuracy, 5× fewer params |
| Qwen3.5 | Math / General | `softmax_only` | 1.08M | ≥95% accuracy, 10× fewer params |
| Qwen3.5 | Code | `all_layers` | 10.82M | Required for this domain |

## Related Papers

This work is the third in a series studying hybrid language model internals:

1. **Paper 1** — [How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models](https://arxiv.org/abs/2603.25325) *(arXiv 2026)*
2. **Paper 2** — [Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures](https://arxiv.org/abs/2603.22473) *(arXiv 2026)*
3. **Paper 3** — This paper: *Where Should LoRA Go?*

Paper 2 established that the recurrent backbone is the "functional backbone" while attention is "refinement." Paper 3 shows this hierarchy is **inverted for adaptation**: the best LoRA target is the minority attention pathway, not the dominant backbone.

## Citation

```bibtex
@article{borobia2026lora,
  title={Where Should LoRA Go? Component-Type Placement for Parameter-Efficient Adaptation of Hybrid Language Models},
  author={Borobia, Hector and Segu{\'i}-Mas, Elies and Tormo-Carb{\'o}, Guillermina},
  journal={arXiv preprint arXiv:2504.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
