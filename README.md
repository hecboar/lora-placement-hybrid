# Where Should LoRA Go? Component-Type Placement in Hybrid Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.22127v2-b31b1b.svg)](https://arxiv.org/abs/2604.22127)

> ### Correction notice
>
> Two defects were found in the evaluation harness that produced the results first
> released here. GSM8K answers were extracted as the **last** `####` match from
> generations run without a stop sequence, and the HumanEval metric never executed,
> so its `NaN` reached the tables as `0.000`.
>
> The corrected analysis is in **[`reanalysis/`](reanalysis/)**, and
> **[`reanalysis/VERDICT.md`](reanalysis/VERDICT.md)** documents both defects claim by
> claim. No model was retrained: the correction re-scores the per-instance outputs
> already in this repository. The findings below are the corrected ones, and match
> arXiv:2604.22127**v2**. Everything under `tables/`, `stats/` and `figures/` is the
> original output and is superseded, retained because v1 of the preprint cites it.

This repository accompanies a controlled study of *where* LoRA should be placed in
hybrid language models that combine softmax attention with a recurrent sequence-mixing
component (state-space models or gated linear attention). Two sub-billion-parameter
models are compared across six placement conditions each, three training domains, and
five benchmarks, at a single seed (3407).

## Findings

After correction, an exhaustive paired bootstrap over all **534** condition comparisons
leaves **52** significant before multiplicity correction and **10** after Holm
correction within each (model, domain, benchmark) family. None of the ten is an
on-target placement gain. Three coherent statements survive:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | The parallel hybrid transfers positively into mathematics from unrelated instruction data | Falcon-H1 gains 8.6–9.0 pp on GSM8K after UltraChat training, across several placements |
| 2 | Off-target degradation is placement-dependent, and it is the **broad** placements that cause it | After UltraChat on Falcon-H1, `all_eligible`, `attention_plus_mlp` and `mlp_only` lose HellaSwag accuracy significantly; `attention_only` and `ssm_only` do not |
| 3 | The two topologies order their single-component placements differently | After GSM8K training on Qwen3.5, `mlp_only` gains 4.3 pp on HellaSwag while `softmax_only` sits 6.1 pp below it |

Every surviving effect is measured either on a log-likelihood benchmark or as
cross-domain transfer. The viable contribution concerns where adaptation does
collateral damage, not which placement maximises target-task gain.

**Withdrawn.** The original release claimed that attention-only placement is the
strongest low-parameter target, that recurrent-backbone adaptation is destructive in
sequential hybrids and constructive in parallel ones, that a destructive interference
anomaly affects `softmax_plus_mlp`, that attention-only placement minimises off-target
degradation, a set of accuracy-per-parameter ratios, and a HumanEval floor effect. None
of these survives correction. HumanEval `pass@1` is in fact 0.213–0.341, not zero.
[`reanalysis/VERDICT.md`](reanalysis/VERDICT.md) gives the before/after numbers.

**Statistical power.** The standard deviation of the paired per-item difference on
GSM8K is 0.45, so detecting a 3 pp difference at 80% power needs 1,764 items and 5 pp
needs 635. The subsets used here hold 128 (Qwen3.5) and 256 (Falcon-H1)
items against true effects of 2–7 pp. The design is underpowered by roughly an order of
magnitude, and the non-significant results should be read as undetectable rather than
absent.

## Repository Structure

```
├── README.md                          ← You are here
├── LICENSE                            ← MIT License
├── REPRODUCIBILITY.md                 ← Hardware, configs, exact protocol
├── DATA_AVAILABILITY.md               ← What is / is not redistributed
├── CITATION.cff                       ← How to cite this work
├── requirements.txt                   ← Python dependencies
├── environment.yml                    ← Conda environment
├── notebook/
│   └── paper3_lora_placement.ipynb    ← Full training/evaluation pipeline
├── results/
│   ├── discovery/                     ← Target-module manifests + condition specs
│   ├── eval_details/                  ← Per-instance evaluation outputs (JSONL)
│   └── summary/paper3_summary.json    ← Machine-readable experiment summary
├── reanalysis/                        ← CORRECTED analysis — start here
│   ├── VERDICT.md                     ← The two defects, claim by claim
│   ├── eval_details_corrected/        ← Corrected per-instance labels
│   ├── *.py                           ← Scripts that produce everything below
│   └── *.csv                          ← Corrected tables, bootstrap, power analysis
├── figures/                           ← Original figures (superseded)
├── tables/                            ← Original aggregate tables (superseded)
└── stats/                             ← Original statistical artifacts (superseded)
```

`results/eval_details/` holds the raw per-instance model outputs. These are unaffected
by the defects and are the common source from which both the original and the corrected
numbers are derived, which is why the correction required no retraining.

## Quick Start

```bash
git clone https://github.com/hecboar/lora-placement-hybrid.git
cd lora-placement-hybrid
```

Inspect the corrected results (no GPU needed):

```python
import pandas as pd

# Corrected accuracy for every model x condition x domain x benchmark,
# with the original value alongside for comparison.
df = pd.read_csv("reanalysis/all_results_corrected.csv")
gsm = df[(df.benchmark == "gsm8k") & (df.train_dataset == "gsm8k_train")]
print(gsm[["model_key", "condition", "acc_reported", "acc_corrected"]].to_string())

# Every pairwise comparison with bootstrap CIs and Holm-corrected p-values.
bs = pd.read_csv("reanalysis/all_pairwise_bootstrap_corrected.csv")
print(bs[bs.sig_holm][["model", "domain", "benchmark", "cond_a", "cond_b", "diff_pp"]])
```

Reproduce the correction from the released per-instance outputs:

```bash
python reanalysis/rescore_gsm8k_and_aggregate.py     # GSM8K extraction fix + tables
python reanalysis/rescore_humaneval_per_instance.py  # executes saved completions
python reanalysis/survivors.py                       # exhaustive bootstrap + Holm
python reanalysis/power_analysis.py                  # required sample sizes
```

Reproduce the experiments from scratch (GPU, ≥24 GB VRAM): set `ROOT_DIR` in
`notebook/paper3_lora_placement.ipynb` and follow the execution order in the final
cell. The pipeline is checkpoint-safe and completed experiments auto-skip. **Fix the
evaluator first** — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Models and Conditions

| Model | Topology | Conditions | Params range |
|-------|----------|------------|-------------|
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | Sequential (18 GDN + 6 softmax attn, 3:1) | `all_layers`, `softmax_only`, `gdn_only`, `mlp_only`, `softmax_plus_mlp`, `gdn_plus_mlp` | 1.08M – 10.82M |
| [Falcon-H1-0.5B-Base](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) | Parallel (attn ∥ Mamba-2 per block) | `all_eligible`, `attention_only`, `ssm_only`, `mlp_only`, `attention_plus_mlp`, `ssm_plus_mlp` | 2.21M – 11.47M |

All conditions use LoRA rank 16, α=32, dropout 0.05, lr 2e-4 (cosine), 3 epochs,
effective batch 16, seq_len 1024, 8-bit Adam, bf16, gradient checkpointing. Full
protocol in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

> **The two condition sets are not equally clean decompositions.** For Qwen3.5 the
> three single-component conditions partition the broad condition exactly
> (1.08 + 4.43 + 5.31 = 10.82M). For Falcon-H1 they do not: 2.21 + 2.52 + 5.31 = 10.04M
> against 11.47M for `all_eligible`, leaving 1.43M (12.4%) in `o_proj` on all 36 blocks
> and in `lm_head`, which appear in no single-component condition. Cross-topology
> comparisons therefore confound topology with adapter coverage. Harmonising the
> condition sets is the first item of follow-up work.

## Training Domains and Evaluation

Each model is fine-tuned independently on three domains of 2,000 examples each and
evaluated on a fixed benchmark suite.

| Domain | Dataset | Role |
|--------|---------|------|
| Mathematics | GSM8K (train split) | Mathematics adaptation |
| Code-instruction | CodeAlpaca | Code instruction-following and domain-shift source |
| General instruction | UltraChat | General instruction adaptation |

| Benchmark | n | Scoring |
|-----------|---|---------|
| MMLU | 512 | log-likelihood over answer labels |
| ARC-Challenge | 299 | log-likelihood over answer labels |
| HellaSwag | 512 | log-likelihood over answer labels |
| GSM8K | 128–256 | greedy generation, numeric answer extraction |
| HumanEval | 164 (157 for one Falcon condition) | greedy generation, unit-test execution |

The three log-likelihood benchmarks involve no free generation and were unaffected by
the defects. The two generative benchmarks are the ones the correction applies to.

## Statistical Notes

- **Single seed.** All runs use seed 3407. Note also that the training configuration
  did not receive the seed, so data ordering and dropout would be identical across
  repeated runs; only the LoRA initialisation would vary. A genuine multi-seed study
  must vary both.
- **Paired bootstrap** over matched `example_id`: 10,000 resamples, percentile 95%
  intervals, bootstrap seed 3407. These quantify evaluation-subset uncertainty for
  fixed trained models, not training-run variance.
- **Multiplicity.** The corrected analysis reports Holm-corrected p-values within each
  (model, domain, benchmark) family over all 534 comparisons, rather than a selected
  subset.
- **Sample sizes** vary by benchmark. GSM8K for Qwen3.5 uses 128 examples for the
  baseline and GSM8K-trained runs and 256 for the CodeAlpaca/UltraChat runs, so those
  cross-domain point estimates are not strictly comparable; paired comparisons use
  shared identifiers only.

## Related Papers

This work is the third in a series studying hybrid language model internals:

1. **Paper 1** — *How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models*
2. **Paper 2** — *Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures*
3. **Paper 3** — This work: *Where Should LoRA Go?*

Paper 2 reported that the recurrent backbone acts as a functional backbone while
attention behaves as refinement. This study asked whether that hierarchy is reflected
in adaptation. The original answer, that the minority attention pathway is the better
adaptation target, did not survive correction.

## Citation

Cite **v2**. Version 1 reports the uncorrected numbers.

```bibtex
@misc{borobia2026lora,
  title  = {Where Should LoRA Go? Component-Type Placement in Hybrid
            Language Models},
  author = {Borobia, H{\'e}ctor and Segu{\'i}-Mas, Elies and Tormo-Carb{\'o}, Guillermina},
  year   = {2026},
  eprint = {2604.22127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note   = {Version 2 corrects two evaluation-harness defects present in v1},
  url    = {https://arxiv.org/abs/2604.22127}
}
```

See also [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
