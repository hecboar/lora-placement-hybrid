# Paper 3 → Neurocomputing: experiment plan indexed to real reviewer demands

Calibrated against **Paper 1**, *Feature survival under weight pruning*, published in
Neurocomputing 704 (2026) 134826 (received 18 May, accepted 13 Aug 2026, two revision
rounds from a "Revise and Reconsider"). That paper cleared this venue with **three
models of 1B–2B and about 30 GPU-hours on a single L4**. Compute was not what carried
it; the control structure and the statistical treatment were.

The two reviewer reports on Paper 1 are the best available evidence of what this venue
asks for. Every experiment below is justified by one of those demands, mapped to its
Paper 3 analogue. Costs are RunPod L4 at €0.40/h, derived from the measured training
times of the 33 complete runs already on disk.

---

## The mapping

| Paper 1 reviewer demand | Paper 3 analogue | What answers it | € |
|---|---|---|---|
| **R2.4** Tables report only means, no variance, CI or significance tests | same objection would apply verbatim | **Already done.** Holm-corrected paired bootstrap over all 534 comparisons, plus a power analysis. This is stronger than what Paper 1 ended up with. | 0 |
| **R1.2** How can the dense perplexity of 410 be explained? | the GSM8K extraction artifact and the HumanEval metric that never ran | **Already done, and found before submission rather than during review.** Documented as a contribution in its own section. | 0 |
| **R1.3** What evaluation metrics are used? List them clearly | same | Consolidated metrics table. Paper 1 added exactly this in revision. | 0 |
| **R1.4** Dense multi-curve plots are hard to read | the v1 radar and Pareto figures | **Already done.** Radar and Pareto dropped; annotated diverging heatmaps and a forest plot with CIs. | 0 |
| **R2.6** Strong mechanistic explanations for 0.1–0.6% differences | v1 explained a 25 pp "anomaly" that was an artifact | **Already done.** Mechanistic claims retracted; effects now reported with intervals and the small ones left uninterpreted. | 0 |
| **R2.8** No causal chain; correlation not mechanism | same | Frame every claim as associational, as Paper 1 did. Costs nothing. | 0 |
| **R2.9** Lacks theoretical depth, no mathematical explanation | same | Paper 1 answered with a scope statement plus two elementary propositions about its estimators, and that was accepted. Do the same: one proposition relating trainable-parameter budget to placement. | 0 |
| **R1.1** Distinguish clearly from existing SAE methodology | the 2025–2026 literature already converged on "attention-only underperforms, target all linear layers" | Reposition: the recurrent backbone is a component type that does not exist in a pure Transformer, so that literature does not answer this question. Sharpen related work. | 0 |
| **R2.1** Conclusions depend on the SAE configuration; different configs give different results | conclusions may depend on LoRA rank | **Rank ablation** r ∈ {8, 16, 32}. Already scripted in `news-experiments/`. | 11 |
| **R2.2** Only residual stream; other structures not analysed | Falcon `attention_only` omits `o_proj`; `ssm_only` covers only `in_proj` | **Harmonised condition sets**, so the single-component conditions partition the broad one in both models. | 23 |
| **R2.3** prune → train SAE → analyse: cannot separate pruning from SAE reconstruction error | train → evaluate: cannot separate placement from trainable-parameter budget | **Budget-matched conditions + random-placement control.** This is the direct analogue of Paper 1's fixed-dictionary control, the single thing that most carried it. | 11 |
| **R2.5** No quantitative correlation with downstream task performance | same | Correlate placement metrics against full-split downstream accuracy. Comes free with the re-evaluation. | 4 |
| **R2.7** Only two similar compression methods; does it hold for others? | only LoRA | **One more PEFT method** under the same placements (DoRA is a drop-in in PEFT). | 11 |
| *(not raised, but Paper 1 used 3 models to Paper 3's 2)* | generality, and topology confounded with family | **The Granite 4.0 pair**, one domain, one seed: a third topology point sharing its mechanism with Falcon and its topology with Qwen, plus a same-family dense control. | 34 |
| *(standing objection at any venue)* | single seed | **3 seeds** on the two domains that carry the surviving findings. | 43 |
| *(absolute reference)* | no full fine-tuning baseline | 2 models × 3 seeds, one domain. | 8 |

---

## The decisive precedent: Paper 4

*Component-Aware Self-Speculative Decoding for Hybrid Language Models*, Computers and
Electrical Engineering, accepted and in production (PII S0045790626005756). Same domain,
the same two anchor models, the same single L4, and — like the corrected Paper 3 — a
**negative headline**.

Its **first submission used 2 hybrid models from 2 families and drew unanimous major
criticism from three reviewers.** What converted that into acceptance was breadth:
**9 models across 5 families**, 6 datasets, 3 executed baselines, and a full statistical
retrofit. Paper 3 today sits at exactly the configuration that failed there.

Three reviewer objections from Paper 4 transfer verbatim:

- **R3:** "single family per paradigm confounds architecture with implementation" — this
  is precisely Paper 3's topology-versus-family confound.
- **R2:** reject claims of architectural determinism drawn from two families and one
  comparison — Paper 3's central claim has the same shape.
- **R2:** "decouple from the unpublished companion ablation paper" — Paper 3's discussion
  leans on `borobia2026functional` for its mechanism. That dependency must be cut.
- **R2:** "drop parameter-count FLOP ratios or measure real kernel timings" — Paper 3's
  accuracy-per-million-parameters is exactly such a proxy. Replace with measured
  wall-clock and peak memory.

Two findings from Paper 4 are strongly encouraging. A negative headline was accepted, and
Reviewer 2 said explicitly it was **more valuable than a manufactured positive result**.
And the paper self-reported two implementation defects and was praised for it. Paper 3's
correction narrative fits the same mould.

One dimension needs *less* than expected: Paper 4 used effectively **one seed** and no
reviewer objected, because greedy decoding is deterministic. Paper 3 cannot use that
defence, since training is stochastic — but it does mean seeds are not where reviewers
at this level look. **Breadth is.** Spend on models, not on extra seeds.

## Packages

Priced from the measured training times, 3 seeds on the two anchor models across both
domains, and 1 seed × 4 conditions × 1 domain on each breadth model.

| | Contents | GPU-h | € | with fast kernels |
|---|---|---|---|---|
| **A** | current 2-model design — *the configuration that failed for Paper 4* | 117 | 47 | 33 |
| **B** | **+ the Granite pair**, 4 conditions, 1 seed, UltraChat | 202 | 81 | 55 |
| **C** | + the Granite pair on both domains | 289 | 116 | 75 |
| **D** | + scale steps (Falcon-H1-1.5B, Qwen3.5-2B) | 371 | 149 | 96 |

Package **B is the one to buy**. The step from A to B costs about €34, or €22 once the
fast kernels are in, and it is the step that converts "two models, one family per
topology" into a design where each contrast varies one thing. That is the difference
between the configuration reviewers rejected for Paper 4 and the one they accepted.

Packages C and D are refinements, not requirements: a second domain on Granite, and a
scale axis. Buy them with what the first runs leave over.

Granite-4.0-H-Micro is 3B and carries the largest per-run cost of any addition, but it
is also the only candidate that breaks the confound, so it is the last thing to cut
rather than the first. `granite-4.0-h-micro` was already loaded on a single L4 for
Paper 4, so memory fit and loading are known rather than assumed.

Add €3–4/month for a 50 GB network volume.

**Biggest single lever on cost:** the original campaign ran without `mamba-ssm` and
`causal-conv1d`, so Falcon used the slow PyTorch fallback and dominates the training
bill at 2.6–3.2 h per run against Qwen's 0.7–1.4 h. If the fast Mamba-2 kernels halve
that, the reviewer-proof package drops by roughly a third. Free to test on the first run.

**Counterintuitive saving:** once the evaluator is batched, evaluating on the *full*
test splits costs 0.25 h per run against 0.64 h for today's 128–256 item subsets. The
power problem that invalidates the current design is fixed by spending less, not more.

---

## The third model: Granite 4.0

The standing weakness, and the objection reviewer 3 raised against the companion
decoding paper, is that topology is confounded with family: with one family per
topology, a difference between the anchors could be topology, recurrent mechanism,
tokenizer, data or anything else. **`ibm-granite/granite-4.0-h-micro-base` resolves it**,
because of what it shares with each anchor.

| Model | Topology | Recurrent mechanism | Ratio |
|---|---|---|---|
| Qwen3.5-0.8B | interleaved | GDN, linear attention | 18 : 6 |
| Falcon-H1-0.5B | parallel | Mamba-2 | both per block |
| **granite-4.0-h-micro** | **interleaved** | **Mamba-2** | **36 : 4** |
| granite-4.0-micro | none, dense | none | — |

- **Falcon vs Granite-H** holds the mechanism and varies the topology.
- **Qwen vs Granite-H** holds the topology and varies the mechanism.
- **Granite-H vs Granite-dense** holds family, data and recipe, and varies only whether
  a recurrent component exists at all. This is a better pure-Transformer control than
  an unrelated Transformer, which varies everything at once.

3B dense, Apache 2.0, base checkpoint available. Also adds a scale point, so one family
buys three axes.

| Plan | GPU-h | € | with fast kernels |
|---|---|---|---|
| Both Granite models, 4 conditions, 1 seed, UltraChat | 85 | 34 | 22 |
| Both, 4 conditions, 1 seed, both domains | 172 | 69 | 42 |

Breadth takes one seed, not three: the seeds exist to measure training variance on the
anchors, not to characterise every model. A Granite result that contradicts the anchors
is what earns it more seeds.

**Considered and rejected**, recorded in `pipeline_v2/models.py` so the choice is
auditable: Zamba2-1.2B, whose attention is a single shared block already carrying its
own LoRA projectors for depth specialisation, so our LoRA would sit on top of built-in
LoRA and `attention_only` would not mean what it means elsewhere; LFM2-1.2B, a
short-convolution mixer with no base checkpoint; Nemotron-H, whose smallest hybrid is
8B; and granite-4.0-h-tiny, which is mixture-of-experts, so its MLP condition is not
comparable.

## Order of execution

1. **Fix the evaluator and batch it.** No GPU. Stop sequences, first-`####` extraction,
   `HF_ALLOW_CODE_EVAL=1`, batched scoring and generation. Everything downstream depends
   on this and nothing should be trained until it is done.
2. **Re-evaluate the 36 existing adapters** on full test splits. ~10 GPU-h, €4. This
   fixes the real baseline and tells us which effects are worth chasing before any
   training budget is committed.
3. **Random-placement and budget-matched controls.** €11. Cheapest experiment with the
   largest effect on defensibility, and it can invalidate the whole premise early: if
   random placement at matched parameter count performs the same as principled
   placement, that is the finding, and it is publishable.
4. **3 seeds on the two domains** that carry the surviving findings. €43.
5. **Harmonised conditions.** €23.
6. **Rank ablation, DoRA, full fine-tuning, third family**, in whatever order the data
   from steps 2–5 suggests.

Step 3 before step 4 is deliberate. It is the cheapest way to learn whether the study
has a subject at all.

---

## What the paper claims

Not "where should LoRA go" — the 2025–2026 literature is already converging on
"not attention alone, target the linear layers including the MLP", and the corrected
results agree with it. Two things that literature does not cover:

1. **Where adaptation does collateral damage in hybrids.** Broad placements degrade
   off-target accuracy where narrow ones do not, and the ordering differs between
   sequential and parallel topologies. This is where all ten Holm-surviving effects are.
2. **How much of a generative-benchmark conclusion is the evaluation protocol.** A
   standard-looking harness detail produced a fully coherent, mechanistically plausible,
   internally consistent and entirely spurious set of conclusions, and we quantify it
   exactly. This is the memorable result and it is positive rather than a null.
