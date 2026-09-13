# Re-analysis of the released per-instance outputs — claim-by-claim verdict

Date: 2026-09-13. Inputs: `github/results/eval_details/*.jsonl` (seed 3407, as published).
Scripts: `rescore_gsm8k_and_aggregate.py`, `rescore_humaneval_per_instance.py`, `power_analysis.py`.
No model was re-run; every number below comes from re-scoring outputs that were already on disk.

---

## 1. The two evaluator defects

### 1.1 GSM8K: last-`####` extraction on untruncated generations

`evaluate_gsm8k()` generates 256 greedy tokens with no stop sequence, then
`extract_final_number()` returns the **last** `#### N` in the text. The few-shot
template repeats a system line plus `Question:`/`Answer:`, so after producing its
real answer the model continues the pattern and hallucinates a fresh question with
its own `#### N`. That trailing number is what was scored.

Prevalence of completions containing more than one `####`:

| Run | multi-`####` / n |
|---|---|
| Qwen base | 113 / 128 |
| Falcon base | 224 / 256 |
| Qwen `gdn_only` (GSM8K-trained) | 118 / 128 |
| Qwen `softmax_only` (GSM8K-trained) | 0 / 128 |

The correction truncates each saved completion at the first stop marker and then
applies the **original, unmodified** extractor and normaliser. Between 0 and 77
labels per file change.

**Robustness of the correction.** Across all 39 GSM8K files, 1,515 labels flip from
incorrect to correct and only 6 flip the other way. The correction is almost purely
recovering answers the model got right and the harness mis-read; it is not an
arbitrary re-scoring that happens to favour some conditions. The six reverse flips
are cases where the model's first `####` was wrong and a later hallucinated block
coincidentally matched the gold answer.

### 1.2 HumanEval: the metric never ran

`evaluate.load("code_eval")` raises unless `HF_ALLOW_CODE_EVAL=1` is set, which the
notebook never does. pass@1 was recorded as NaN and reported as 0/164 for all twelve
conditions, then written up as a "scale-limited floor effect". Re-executing every saved
completion against the official unit tests, with standard stop-sequence truncation,
recovers real pass@1 for all twelve conditions. The negative result was an
infrastructure failure, not a property of the models.

| Model | Condition | Reported | pass@1 recovered |
|---|---|---|---|
| Falcon | `attention_only` | 0.000 | **0.341** |
| Falcon | `ssm_only` | 0.000 | **0.341** |
| Falcon | `all_eligible` | 0.000 | **0.331** |
| Falcon | `mlp_only` | 0.000 | **0.323** |
| Falcon | `attention_plus_mlp` | 0.000 | **0.317** |
| Falcon | `ssm_plus_mlp` | 0.000 | **0.311** |
| Qwen | `gdn_only` | 0.000 | **0.256** |
| Qwen | `softmax_plus_mlp` | 0.000 | **0.244** |
| Qwen | `all_layers` | 0.000 | **0.238** |
| Qwen | `gdn_plus_mlp` | 0.000 | **0.226** |
| Qwen | `mlp_only` | 0.000 | **0.226** |
| Qwen | `softmax_only` | 0.000 | **0.213** |

Without truncation (scoring the raw 256-token generation as the pipeline would have)
pass@1 is 0.12–0.23, so even the unmodified harness would not have produced zeros had
the metric run at all. Placement makes no detectable difference here: across the 30
within-model comparisons the largest gap is 4.3 pp against a median CI half-width of
4.3 pp, and none reaches significance. HumanEval is a usable secondary benchmark for
this study, but only for reporting that CodeAlpaca adaptation works, not for ranking
placements.

---

## 2. GSM8K accuracy, as reported vs corrected

| Model | Domain | Condition | Reported | Corrected |
|---|---|---|---|---|
| Qwen | — | base | .297 | **.422** |
| Qwen | GSM8K | `softmax_only` | .398 | .398 |
| Qwen | GSM8K | `mlp_only` | .383 | .414 |
| Qwen | GSM8K | `gdn_only` | .148 | **.438** |
| Qwen | GSM8K | `softmax_plus_mlp` | .148 | **.438** |
| Qwen | GSM8K | `gdn_plus_mlp` | .203 | **.492** |
| Qwen | GSM8K | `all_layers` | .375 | **.492** |
| Falcon | — | base | .383 | **.531** |
| Falcon | GSM8K | `attention_only` | .555 | .590 |
| Falcon | GSM8K | `ssm_plus_mlp` | .508 | .570 |
| Falcon | GSM8K | `all_eligible` | .504 | .555 |
| Falcon | GSM8K | `ssm_only` | .469 | .551 |
| Falcon | GSM8K | `mlp_only` | .508 | .543 |
| Falcon | GSM8K | `attention_plus_mlp` | .492 | .520 |

Note the mechanism: `softmax_only` is the single condition whose score does not move,
because it is the only one that learned to emit an end-of-sequence token. Much of what
the paper measured as "adaptation quality" was whether a condition learned to stop
generating.

Full matrices: `result_matrix_reported.csv` vs `result_matrix_corrected.csv`.
MMLU, ARC-Challenge and HellaSwag use log-likelihood scoring and are unaffected.

---

## 3. Paired bootstrap on corrected labels

10,000 resamples, percentile 95%, bootstrap seed 3407, matched `example_id` — identical
settings to the manuscript, applied to corrected correctness labels.

| Comparison | Reported (pp) | Corrected (pp) | 95% CI |
|---|---|---|---|
| Qwen GSM8K `softmax_only` vs base | +10.2 | **−2.3** | [−10.2, +5.5] |
| Qwen GSM8K `gdn_only` vs base | −14.8 | **+1.6** | [−7.8, +10.9] |
| Qwen GSM8K `softmax_plus_mlp` vs base | −14.8 | **+1.6** | [−7.8, +10.9] |
| Qwen GSM8K `softmax_plus_mlp` vs `mlp_only` | −23.4 | **+2.3** | [−5.5, +10.2] |
| Qwen GSM8K `softmax_only` vs `all_layers` | — | **−9.4** | [−17.2, −1.6] |
| Falcon GSM8K `attention_only` vs base | +17.2 | **+5.9** | [0.0, +11.7] |
| Falcon GSM8K `attention_only` vs `all_eligible` | +5.1 | **+3.5** | [−2.3, +9.4] |
| Falcon GSM8K `attention_only` vs `ssm_only` | +8.6 | **+3.9** | [−2.3, +10.2] |
| Falcon GSM8K `ssm_only` vs base | +8.6 | **+2.0** | [−4.3, +8.2] |
| Falcon UltraChat GSM8K `attention_only` vs `all_eligible` | +5.5 | **−3.1** | [−9.0, +2.7] |
| Falcon UltraChat HellaSwag `attention_only` vs `all_eligible` | +2.7 | **+2.7** | [+0.8, +4.9] |

---

## 4. Verdict on each headline claim

| # | Claim in the manuscript | Verdict |
|---|---|---|
| 1 | Attention-only is the strongest low-parameter target | **Fails.** In Qwen it is −2.3 pp vs base and significantly *worse* than `all_layers` (−9.4 pp, CI excludes 0) — the reverse of the thesis. In Falcon it is +5.9 pp vs base (CI touches 0) and not separable from `all_eligible` or `ssm_only`. |
| 2 | Recurrent-backbone adaptation differs by topology (Qwen −14.8 vs Falcon +8.6) | **Fails.** Corrected: Qwen +1.6, Falcon +2.0, both n.s. The 23-point contrast that motivates Sections 6.2 and 6.4 becomes 0.4 points. |
| 3 | Off-target transfer patterns differ between topologies | **Partly survives.** Qwen worst off-target after UltraChat goes from −16.0 to −5.9 pp; Falcon best goes from +10.9 to +9.0 pp. The asymmetry shrinks from ~27 to ~15 points but the sign difference holds. It is not placement-specific: every Falcon condition shows it. |
| 4 | Attention-only shows the smallest off-target degradation | **Fails.** After UltraChat on Qwen, `softmax_only` is −5.1 pp, ranking fourth of six behind `gdn_plus_mlp` (0.0) and `mlp_only` (−2.3). |
| 5 | Higher accuracy-per-parameter (9.4 vs 0.7 pp/M; 7.8 vs 1.1) | **Fails for Qwen** (the numerator is now negative), **weakens for Falcon** (2.7 vs 0.2 pp/M, both from n.s. effects). |
| 6 | HumanEval floor effect, 0/164 everywhere | **Fails.** Real pass@1 is 0.21–0.34. This reverses a reported negative result. |
| 7 | `softmax_plus_mlp` destructive-interference anomaly (Section 6.3) | **Fails.** .148 → .438; +2.3 pp vs `mlp_only`. The anomaly was entirely the extraction artifact. |

### What does survive

- **Falcon `attention_only` beats `all_eligible` on HellaSwag after UltraChat**: +2.7 pp, CI [+0.8, +4.9]. Log-likelihood benchmark, untouched by the bug. The only comparison in the paper that is both significant and unchanged.
- **Cross-domain transfer into GSM8K in the parallel hybrid**: training Falcon on UltraChat raises corrected GSM8K by 5–9 pp in every condition. Real, but a property of the model, not of placement.
- **The module-discovery and verification methodology** (manifests, exact dotted target lists, trainable-host assertions) is sound and reusable. It is the part of the work worth keeping intact.
- Parameter accounting, condition construction and the checkpointing infrastructure are fine.

### One anomaly the corrected numbers expose

Falcon trained on **CodeAlpaca** scores .605 on GSM8K, above both the base model (.531)
and every GSM8K-trained condition (.520–.590). Training on code beats training on maths,
on maths. With n=256 this is inside the noise band, but it points the same way as
everything else: at this scale the fine-tuning is mostly teaching output format and
stopping behaviour, not reasoning.

---

## 4b. What is actually left: an exhaustive search

`survivors.py` runs the paired bootstrap over **every** condition-vs-condition and
condition-vs-base comparison in the corrected data: 534 comparisons across both models,
three training domains and five benchmarks. Holm correction is applied within each
(model, domain, benchmark) family.

- 51 comparisons are significant before multiplicity correction.
- **10 survive Holm correction.** None of them is an on-target GSM8K placement effect
  of the kind the paper is built on, and none is on HumanEval.

| Model | Domain | Benchmark | Comparison | Diff (pp) | 95% CI | Holm p |
|---|---|---|---|---|---|---|
| Qwen | GSM8K | HellaSwag | `mlp_only` − `softmax_plus_mlp` | +4.3 | [+2.2, +6.6] | <0.01 |
| Qwen | GSM8K | HellaSwag | `softmax_only` − `mlp_only` | −6.1 | [−8.8, −3.3] | <0.01 |
| Qwen | GSM8K | HellaSwag | `softmax_only` − `gdn_only` | −5.3 | [−8.4, −2.3] | 0.01 |
| Qwen | GSM8K | HellaSwag | `mlp_only` − base | +4.3 | [+1.8, +6.8] | 0.01 |
| Falcon | UltraChat | HellaSwag | `all_eligible` − base | −4.3 | [−6.6, −2.0] | <0.01 |
| Falcon | UltraChat | HellaSwag | `attention_plus_mlp` − base | −3.9 | [−6.3, −1.6] | 0.02 |
| Falcon | UltraChat | HellaSwag | `mlp_only` − base | −3.7 | [−6.3, −1.2] | 0.04 |
| Falcon | UltraChat | GSM8K | `ssm_only` − base | +9.0 | [+3.1, +14.8] | 0.04 |
| Falcon | UltraChat | GSM8K | `attention_plus_mlp` − base | +8.6 | [+3.1, +14.1] | 0.04 |
| Falcon | UltraChat | GSM8K | `ssm_plus_mlp` − base | +8.6 | [+3.1, +14.1] | 0.04 |

Three coherent findings come out of this, and they are not the ones in the manuscript:

1. **Cross-domain positive transfer into maths in the parallel hybrid.** Training Falcon
   on UltraChat or CodeAlpaca raises corrected GSM8K by 5–9 pp against base, across
   several placements. Robust and replicated within the data.
2. **Off-target degradation is placement-dependent, and it is the broad placements that
   degrade.** After UltraChat on Falcon, `all_eligible`, `attention_plus_mlp` and
   `mlp_only` all significantly lose HellaSwag accuracy, while `attention_only` and
   `ssm_only` do not. This is a weaker, better-supported version of the manuscript's
   fourth claim — stated as "narrow placements degrade less", not "attention-only wins".
3. **In the sequential hybrid the ordering is reversed and MLP-only is the benign one.**
   After GSM8K training on Qwen, `mlp_only` *improves* HellaSwag by 4.3 pp while
   `softmax_only` sits 6.1 pp below it. Attention-only is the most damaging single-component
   placement here, the opposite of the paper's thesis.

Every surviving effect is measured on log-likelihood benchmarks or on cross-domain
transfer — quantities the extraction bug did not touch and where n is 256–512. The
paper's viable contribution is about **where adaptation does collateral damage**, not
about which placement maximises on-target gain. That reframing is also cheaper to
defend: it needs no new training runs, only a corrected evaluation of the adapters that
already exist.

---

## 5. The design is underpowered by roughly an order of magnitude

From the corrected per-instance labels, the standard deviation of the paired per-item
difference between two conditions is about 0.46. For 80% power at alpha 0.05
(`power_analysis_gsm8k.csv`):

| To detect | Items needed | Current n |
|---|---|---|
| 3 pp | ~1,850 | 128 (Qwen) / 256 (Falcon) |
| 5 pp | ~665 | same |
| 10 pp | ~167 | same |

The true effects are 2–7 pp. The current subsets give a CI half-width of 6.8 pp (median),
so the study could only ever have detected effects it does not have. This is the single
most important number for planning the new runs: **GSM8K must be evaluated on the full
1,319-item test split**, which buys detection down to roughly 3.5–4 pp, and multiple
seeds must be averaged to go below that.

---

## 6. What to fix before spending GPU money

**Evaluator**

1. Pass stop sequences to `generate_text` (`\nQuestion:`, the system-prompt prefix, EOS) or take the first `####`. Keep both and assert they agree.
2. Set `HF_ALLOW_CODE_EVAL=1`, or execute completions directly as `rescore_humaneval_per_instance.py` does, and apply the standard HumanEval stop list.
3. Batch the evaluation. `score_continuation_logprob` runs one forward pass per (example, choice) and `generate_text` one sequence at a time — roughly 4,000 sequential forwards plus 256 sequential generations per evaluation. This, not training, is why runs took 0.7–3.2 h. Batching at 16–32, or serving through vLLM, should cut evaluation by an order of magnitude and makes full test splits affordable.
4. Use full test splits: GSM8K 1,319, ARC-Challenge 1,172, HellaSwag and MMLU at 2,048 or full. Moving to `lm-eval-harness` gets all of this plus comparability with published numbers, at the cost of rewriting the harness.

**Design**

5. Harmonise the conditions before comparing topologies. Audited from `results/discovery/*__condition_specs.json`:

   | Model | Single-component sum | Broad condition | Unaccounted |
   |---|---|---|---|
   | Qwen (`all_layers`) | 10,822,656 | 10,822,656 | 0 (0.0%) |
   | Falcon (`all_eligible`) | 10,040,832 | 11,466,496 | 1,425,664 (12.4%) |

   In Qwen the three single-component conditions partition the broad condition exactly.
   In Falcon they do not: `all_eligible` additionally carries `o_proj` on all 36 blocks
   and `lm_head`, so 12.4% of its adapter budget sits in modules that appear in **no**
   single-component condition. `attention_only` is therefore not the attention part of
   `all_eligible` — it is q, k, v without the output projection, while `all_eligible`
   has it. `ssm_only` covers only `in_proj` (one module per block) against five GDN
   projections per layer in Qwen. The headline "sequential vs parallel" contrast is
   partly a contrast between a clean decomposition and a ragged one, and `lm_head`
   adaptation changes the output distribution directly, which is not a placement effect
   at all.
6. Make the seeds real. `build_sft_config` passes no `seed`/`data_seed`, so the HF Trainer resets to 42 and only the LoRA initialisation varies across "seeds". Pass `seed=seed, data_seed=seed`.
7. Add the missing controls: full fine-tuning, a parameter-matched random-module placement, and rank scaled so that every condition has the same trainable-parameter budget. Without them "accuracy per parameter" is not interpretable.

**Then, and only then**, run the multi-seed and rank-ablation grid. Note that the runner as written also reads `train_dataset_key` where the checkpoints store `train_dataset`, re-runs `pip install` from git `main` on import, and lets the base Section 5/6 cell recompute the manuscript tables while rank-8/32 runs are present in the same checkpoint store, which contaminates heatmaps, the bootstrap table and the recipe table.

---

## 7. Files produced

| File | Contents |
|---|---|
| `all_results_corrected.csv` | One row per model × condition × domain × benchmark, reported and corrected accuracy, deltas vs base, count of labels changed |
| `result_matrix_reported.csv` / `result_matrix_corrected.csv` | Side-by-side accuracy matrices |
| `off_target_corrected.csv` | On-target and off-target deltas, both versions |
| `key_bootstrap_corrected.csv` | Paired bootstrap on corrected labels |
| `power_analysis_gsm8k.csv` | Per-pair SD, achieved CI half-width, required n |
| `humaneval_rescore_summary.json` | pass@1 recovered by executing saved completions |
| `eval_details_corrected/*.jsonl` | Corrected per-instance labels, both GSM8K and HumanEval |
