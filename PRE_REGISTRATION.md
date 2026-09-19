# Pre-registration: component-type LoRA placement in hybrid language models

**Written 2026-09-19, before any run of the follow-up campaign.** The git commit that
adds this file predates every checkpoint the campaign produces, which is the point: a
paper that has already been corrected once cannot ask to be believed on decisions made
after seeing the data.

Everything below is fixed now. Section 8 is the only place where departures from it may
be recorded, and each entry must say what changed, when, and why.

---

## 1. What is being asked

Does the *component type* a LoRA adapter is placed on change the outcome of adaptation
in hybrid language models, beyond what the trainable-parameter budget alone explains?

Two sub-questions, which the study separates because v1 conflated them:

- **On-target.** Does placement change accuracy on the task that was trained?
- **Off-target.** Does placement change accuracy on tasks that were *not* trained?

## 2. What would count as an answer

The decisive comparison is against a **random placement at the same parameter budget**,
not against another principled placement. Stated in advance:

| Outcome | Reading |
|---|---|
| A principled placement beats its budget-matched random control, and the effect holds across seeds | Component type matters. This is the positive result. |
| A principled placement does not separate from its random control | Component type does not matter at this scale; what v1 measured was budget and evaluation protocol. **This is a publishable result and will be reported as the headline.** |
| Placements separate on off-target but not on-target benchmarks | The contribution is about collateral damage, not about target-task gain. |

We commit to reporting the second outcome as prominently as the first. The companion
decoding study was accepted at an Elsevier journal with a negative headline, and a
reviewer stated that the negative was worth more than a manufactured positive; there is
no incentive here to bury one.

## 3. Design, fixed now

**Models.** Anchors `Qwen3.5-0.8B-Base` (sequential) and `Falcon-H1-0.5B-Base`
(parallel). Breadth, budget permitting and in this order of priority: a third hybrid
family at ~1.2B, a pure-Transformer control at ~0.5B, then scale steps within each
anchor family. Every model must pass the classification guard in
`pipeline_v2/models.py` before it is trained on.

**Conditions.** The six harmonised conditions in `pipeline_v2/conditions.py`:
`attention_only`, `recurrent_only`, `mlp_only`, `attention_plus_mlp`,
`recurrent_plus_mlp`, `all_components`. Single components partition the broad condition
exactly in every model; `lm_head` is excluded everywhere.

**Controls.** Budget-matched variants of all four primary conditions, and random
placements at each single-component budget, three seeds each.

**Seeds.** 3407, 1337, 2026. Both `seed` and `data_seed` are set, so data order and
dropout vary and not only the adapter initialisation.

**Domains.** GSM8K-train and UltraChat. CodeAlpaca is dropped from the primary design;
it carried no surviving effect in the corrected re-analysis and doubles the bill.

**Evaluation.** Full test splits, not subsets: GSM8K 1,319, ARC-Challenge 1,172,
HellaSwag and MMLU capped at 2,048, HumanEval 164. Subsets are built as supersets of
every subset already released, so prior results stay paired with new ones. Scoring uses
`evaluator_v2/evaluator.py`: stop sequences, first-`####` extraction, HumanEval executed.

**Primary outcome.** Accuracy on the benchmark matching the training domain (GSM8K for
GSM8K-train; the mean over MMLU, ARC-Challenge and HellaSwag for UltraChat).
**Primary contrast.** Each single-component condition against its budget-matched random
control, on the primary outcome, pooled over seeds.

## 4. Analysis plan, fixed now

Implemented in `pipeline_v2/aggregate.py` and frozen before the first run.

- Paired percentile bootstrap over items, 10,000 resamples, 95% intervals, bootstrap
  seed 3407, on the per-item mean across seeds. Items are the clustering unit.
- Items are matched by **benchmark index**, never by position within a subset.
- Across-seed standard deviation of the per-seed difference is reported for every
  comparison, with the sign-consistency flag. An effect whose sign flips across seeds is
  reported as seed-sensitive regardless of its interval.
- Holm correction within each (model, domain, benchmark) family, over every comparison
  computed in that family, not a subset chosen afterwards.
- An effect is called **detected** only if its Holm-adjusted interval excludes zero
  *and* its sign is consistent across all three seeds. Anything else is reported as not
  detected, with the interval, never as absence of an effect.

## 5. Out-of-sample predictions

The corrected single-seed re-analysis gives point estimates. They are recorded here so
the campaign is a test of them rather than a description of whatever comes back. These
conditions are unchanged by harmonisation, so the predictions transfer directly.

| Comparison (GSM8K-trained, GSM8K) | Predicted (pp) |
|---|---|
| Qwen `attention_only` vs base | −2.3 |
| Qwen `recurrent_only` vs base | +1.6 |
| Qwen `attention_only` vs `all_components` | −9.4 |
| Qwen `recurrent_only` vs `attention_only` | +3.9 |
| Qwen `attention_plus_mlp` vs `mlp_only` | −1.6 |
| Falcon `recurrent_only` vs base | +2.0 |

Predictions are **not** recorded for Falcon `attention_only` or `all_components`,
because harmonisation changed those conditions: attention gains its output projection
and the broad condition loses `lm_head`. Carrying the old estimate across would be
comparing two different things.

Stated in advance: these are single-seed estimates on 128 and 256 items, so the
campaign's intervals will be narrower and movement of a few points is expected. A
prediction that misses by more than its own v1 interval will be reported as a miss.

## 6. Sample size and power

From the corrected labels, the standard deviation of the paired per-item difference on
GSM8K is 0.45. The smallest difference resolvable at 80% power, α = 0.05 two-sided:

| Design | Resolvable |
|---|---|
| 256 items, 1 seed (what v1 had) | 5.6 pp |
| 1,319 items, 1 seed | 2.5 pp |
| 1,319 items, 3 seeds | 1.4 pp |

The corrected effects sit at 2–7 pp, so v1 could not have detected most of them and
this design can. No interim analysis will be used to decide whether to add seeds.

## 7. Stopping and budget

The campaign stops when the planned runs are complete or the GPU budget is exhausted,
whichever comes first; the budget guard is in `pipeline_v2/run_campaign.py`. If the
budget binds, runs are dropped from the **end** of the priority order in section 3, and
the shortfall is reported. Runs are never dropped because of what they showed.

## 8. Deviations

Any departure from sections 1–7 is recorded here with date and reason, before the
affected analysis is run.

*(none yet)*
