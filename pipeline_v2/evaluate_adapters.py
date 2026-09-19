"""Batched re-evaluation of trained adapters on full test splits.

The runner can patch the notebook's answer extractor, and that fixes correctness, but
it cannot make the notebook's evaluation fast: `generate_text` produces one sequence at
a time with no stop criterion, and `score_continuation_logprob` runs one forward pass
per (example, answer option). On the released subsets that is about 0.6 GPU-hours per
evaluation; on full test splits it would be roughly 2.5, which turns re-evaluating the
36 existing adapters from a four-euro job into a forty-euro one.

This driver therefore replaces the evaluation path rather than patching it. It borrows
the notebook's prompt construction, so prompts are identical to the ones already used,
and routes everything else through `evaluator_v2`: stop sequences, first-marker
extraction, executed HumanEval, and batching everywhere.

Per-instance outputs carry the benchmark index as `example_id`, so new results pair
with the released ones through `results/eval_indices/`.

    python evaluate_adapters.py --self-test          # offline, no GPU
    python evaluate_adapters.py --plan               # what would run, and the cost
    python evaluate_adapters.py --run --project-dir /workspace/lora-placement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "evaluator_v2")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Full-split targets. MMLU and HellaSwag are capped: beyond a couple of thousand items
# the interval stops narrowing fast enough to be worth the wall clock.
TARGETS = {"gsm8k": 1319, "arc_challenge": 1172, "hellaswag": 2048,
           "mmlu": 2048, "humaneval": 164}
MC_BENCHMARKS = ("mmlu", "arc_challenge", "hellaswag")


# Seconds per item on an L4 for a sub-1B model. Generation: a stop-terminated GSM8K
# answer runs to roughly 100 new tokens, and batching at 16 gives on the order of
# 600 tokens/s across the batch, so ~0.17 s/item; 0.25 is carried as the conservative
# figure. Multiple choice needs one short forward pass per option and comes out about
# six times cheaper per item. These are estimates and the first real run replaces them,
# which is what the campaign's budget guard exists for.
GEN_SECONDS_PER_ITEM = 0.25
MC_SECONDS_PER_ITEM = GEN_SECONDS_PER_ITEM / 6

# What the unbatched notebook path costs, measured from the original campaign: about
# 0.64 GPU-hours per evaluation on the released 128-256 item subsets.
UNBATCHED_HOURS_PER_EVAL_ON_SUBSETS = 0.64
RELEASED_SUBSET_ITEMS = 128 + 256 + 299 + 512 + 512


def plan_evaluations(adapters: Sequence[str], benchmarks: Sequence[str],
                     gen_seconds: float = GEN_SECONDS_PER_ITEM) -> Dict[str, Any]:
    """Cost of the sweep, and of doing the same thing unbatched, for comparison."""
    items = sum(TARGETS[b] for b in benchmarks)
    gen_items = sum(TARGETS[b] for b in benchmarks if b in ("gsm8k", "humaneval"))
    mc_items = items - gen_items
    hours = len(adapters) * (gen_items * gen_seconds
                             + mc_items * gen_seconds / 6) / 3600
    # the notebook path scales with items at its measured per-item rate
    unbatched = (len(adapters) * UNBATCHED_HOURS_PER_EVAL_ON_SUBSETS
                 * items / RELEASED_SUBSET_ITEMS)
    return {"adapters": len(adapters), "items_per_adapter": items,
            "gpu_hours": hours, "eur": hours * 0.40,
            "unbatched_gpu_hours": unbatched, "unbatched_eur": unbatched * 0.40}


def evaluate_one(model, tokenizer, benchmarks: Sequence[str], nb: Dict[str, Any],
                 index_dir: str, out_dir: str, label: str,
                 batch_size: int = 16, progress: bool = True) -> Dict[str, Any]:
    """Evaluate one loaded model on `benchmarks`, writing per-instance JSONL.

    `nb` is the notebook namespace: the prompt builders come from there so the prompts
    match the released runs exactly. Nothing else is taken from it.
    """
    import evaluator as ev
    import eval_data as ed

    summary: Dict[str, Any] = {}
    for bench in benchmarks:
        ds = nb["get_benchmark_split"](bench, ev_split(bench, nb))
        idx = ed.nested_indices(bench, len(ds), TARGETS[bench], seed=nb.get("SEED", 3407),
                                index_dir=index_dir)
        ed.write_index_file(index_dir, bench, idx)

        t0 = time.time()
        if bench in MC_BENCHMARKS:
            shots = nb["get_shot_pool"](bench)
            examples = ed.build_mc_examples(
                ds, idx, bench,
                prompt_fn=lambda e, b=bench: nb["build_fewshot_prompt"](tokenizer, b, e, shots),
                mc_fn=nb["get_mc_choices"], labels_fn=nb["choice_labels"])
            res = ev.evaluate_multiple_choice(model, tokenizer, examples, bench,
                                              batch_size=batch_size)
        elif bench == "gsm8k":
            shots = nb["get_shot_pool"]("gsm8k")
            examples = ed.build_gsm8k_examples(
                ds, idx,
                prompt_fn=lambda e: nb["build_fewshot_prompt"](tokenizer, "gsm8k", e, shots),
                answer_fn=nb["gsm8k_answer"])
            res = ev.evaluate_gsm8k(model, tokenizer, examples, batch_size=batch_size)
        elif bench == "humaneval":
            problems = ed.build_humaneval_problems(ds, idx)
            res = ev.evaluate_humaneval(model, tokenizer, problems,
                                        batch_size=max(1, batch_size // 2))
        else:
            raise ValueError(f"unknown benchmark {bench}")

        ev.write_jsonl(os.path.join(out_dir, f"{label}__{bench}.jsonl"), res.rows)
        summary[bench] = res.summary()
        summary[bench]["seconds"] = round(time.time() - t0, 1)
        if progress:
            print(f"    {bench:14s} {res.accuracy:.4f}  n={res.n_eval:5d}  "
                  f"{summary[bench]['seconds']:6.1f}s")
    return summary


def ev_split(bench: str, nb: Dict[str, Any]) -> str:
    return nb["BENCHMARK_REGISTRY"][bench]["eval_split"]


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test() -> int:
    failures: List[str] = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("\n1. the sweep is planned against full splits, not the released subsets")
    benches = ["gsm8k", "mmlu", "arc_challenge", "hellaswag"]
    p = plan_evaluations(["a"] * 38, benches)
    check("GSM8K is the full test split", TARGETS["gsm8k"] == 1319)
    check("ARC-Challenge is the full validation split", TARGETS["arc_challenge"] == 1172)
    check("every target is at least as large as the released subset",
          TARGETS["gsm8k"] >= 256 and TARGETS["mmlu"] >= 512
          and TARGETS["hellaswag"] >= 512 and TARGETS["arc_challenge"] >= 299)
    print(f"    38 adapters x {p['items_per_adapter']:,} items")
    print(f"      batched   {p['gpu_hours']:5.1f} GPU-h  EUR {p['eur']:5.1f}")
    print(f"      unbatched {p['unbatched_gpu_hours']:5.1f} GPU-h  EUR "
          f"{p['unbatched_eur']:5.1f}   <- the notebook path")
    # A cost that rounds to zero would mean the model is wrong, not that the job is free.
    check("the batched estimate is in a physically plausible band",
          2.0 < p["gpu_hours"] < 20.0, f"{p['gpu_hours']:.1f} GPU-h")
    check("batching is worth an order of magnitude, which is why this driver exists",
          p["unbatched_gpu_hours"] / p["gpu_hours"] > 8,
          f"{p['unbatched_gpu_hours']/p['gpu_hours']:.1f}x")
    check("the unbatched path on full splits is what a forty-euro mistake looks like",
          p["unbatched_eur"] > 25, f"EUR {p['unbatched_eur']:.0f}")

    print("\n2. the driver calls the batched evaluator, not the notebook's")
    src = open(os.path.join(HERE, "evaluate_adapters.py"), encoding="utf-8").read()
    body = src.split("def _self_test")[0]
    check("multiple choice goes through evaluator_v2",
          "ev.evaluate_multiple_choice" in body)
    check("GSM8K goes through evaluator_v2", "ev.evaluate_gsm8k" in body)
    check("HumanEval goes through evaluator_v2", "ev.evaluate_humaneval" in body)
    check("the notebook's own evaluators are never called",
          "nb[\"evaluate_gsm8k\"]" not in body
          and "nb[\"evaluate_model_suite\"]" not in body
          and "nb[\"evaluate_multiple_choice\"]" not in body)
    check("prompts still come from the notebook, so they match the released runs",
          'nb["build_fewshot_prompt"]' in body)
    check("a batch size is passed to every evaluator call",
          body.count("batch_size=") >= 3)

    print("\n3. subsets and identifiers")
    check("indices come from eval_data, so they are supersets of what was released",
          "ed.nested_indices" in body)
    check("the index file is written so the mapping is reproducible",
          "ed.write_index_file" in body)
    check("examples are built by eval_data, which keys on the benchmark index",
          "ed.build_mc_examples" in body and "ed.build_gsm8k_examples" in body)

    print("\n4. the plan is honest about what drives the cost")
    gen_only = plan_evaluations(["a"], ["gsm8k"])
    mc_only = plan_evaluations(["a"], ["hellaswag"])
    check("generation costs more per item than multiple choice",
          gen_only["gpu_hours"] / TARGETS["gsm8k"]
          > mc_only["gpu_hours"] / TARGETS["hellaswag"])
    check("cost scales with the number of adapters",
          abs(plan_evaluations(["a"] * 2, benches)["gpu_hours"]
              - 2 * plan_evaluations(["a"], benches)["gpu_hours"]) < 1e-9)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--project-dir", default="/workspace/lora-placement")
    ap.add_argument("--notebook", default=os.path.join(
        ROOT, "news-experiments", "paper3_lora_placement_reproducible.ipynb"))
    ap.add_argument("--benchmarks", nargs="+",
                    default=["gsm8k", "mmlu", "arc_challenge", "hellaswag"])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    models_dir = os.path.join(a.project_dir, "models")
    adapters = sorted(d for d in os.listdir(models_dir)
                      if os.path.isdir(os.path.join(models_dir, d, "final_adapter"))
                      ) if os.path.isdir(models_dir) else []
    p = plan_evaluations(adapters or ["?"] * 36, a.benchmarks)
    print(f"{len(adapters)} adapters found, {p['items_per_adapter']:,} items each")
    print(f"estimated {p['gpu_hours']:.1f} GPU-h, about EUR {p['eur']:.0f}")
    if not a.run:
        print("\nPass --run to execute. Nothing has been launched.")
        return 0

    import run_campaign as rc
    ns = rc.load_pipeline(a.notebook, a.project_dir)
    index_dir = os.path.join(a.project_dir, "results", "eval_indices")
    out_dir = os.path.join(a.project_dir, "results", "eval_details_v2")
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for i, name in enumerate(adapters, 1):
        print("=" * 90)
        print(f"[{i}/{len(adapters)}] {name}")
        model_key = next((k for k in ns["MODEL_REGISTRY"] if k in name), None)
        if model_key is None:
            print("    cannot infer the base model from the adapter name; skipping")
            continue
        adapter = os.path.join(models_dir, name, "final_adapter")
        model, tok = ns["load_adapter_for_eval"](model_key, adapter)
        try:
            evaluate_one(model, tok, a.benchmarks, ns, index_dir, out_dir,
                         label=f"{name}__eval_v2", batch_size=a.batch_size)
        finally:
            del model
            ns["clear_memory"]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
