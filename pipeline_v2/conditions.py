"""Harmonised placement conditions and the two missing controls.

Three problems with the original condition sets are fixed here.

1. The two models' conditions were not comparable decompositions. For Qwen the three
   single-component conditions partitioned the broad condition exactly
   (1.08 + 4.43 + 5.31 = 10.82M). For Falcon they did not: `attention_only` covered
   q, k, v but not `o_proj`, while `all_eligible` covered `o_proj` *and* `lm_head`,
   leaving 1.43M (12.4% of the broad budget) in modules present in no single-component
   condition. Any cross-topology comparison therefore confounded topology with adapter
   coverage.

2. There was no budget-matched control. Every condition ran at rank 16, so conditions
   differed in trainable-parameter count as well as in placement, and a difference
   between them could not be attributed to placement.

3. There was no null. Without a random placement at the same parameter budget there is
   no baseline against which a principled placement can be said to do anything.

The fix for (1) is to give Falcon's attention condition its output projection and to
drop `lm_head` from the broad condition in both models, so that in both models
single components partition the broad condition exactly.

One asymmetry is left in place deliberately. Falcon's `ssm_only` covers `in_proj` only,
because the official Falcon-H1 recipe excludes `conv1d` and `out_proj` from adaptation,
while Qwen's `gdn_only` covers five projections including `out_proj`. That is a
property of the two architectures' published guidance, not a choice, and it is reported
rather than engineered away.

`python conditions.py --self-test` checks the invariants against the real manifests.
"""
from __future__ import annotations


def _results_dir(*parts):
    """Locate `results/<parts>` from either the repository or its parent working copy."""
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for prefix in ("", "github"):
            cand = os.path.join(here, prefix, "results", *parts) if prefix                 else os.path.join(here, "results", *parts)
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", *parts)

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Component labels differ between the two manifests.
RECURRENT_COMPONENTS = ("ssm", "gdn")
ATTENTION_COMPONENTS = ("attention", "softmax_attn")
MLP_COMPONENTS = ("mlp",)
# Never adapted: it is not a sequence-mixing component, it changes the output
# distribution directly, and including it in one model but not the other was the bug.
NEVER_ADAPT = ("lm_head", "embedding", "norm", "other")


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eligible_records(manifest: Dict[str, Any],
                     exclude: Sequence[str] = NEVER_ADAPT) -> List[Dict[str, Any]]:
    return [r for r in manifest["records"]
            if r.get("lora_eligible") and r.get("component") not in exclude]


def lora_params(record: Dict[str, Any], rank: int) -> int:
    """Trainable parameters LoRA adds to one linear module.

    `direct_param_shapes["weight"]` is [out_features, in_features] (torch convention),
    and LoRA adds A (r x in) and B (out x r), so r*(in + out).
    """
    shapes = record.get("direct_param_shapes") or {}
    w = shapes.get("weight")
    if not w or len(w) != 2:
        return 0
    out_f, in_f = int(w[0]), int(w[1])
    return rank * (in_f + out_f)


def spec_params(records: Sequence[Dict[str, Any]], rank: int) -> int:
    return sum(lora_params(r, rank) for r in records)


def component_of(record: Dict[str, Any]) -> Optional[str]:
    c = record.get("component")
    if c in RECURRENT_COMPONENTS:
        return "recurrent"
    if c in ATTENTION_COMPONENTS:
        return "attention"
    if c in MLP_COMPONENTS:
        return "mlp"
    return None


# ───────────────────────────────────────────────────── harmonised conditions

def harmonised_specs(manifest: Dict[str, Any], rank: int = 16) -> Dict[str, Dict[str, Any]]:
    """Six conditions in which single components partition the broad condition.

    Names are kept model-neutral so the two models' tables line up row by row.
    """
    els = eligible_records(manifest)
    buckets: Dict[str, List[Dict[str, Any]]] = {"attention": [], "recurrent": [], "mlp": []}
    for r in els:
        c = component_of(r)
        if c:
            buckets[c].append(r)

    def spec(recs: Sequence[Dict[str, Any]], note: str) -> Dict[str, Any]:
        recs = sorted(recs, key=lambda r: r["full_name"])
        return {"target_modules": [r["full_name"] for r in recs],
                "n_modules": len(recs),
                "trainable_params_at_rank": {str(rank): spec_params(recs, rank)},
                "notes": note}

    a, rec, m = buckets["attention"], buckets["recurrent"], buckets["mlp"]
    return {
        "attention_only": spec(a, "all attention projections including the output projection"),
        "recurrent_only": spec(rec, "all eligible recurrent projections"),
        "mlp_only": spec(m, "all MLP projections"),
        "attention_plus_mlp": spec(a + m, "attention and MLP, recurrent frozen"),
        "recurrent_plus_mlp": spec(rec + m, "recurrent and MLP, attention frozen"),
        "all_components": spec(a + rec + m,
                               "every eligible component module; excludes lm_head"),
    }


def partition_check(specs: Dict[str, Dict[str, Any]], rank: int = 16) -> Dict[str, Any]:
    """The invariant the original conditions violated for Falcon."""
    k = str(rank)
    singles = sum(specs[c]["trainable_params_at_rank"][k]
                  for c in ("attention_only", "recurrent_only", "mlp_only"))
    broad = specs["all_components"]["trainable_params_at_rank"][k]
    n_singles = sum(specs[c]["n_modules"] for c in ("attention_only", "recurrent_only", "mlp_only"))
    return {"sum_of_singles": singles, "broad": broad,
            "unaccounted": broad - singles,
            "modules_singles": n_singles, "modules_broad": specs["all_components"]["n_modules"],
            "partitions": singles == broad and n_singles == specs["all_components"]["n_modules"]}


# ────────────────────────────────────────────────────────────── the controls

def budget_matched_rank(manifest: Dict[str, Any], spec: Dict[str, Any],
                        target_params: int, min_rank: int = 1,
                        max_rank: int = 512) -> int:
    """Rank at which `spec` costs about `target_params` trainable parameters.

    LoRA parameters are exactly linear in rank, so the ideal rank is a ratio; it is
    rounded to an integer and clamped, and the caller should report the achieved count
    rather than assume the target was hit.
    """
    by_name = {r["full_name"]: r for r in manifest["records"]}
    recs = [by_name[n] for n in spec["target_modules"] if n in by_name]
    per_rank = spec_params(recs, 1)
    if per_rank == 0:
        return min_rank
    return max(min_rank, min(max_rank, round(target_params / per_rank)))


def random_placement_spec(manifest: Dict[str, Any], target_params: int, seed: int,
                          rank: int = 16) -> Dict[str, Any]:
    """A uniformly random set of eligible modules costing about `target_params`.

    This is the null the study lacked. If a principled placement cannot beat a random
    one at the same parameter budget, then what is being measured is the budget, not
    the placement. Modules are shuffled with a seeded RNG and added while they fit, so
    the result is reproducible and never overshoots the budget.
    """
    els = eligible_records(manifest)
    rng = random.Random(seed)
    order = sorted(els, key=lambda r: r["full_name"])
    rng.shuffle(order)

    chosen: List[Dict[str, Any]] = []
    total = 0
    for r in order:
        c = lora_params(r, rank)
        if total + c <= target_params:
            chosen.append(r)
            total += c
    chosen = sorted(chosen, key=lambda r: r["full_name"])

    mix: Dict[str, int] = {}
    for r in chosen:
        mix[component_of(r) or "other"] = mix.get(component_of(r) or "other", 0) + 1
    return {"target_modules": [r["full_name"] for r in chosen],
            "n_modules": len(chosen),
            "trainable_params_at_rank": {str(rank): total},
            "target_params": target_params,
            "component_mix": mix,
            "seed": seed,
            "notes": f"uniformly random eligible modules, budget-matched, seed {seed}"}


def choose_budget(manifest: Dict[str, Any], rank: int = 16,
                  mode: str = "broad") -> int:
    """The parameter budget every control is matched to.

    `mode="broad"` pegs to the broad condition and scales the narrow ones *up*;
    `mode="min"` pegs to the smallest single component and scales the broad one *down*.

    Prefer "broad". LoRA parameters are linear in rank but rank is an integer, so the
    achievable granularity is one rank step. Scaling down forces the broad condition to
    rank 1-2, where one step is a 50-100% change in budget and the match cannot be made
    (on Qwen the best attainable is 25% over). Scaling up puts every condition at rank
    30-160, where one step is a percent or two and all conditions match closely.
    """
    specs = harmonised_specs(manifest, rank)
    k = str(rank)
    if mode == "broad":
        return specs["all_components"]["trainable_params_at_rank"][k]
    return min(specs[c]["trainable_params_at_rank"][k]
               for c in ("attention_only", "recurrent_only", "mlp_only"))


def build_control_suite(manifest: Dict[str, Any], rank: int = 16,
                        seeds: Sequence[int] = (3407, 1337, 2026),
                        budget_mode: str = "broad") -> Dict[str, Dict[str, Any]]:
    """The two controls the study lacked. They answer different questions.

    **Budget-matched** puts every placement at one common parameter budget by scaling
    its rank, so a difference between placements cannot be a difference in budget.
    The budget is the broad condition's, and the narrow placements scale up to it.

    **Random placement** pairs each *single-component* condition with a uniformly random
    set of modules costing the same. It is the null: if a principled narrow placement
    does not beat a random one of the same size, what is being measured is the budget.
    It is deliberately not generated at the broad budget, because the broad condition
    already contains every eligible module and a "random" subset of everything is
    everything.
    """
    specs = harmonised_specs(manifest, rank)
    by_name = {rec["full_name"]: rec for rec in manifest["records"]}
    total = specs["all_components"]["trainable_params_at_rank"][str(rank)]
    budget = choose_budget(manifest, rank, budget_mode)

    out: Dict[str, Dict[str, Any]] = {}
    for name in ("attention_only", "recurrent_only", "mlp_only", "all_components"):
        r = budget_matched_rank(manifest, specs[name], budget)
        recs = [by_name[n] for n in specs[name]["target_modules"]]
        out[f"budget_matched__{name}"] = {
            "target_modules": specs[name]["target_modules"],
            "n_modules": specs[name]["n_modules"],
            "rank": r,
            "trainable_params_at_rank": {str(r): spec_params(recs, r)},
            "target_params": budget,
            "notes": (f"{name} with rank scaled to the common budget of {budget:,} "
                      f"parameters; isolates placement from parameter count")}

    for comp in ("attention_only", "recurrent_only", "mlp_only"):
        comp_budget = specs[comp]["trainable_params_at_rank"][str(rank)]
        if comp_budget >= total:
            continue            # nothing left to randomise against
        short = comp.replace("_only", "")
        for s in seeds:
            spec = random_placement_spec(manifest, comp_budget, s, rank)
            spec["notes"] = (f"uniformly random eligible modules at the {short} budget "
                             f"({comp_budget:,} params), seed {s}; the null for {comp}")
            spec["controls_for"] = comp
            out[f"random_placement__{short}__seed{s}"] = spec
    return out


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test(discovery_dir: str) -> int:
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    for mk, old_attn, old_broad in [("falcon_h1_0_5b_base", 2_211_840, 11_466_496),
                                    ("qwen3_5_0_8b_base", 1_081_344, 10_822_656)]:
        path = os.path.join(discovery_dir, f"{mk}__manifest.json")
        if not os.path.exists(path):
            print(f"  SKIP  {mk}: manifest not found")
            continue
        man = load_manifest(path)
        specs = harmonised_specs(man, 16)
        pc = partition_check(specs, 16)
        print(f"\n{mk}")
        for name, s in specs.items():
            print(f"    {name:22s} {s['n_modules']:4d} modules  "
                  f"{s['trainable_params_at_rank']['16']:>12,} params")
        check("single components partition the broad condition", pc["partitions"],
              f"singles {pc['sum_of_singles']:,} vs broad {pc['broad']:,}, "
              f"unaccounted {pc['unaccounted']:,}")
        check("lm_head is excluded everywhere",
              not any("lm_head" in n for s in specs.values() for n in s["target_modules"]))

        # The Falcon fix must actually add the output projection.
        if mk.startswith("falcon"):
            attn = specs["attention_only"]
            check("Falcon attention now includes o_proj",
                  any(n.endswith("o_proj") for n in attn["target_modules"]))
            check("Falcon attention grew relative to the original condition",
                  attn["trainable_params_at_rank"]["16"] > old_attn,
                  f"{attn['trainable_params_at_rank']['16']:,} vs {old_attn:,} before")
            check("the broad condition shrank once lm_head was dropped",
                  specs["all_components"]["trainable_params_at_rank"]["16"] < old_broad,
                  f"{specs['all_components']['trainable_params_at_rank']['16']:,} vs {old_broad:,}")
        else:
            check("Qwen attention already had o_proj, and is unchanged in size",
                  specs["attention_only"]["trainable_params_at_rank"]["16"] == old_attn,
                  f"{specs['attention_only']['trainable_params_at_rank']['16']:,} vs {old_attn:,}")

        # Controls, both budget directions, to show why "broad" is the right peg
        worst_by_mode = {}
        for mode in ("min", "broad"):
            c = build_control_suite(man, 16, seeds=(3407, 1337), budget_mode=mode)
            b = choose_budget(man, 16, mode)
            worst_by_mode[mode] = max(
                abs(list(s["trainable_params_at_rank"].values())[0] - b) / b
                for s in c.values())
        print(f"    worst budget deviation: scaling down {worst_by_mode['min']:.1%}, "
              f"scaling up {worst_by_mode['broad']:.1%}")

        ctrl = build_control_suite(man, 16, seeds=(3407, 1337), budget_mode="broad")
        budget = choose_budget(man, 16, "broad")
        print(f"    budget for the controls: {budget:,} params")
        for name, s in ctrl.items():
            got = list(s["trainable_params_at_rank"].values())[0]
            print(f"    {name:34s} {s['n_modules']:4d} modules  {got:>10,} params"
                  + (f"  rank {s['rank']}" if "rank" in s else "")
                  + (f"  mix {s['component_mix']}" if "component_mix" in s else ""))
        bm = {k: v for k, v in ctrl.items() if k.startswith("budget_matched__")}
        rp = {k: v for k, v in ctrl.items() if k.startswith("random_placement__")}
        worst_bm = max(abs(list(s["trainable_params_at_rank"].values())[0] - budget) / budget
                       for s in bm.values())
        check("budget-matched conditions land within 5% of the common budget",
              worst_bm < 0.05, f"worst deviation {worst_bm:.1%}")
        check("scaling up beats scaling down, which is why it is the default",
              worst_by_mode["broad"] <= worst_by_mode["min"],
              f"{worst_by_mode['broad']:.1%} vs {worst_by_mode['min']:.1%}")

        a3407 = rp["random_placement__attention__seed3407"]
        a1337 = rp["random_placement__attention__seed1337"]
        check("a random placement is a strict subset, never the whole model",
              all(s["n_modules"] < specs["all_components"]["n_modules"] for s in rp.values()),
              str({k: s["n_modules"] for k, s in rp.items()}))
        check("random placements differ between seeds",
              a3407["target_modules"] != a1337["target_modules"])
        check("random placement is reproducible for a fixed seed",
              random_placement_spec(man, a3407["target_params"], 3407)["target_modules"]
              == a3407["target_modules"])
        check("random placement mixes components rather than picking one",
              len(a3407["component_mix"]) >= 2, str(a3407["component_mix"]))
        check("each random placement matches the budget of the condition it controls",
              all(abs(list(s["trainable_params_at_rank"].values())[0] - s["target_params"])
                  / s["target_params"] < 0.05 for s in rp.values()))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--discovery-dir", default=_results_dir("discovery"))
    a = ap.parse_args()
    raise SystemExit(_self_test(a.discovery_dir) if a.self_test else print(__doc__))
