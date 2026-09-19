"""Candidate models for the follow-up campaign, and a guard against mis-classification.

Adding a model means the discovery pipeline has to sort its modules into attention,
recurrent and MLP. The original pipeline does that with two hand-written classifiers,
one per family, and a fallback that branches on `model_family == "qwen3_5"` or
`"falcon_h1"`; anything else falls through to "other". A new family would therefore
produce a condition set that looks valid and contains nothing.

`classify_module` here is family-agnostic and token-based, so a new family has a chance
of working unmodified. It is a heuristic and it is written against naming conventions
rather than against the models themselves, which cannot be inspected without
downloading them. That is exactly why `assert_classification_sane` exists: it runs on
the machine, before any training, and refuses to continue if a model's modules do not
sort into three non-empty components with almost nothing left over. Spending GPU hours
on a silently empty condition is the failure this guards against, and it is the same
class of error that has already cost this project two corrections.

`python models.py --self-test` checks the classifier and the guard offline.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

# Candidates. The priority is not breadth for its own sake but breaking the confound
# the companion decoding study was criticised for: with one family per topology, a
# difference between the two anchors could be topology, recurrent mechanism, tokenizer,
# data or anything else. Granite-4.0-H-Micro resolves it because of what it shares with
# each anchor:
#
#   Falcon vs Granite-H  same mechanism (Mamba-2), different topology  -> isolates topology
#   Qwen   vs Granite-H  same topology (interleaved), different mechanism -> isolates mechanism
#   Granite-H vs Granite-dense  same family, data and recipe, one with a recurrent
#                               component and one without -> the cleanest control available
#
# `layers=None` means the guard must read the count from the model rather than assume it.
CANDIDATES: Dict[str, Dict[str, Any]] = {
    # anchors, already in the study
    "qwen3_5_0_8b_base": dict(hf_id="Qwen/Qwen3.5-0.8B-Base", family="qwen3_5",
                              topology="interleaved", mechanism="gdn", params_b=0.76,
                              layers=24, ratio="18 GDN : 6 attention", priority=0,
                              role="anchor, interleaved hybrid with linear attention"),
    "falcon_h1_0_5b_base": dict(hf_id="tiiuae/Falcon-H1-0.5B-Base", family="falcon_h1",
                                topology="parallel", mechanism="mamba2", params_b=0.52,
                                layers=36, ratio="both per block", priority=0,
                                role="anchor, parallel hybrid with Mamba-2"),
    # the confound breaker, and its own same-family control
    "granite_4_h_micro_base": dict(hf_id="ibm-granite/granite-4.0-h-micro-base",
                                   family="granite4", topology="interleaved",
                                   mechanism="mamba2", params_b=3.0, layers=40,
                                   ratio="36 Mamba-2 : 4 attention", priority=1,
                                   role="third topology point; shares mechanism with "
                                        "Falcon and topology with Qwen, so the two "
                                        "contrasts separate"),
    "granite_4_micro_base": dict(hf_id="ibm-granite/granite-4.0-micro-base",
                                 family="granite4", topology="none", mechanism="none",
                                 params_b=3.0, layers=None, ratio="dense", priority=1,
                                 role="pure-Transformer control from the SAME family, "
                                      "data and recipe as granite_4_h_micro_base"),
    # scale axis, family held fixed
    "falcon_h1_1_5b_base": dict(hf_id="tiiuae/Falcon-H1-1.5B-Base", family="falcon_h1",
                                topology="parallel", mechanism="mamba2", params_b=1.5,
                                layers=24, ratio="both per block", priority=2,
                                role="scale step within the parallel family"),
    "qwen3_5_2b_base": dict(hf_id="Qwen/Qwen3.5-2B-Base", family="qwen3_5",
                            topology="interleaved", mechanism="gdn", params_b=2.0,
                            layers=None, ratio=None, priority=2,
                            role="scale step within the interleaved family"),
}

# Considered and not selected. Recorded so the choice is auditable rather than implied.
REJECTED: Dict[str, str] = {
    "Zyphra/Zamba2-1.2B":
        "Attention is a single SHARED block reused across depth, and the architecture "
        "already carries its own LoRA projectors for depth specialisation. Placing our "
        "LoRA on top of built-in LoRA breaks both the parameter accounting and the "
        "meaning of 'attention_only', which elsewhere is one attention module per layer. "
        "The model card also calls its HuggingFace implementation temporary.",
    "LiquidAI/LFM2-1.2B":
        "Short-convolution mixer rather than a recurrent state, and no base (non-instruct) "
        "checkpoint, so it matches neither the component taxonomy nor the base-model design.",
    "nvidia/Nemotron-H-8B":
        "Smallest hybrid in the family is 8B, which does not train on a 24 GB card at "
        "this sequence length.",
    "ibm-granite/granite-4.0-h-tiny-base":
        "Mixture-of-experts, so the MLP component is not one dense block per layer and "
        "the MLP condition is not comparable with the other models.",
    "Qwen/Qwen2.5-0.5B":
        "Viable pure-Transformer control, but superseded by granite-4.0-micro-base, which "
        "holds family, data and recipe fixed against its own hybrid sibling instead of "
        "varying all three.",
}

# Token sets, most specific first. Order matters: a Mamba block often contains a module
# whose name mentions projections that would otherwise read as attention.
RECURRENT_TOKENS = ("mamba", "ssm", "state_space", "gated_delta", "gateddelta",
                    "deltanet", "linear_attn", "linearattention", "gla", "rwkv",
                    "retention", "short_conv", "shortconv", "conv_mixer")
ATTENTION_TOKENS = ("self_attn", "attention", "attn")
MLP_TOKENS = ("mlp", "ffn", "feed_forward", "feedforward", "moe", "experts")
SKIP_TOKENS = ("embed", "wte", "lm_head", "norm", "rotary", "router", "gate_up_router")


def classify_module(full_name: str, class_name: str = "",
                    layer_type: Optional[str] = None) -> str:
    """Sort one module into attention / recurrent / mlp / skip / other.

    `layer_type` disambiguates the sequential-hybrid case, where attention and the
    recurrent mixer can both live under `self_attn` and only the layer's declared type
    says which one this is.
    """
    f, c = full_name.lower(), (class_name or "").lower()
    lt = (layer_type or "").lower()
    blob = f + " " + c

    if any(t in f for t in SKIP_TOKENS) or "norm" in c:
        return "skip"
    if any(t in blob for t in MLP_TOKENS):
        return "mlp"
    # A declared layer type beats the module name, which is how Qwen3.5 packs two
    # different mixers under the same attribute.
    if lt:
        if "linear" in lt or "delta" in lt or "recurrent" in lt or "mamba" in lt:
            if any(t in blob for t in ATTENTION_TOKENS + RECURRENT_TOKENS):
                return "recurrent"
        if "full" in lt or "attention" in lt or "softmax" in lt:
            if any(t in blob for t in ATTENTION_TOKENS):
                return "attention"
    if any(t in blob for t in RECURRENT_TOKENS):
        return "recurrent"
    if any(t in blob for t in ATTENTION_TOKENS):
        return "attention"
    return "other"


def classification_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per component over the modules that carry trainable weights."""
    counts: Dict[str, int] = {}
    for r in records:
        if not r.get("direct_param_count"):
            continue
        comp = classify_module(r.get("full_name", ""), r.get("class_name", ""),
                               r.get("layer_type"))
        counts[comp] = counts.get(comp, 0) + 1
    considered = sum(v for k, v in counts.items() if k != "skip")
    return {"counts": counts, "considered": considered,
            "other_fraction": (counts.get("other", 0) / considered) if considered else 1.0}


def assert_classification_sane(report: Dict[str, Any], model_key: str,
                               require_recurrent: bool = True,
                               max_other_fraction: float = 0.10) -> None:
    """Refuse to proceed on a model whose modules did not sort cleanly.

    Called on the machine before training. A component with zero modules means the
    condition built from it would train nothing while reporting success, which is the
    outcome this exists to prevent.
    """
    c = report["counts"]
    problems: List[str] = []
    for comp in ("attention", "mlp"):
        if c.get(comp, 0) == 0:
            problems.append(f"no modules classified as {comp}")
    if require_recurrent and c.get("recurrent", 0) == 0:
        problems.append("no modules classified as recurrent, but this model is "
                        "registered as a hybrid")
    if report["other_fraction"] > max_other_fraction:
        problems.append(f"{report['other_fraction']:.0%} of weight-carrying modules "
                        f"are unclassified (limit {max_other_fraction:.0%})")
    if problems:
        raise RuntimeError(
            f"Module classification for {model_key} is not usable:\n  - "
            + "\n  - ".join(problems)
            + f"\n  counts: {c}\n"
            "Add a rule for this family in pipeline_v2/models.py and re-run discovery "
            "before training anything on it.")


# ──────────────────────────────────────────────────────────────── self-tests

def _synthetic(family: str) -> List[Dict[str, Any]]:
    """Module trees in the naming style each family is expected to use.

    These encode an expectation, not a verified fact: the real names are only visible
    once the model is downloaded. The guard is what catches it when the expectation is
    wrong.
    """
    def lin(name, cls="Linear", lt=None, n=1024):
        return dict(full_name=name, class_name=cls, layer_type=lt, direct_param_count=n)

    if family == "falcon_h1":                      # parallel: both mixers per block
        rows = []
        for i in range(2):
            rows += [lin(f"model.layers.{i}.self_attn.q_proj"),
                     lin(f"model.layers.{i}.self_attn.o_proj"),
                     lin(f"model.layers.{i}.mamba.in_proj"),
                     lin(f"model.layers.{i}.feed_forward.down_proj"),
                     lin(f"model.layers.{i}.input_layernorm", "RMSNorm")]
        return rows
    if family == "qwen3_5":                        # sequential: type decides the mixer
        rows = []
        rows += [lin("model.layers.0.self_attn.q_proj", lt="full_attention"),
                 lin("model.layers.0.mlp.down_proj"),
                 lin("model.layers.1.self_attn.in_proj_qkv", "GatedDeltaNet",
                     lt="linear_attention"),
                 lin("model.layers.1.mlp.up_proj"),
                 lin("model.norm", "RMSNorm"), lin("lm_head")]
        return rows
    if family == "zamba2":
        return [lin("model.layers.0.mamba.in_proj", "Mamba2"),
                lin("model.layers.0.self_attn.q_proj"),
                lin("model.layers.0.feed_forward.up_proj"),
                lin("model.embed_tokens", "Embedding")]
    if family == "lfm2":
        return [lin("model.layers.0.conv.in_proj", "ShortConv"),
                lin("model.layers.1.self_attn.k_proj"),
                lin("model.layers.0.feed_forward.w1"),
                lin("model.norm", "RMSNorm")]
    if family == "transformer":
        return [lin("model.layers.0.self_attn.q_proj"),
                lin("model.layers.0.mlp.down_proj")]
    return []


def _self_test() -> int:
    failures: List[str] = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("\n1. the sequential-hybrid case, where the module name is ambiguous")
    check("a self_attn under a full-attention layer is attention",
          classify_module("model.layers.0.self_attn.q_proj", "Linear", "full_attention")
          == "attention")
    check("a self_attn under a linear-attention layer is recurrent",
          classify_module("model.layers.1.self_attn.q_proj", "Linear", "linear_attention")
          == "recurrent")
    check("the class name alone also identifies the recurrent mixer",
          classify_module("model.layers.1.self_attn.in_proj", "GatedDeltaNet") == "recurrent")

    print("\n2. component tokens")
    for name, cls, expect in [("model.layers.0.mamba.in_proj", "Mamba2", "recurrent"),
                              ("model.layers.0.conv.in_proj", "ShortConv", "recurrent"),
                              ("model.layers.0.feed_forward.w1", "Linear", "mlp"),
                              ("model.layers.0.mlp.down_proj", "Linear", "mlp"),
                              ("model.layers.0.self_attn.k_proj", "Linear", "attention"),
                              ("lm_head", "Linear", "skip"),
                              ("model.embed_tokens", "Embedding", "skip"),
                              ("model.layers.0.input_layernorm", "RMSNorm", "skip")]:
        check(f"{name.split('.')[-2] if '.' in name else name} -> {expect}",
              classify_module(name, cls) == expect, classify_module(name, cls))
    check("MLP wins over attention when both tokens appear",
          classify_module("model.layers.0.self_attn.mlp_proj", "Linear") == "mlp")

    print("\n3. each candidate family sorts into three non-empty components")
    for fam in ("falcon_h1", "qwen3_5", "zamba2", "lfm2"):
        rep = classification_report(_synthetic(fam))
        c = rep["counts"]
        print(f"    {fam:12s} {c}  unclassified {rep['other_fraction']:.0%}")
        check(f"{fam}: all three components are present",
              all(c.get(k, 0) > 0 for k in ("attention", "recurrent", "mlp")))
        try:
            assert_classification_sane(rep, fam)
            check(f"{fam}: passes the guard", True)
        except RuntimeError as e:
            check(f"{fam}: passes the guard", False, str(e).splitlines()[0])

    print("\n4. the guard actually refuses bad input")
    rep = classification_report(_synthetic("transformer"))
    try:
        assert_classification_sane(rep, "qwen2_5_0_5b", require_recurrent=True)
        check("a Transformer registered as hybrid is rejected", False)
    except RuntimeError as e:
        check("a Transformer registered as hybrid is rejected", True)
        check("and the message names the missing component", "recurrent" in str(e))
    try:
        assert_classification_sane(rep, "qwen2_5_0_5b", require_recurrent=False)
        check("the same model passes when registered as a control", True)
    except RuntimeError:
        check("the same model passes when registered as a control", False)

    unknown = [dict(full_name=f"model.layers.0.mystery_{i}.proj", class_name="Linear",
                    layer_type=None, direct_param_count=10) for i in range(9)]
    rep2 = classification_report(_synthetic("falcon_h1") + unknown)
    try:
        assert_classification_sane(rep2, "unknown_family")
        check("a family with many unclassified modules is rejected", False,
              str(rep2["counts"]))
    except RuntimeError as e:
        check("a family with many unclassified modules is rejected", True)
        check("and the message quantifies how many", "unclassified" in str(e))

    print("\n5. the registry")
    check("every candidate has an HF id, a role and a priority",
          all(v.get("hf_id") and v.get("role") and "priority" in v
              for v in CANDIDATES.values()))
    check("the control is marked as having no recurrent component",
          CANDIDATES["granite_4_micro_base"]["topology"] == "none")
    check("the control shares its family with a hybrid sibling",
          CANDIDATES["granite_4_micro_base"]["family"]
          == CANDIDATES["granite_4_h_micro_base"]["family"])

    # The point of the roster: each contrast must vary exactly one thing.
    q = CANDIDATES["qwen3_5_0_8b_base"]
    f = CANDIDATES["falcon_h1_0_5b_base"]
    g = CANDIDATES["granite_4_h_micro_base"]
    check("Falcon vs Granite isolates topology: mechanism held, topology varies",
          f["mechanism"] == g["mechanism"] and f["topology"] != g["topology"],
          f"{f['mechanism']}/{g['mechanism']}, {f['topology']}/{g['topology']}")
    check("Qwen vs Granite isolates mechanism: topology held, mechanism varies",
          q["topology"] == g["topology"] and q["mechanism"] != g["mechanism"],
          f"{q['topology']}/{g['topology']}, {q['mechanism']}/{g['mechanism']}")
    check("the two anchors alone cannot separate the two, which is the confound",
          q["topology"] != f["topology"] and q["mechanism"] != f["mechanism"])

    check("there are two scale points in at least one family",
          sum(1 for v in CANDIDATES.values() if v["family"] == "falcon_h1") >= 2)
    check("every rejected model records why it was rejected",
          all(len(v) > 40 for v in REJECTED.values()) and len(REJECTED) >= 4)
    check("nothing is both selected and rejected",
          not ({v["hf_id"] for v in CANDIDATES.values()} & set(REJECTED)))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_self_test() if a.self_test else print(__doc__))
