"""Verify the machine can actually run the campaign, before any GPU time is spent.

The original campaign needed pre-release builds of transformers, peft and trl because
Qwen3.5 and Falcon-H1 support existed only there. If the stable releases installed today
cannot build a config for those architectures, nothing downstream works, and it is much
cheaper to learn that from a config lookup than from a training run that dies an hour in.

Checks, cheapest first, stopping at the first that matters:
  1. torch, CUDA, the device and its memory
  2. every library the pipeline imports, with versions
  3. the fast Mamba-2 kernels, which are optional but halve Falcon's training time
  4. that each model in the roster resolves to a config class, which is the real question
  5. that PEFT can build a LoRA config with exact dotted module names

Downloads only config.json for step 4, a few kilobytes per model, never the weights.

    python check_environment.py              # everything
    python check_environment.py --offline    # skip anything needing the network
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List, Tuple

ROSTER = [
    ("Qwen/Qwen3.5-0.8B-Base", "anchor, interleaved hybrid", True),
    ("tiiuae/Falcon-H1-0.5B-Base", "anchor, parallel hybrid", True),
    ("ibm-granite/granite-4.0-h-micro-base", "third topology point", False),
    ("ibm-granite/granite-4.0-micro-base", "same-family dense control", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    problems: List[str] = []
    warnings: List[str] = []

    print("1. compute")
    try:
        import torch
        print(f"   torch {torch.__version__}  cuda {torch.version.cuda}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"   {p.name}  {p.total_memory/1e9:.1f} GB  capability {p.major}.{p.minor}")
            print(f"   bf16 supported: {torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                problems.append("bf16 is not supported; the protocol trains in bf16")
            if p.total_memory / 1e9 < 20:
                warnings.append(f"only {p.total_memory/1e9:.1f} GB of VRAM; 3B models "
                                "will need batch 2 with accumulation 8")
        else:
            problems.append("CUDA is not available")
    except Exception as e:
        problems.append(f"torch import failed: {e}")
        print("\n".join("   PROBLEM: " + p for p in problems))
        return 1

    print("\n2. libraries")
    for mod, needed in [("transformers", True), ("peft", True), ("trl", True),
                        ("datasets", True), ("accelerate", True),
                        ("bitsandbytes", True), ("numpy", True), ("pandas", True)]:
        try:
            m = importlib.import_module(mod)
            print(f"   {mod:14s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            (problems if needed else warnings).append(
                f"{mod} missing ({type(e).__name__})")
            print(f"   {mod:14s} MISSING")

    print("\n3. fast Mamba-2 kernels (optional; halves Falcon training time)")
    for mod in ("mamba_ssm", "causal_conv1d"):
        try:
            importlib.import_module(mod)
            print(f"   {mod:14s} present")
        except Exception:
            print(f"   {mod:14s} absent")
            warnings.append(f"{mod} absent: Falcon and Granite train about twice as "
                            "slowly, which roughly doubles their share of the bill")

    print("\n4. architecture support (the question the stable releases might fail)")
    if a.offline:
        print("   skipped (--offline)")
    else:
        try:
            from transformers import AutoConfig
        except Exception as e:
            problems.append(f"cannot import AutoConfig: {e}")
            AutoConfig = None
        if AutoConfig is not None:
            for repo, role, required in ROSTER:
                try:
                    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
                    layers = (getattr(cfg, "num_hidden_layers", None)
                              or getattr(getattr(cfg, "text_config", cfg),
                                         "num_hidden_layers", "?"))
                    print(f"   OK      {repo}")
                    print(f"           {type(cfg).__name__}, {layers} layers  [{role}]")
                except Exception as e:
                    msg = str(e).splitlines()[0][:110]
                    print(f"   FAILED  {repo}\n           {type(e).__name__}: {msg}")
                    (problems if required else warnings).append(
                        f"{repo} does not resolve: {type(e).__name__}")

    print("\n5. PEFT accepts exact dotted module names, which every condition relies on")
    try:
        from peft import LoraConfig, TaskType
        c = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                       lora_dropout=0.05, bias="none",
                       target_modules=["model.layers.0.self_attn.q_proj",
                                       "model.layers.1.mlp.down_proj"])
        ok = list(c.target_modules) == ["model.layers.0.self_attn.q_proj",
                                        "model.layers.1.mlp.down_proj"]
        print(f"   exact target_modules preserved: {ok}")
        if not ok:
            problems.append("PEFT rewrote the target module list; conditions would not "
                            "target what they claim to")
    except Exception as e:
        problems.append(f"PEFT LoraConfig failed: {e}")
        print(f"   FAILED: {e}")

    print("\n" + "=" * 70)
    for w in warnings:
        print(f"  WARNING  {w}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    if problems:
        print(f"\n{len(problems)} problem(s): do not start the campaign yet.")
        return 1
    print(f"\nReady.{'  ' + str(len(warnings)) + ' warning(s), none blocking.' if warnings else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
