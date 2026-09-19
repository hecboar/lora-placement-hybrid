"""Training configuration with real seeds, plus the run plan for the campaign.

The original `build_sft_config` never passed a seed to the trainer. HuggingFace's
`TrainingArguments` defaults to `seed=42` and calls `set_seed` in `Trainer.__init__`,
so every run shared one data order and one dropout stream regardless of the `seed`
argument threaded through `train_condition`. Only the LoRA initialisation varied,
because `seed_everything(seed)` ran before `get_peft_model`.

That is not a multi-seed study, and a reviewer who reads the config will say so. Two
runs that differ only in initialisation understate run-to-run variance, which is the
quantity a multi-seed experiment exists to measure.

`build_sft_config_v2` passes both `seed` and `data_seed`. Everything else is carried
over unchanged so that the seed-3407 runs already on disk stay comparable: with
seed=3407 the only behavioural difference from the original is the data order, which
is the thing that was wrong.

`python training.py --self-test` checks the kwarg assembly without importing trl.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence


def sft_kwargs(output_dir: str,
               seed: int,
               *,
               num_train_epochs: int = 3,
               learning_rate: float = 2e-4,
               per_device_batch_size: int = 4,
               gradient_accumulation_steps: int = 4,
               logging_steps: int = 10,
               save_steps: int = 200,
               max_seq_length: int = 1024,
               use_bf16: bool = True,
               use_fp16: bool = False,
               eval_during_training: bool = True,
               eval_every_n_steps: int = 200,
               eos_token: Optional[str] = None) -> Dict[str, Any]:
    """The keyword arguments for SFTConfig, as a plain dict so it can be tested.

    `seed` controls parameter initialisation and dropout; `data_seed` controls the
    sampler. Passing only the first leaves the data order fixed across "seeds".
    """
    kw: Dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_batch_size,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=[],
        bf16=use_bf16,
        fp16=use_fp16,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # the correction
        seed=seed,
        data_seed=seed,
    )
    if eos_token is not None:
        kw["eos_token"] = eos_token
    if eval_during_training:
        kw["eval_steps"] = eval_every_n_steps
    return kw


def build_sft_config_v2(output_dir: str, seed: int, *, eval_during_training: bool = True,
                        **kw):
    """Construct an SFTConfig, tolerating the eval-strategy rename across TRL versions."""
    from trl import SFTConfig  # imported lazily so the module is testable without trl

    common = sft_kwargs(output_dir, seed, eval_during_training=eval_during_training, **kw)
    strategy = "steps" if eval_during_training else "no"
    try:
        return SFTConfig(eval_strategy=strategy, **common)
    except TypeError:
        return SFTConfig(evaluation_strategy=strategy, **common)


# ───────────────────────────────────────────────────────────────── run plan

def campaign_jobs(models: Sequence[str],
                  conditions: Sequence[str],
                  domains: Sequence[str],
                  seeds: Sequence[int],
                  rank_by_condition: Optional[Dict[str, int]] = None,
                  default_rank: int = 16) -> list[Dict[str, Any]]:
    """Every run of the campaign, ordered cheapest-model-first.

    The experiment key carries the rank explicitly whenever it is not the default, so
    budget-matched runs never collide with the rank-16 grid in the checkpoint store.
    That collision is what made the original runner's suffixes ambiguous.
    """
    jobs = []
    for model in models:
        for domain in domains:
            for cond in conditions:
                r = (rank_by_condition or {}).get(cond, default_rank)
                for seed in seeds:
                    suffix = "" if r == default_rank else f"__rank{r}"
                    jobs.append(dict(
                        model_key=model, condition_name=cond, train_dataset_key=domain,
                        seed=seed, rank=r, lora_alpha=2 * r,
                        experiment_suffix=suffix,
                        exp_key=(f"train__{model}__{cond}__{domain}"
                                 f"__seed{seed}{suffix}")))
    return jobs


def _self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("\n1. the seed correction")
    kw = sft_kwargs("/tmp/out", seed=1337)
    check("seed is passed to the trainer", kw.get("seed") == 1337)
    check("data_seed is passed too, so the data order actually varies",
          kw.get("data_seed") == 1337)
    a, b = sft_kwargs("/tmp/o", 1337), sft_kwargs("/tmp/o", 2026)
    diff = {k for k in a if a[k] != b.get(k)}
    check("two seeds differ in exactly the two seed fields", diff == {"seed", "data_seed"},
          str(sorted(diff)))
    check("everything else is carried over unchanged",
          a["learning_rate"] == 2e-4 and a["optim"] == "paged_adamw_8bit"
          and a["max_length"] == 1024 and a["gradient_checkpointing"] is True)
    check("eos_token is only set when supplied",
          "eos_token" not in sft_kwargs("/tmp/o", 1)
          and sft_kwargs("/tmp/o", 1, eos_token="</s>")["eos_token"] == "</s>")
    check("eval_steps only appears when evaluating during training",
          "eval_steps" in sft_kwargs("/tmp/o", 1)
          and "eval_steps" not in sft_kwargs("/tmp/o", 1, eval_during_training=False))

    print("\n2. the run plan")
    jobs = campaign_jobs(["m1"], ["attention_only", "mlp_only"], ["gsm8k_train"],
                         [3407, 1337, 2026])
    check("one job per condition x seed", len(jobs) == 6)
    check("experiment keys are unique", len({j["exp_key"] for j in jobs}) == 6)
    check("the rank-16 key has no suffix, matching the existing checkpoints",
          jobs[0]["exp_key"] == "train__m1__attention_only__gsm8k_train__seed3407")

    bm = campaign_jobs(["m1"], ["attention_only"], ["gsm8k_train"], [3407],
                       rank_by_condition={"attention_only": 160})
    check("a budget-matched run carries its rank in the key",
          bm[0]["exp_key"] == "train__m1__attention_only__gsm8k_train__seed3407__rank160")
    check("a budget-matched run cannot collide with the rank-16 grid",
          bm[0]["exp_key"] != jobs[0]["exp_key"])
    check("alpha tracks rank at the usual 2x", bm[0]["lora_alpha"] == 320)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_self_test() if a.self_test else print(__doc__))
