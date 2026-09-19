"""Headless campaign runner.

The pipeline lives in a notebook, so it has to be loaded by executing its definition
cells. The original runner did that too, but executed *every* code cell except three,
which meant that opening it re-ran `pip install git+…@main` for transformers, peft and
trl, and re-ran module discovery and verification (twelve model loads). Two
consequences: library versions could change between runs of the same campaign, and
several minutes of paid GPU time were spent before any training started.

This runner skips by intent rather than by exception: it executes only the cells that
define functions and configuration, and refuses to execute the install cell at all. It
then replaces three pieces with their corrected versions:

  * the evaluator            -> evaluator_v2/evaluator.py   (stop sequences, first-####,
                                HumanEval actually executed, everything batched)
  * the placement conditions -> pipeline_v2/conditions.py   (harmonised, plus the
                                budget-matched and random-placement controls)
  * the trainer config       -> pipeline_v2/training.py     (seed AND data_seed)

Modes:
    --dry-run     print the plan and what is already done. No torch, no GPU, no notebook.
    --self-test   check the planning and cell-selection logic offline.
    --run         execute. Requires the notebook, the GPU and the dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluator_v2"))

# Cells that must never run on import. The first is the reason a campaign could end up
# spanning two different versions of transformers.
FORBIDDEN_MARKERS = ("SECTION 0.1",)
# Cells that define things we want but are expensive or execute the grid.
SKIP_MARKERS = (
    "SECTION 1.4",                              # runs discovery + verification
    "SECTION 2.2",                              # re-prepares every dataset
    "SECTION 3.3 / 4.3",                        # the experiment grid
    "SECTION 5 / 6",                            # the whole analysis and plotting pass
    "OPTIONAL SECTION — Run the Layer Card",    # optional diagnostic
)



def default_notebook() -> str:
    """The pipeline notebook, wherever it lives.

    The repository ships it as `notebook/paper3_lora_placement.ipynb`; the development
    tree also has an identical copy under `news-experiments/`. Defaulting to one of
    them breaks the other, and the one that breaks is the clone.
    """
    for rel in ("notebook/paper3_lora_placement.ipynb",
                "news-experiments/paper3_lora_placement_reproducible.ipynb"):
        p = os.path.join(ROOT, *rel.split("/"))
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "notebook", "paper3_lora_placement.ipynb")

def select_cells(notebook_path: str) -> tuple[List[str], List[str]]:
    """Return (cells to execute, human-readable reasons for the ones skipped)."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    run, skipped = [], []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        head = next((l for l in src.splitlines() if l.strip()), "")[:70]
        if any(m in src for m in FORBIDDEN_MARKERS):
            skipped.append(f"REFUSED  {head}   (installs packages)")
        elif any(m in src for m in SKIP_MARKERS):
            skipped.append(f"skipped  {head}")
        else:
            run.append(src)
    return run, skipped


def load_pipeline(notebook_path: str, project_dir: str) -> Dict[str, Any]:
    """Execute the definition cells and return the resulting namespace."""
    run, skipped = select_cells(notebook_path)
    ns: Dict[str, Any] = {"__name__": "__pipeline__"}
    # ROOT_DIR is read by the notebook's section 0.2; set it before anything runs.
    os.environ.setdefault("PAPER3_ROOT_DIR", project_dir)
    for src in run:
        src = re.sub(r'^\s*ROOT_DIR\s*=.*$',
                     f'ROOT_DIR = Path(r"{project_dir}")', src, flags=re.M)
        exec(compile(src, "<pipeline>", "exec"), ns)
    for line in skipped:
        print("   ", line)
    return ns


def patch_pipeline(ns: Dict[str, Any]) -> List[str]:
    """Swap in the corrected evaluator and trainer config. Returns what was replaced."""
    import evaluator as ev
    import training as tr

    replaced = []

    def sft(output_dir, eos_token=None, _seed_holder=ns):
        return tr.build_sft_config_v2(
            str(output_dir), seed=int(_seed_holder.get("_ACTIVE_SEED", 3407)),
            eos_token=eos_token,
            eval_during_training=bool(ns.get("EVAL_DURING_TRAINING", True)))

    ns["build_sft_config"] = sft
    replaced.append("build_sft_config -> training.build_sft_config_v2 (seed + data_seed)")

    ns["extract_final_number"] = lambda t: ev.extract_gsm8k_answer(t)
    replaced.append("extract_final_number -> evaluator.extract_gsm8k_answer (first ####)")

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    replaced.append("HF_ALLOW_CODE_EVAL=1 (the metric that never ran)")
    return replaced


def is_done(ckpt, exp_key: str) -> bool:
    cached = ckpt.load(exp_key)
    return bool(cached and cached.get("status") == "complete")


def run_campaign(ns: Dict[str, Any], jobs: Sequence[Dict[str, Any]],
                 budget_hours: float, rate_eur_h: float,
                 log_path: Optional[str] = None) -> Dict[str, Any]:
    ckpt = ns["ckpt"]
    train_condition = ns["train_condition"]
    pending = [j for j in jobs if not is_done(ckpt, j["exp_key"])]
    print(f"{len(jobs) - len(pending)} already complete, {len(pending)} pending, "
          f"budget {budget_hours:.1f} GPU-h (about EUR {budget_hours * rate_eur_h:.0f})")

    spent, done, failed = 0.0, [], []
    per_run = 1.5
    for i, j in enumerate(pending, 1):
        if spent + per_run > budget_hours:
            print(f"\nStopping: the next run would exceed the budget "
                  f"({spent:.1f} + {per_run:.1f} > {budget_hours:.1f} GPU-h).")
            break
        print("=" * 90)
        print(f"[{i}/{len(pending)}] {j['exp_key']}   spent {spent:.2f} h")
        ns["_ACTIVE_SEED"] = j["seed"]          # read by the patched build_sft_config
        t0 = time.time()
        try:
            res = train_condition(
                model_key=j["model_key"], condition_name=j["condition_name"],
                train_dataset_key=j["train_dataset_key"], seed=j["seed"],
                rank=j["rank"], lora_alpha=j["lora_alpha"],
                experiment_suffix=j["experiment_suffix"])
            dt = (time.time() - t0) / 3600.0
            if dt > 0.02:                       # a cache hit is not a run
                per_run = 0.5 * per_run + 0.5 * dt
                spent += dt
            ok = isinstance(res, dict) and res.get("status") == "complete"
            (done if ok else failed).append(j["exp_key"])
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.strftime("%F %T"),
                                        "exp_key": j["exp_key"], "hours": round(dt, 4),
                                        "status": "complete" if ok else "failed"}) + "\n")
        except Exception as e:                  # one bad run must not kill the campaign
            import traceback
            traceback.print_exc()
            failed.append(j["exp_key"])
            try:
                ns["clear_memory"]()
            except Exception:
                pass
    print(f"\n{len(done)} completed, {len(failed)} failed, {spent:.2f} GPU-h "
          f"(about EUR {spent * rate_eur_h:.0f})")
    return {"completed": done, "failed": failed, "hours": spent}


# ───────────────────────────────────────────────────────────────── self-test

def _self_test() -> int:
    import training as tr
    failures = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    nb = os.path.join(ROOT, "news-experiments", "paper3_lora_placement_reproducible.ipynb")
    print("\n1. cell selection")
    if os.path.exists(nb):
        run, skipped = select_cells(nb)
        joined = "\n".join(run)
        check("the install cell is refused, not merely skipped",
              any("REFUSED" in s for s in skipped))
        check("no executed cell installs packages",
              "pip install" not in joined and "pip_install(" not in joined)
        check("no executed cell runs the experiment grid",
              "run_experiment_grid()" not in joined.replace("def run_experiment_grid()", ""))
        check("the trainer and evaluator definitions are still executed",
              "def train_condition" in joined and "def evaluate_gsm8k" in joined)
        check("something is actually skipped", len(skipped) >= 5, f"{len(skipped)} skipped")
        print(f"    {len(run)} cells executed, {len(skipped)} skipped")
    else:
        print("  SKIP  notebook not found at", nb)

    print("\n2. resume logic")
    class FakeCkpt:
        def __init__(self, done): self.done = done
        def load(self, k): return {"status": "complete"} if k in self.done else None
    jobs = tr.campaign_jobs(["m"], ["a", "b"], ["d"], [1, 2])
    ck = FakeCkpt({jobs[0]["exp_key"], jobs[3]["exp_key"]})
    pending = [j for j in jobs if not is_done(ck, j["exp_key"])]
    check("completed runs are skipped on resume", len(pending) == 2)
    check("the right ones are skipped",
          {j["exp_key"] for j in pending} == {jobs[1]["exp_key"], jobs[2]["exp_key"]})
    check("a failed run is retried, not treated as done",
          not is_done(FakeCkpt(set()), jobs[0]["exp_key"]))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--project-dir", default="/workspace/lora-placement")
    ap.add_argument("--notebook", default=default_notebook())
    ap.add_argument("--models", nargs="+",
                    default=["qwen3_5_0_8b_base", "falcon_h1_0_5b_base"])
    ap.add_argument("--conditions", nargs="+",
                    default=["attention_only", "recurrent_only", "mlp_only",
                             "attention_plus_mlp", "recurrent_plus_mlp", "all_components"])
    ap.add_argument("--domains", nargs="+", default=["gsm8k_train", "ultrachat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 1337, 2026])
    ap.add_argument("--budget-hours", type=float, default=40.0)
    ap.add_argument("--rate", type=float, default=0.40)
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    import training as tr
    jobs = tr.campaign_jobs(a.models, a.conditions, a.domains, a.seeds)

    if a.dry_run or not a.run:
        print(f"{len(jobs)} runs planned\n")
        for j in jobs[:15]:
            print("   ", j["exp_key"])
        if len(jobs) > 15:
            print(f"    ... and {len(jobs) - 15} more")
        print(f"\nAt roughly 1.5 GPU-h per run this is about "
              f"{len(jobs) * 1.5:.0f} GPU-h, or EUR {len(jobs) * 1.5 * a.rate:.0f}.")
        print("Pass --run to execute. Nothing has been launched.")
        return 0

    print("Loading the pipeline; skipped cells:")
    ns = load_pipeline(a.notebook, a.project_dir)
    print("Patches applied:")
    for p in patch_pipeline(ns):
        print("   ", p)
    return 0 if run_campaign(
        ns, jobs, a.budget_hours, a.rate,
        os.path.join(a.project_dir, "logs", "campaign.jsonl"))["failed"] == [] else 1


if __name__ == "__main__":
    raise SystemExit(main())
