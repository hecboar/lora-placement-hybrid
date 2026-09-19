"""Install the harmonised conditions and controls into the pipeline's spec files.

`train_condition` resolves a condition name through `condition_specs_for_model`, which
reads `results/discovery/<model>__condition_specs.json`. Building better conditions in
`conditions.py` therefore changes nothing until they are written to that file: a run of
`attention_only` on Falcon would still use the original definition, without the output
projection.

This writes them, keeping the exact schema the pipeline expects (`target_modules`,
`target_parameters`, `notes`), backing up whatever was there, and verifying the result
round-trips. The original definitions are preserved under `legacy__` names so the
released seed-3407 runs remain reproducible from the same file.

    python install_conditions.py --check                 # report, change nothing
    python install_conditions.py --install               # write, after backing up
    python install_conditions.py --self-test             # offline
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import conditions as cond  # noqa: E402


def _results_dir(*parts):
    here = HERE
    for _ in range(4):
        for prefix in ("", "github"):
            cand = (os.path.join(here, prefix, "results", *parts) if prefix
                    else os.path.join(here, "results", *parts))
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.join(HERE, "results", *parts)


def to_pipeline_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Only the three keys the pipeline reads; the rest would be silently ignored."""
    return {"target_modules": list(spec["target_modules"]),
            "target_parameters": [],
            "notes": spec.get("notes", "")}


def build_all(manifest: Dict[str, Any], rank: int = 16,
              seeds=(3407, 1337, 2026)) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in cond.harmonised_specs(manifest, rank).items():
        out[name] = to_pipeline_schema(spec)
    for name, spec in cond.build_control_suite(manifest, rank, seeds).items():
        s = to_pipeline_schema(spec)
        if "rank" in spec:
            s["notes"] += f" [run at rank {spec['rank']}]"
        out[name] = s
    return out


def install(discovery_dir: str, model_key: str, rank: int = 16,
            seeds=(3407, 1337, 2026), dry_run: bool = True) -> Dict[str, Any]:
    man_path = os.path.join(discovery_dir, f"{model_key}__manifest.json")
    spec_path = os.path.join(discovery_dir, f"{model_key}__condition_specs.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(man_path)

    manifest = cond.load_manifest(man_path)
    legacy: Dict[str, Any] = {}
    if os.path.exists(spec_path):
        with open(spec_path, encoding="utf-8") as f:
            legacy = json.load(f)

    new = build_all(manifest, rank, seeds)
    # Keep the originals reachable so the released runs stay reproducible.
    for name, spec in legacy.items():
        if not name.startswith("legacy__"):
            new[f"legacy__{name}"] = spec

    check = cond.partition_check(cond.harmonised_specs(manifest, rank), rank)
    report = {"model_key": model_key, "path": spec_path,
              "legacy_conditions": sorted(k for k in legacy),
              "new_conditions": sorted(k for k in new if not k.startswith("legacy__")),
              "controls": sorted(k for k in new if k.startswith(("budget_matched__",
                                                                 "random_placement__"))),
              "partitions": check["partitions"], "unaccounted": check["unaccounted"],
              "written": False}

    if dry_run:
        return report

    if os.path.exists(spec_path):
        backup = spec_path.replace(".json", ".pre_harmonisation.json")
        if not os.path.exists(backup):
            shutil.copy2(spec_path, backup)
            report["backup"] = backup
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=1)

    with open(spec_path, encoding="utf-8") as f:
        back = json.load(f)
    if set(back) != set(new):
        raise RuntimeError("round-trip failed: the written file does not match")
    report["written"] = True
    return report


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test(discovery_dir: str) -> int:
    failures: List[str] = []

    def check(name, c, detail=""):
        print(("  PASS  " if c else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not c:
            failures.append(name)

    for mk in ("falcon_h1_0_5b_base", "qwen3_5_0_8b_base"):
        man_path = os.path.join(discovery_dir, f"{mk}__manifest.json")
        if not os.path.exists(man_path):
            print(f"  SKIP  {mk}: manifest not found")
            continue
        print(f"\n{mk}")
        manifest = cond.load_manifest(man_path)
        built = build_all(manifest, 16, seeds=(3407, 1337))

        check("the schema is exactly what the pipeline reads",
              all(set(v) == {"target_modules", "target_parameters", "notes"}
                  for v in built.values()))
        check("target_parameters is empty, as in the original specs",
              all(v["target_parameters"] == [] for v in built.values()))
        check("every condition names at least one module",
              all(len(v["target_modules"]) > 0 for v in built.values()))
        check("module names are unique inside each condition",
              all(len(set(v["target_modules"])) == len(v["target_modules"])
                  for v in built.values()))
        check("no condition targets lm_head",
              not any(any("lm_head" in m for m in v["target_modules"])
                      for v in built.values()))
        check("the six harmonised conditions are present",
              {"attention_only", "recurrent_only", "mlp_only", "attention_plus_mlp",
               "recurrent_plus_mlp", "all_components"} <= set(built))
        n_bm = sum(1 for k in built if k.startswith("budget_matched__"))
        n_rp = sum(1 for k in built if k.startswith("random_placement__"))
        check("both controls are installed", n_bm >= 4 and n_rp >= 6,
              f"{n_bm} budget-matched, {n_rp} random")
        check("a budget-matched condition records the rank it must run at",
              all("rank" in built[k]["notes"] for k in built
                  if k.startswith("budget_matched__")))

        rep = install(discovery_dir, mk, dry_run=True)
        check("a dry run reports without writing", rep["written"] is False)
        check("the harmonised set partitions cleanly", rep["partitions"],
              f"unaccounted {rep['unaccounted']:,}")
        check("the original conditions are preserved under legacy names",
              all(f"legacy__{c}" not in rep["new_conditions"]
                  for c in rep["legacy_conditions"])
              and len(rep["legacy_conditions"]) == 6,
              str(rep["legacy_conditions"]))
        print(f"    installs {len(rep['new_conditions'])} conditions "
              f"+ {len(rep['controls'])} controls, "
              f"{len(rep['legacy_conditions'])} originals kept as legacy__")

        # Falcon is the one whose definitions actually change.
        if mk.startswith("falcon"):
            old = {c["target_modules"] and len(c["target_modules"])
                   for c in [json.load(open(os.path.join(
                       discovery_dir, f"{mk}__condition_specs.json"),
                       encoding="utf-8"))["attention_only"]]}
            check("Falcon attention_only gains modules relative to the original",
                  len(built["attention_only"]["target_modules"]) > list(old)[0],
                  f"{len(built['attention_only']['target_modules'])} vs {list(old)[0]}")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--discovery-dir", default=None)
    ap.add_argument("--models", nargs="+",
                    default=["qwen3_5_0_8b_base", "falcon_h1_0_5b_base"])
    a = ap.parse_args()
    d = a.discovery_dir or _results_dir("discovery")

    if a.self_test:
        return _self_test(d)
    for mk in a.models:
        rep = install(d, mk, dry_run=not a.install)
        print(json.dumps(rep, indent=1))
    if not a.install:
        print("\nNothing was written. Pass --install to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
