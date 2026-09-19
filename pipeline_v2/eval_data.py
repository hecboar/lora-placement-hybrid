"""Evaluation subsets for the follow-up campaign, and the bridge to the evaluator.

Two problems in the original evaluation data layer.

**Subsets were drawn independently per size, so they do not nest.** `get_fixed_eval_indices`
resampled from scratch for each `max_n`, so the 128- and 256-item GSM8K subsets share 22
items and agree on no position. Any comparison between runs that used different sizes was
therefore over different questions. The new subsets are built as supersets of every subset
already released for that benchmark, so everything computed so far stays paired against
everything computed from now on.

**The stored identifier was a position, not an identifier.** `example_id` was the row's
position inside the subset. Two runs with different subsets reused the same positions for
different questions. Examples built here carry the benchmark index as `example_id`, which
is stable across subset sizes, and `subset_position` is kept only for tracing back to the
released files.

The builders take the notebook's own prompt and schema helpers as arguments rather than
importing them, so this module is testable without the notebook, without `datasets` and
without a GPU. `python eval_data.py --self-test` runs the checks.
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
import glob
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

SPLIT_OF = {"gsm8k": "test", "mmlu": "validation", "arc_challenge": "validation",
            "hellaswag": "validation", "humaneval": "test"}

# Full split sizes, for reference when choosing a target.
FULL_SPLIT = {"gsm8k": 1319, "arc_challenge": 299, "hellaswag": 10042,
              "mmlu": 1531, "humaneval": 164}


def released_index_sets(index_dir: str, benchmark: str) -> Dict[int, List[int]]:
    """Every subset already released for this benchmark, keyed by size."""
    out: Dict[int, List[int]] = {}
    pattern = os.path.join(index_dir, f"indices__{benchmark}__*.json")
    for fp in sorted(glob.glob(pattern)):
        m = re.search(r"__(\d+)\.json$", fp)
        if not m:
            continue
        with open(fp, encoding="utf-8") as f:
            idx = json.load(f)
        if len(idx) == int(m.group(1)):
            out[int(m.group(1))] = idx
    return out


def nested_indices(benchmark: str, n_total: int, target_n: int, seed: int,
                   index_dir: Optional[str] = None) -> List[int]:
    """A subset of size `target_n` that contains every previously released subset.

    Order of business: take the union of what has already been used, then top up with a
    seeded draw from the remainder. If the union already exceeds the target, the target
    is raised to the union rather than dropping items, because dropping one would break
    the pairing it exists to preserve.
    """
    import numpy as np

    prior: set[int] = set()
    if index_dir and os.path.isdir(index_dir):
        for idx in released_index_sets(index_dir, benchmark).values():
            prior.update(int(i) for i in idx)
    prior = {i for i in prior if 0 <= i < n_total}

    target_n = min(target_n, n_total)
    if len(prior) >= target_n:
        return sorted(prior)

    remaining = np.array(sorted(set(range(n_total)) - prior))
    rng = np.random.default_rng(seed)
    extra = rng.choice(remaining, size=target_n - len(prior), replace=False)
    return sorted(prior | {int(i) for i in extra})


def write_index_file(index_dir: str, benchmark: str, indices: Sequence[int]) -> str:
    os.makedirs(index_dir, exist_ok=True)
    fp = os.path.join(index_dir,
                      f"indices__{benchmark}__{SPLIT_OF.get(benchmark,'test')}__{len(indices)}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(list(indices), f)
    return fp


# ───────────────────────────────────────────── examples in evaluator_v2 format

def build_mc_examples(dataset: Sequence[Dict[str, Any]],
                      indices: Sequence[int],
                      benchmark: str,
                      prompt_fn: Callable[[Dict[str, Any]], str],
                      mc_fn: Callable[[Dict[str, Any]], tuple],
                      labels_fn: Callable[[int], List[str]]) -> List[Dict[str, Any]]:
    """`{prompt, choices, gold, example_id}` for `evaluator.evaluate_multiple_choice`.

    `choices` are the answer LABELS ("A", "B", ...), not the answer texts, because the
    original protocol scored the label token after an "Answer:" prompt. Keeping that
    makes the new numbers comparable with the released ones.
    """
    out = []
    for pos, di in enumerate(indices):
        ex = dataset[di]
        _, choices, gold = mc_fn(ex)
        out.append({"prompt": prompt_fn(ex),
                    "choices": labels_fn(len(choices)),
                    "gold": int(gold),
                    "example_id": int(di),
                    "subset_position": pos})
    return out


def build_gsm8k_examples(dataset: Sequence[Dict[str, Any]],
                         indices: Sequence[int],
                         prompt_fn: Callable[[Dict[str, Any]], str],
                         answer_fn: Callable[[Dict[str, Any]], str]) -> List[Dict[str, Any]]:
    return [{"prompt": prompt_fn(dataset[di]),
             "gold": answer_fn(dataset[di]),
             "example_id": int(di),
             "subset_position": pos}
            for pos, di in enumerate(indices)]


def build_humaneval_problems(dataset: Sequence[Dict[str, Any]],
                             indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [{"prompt": dataset[di]["prompt"], "test": dataset[di]["test"],
             "entry_point": dataset[di]["entry_point"], "task_id": dataset[di]["task_id"],
             "example_id": int(di), "subset_position": pos}
            for pos, di in enumerate(indices)]


def remap_legacy_rows(rows: Sequence[Dict[str, Any]], index_map: Sequence[int]) -> List[Dict[str, Any]]:
    """Rewrite a released per-instance file so `example_id` is the benchmark index."""
    out = []
    for r in rows:
        pos = int(r["example_id"])
        d = dict(r)
        d["subset_position"] = pos
        d["example_id"] = int(index_map[pos])
        out.append(d)
    return out


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test(index_dir: str) -> int:
    failures: List[str] = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("\n1. the released subsets are not nested, which is the problem")
    sets = released_index_sets(index_dir, "gsm8k")
    if {128, 256} <= set(sets):
        a, b = set(sets[128]), set(sets[256])
        check("the released 128 is not a subset of the released 256", not a <= b,
              f"overlap {len(a & b)} of 128")
        check("no position agrees between them",
              sum(1 for i in range(128) if sets[128][i] == sets[256][i]) == 0)
    else:
        print("  SKIP  released gsm8k index files not found in", index_dir)

    print("\n2. new subsets are supersets of everything already released")
    for bench, n_total, target in [("gsm8k", 1319, 1319), ("hellaswag", 10042, 2048),
                                   ("mmlu", 1531, 1531)]:
        idx = nested_indices(bench, n_total, target, seed=3407, index_dir=index_dir)
        prior = released_index_sets(index_dir, bench)
        ok = all(set(v) <= set(idx) for v in prior.values())
        check(f"{bench}: contains every released subset", ok,
              f"sizes released {sorted(prior)}")
        check(f"{bench}: has the requested size", len(idx) == min(target, n_total),
              f"{len(idx)} vs {min(target, n_total)}")
        check(f"{bench}: indices are unique and in range",
              len(set(idx)) == len(idx) and min(idx) >= 0 and max(idx) < n_total)
    idx_a = nested_indices("hellaswag", 10042, 2048, 3407, index_dir)
    idx_b = nested_indices("hellaswag", 10042, 2048, 3407, index_dir)
    check("the draw is reproducible for a fixed seed", idx_a == idx_b)
    small = nested_indices("hellaswag", 10042, 2048, 3407, index_dir)
    big = nested_indices("hellaswag", 10042, 4096, 3407, index_dir)
    check("a larger target still contains every released subset",
          all(set(v) <= set(big) for v in released_index_sets(index_dir, "hellaswag").values()))
    print(f"    hellaswag 2048 covers the released 512: "
          f"{set(released_index_sets(index_dir,'hellaswag').get(512, [])) <= set(small)}")

    print("\n3. never drop a released item to hit a smaller target")
    tiny = nested_indices("gsm8k", 1319, 10, 3407, index_dir)
    prior_union = set()
    for v in released_index_sets(index_dir, "gsm8k").values():
        prior_union |= set(v)
    check("the target is raised rather than dropping released items",
          set(tiny) == prior_union and len(tiny) >= 10,
          f"{len(tiny)} items kept, union is {len(prior_union)}")

    print("\n4. examples carry the benchmark index, not the position")
    ds = [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4,
           "answer_text": "x"} for i in range(50)]
    mc = lambda ex: (ex["question"], ex["choices"], ex["answer"])
    labels = lambda n: [chr(65 + i) for i in range(n)]
    ex = build_mc_examples(ds, [7, 19, 33], "mmlu", lambda e: e["question"], mc, labels)
    check("example_id is the benchmark index", [e["example_id"] for e in ex] == [7, 19, 33])
    check("the position is kept for tracing", [e["subset_position"] for e in ex] == [0, 1, 2])
    check("choices are labels, matching the original protocol",
          ex[0]["choices"] == ["A", "B", "C", "D"])
    check("gold survives the round trip", [e["gold"] for e in ex] == [7 % 4, 19 % 4, 33 % 4])

    g = build_gsm8k_examples(ds, [3, 5], lambda e: "P:" + e["question"],
                             lambda e: "#### 42")
    check("gsm8k examples carry prompt, gold and index",
          g[0]["prompt"] == "P:q3" and g[0]["gold"] == "#### 42" and g[0]["example_id"] == 3)

    print("\n5. legacy files can be remapped onto benchmark indices")
    legacy = [{"example_id": 0, "correct": 1}, {"example_id": 1, "correct": 0}]
    mapped = remap_legacy_rows(legacy, [77, 88])
    check("positions become benchmark indices",
          [r["example_id"] for r in mapped] == [77, 88])
    check("correctness is untouched", [r["correct"] for r in mapped] == [1, 0])
    check("the original position is preserved",
          [r["subset_position"] for r in mapped] == [0, 1])

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--index-dir", default=_results_dir("eval_indices"))
    a = ap.parse_args()
    raise SystemExit(_self_test(a.index_dir) if a.self_test else print(__doc__))
