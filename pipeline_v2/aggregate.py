"""Multi-seed aggregation and statistics for the follow-up campaign.

A multi-seed study has two sources of uncertainty and they answer different questions.

*Evaluation-subset uncertainty* is what the v1 bootstrap measured: given these trained
models, how much would the estimate move on a different sample of questions? That is a
paired bootstrap over items.

*Training-run variance* is what a single seed cannot see at all, and what the extra
seeds are bought for: given this recipe, how much does the outcome move when you train
again? That is the spread across seeds.

Reporting only the first, as v1 did, understates uncertainty; reporting only the second
wastes the per-item data. Both are produced here, and the headline interval is a paired
bootstrap over items on the per-item mean across seeds, with the clustering unit stated,
which is the treatment that cleared review for the companion pruning study.

Holm correction is applied within families, so a family is one (model, domain,
benchmark) triple and the correction does not leak across unrelated comparisons.

`python aggregate.py --self-test` checks the statistics against cases with known
answers. No GPU, no data required.
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────── seed summary

def seed_level_accuracy(labels_by_seed: Dict[int, Dict[int, int]]) -> Dict[int, float]:
    """Accuracy per seed, each over that seed's own items."""
    return {s: (sum(v.values()) / len(v) if v else float("nan"))
            for s, v in labels_by_seed.items()}


def across_seed_summary(labels_by_seed: Dict[int, Dict[int, int]]) -> Dict[str, Any]:
    """Mean and spread of accuracy across seeds.

    The standard deviation uses ddof=1 because the seeds are a sample of training runs,
    not the population of them. With one seed it is undefined and reported as NaN rather
    than as zero, which would claim the recipe is deterministic.
    """
    acc = seed_level_accuracy(labels_by_seed)
    vals = np.array([v for v in acc.values() if not math.isnan(v)], dtype=float)
    n = len(vals)
    return {"n_seeds": n,
            "mean": float(vals.mean()) if n else float("nan"),
            "sd": float(vals.std(ddof=1)) if n > 1 else float("nan"),
            "sem": float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
            "per_seed": acc}


def per_item_mean(labels_by_seed: Dict[int, Dict[int, int]]) -> Dict[int, float]:
    """Mean correctness of each item across the seeds that evaluated it.

    Only items every seed saw are kept, so the average is over a constant set of runs.
    """
    if not labels_by_seed:
        return {}
    shared = set.intersection(*(set(v) for v in labels_by_seed.values()))
    return {i: float(np.mean([labels_by_seed[s][i] for s in labels_by_seed]))
            for i in sorted(shared)}


# ────────────────────────────────────────────────────────────────── bootstrap

def paired_bootstrap(a: Dict[int, float], b: Dict[int, float],
                     n_boot: int = 10000, seed: int = 3407) -> Dict[str, Any]:
    """Percentile CI for the paired mean difference, resampling items.

    Items are the clustering unit. `a` and `b` must be keyed by the same identifier
    space, which after `eval_data` means benchmark indices rather than positions.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return {"n_shared": len(shared), "diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
    d = np.array([a[i] for i in shared]) - np.array([b[i] for i in shared])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"n_shared": len(shared), "diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "p": float(min(p, 1.0))}


def compare_conditions(a_by_seed: Dict[int, Dict[int, int]],
                       b_by_seed: Dict[int, Dict[int, int]],
                       n_boot: int = 10000, seed: int = 3407) -> Dict[str, Any]:
    """The full comparison: item-level interval plus across-seed spread.

    `seed_diff_sd` is the standard deviation of the per-seed difference, computed only
    over seeds present in both arms. It is the number that says whether an effect
    survives retraining, and it is the one v1 could not report.
    """
    res = paired_bootstrap(per_item_mean(a_by_seed), per_item_mean(b_by_seed), n_boot, seed)
    common = sorted(set(a_by_seed) & set(b_by_seed))
    diffs = []
    for s in common:
        sh = sorted(set(a_by_seed[s]) & set(b_by_seed[s]))
        if sh:
            diffs.append(float(np.mean([a_by_seed[s][i] - b_by_seed[s][i] for i in sh])))
    res["n_seeds"] = len(diffs)
    res["seed_diff_mean"] = float(np.mean(diffs)) if diffs else float("nan")
    res["seed_diff_sd"] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    res["sign_consistent"] = (bool(all(d > 0 for d in diffs) or all(d < 0 for d in diffs))
                              if len(diffs) > 1 else None)
    return res


# ───────────────────────────────────────────────────────────── multiplicity

def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, order preserved, monotone non-decreasing."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = val
        running = val
    return adj


def holm_within_families(rows: Sequence[Dict[str, Any]],
                         family_keys: Sequence[str] = ("model", "domain", "benchmark"),
                         p_key: str = "p") -> List[Dict[str, Any]]:
    """Apply Holm inside each family, leaving the input order intact."""
    fams: Dict[Tuple, List[int]] = {}
    for i, r in enumerate(rows):
        fams.setdefault(tuple(r[k] for k in family_keys), []).append(i)
    out = [dict(r) for r in rows]
    for idxs in fams.values():
        adj = holm([rows[i][p_key] for i in idxs])
        for i, a in zip(idxs, adj):
            out[i]["p_holm"] = a
            out[i]["sig_holm"] = a < 0.05
    return out


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test() -> int:
    failures: List[str] = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("\n1. Holm correction")
    # Textbook case: m=4, sorted p, adjusted = max so far of (m-rank)*p, capped at 1.
    p = [0.01, 0.02, 0.03, 0.04]
    check("adjusted values match the definition",
          [round(x, 4) for x in holm(p)] == [0.04, 0.06, 0.06, 0.06],
          str([round(x, 4) for x in holm(p)]))
    check("adjusted p never decreases as raw p increases",
          all(x <= y + 1e-12 for x, y in zip(holm(p), holm(p)[1:])))
    check("order of the input is preserved",
          holm([0.04, 0.01])[0] > holm([0.04, 0.01])[1])
    check("adjustment is capped at 1", max(holm([0.9, 0.95])) <= 1.0)
    check("a single test is unadjusted", holm([0.03]) == [0.03])
    check("empty input is handled", holm([]) == [])

    print("\n2. families are corrected separately")
    rows = [{"model": "m1", "domain": "d", "benchmark": "b", "p": 0.01},
            {"model": "m1", "domain": "d", "benchmark": "b", "p": 0.02},
            {"model": "m2", "domain": "d", "benchmark": "b", "p": 0.01}]
    out = holm_within_families(rows)
    check("a family of two is corrected by two, not by three",
          abs(out[0]["p_holm"] - 0.02) < 1e-9, f"{out[0]['p_holm']}")
    check("a family of one is not corrected",
          abs(out[2]["p_holm"] - 0.01) < 1e-9, f"{out[2]['p_holm']}")

    print("\n3. across-seed spread")
    # Three seeds with accuracies 0.4, 0.5, 0.6 on 10 items each.
    def labels(acc, n=10, offset=0):
        k = int(round(acc * n))
        return {i: (1 if i < k else 0) for i in range(offset, offset + n)}
    by_seed = {1: labels(0.4), 2: labels(0.5), 3: labels(0.6)}
    s = across_seed_summary(by_seed)
    check("mean across seeds is right", abs(s["mean"] - 0.5) < 1e-9)
    check("sd uses ddof=1", abs(s["sd"] - np.std([0.4, 0.5, 0.6], ddof=1)) < 1e-12)
    check("three seeds are counted", s["n_seeds"] == 3)
    one = across_seed_summary({1: labels(0.4)})
    check("one seed reports NaN spread, not zero", math.isnan(one["sd"]))

    print("\n4. per-item mean across seeds")
    a = {1: {10: 1, 11: 0}, 2: {10: 1, 11: 1}}
    check("items are averaged over seeds", per_item_mean(a) == {10: 1.0, 11: 0.5})
    b = {1: {10: 1, 11: 0}, 2: {10: 1, 12: 1}}
    check("only items every seed saw are kept", set(per_item_mean(b)) == {10})

    print("\n5. paired bootstrap")
    # Arm a beats arm b on every item, by exactly 1.0
    a1 = {i: 1.0 for i in range(200)}
    b1 = {i: 0.0 for i in range(200)}
    r = paired_bootstrap(a1, b1)
    check("a deterministic difference is recovered exactly", abs(r["diff"] - 1.0) < 1e-12)
    check("its interval has zero width", r["ci_low"] == r["ci_high"] == 1.0)
    check("identical arms give a zero difference",
          abs(paired_bootstrap(a1, a1)["diff"]) < 1e-12)
    rng = np.random.default_rng(0)
    noise_a = {i: float(v) for i, v in enumerate(rng.integers(0, 2, 400))}
    noise_b = {i: float(v) for i, v in enumerate(rng.integers(0, 2, 400))}
    rn = paired_bootstrap(noise_a, noise_b)
    check("two independent coin flips are not distinguishable",
          rn["ci_low"] < 0 < rn["ci_high"], f"[{rn['ci_low']:.3f}, {rn['ci_high']:.3f}]")
    check("the bootstrap is reproducible",
          paired_bootstrap(noise_a, noise_b)["ci_low"] == rn["ci_low"])
    check("only shared items are used",
          paired_bootstrap({1: 1.0, 2: 1.0}, {2: 0.0, 3: 0.0})["n_shared"] == 1)

    print("\n6. seed variance is reported alongside the item interval")
    # Same mean difference every seed: sd = 0, sign consistent.
    stable_a = {s: {i: 1 for i in range(50)} for s in (1, 2, 3)}
    stable_b = {s: {i: 0 for i in range(50)} for s in (1, 2, 3)}
    c = compare_conditions(stable_a, stable_b)
    check("a perfectly stable effect has zero seed spread", abs(c["seed_diff_sd"]) < 1e-12)
    check("and is flagged sign-consistent", c["sign_consistent"] is True)
    # An effect that flips sign between seeds must not look stable.
    flip_a = {1: {i: 1 for i in range(50)}, 2: {i: 0 for i in range(50)}}
    flip_b = {1: {i: 0 for i in range(50)}, 2: {i: 1 for i in range(50)}}
    f = compare_conditions(flip_a, flip_b)
    check("an effect that flips sign is not sign-consistent", f["sign_consistent"] is False)
    check("its seed spread is large", f["seed_diff_sd"] > 1.0, f"{f['seed_diff_sd']:.3f}")
    check("yet its item-level mean difference is zero, which is why both are reported",
          abs(f["diff"]) < 1e-12)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_self_test() if a.self_test else print(__doc__))
