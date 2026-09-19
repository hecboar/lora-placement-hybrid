"""Exhaustive search for comparisons that remain statistically detectable once
the GSM8K extractor is fixed.

Runs a paired percentile bootstrap (10,000 resamples, seed 3407, matched
example_id) over every condition-vs-condition and condition-vs-base comparison,
for every model x training domain x benchmark, and reports the ones whose 95%
interval excludes zero. Holm correction is applied within each
(model, domain, benchmark) family.
"""
import json, glob, os, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "github", "results", "eval_details")
CORR = os.path.join(ROOT, "reanalysis", "eval_details_corrected")
OUT = os.path.join(ROOT, "reanalysis")

MC = {"mmlu", "arc_challenge", "hellaswag"}

# The per-instance files record `example_id` as the POSITION within that run's fixed
# evaluation subset, not as the index of the item in the benchmark. Two runs that used
# subsets of different sizes therefore share positions 0..n-1 while those positions
# point at different questions: for GSM8K the 128- and 256-item subsets have only 22
# items in common and agree on none of the 128 positions. Pairing on `example_id` alone
# silently compares different questions, which is the same class of error as the
# extraction defect. Positions are mapped back to benchmark indices here so that every
# paired comparison is over the same questions.
IDX_DIR = os.path.join(ROOT, "github", "results", "eval_indices")
_SPLIT = {"gsm8k": "test", "mmlu": "validation", "arc_challenge": "validation",
          "hellaswag": "validation", "humaneval": "test"}


def _index_map(bench, n):
    """Position -> benchmark index for the fixed subset of size n, if it is released."""
    fp = os.path.join(IDX_DIR, f"indices__{bench}__{_SPLIT.get(bench, 'test')}__{n}.json")
    if not os.path.exists(fp):
        return None
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def load_labels(model, condition, domain, bench):
    """Corrected labels where a corrected file exists, original otherwise.

    Keys are benchmark indices, so comparisons pair the same questions.
    """
    if condition == "__base__":
        stem = model + "__base__" + bench + ".jsonl"
    else:
        stem = ("train__" + model + "__" + condition + "__" + domain
                + "__seed3407__inline_eval__" + bench + ".jsonl")
    for d in (CORR, SRC):
        fp = os.path.join(d, stem)
        if not os.path.exists(fp):
            continue
        try:
            rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
        except json.JSONDecodeError:
            continue                      # file still being written
        if not rows or "correct" not in rows[0]:
            continue                      # e.g. raw HumanEval: completions only
        pos2idx = _index_map(bench, len(rows))
        if pos2idx and len(pos2idx) == len(rows):
            return {pos2idx[int(r["example_id"])]: r["correct"] for r in rows}
        return {r["example_id"]: r["correct"] for r in rows}
    return None


def paired_bootstrap(a, b, n_boot=10000, seed=3407):
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    d = (np.array([a[i] for i in shared], float)
         - np.array([b[i] for i in shared], float))
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided bootstrap p-value: proportion of resamples on the other side of 0
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(n_shared=len(d), diff_pp=d.mean() * 100,
                ci_low_pp=lo * 100, ci_high_pp=hi * 100, p=min(p, 1.0))


CONDS = {
    "qwen3_5_0_8b_base": ["all_layers", "softmax_only", "gdn_only", "mlp_only",
                          "softmax_plus_mlp", "gdn_plus_mlp"],
    "falcon_h1_0_5b_base": ["all_eligible", "attention_only", "ssm_only",
                            "mlp_only", "attention_plus_mlp", "ssm_plus_mlp"],
}
DOMAINS = ["gsm8k_train", "ultrachat", "codealpaca"]
BENCHES = ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "humaneval"]

rows = []
for model, conds in CONDS.items():
    for domain in DOMAINS:
        for bench in BENCHES:
            labels = {}
            for c in conds + ["__base__"]:
                lb = load_labels(model, c, domain, bench)
                if lb:
                    labels[c] = lb
            present = [c for c in conds if c in labels]
            pairs = list(itertools.combinations(present, 2))
            pairs += [(c, "__base__") for c in present if "__base__" in labels]
            for a, b in pairs:
                r = paired_bootstrap(labels[a], labels[b])
                if r is None:
                    continue
                r.update(model=model.split("_")[0], domain=domain,
                         benchmark=bench, cond_a=a, cond_b=b)
                rows.append(r)

df = pd.DataFrame(rows)

# Holm correction within each (model, domain, benchmark) family
df["p_holm"] = np.nan
for _, idx in df.groupby(["model", "domain", "benchmark"]).groups.items():
    sub = df.loc[idx].sort_values("p")
    m = len(sub)
    adj, prev = [], 0.0
    for rank, p in enumerate(sub.p.values):
        v = max(prev, min(1.0, (m - rank) * p))
        adj.append(v)
        prev = v
    df.loc[sub.index, "p_holm"] = adj

df["sig_raw"] = (df.ci_low_pp > 0) | (df.ci_high_pp < 0)
df["sig_holm"] = df.p_holm < 0.05
df = df.sort_values("p")
df.to_csv(os.path.join(OUT, "all_pairwise_bootstrap_corrected.csv"), index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
cols = ["model", "domain", "benchmark", "cond_a", "cond_b", "n_shared",
        "diff_pp", "ci_low_pp", "ci_high_pp", "p", "p_holm"]
print("Total comparisons:", len(df))
print("Significant before correction:", int(df.sig_raw.sum()))
print("Significant after Holm within family:", int(df.sig_holm.sum()))
print("\n===== SURVIVORS (Holm-corrected, p < 0.05) =====")
s = df[df.sig_holm]
print(s[cols].round(2).to_string(index=False) if len(s) else "  none")
print("\n===== Uncorrected-significant, GSM8K only (what the paper argued about) =====")
g = df[(df.benchmark == "gsm8k") & df.sig_raw]
print(g[cols].round(2).to_string(index=False) if len(g) else "  none")
