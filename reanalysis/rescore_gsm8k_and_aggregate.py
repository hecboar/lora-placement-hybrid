"""Re-score the released per-instance outputs with a corrected GSM8K answer
extractor, and rebuild the accuracy / delta / off-target tables.

Bug being corrected
-------------------
evaluate_gsm8k() generated 256 greedy tokens with NO stop sequence and then
called extract_final_number(), which returns the LAST "#### N" in the text.
The few-shot prompt template repeats

    You are a careful mathematician. ... end with '#### <final answer>'.
    Question: {q}
    Answer:

so after emitting its real answer the model simply continues the pattern,
hallucinating a fresh "Question:" block with its own "#### N". That trailing
number is what got scored.

Correction applied here: truncate each saved completion at the first stop
marker a correctly configured harness would have used, then apply the
ORIGINAL extractor and the ORIGINAL normaliser unchanged. This reproduces
what the pipeline would have recorded with stop sequences enabled.
"""
import json, glob, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "github", "results", "eval_details")
OUT = os.path.join(ROOT, "reanalysis")
CORR = os.path.join(OUT, "eval_details_corrected")
os.makedirs(CORR, exist_ok=True)


# ---------------------------------------------------------------- extractors
def extract_final_number(text):
    """First '####' in the (already truncated) text, else the last bare number.

    The original pipeline took the LAST '####'. Truncating at the stop marker removes
    the hallucinated follow-up question, but 13 of the 8,832 released completions still
    carry more than one '####' inside the answer block itself, where the model kept
    rambling after giving its answer. Generation with a working stop criterion would
    never have produced that tail, so the first marker is the answer. The rule is
    chosen from what a correctly configured harness emits, not from which rule scores
    higher; it is the same rule `evaluator_v2/evaluator.py` applies to new runs, so the
    correction of v1 and the follow-up campaign are scored identically.
    """
    m = re.search(r"####\s*([-+]?[0-9][0-9,\.]*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.findall(r"[-+]?[0-9][0-9,\.]*", text)
    if m:
        return m[-1].replace(",", "").strip()
    return None


def normalize_number_string(s):
    """Verbatim copy of the pipeline normaliser."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        val = float(s)
        return str(int(val)) if val.is_integer() else str(val)
    except Exception:
        return s


STOPS = ["You are a careful mathematician", "\nQuestion:", "Question:"]


def truncate_at_stop(completion):
    cut = len(completion)
    for s in STOPS:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]


# ------------------------------------------------------------ file name parse
def parse_name(fn):
    base = fn[:-len(".jsonl")]
    m = re.match(r"^(.+?)__base__(\w+)$", base)
    if m:
        return dict(model_key=m.group(1), condition="__base__",
                    train_dataset="base", benchmark=m.group(2))
    m = re.match(r"^train__(.+?)__(.+?)__(gsm8k_train|ultrachat|codealpaca)"
                 r"__seed(\d+)__inline_eval__(\w+)$", base)
    if m:
        return dict(model_key=m.group(1), condition=m.group(2),
                    train_dataset=m.group(3), seed=int(m.group(4)),
                    benchmark=m.group(5))
    return None


# ------------------------------------------------------------------ rescoring
records, per_instance = [], {}
for fp in sorted(glob.glob(os.path.join(SRC, "*.jsonl"))):
    meta = parse_name(os.path.basename(fp))
    if meta is None:
        print("SKIP unparsed:", os.path.basename(fp))
        continue
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    bench = meta["benchmark"]

    if bench == "humaneval":
        continue                       # handled by the code-execution script

    if bench == "gsm8k":
        out_rows, n_rep, n_corr, n_changed, n_multi = [], 0, 0, 0, 0
        for r in rows:
            comp = r["raw_completion"]
            gold = normalize_number_string(r["gold_answer"])
            pred_corr = normalize_number_string(
                extract_final_number(truncate_at_stop(comp)))
            ok_corr = int(pred_corr == gold)
            n_rep += r["correct"]
            n_corr += ok_corr
            n_changed += int(ok_corr != r["correct"])
            n_multi += int(comp.count("####") > 1)
            out_rows.append({"benchmark": "gsm8k", "example_id": r["example_id"],
                             "correct": ok_corr, "correct_original": r["correct"],
                             "pred_answer": pred_corr,
                             "pred_answer_original": r["pred_answer"],
                             "gold_answer": gold})
        with open(os.path.join(CORR, os.path.basename(fp)), "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        meta.update(n=len(rows), acc_reported=n_rep / len(rows),
                    acc_corrected=n_corr / len(rows),
                    n_labels_changed=n_changed, n_multi_hash=n_multi)
        per_instance[(meta["model_key"], meta["condition"],
                      meta["train_dataset"], "gsm8k")] = \
            {r["example_id"]: r["correct"] for r in out_rows}
    else:                              # log-likelihood MC: unaffected by the bug
        n_rep = sum(r["correct"] for r in rows)
        meta.update(n=len(rows), acc_reported=n_rep / len(rows),
                    acc_corrected=n_rep / len(rows),
                    n_labels_changed=0, n_multi_hash=0)
        per_instance[(meta["model_key"], meta["condition"],
                      meta["train_dataset"], bench)] = \
            {r["example_id"]: r["correct"] for r in rows}
    records.append(meta)

# ---------------------------------------------------------- HumanEval (if run)
for fp in sorted(glob.glob(os.path.join(CORR, "*humaneval*.jsonl"))):
    meta = parse_name(os.path.basename(fp))
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    n_ok = sum(r["correct"] for r in rows)
    meta.update(n=len(rows), acc_reported=0.0, acc_corrected=n_ok / len(rows),
                n_labels_changed=n_ok, n_multi_hash=0)
    per_instance[(meta["model_key"], meta["condition"],
                  meta["train_dataset"], "humaneval")] = \
        {r["example_id"]: r["correct"] for r in rows}
    records.append(meta)

df = pd.DataFrame(records)
if "seed" not in df.columns:
    df["seed"] = 3407
df["seed"] = df["seed"].fillna(3407).astype(int)

# --------------------------------------------------------------- deltas vs base
base = (df[df.condition == "__base__"]
        [["model_key", "benchmark", "acc_reported", "acc_corrected"]]
        .rename(columns={"acc_reported": "base_reported",
                         "acc_corrected": "base_corrected"}))
df = df.merge(base, on=["model_key", "benchmark"], how="left")
df["delta_reported"] = df.acc_reported - df.base_reported
df["delta_corrected"] = df.acc_corrected - df.base_corrected
df = df.sort_values(["model_key", "train_dataset", "condition", "benchmark"])
df.to_csv(os.path.join(OUT, "all_results_corrected.csv"), index=False)

# ------------------------------------------------------------ accuracy matrices
for tag in ["reported", "corrected"]:
    piv = df[df.condition != "__base__"].pivot_table(
        index=["model_key", "train_dataset", "condition"],
        columns="benchmark", values="acc_" + tag)
    piv.to_csv(os.path.join(OUT, "result_matrix_" + tag + ".csv"))

# -------------------------------------------------------- off-target summaries
TARGET = {"gsm8k_train": {"gsm8k"},
          "ultrachat": {"mmlu", "arc_challenge", "hellaswag"},
          "codealpaca": {"humaneval"}}
off = []
for (mk, cond, ds), g in df[df.condition != "__base__"].groupby(
        ["model_key", "condition", "train_dataset"]):
    tgt = TARGET.get(ds, set())
    on = g[g.benchmark.isin(tgt)]
    ofb = g[~g.benchmark.isin(tgt)]
    off.append(dict(model_key=mk, condition=cond, train_dataset=ds,
                    on_target_delta_reported=on.delta_reported.mean(),
                    on_target_delta_corrected=on.delta_corrected.mean(),
                    off_target_delta_reported=ofb.delta_reported.mean(),
                    off_target_delta_corrected=ofb.delta_corrected.mean(),
                    mean_acc_reported=g.acc_reported.mean(),
                    mean_acc_corrected=g.acc_corrected.mean()))
pd.DataFrame(off).to_csv(os.path.join(OUT, "off_target_corrected.csv"), index=False)


# ----------------------------------------------------------- paired bootstrap
def paired_bootstrap(a, b, n_boot=10000, seed=3407):
    shared = sorted(set(a) & set(b))
    if not shared:
        return dict(n_shared=0, mean_diff=np.nan, ci_low=np.nan, ci_high=np.nan)
    d = (np.array([a[i] for i in shared], dtype=np.float64)
         - np.array([b[i] for i in shared], dtype=np.float64))
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return dict(n_shared=len(d), mean_diff=float(d.mean()),
                ci_low=float(np.percentile(boot, 2.5)),
                ci_high=float(np.percentile(boot, 97.5)))


KEY = [
    ("Qwen GSM8K: softmax_only vs base", "qwen3_5_0_8b_base", "gsm8k_train",
     "gsm8k", "softmax_only", "__base__"),
    ("Qwen GSM8K: gdn_only vs base", "qwen3_5_0_8b_base", "gsm8k_train",
     "gsm8k", "gdn_only", "__base__"),
    ("Qwen GSM8K: softmax_plus_mlp vs base", "qwen3_5_0_8b_base", "gsm8k_train",
     "gsm8k", "softmax_plus_mlp", "__base__"),
    ("Qwen GSM8K: softmax_plus_mlp vs mlp_only", "qwen3_5_0_8b_base",
     "gsm8k_train", "gsm8k", "softmax_plus_mlp", "mlp_only"),
    ("Qwen GSM8K: softmax_only vs all_layers", "qwen3_5_0_8b_base",
     "gsm8k_train", "gsm8k", "softmax_only", "all_layers"),
    ("Qwen GSM8K: gdn_only vs softmax_only", "qwen3_5_0_8b_base", "gsm8k_train",
     "gsm8k", "gdn_only", "softmax_only"),
    ("Falcon GSM8K: attention_only vs base", "falcon_h1_0_5b_base",
     "gsm8k_train", "gsm8k", "attention_only", "__base__"),
    ("Falcon GSM8K: attention_only vs all_eligible", "falcon_h1_0_5b_base",
     "gsm8k_train", "gsm8k", "attention_only", "all_eligible"),
    ("Falcon GSM8K: attention_only vs ssm_only", "falcon_h1_0_5b_base",
     "gsm8k_train", "gsm8k", "attention_only", "ssm_only"),
    ("Falcon GSM8K: ssm_only vs base", "falcon_h1_0_5b_base", "gsm8k_train",
     "gsm8k", "ssm_only", "__base__"),
    ("Falcon UltraChat GSM8K: attention_only vs all_eligible",
     "falcon_h1_0_5b_base", "ultrachat", "gsm8k", "attention_only", "all_eligible"),
    ("Falcon UltraChat HellaSwag: attention_only vs all_eligible",
     "falcon_h1_0_5b_base", "ultrachat", "hellaswag", "attention_only", "all_eligible"),
]
rows = []
for label, mk, ds, bench, ca, cb in KEY:
    ka = (mk, ca, ds, bench)
    kb = (mk, cb, "base" if cb == "__base__" else ds, bench)
    if ka not in per_instance or kb not in per_instance:
        print("missing:", label)
        continue
    r = paired_bootstrap(per_instance[ka], per_instance[kb])
    r.update(label=label, model_key=mk, train_dataset=ds, benchmark=bench,
             condition_a=ca, condition_b=cb)
    for k in ["mean_diff", "ci_low", "ci_high"]:
        r[k + "_pp"] = r[k] * 100
    rows.append(r)
bs = pd.DataFrame(rows)
bs.to_csv(os.path.join(OUT, "key_bootstrap_corrected.csv"), index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
print("\n===== GSM8K: reported vs corrected =====")
print(df[df.benchmark == "gsm8k"][
    ["model_key", "train_dataset", "condition", "n", "acc_reported",
     "acc_corrected", "n_labels_changed", "n_multi_hash"]].to_string(index=False))
print("\n===== Key paired bootstrap on CORRECTED labels (pp) =====")
print(bs[["label", "n_shared", "mean_diff_pp", "ci_low_pp",
          "ci_high_pp"]].to_string(index=False))
print("\nWrote artifacts to:", OUT)
