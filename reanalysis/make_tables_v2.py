"""Generate every numeric LaTeX table for arXiv v2 directly from the corrected
data, so no number is transcribed by hand.

Writes arxiv_v2/generated_tables.json: marker name -> LaTeX block. assemble_v2.py
splices these into the template.
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reanalysis")
OUT = os.path.join(ROOT, "arxiv_v2")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "all_results_corrected.csv"))
bs = pd.read_csv(os.path.join(DATA, "key_bootstrap_corrected.csv"))
pw = pd.read_csv(os.path.join(DATA, "power_analysis_gsm8k.csv"))
allbs = pd.read_csv(os.path.join(DATA, "all_pairwise_bootstrap_corrected.csv"))

QW, FA = "qwen3_5_0_8b_base", "falcon_h1_0_5b_base"
MACRO = {QW: r"\modelqwen", FA: r"\modelfalcon"}
ORDER = {QW: ["all_layers", "softmax_only", "gdn_only", "mlp_only",
              "softmax_plus_mlp", "gdn_plus_mlp"],
         FA: ["all_eligible", "attention_only", "ssm_only", "mlp_only",
              "attention_plus_mlp", "ssm_plus_mlp"]}
BENCH = ["mmlu", "gsm8k", "arc_challenge", "hellaswag"]
PARAMS = {  # trainable parameters in millions, rank 16 (unchanged from v1)
    (QW, "all_layers"): 10.82, (QW, "softmax_only"): 1.08, (QW, "gdn_only"): 4.43,
    (QW, "mlp_only"): 5.31, (QW, "softmax_plus_mlp"): 6.39, (QW, "gdn_plus_mlp"): 9.74,
    (FA, "all_eligible"): 11.47, (FA, "attention_only"): 2.21, (FA, "ssm_only"): 2.52,
    (FA, "mlp_only"): 5.31, (FA, "attention_plus_mlp"): 7.52, (FA, "ssm_plus_mlp"): 7.83,
}


def tt(c):
    return r"\texttt{" + c.replace("_", r"\_") + "}"


def acc(model, cond, domain, bench, col="acc_corrected"):
    r = df[(df.model_key == model) & (df.condition == cond)
           & (df.train_dataset == domain) & (df.benchmark == bench)]
    return float(r.iloc[0][col]) if len(r) else np.nan


def f3(x):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"


def pp(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{abs(x)*100:.1f}"
    if not signed:
        return s
    return ("+" if x >= 0 else "$-$") + s


T = {}

# ------------------------------------------------------------- baselines
rows = []
for m in (QW, FA):
    vals = [acc(m, "__base__", "base", b) for b in BENCH]
    old = [acc(m, "__base__", "base", b, "acc_reported") for b in BENCH]
    rows.append(f"{MACRO[m]}  & {f3(vals[0])} & {f3(vals[1])} ({f3(old[1])}) "
                f"& {f3(vals[2])} & {f3(vals[3])} \\\\")
T["BASELINES"] = "\n".join(rows)

# ------------------------------------------------------- main GSM8K table
blocks = []
for m in (QW, FA):
    conds = sorted(ORDER[m], key=lambda c: -acc(m, c, "gsm8k_train", "gsm8k"))
    body = []
    best = conds[0]
    for c in conds:
        g = acc(m, c, "gsm8k_train", "gsm8k")
        d = g - acc(m, "__base__", "base", "gsm8k")
        gs = (r"\textbf{" + f3(g) + "}") if c == best else f3(g)
        body.append(f"  & {tt(c)} & {f3(acc(m,c,'gsm8k_train','mmlu'))} & {gs} "
                    f"& {f3(acc(m,c,'gsm8k_train','arc_challenge'))} "
                    f"& {f3(acc(m,c,'gsm8k_train','hellaswag'))} & {pp(d)} \\\\")
    blocks.append(r"\multirow{6}{*}{\rotatebox[origin=c]{90}{\footnotesize "
                  + MACRO[m] + "}}\n" + "\n".join(body))
T["MAIN_GSM8K"] = "\n\\midrule\n".join(blocks)

# --------------------------------------------------------- efficiency table
blocks = []
for m in (QW, FA):
    broad = "all_layers" if m == QW else "all_eligible"
    narrow = "softmax_only" if m == QW else "attention_only"
    body = []
    for c in [narrow, "mlp_only", broad]:
        d = acc(m, c, "gsm8k_train", "gsm8k") - acc(m, "__base__", "base", "gsm8k")
        p = PARAMS[(m, c)]
        ratio = d * 100 / p
        body.append(f"  & {tt(c)} & {p:.2f} & {pp(d)} & "
                    + (f"{ratio:.1f}" if ratio >= 0 else f"$-${abs(ratio):.1f}") + r" \\")
    blocks.append(r"\multirow{3}{*}{" + MACRO[m] + "}\n" + "\n".join(body))
T["EFFICIENCY"] = "\n\\midrule\n".join(blocks)

# ------------------------------------------------------------ bootstrap key
V1 = {  # the corresponding value as printed in v1, for the side-by-side column
    "Qwen GSM8K: softmax_only vs base": "+10.2",
    "Qwen GSM8K: gdn_only vs base": "$-$14.8",
    "Qwen GSM8K: softmax_plus_mlp vs base": "$-$14.8",
    "Qwen GSM8K: softmax_plus_mlp vs mlp_only": "$-$23.4",
    "Qwen GSM8K: softmax_only vs all_layers": "--",
    "Qwen GSM8K: gdn_only vs softmax_only": "--",
    "Falcon GSM8K: attention_only vs base": "+17.2",
    "Falcon GSM8K: attention_only vs all_eligible": "+5.1",
    "Falcon GSM8K: attention_only vs ssm_only": "+8.6",
    "Falcon GSM8K: ssm_only vs base": "+8.6",
    "Falcon UltraChat GSM8K: attention_only vs all_eligible": "+5.5",
    "Falcon UltraChat HellaSwag: attention_only vs all_eligible": "+2.7",
}
rows = []
for _, r in bs.iterrows():
    lab = r.label.replace("_plus_", " + ").replace("_", r"\_")
    sig = r"$\ast$" if (r.ci_low_pp > 0 or r.ci_high_pp < 0) else ""
    rows.append(f"{lab} & {int(r.n_shared)} & {V1.get(r.label,'--')} & "
                f"{pp(r.mean_diff/1.0)} & [{pp(r.ci_low)}, {pp(r.ci_high)}] & {sig} \\\\")
T["BOOTSTRAP"] = "\n".join(rows)

# ----------------------------------------------------------- Holm survivors
sv = allbs[allbs.sig_holm].sort_values("p_holm")
rows = []
for _, r in sv.iterrows():
    mod = r"\modelqwen" if r.model == "qwen3" else r"\modelfalcon"
    dom = {"gsm8k_train": "GSM8K", "ultrachat": "UltraChat",
           "codealpaca": "CodeAlpaca"}[r.domain]
    ben = {"mmlu": "MMLU", "gsm8k": "GSM8K", "arc_challenge": "ARC-C",
           "hellaswag": "HellaSwag", "humaneval": "HumanEval"}[r.benchmark]
    a = r.cond_a.replace("_plus_", " + ").replace("_", r"\_")
    b = "base" if r.cond_b == "__base__" else r.cond_b.replace("_plus_", " + ").replace("_", r"\_")
    rows.append(f"{mod} & {dom} & {ben} & \\texttt{{{a}}} $-$ \\texttt{{{b}}} & "
                f"{pp(r.diff_pp/100)} & [{pp(r.ci_low_pp/100)}, {pp(r.ci_high_pp/100)}] & "
                f"{r.p_holm:.3f} \\\\")
T["SURVIVORS"] = "\n".join(rows)

# -------------------------------------------------------------- HumanEval
he = df[(df.benchmark == "humaneval") & (df.condition != "__base__")].copy()
he = he.sort_values(["model_key", "acc_corrected"], ascending=[True, False])
rows = []
for m in (QW, FA):
    sub = he[he.model_key == m]
    for i, (_, r) in enumerate(sub.iterrows()):
        lead = r"\multirow{6}{*}{" + MACRO[m] + "}" if i == 0 else ""
        rows.append(f"{lead} & {tt(r.condition)} & 0.000 & {r.acc_corrected:.3f} \\\\")
    if m == QW:
        rows.append(r"\midrule")
T["HUMANEVAL"] = "\n".join(rows)

# ------------------------------------------------------- full result matrices
DOM = [("gsm8k_train", "GSM8K"), ("codealpaca", "CodeAlpaca"), ("ultrachat", "UltraChat")]
for m, key in ((QW, "FULL_QWEN"), (FA, "FULL_FALCON")):
    blocks = []
    for dkey, dlab in DOM:
        body = []
        conds = sorted(ORDER[m], key=lambda c: -np.nanmean(
            [acc(m, c, dkey, b) for b in BENCH]))
        for c in conds:
            body.append(f"  & {tt(c)} & " + " & ".join(
                f3(acc(m, c, dkey, b)) for b in BENCH) + r" \\")
        blocks.append(r"\multirow{6}{*}{" + dlab + "}\n" + "\n".join(body))
    T[key] = "\n\\midrule\n".join(blocks)

# -------------------------------------------------------------- power table
med = pw.sd_of_paired_diff.median()
rows = []
for eff, col in [(3, "n_for_3pp"), (5, "n_for_5pp"), (10, "n_for_10pp")]:
    rows.append(f"{eff} & {int(pw[col].median()):,} \\\\".replace(",", "{,}"))
T["POWER"] = "\n".join(rows)
T["POWER_SD"] = f"{med:.2f}"
T["POWER_HALFWIDTH"] = f"{pw.ci_half_width_pp.median():.1f}"

# ------------------------------------------------------- inline scalar macros
T["N_SIG_RAW"] = str(int(allbs.sig_raw.sum()))
T["N_SIG_HOLM"] = str(int(allbs.sig_holm.sum()))
T["N_COMPARISONS"] = str(len(allbs))

json.dump(T, open(os.path.join(OUT, "generated_tables.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "generated_tables.json"))
for k in T:
    v = T[k]
    print(f"  {k:16s} {len(v.splitlines()) if len(v) > 40 else v}")
