"""How large must the evaluation subset be to detect the effects that actually
exist once the GSM8K extractor is fixed?

Uses the corrected per-instance labels to estimate the standard deviation of
the paired per-item difference, then solves for the n needed to detect a given
effect at 80% power / alpha 0.05 (two-sided, normal approximation), and
reports the achieved half-width of the current subsets.
"""
import json, glob, os, itertools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORR = os.path.join(ROOT, "reanalysis", "eval_details_corrected")

Z_A, Z_B = 1.959964, 0.8416212      # alpha=0.05 two-sided, power=0.80


def load(fp):
    return {json.loads(l)["example_id"]: json.loads(l)["correct"]
            for l in open(fp, encoding="utf-8")}


def key(fp):
    b = os.path.basename(fp)[:-len(".jsonl")]
    parts = b.replace("train__", "").split("__")
    return parts[0] + "/" + parts[1] + "/" + parts[2]


rows = []
for model_tag in ["qwen3_5_0_8b_base", "falcon_h1_0_5b_base"]:
    files = sorted(glob.glob(os.path.join(
        CORR, "train__" + model_tag + "__*gsm8k_train*gsm8k.jsonl")))
    conds = {key(f).split("/")[1]: load(f) for f in files}
    for a, b in itertools.combinations(sorted(conds), 2):
        ka, kb = conds[a], conds[b]
        shared = sorted(set(ka) & set(kb))
        d = (np.array([ka[i] for i in shared], float)
             - np.array([kb[i] for i in shared], float))
        sd = d.std(ddof=1)
        n_cur = len(d)
        half_width = Z_A * sd / np.sqrt(n_cur)
        rows.append(dict(model=model_tag.split("_")[0], pair=a + " vs " + b,
                         n_current=n_cur,
                         observed_diff_pp=round(d.mean() * 100, 1),
                         sd_of_paired_diff=round(sd, 3),
                         ci_half_width_pp=round(half_width * 100, 1),
                         n_for_3pp=int(np.ceil(((Z_A + Z_B) * sd / 0.03) ** 2)),
                         n_for_5pp=int(np.ceil(((Z_A + Z_B) * sd / 0.05) ** 2)),
                         n_for_10pp=int(np.ceil(((Z_A + Z_B) * sd / 0.10) ** 2))))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "reanalysis", "power_analysis_gsm8k.csv"), index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
print(df.to_string(index=False))
print("\n---- summary over all condition pairs (GSM8K-trained, corrected) ----")
print("median SD of paired difference :", round(df.sd_of_paired_diff.median(), 3))
print("median achieved CI half-width  :", round(df.ci_half_width_pp.median(), 1), "pp")
print("median n needed for 3 pp        :", int(df.n_for_3pp.median()))
print("median n needed for 5 pp        :", int(df.n_for_5pp.median()))
print("median n needed for 10 pp       :", int(df.n_for_10pp.median()))
print("\nGSM8K full test split has 1319 items.")
