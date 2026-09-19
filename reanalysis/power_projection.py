"""What becomes detectable once the evaluation is done on full test splits?

The current subsets (128 Qwen / 256 Falcon GSM8K items) can only resolve ~10 pp.
The corrected point estimates are 2-7 pp. So "not significant" today is a statement
about the instrument, not about the effects. This projects each corrected comparison
onto the full GSM8K test split (1319 items) and onto 3 seeds, using the measured
standard deviation of the paired per-item difference.
"""
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bs = pd.read_csv(os.path.join(ROOT, "reanalysis", "key_bootstrap_corrected.csv"))
pw = pd.read_csv(os.path.join(ROOT, "reanalysis", "power_analysis_gsm8k.csv"))

SD = float(pw.sd_of_paired_diff.median())          # 0.46, measured
Z = 1.959964
FULL = {"gsm8k": 1319, "hellaswag": 10042, "arc_challenge": 1172, "mmlu": 14042}
# We would cap the very large multiple-choice sets; 2048 is the planned cap.
CAP = 2048


def halfwidth(n, seeds=1):
    # seed averaging shrinks the standard error of the mean difference by sqrt(seeds),
    # ignoring between-seed variance, which is why this is an optimistic bound
    return Z * SD / np.sqrt(n * seeds) * 100


rows = []
for _, r in bs.iterrows():
    bench = r.benchmark
    n_now = int(r.n_shared)
    n_full = min(FULL.get(bench, n_now), CAP) if bench != "gsm8k" else FULL["gsm8k"]
    d = r.mean_diff_pp
    hw_now = (r.ci_high_pp - r.ci_low_pp) / 2
    hw_1 = halfwidth(n_full, 1)
    hw_3 = halfwidth(n_full, 3)
    rows.append(dict(
        comparison=r.label.replace("_plus_", "+").replace("_", "-"),
        n_now=n_now, n_full=n_full, effect_pp=round(d, 1),
        sig_now="yes" if (r.ci_low_pp > 0 or r.ci_high_pp < 0) else "no",
        hw_now_pp=round(hw_now, 1), hw_full_pp=round(hw_1, 1), hw_3seed_pp=round(hw_3, 1),
        sig_full="yes" if abs(d) > hw_1 else "no",
        sig_3seed="yes" if abs(d) > hw_3 else "no"))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 60)
print("Projection of each corrected comparison onto full test splits\n")
print(df.to_string(index=False))

flips1 = df[(df.sig_now == "no") & (df.sig_full == "yes")]
flips3 = df[(df.sig_now == "no") & (df.sig_3seed == "yes")]
print(f"\nSignificant today                      : {(df.sig_now == 'yes').sum()} / {len(df)}")
print(f"Would become significant, full splits  : {len(flips1)} more")
print(f"Would become significant, + 3 seeds    : {len(flips3)} more")
if len(flips3):
    print("\nComparisons that flip from 'undetectable' to 'detectable':")
    for _, r in flips3.iterrows():
        print(f"  {r.comparison:58s} {r.effect_pp:+6.1f} pp   "
              f"(half-width {r.hw_now_pp:.1f} -> {r.hw_3seed_pp:.1f} pp)")

print(f"""
Smallest effect resolvable, by design:
  today, GSM8K 256 items, 1 seed     : {halfwidth(256,1):.1f} pp
  full split 1319 items, 1 seed      : {halfwidth(1319,1):.1f} pp
  full split 1319 items, 3 seeds     : {halfwidth(1319,3):.1f} pp

The corrected point estimates sit at 2-7 pp. The current instrument cannot see them.
The planned one can. Whether they are real is exactly what the campaign decides, and
that is a different statement from 'the result will be negative'.""")
