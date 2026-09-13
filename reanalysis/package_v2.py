"""Verify every referenced figure exists and stage the arXiv submission tree."""
import os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "arxiv_v2")
TEX = os.path.join(V2, "paper3_lora_placement_v2.tex")
SUB = os.path.join(V2, "submission")

tex = open(TEX, encoding="utf-8").read()
refs = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)))
print("figures referenced:", len(refs))

missing = []
for r in refs:
    p = os.path.join(V2, "figures", r)
    ok = os.path.exists(p)
    print(("  OK   " if ok else "  MISS "), r,
          (f"{os.path.getsize(p):,} B" if ok else ""))
    if not ok:
        missing.append(r)
if missing:
    sys.exit("missing figures: " + ", ".join(missing))

if os.path.isdir(SUB):
    shutil.rmtree(SUB)
os.makedirs(os.path.join(SUB, "figures"))
shutil.copy(TEX, SUB)
for r in refs:
    shutil.copy(os.path.join(V2, "figures", r), os.path.join(SUB, "figures", r))

total = sum(os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(SUB) for f in fs)
print(f"\nstaged {SUB}")
print(f"  {len(refs) + 1} files, {total/1e6:.2f} MB uncompressed")
