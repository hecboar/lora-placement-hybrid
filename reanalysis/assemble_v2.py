"""Splice the generated LaTeX tables into the v2 template.

Every number in the manuscript body comes from generated_tables.json, which is
produced directly from the corrected data, so nothing is transcribed by hand.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "arxiv_v2")

tables = json.load(open(os.path.join(V2, "generated_tables.json"), encoding="utf-8"))
src = open(os.path.join(V2, "template_v2.tex"), encoding="utf-8").read()

used, missing = set(), []
for m in sorted(set(re.findall(r"%%([A-Z0-9_]+)%%", src))):
    if m not in tables:
        missing.append(m)
    else:
        used.add(m)
if missing:
    sys.exit("Template references unknown markers: " + ", ".join(missing))

unused = sorted(set(tables) - used)
for m, v in tables.items():
    src = src.replace("%%" + m + "%%", v)

leftover = re.findall(r"%%([A-Z0-9_]+)%%", src)
if leftover:
    sys.exit("Unsubstituted markers remain: " + ", ".join(leftover))

out = os.path.join(V2, "paper3_lora_placement_v2.tex")
open(out, "w", encoding="utf-8").write(src)
print("wrote", out)
print("substituted", len(used), "markers")
if unused:
    print("generated but unused:", ", ".join(unused))
