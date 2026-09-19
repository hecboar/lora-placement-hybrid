"""Re-execute saved HumanEval completions against the official unit tests.

The original pipeline called evaluate.load("code_eval") without setting
HF_ALLOW_CODE_EVAL=1, so it raised and pass@1 was recorded as NaN / 0.
This script recovers the real pass@1 from the saved completions and writes
per-instance correctness so paired bootstrap can be computed.
"""
import json, glob, os, sys, subprocess, tempfile
import pyarrow as pa, pyarrow.ipc as ipc


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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reanalysis", "eval_details_corrected")
os.makedirs(OUT, exist_ok=True)

arrow = os.path.join(ROOT, "3-LoRA-Placement", "3-LoRA-Placement", "data_cache",
                     "benchmark__humaneval", "test", "data-00000-of-00001.arrow")
with pa.memory_map(arrow) as src:
    try:
        tbl = ipc.open_stream(src).read_all()
    except Exception:
        tbl = ipc.open_file(src).read_all()
HE = {r["task_id"]: r for r in tbl.to_pylist()}

# Standard HumanEval stop sequences (Chen et al. 2021).
STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```", "\n@"]

def truncate(completion):
    cut = len(completion)
    for s in STOPS:
        i = completion.find(s)
        if i != -1:
            cut = min(cut, i)
    return completion[:cut]

def passes(task, completion, timeout=10):
    prog = (task["prompt"] + completion + "\n\n" + task["test"] + "\n"
            + f"check({task['entry_point']})\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass

files = sorted(glob.glob(os.path.join(_results_dir("eval_details"), "*humaneval*.jsonl")))
summary = []
for fp in files:
    rows = [json.loads(l) for l in open(fp, encoding="utf-8")]
    out_rows, n_raw, n_trunc = [], 0, 0
    for r in rows:
        task = HE[r["task_id"]]
        raw_ok = passes(task, r["completion"])
        tr_ok = passes(task, truncate(r["completion"]))
        n_raw += raw_ok; n_trunc += tr_ok
        out_rows.append({"benchmark": "humaneval", "example_id": r["example_id"],
                         "task_id": r["task_id"], "correct": int(tr_ok),
                         "correct_untruncated": int(raw_ok)})
    base = os.path.basename(fp)
    with open(os.path.join(OUT, base), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summary.append({"file": base, "n": n, "pass_raw": n_raw, "pass_truncated": n_trunc,
                    "acc_raw": n_raw / n, "acc_truncated": n_trunc / n})
    print(f"{base.replace('__inline_eval__humaneval.jsonl',''):62s} n={n:3d} "
          f"raw={n_raw/n:.3f} truncated={n_trunc/n:.3f}", flush=True)

json.dump(summary, open(os.path.join(ROOT, "reanalysis", "humaneval_rescore_summary.json"), "w"), indent=1)
print("DONE")
