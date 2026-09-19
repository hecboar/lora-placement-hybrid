"""Corrected, batched evaluator for the LoRA-placement study.

Replaces the three `evaluate_*` functions of the original pipeline. Two defects are
fixed and one performance problem is removed.

Defect 1 - GSM8K answer extraction
    The original generated 256 tokens with no stop criterion and took the LAST
    "#### N" match. The few-shot template is self-repeating, so a model that answered
    correctly and did not emit EOS continued the pattern and invented a further
    question with its own "####", and that trailing value was scored. Here generation
    stops at the first stop string and extraction takes the FIRST "####".

Defect 2 - HumanEval never executed
    `evaluate.load("code_eval")` raises unless HF_ALLOW_CODE_EVAL=1. The original
    caught the exception and recorded NaN, which reached the tables as 0.000. Here the
    completions are executed directly in a subprocess with a timeout, so no
    environment variable and no network dependency is involved.

Performance - the original ran batch size 1 everywhere: one forward pass per
    (example, answer option) and one generation at a time. That is why a single run
    took hours and why full test splits were unaffordable. Everything here is batched.

The module deliberately has no dependency on the notebook's globals: pass what you
need. `python evaluator.py --self-test` runs the offline tests.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

# ─────────────────────────────────────────────────────────────── GSM8K answers

# The few-shot template repeats this preamble and a "Question:" field, so either
# marker means the model has rolled on into a question it invented.
GSM8K_STOP_STRINGS: tuple[str, ...] = (
    "You are a careful mathematician",
    "\nQuestion:",
    "Question:",
)

_HASH_NUM = re.compile(r"####\s*([-+]?[0-9][0-9,\.]*)")
_ANY_NUM = re.compile(r"[-+]?[0-9][0-9,\.]*")


def truncate_at_stop(text: str, stops: Sequence[str] = GSM8K_STOP_STRINGS) -> str:
    """Cut `text` at the earliest occurrence of any stop string."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def normalize_number(s: Optional[str]) -> Optional[str]:
    """Unchanged from the original pipeline, so corrected and original labels are
    comparable on the same normalisation."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return s


def extract_gsm8k_answer(completion: str,
                         stops: Sequence[str] = GSM8K_STOP_STRINGS) -> Optional[str]:
    """First '#### N' after truncation; fall back to the last bare number.

    The fallback matches the original's behaviour for completions with no '####' at
    all, so the only behavioural change is the one being corrected.
    """
    body = truncate_at_stop(completion, stops)
    m = _HASH_NUM.search(body)
    if m:
        return normalize_number(m.group(1))
    nums = _ANY_NUM.findall(body)
    return normalize_number(nums[-1]) if nums else None


def gsm8k_gold(answer_field: str) -> Optional[str]:
    """Gold answers in GSM8K always carry exactly one '#### N' at the end."""
    m = _HASH_NUM.findall(answer_field)
    return normalize_number(m[-1]) if m else None


# ───────────────────────────────────────────────────────────── batched helpers

def _chunk(xs: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@torch.no_grad()
def batched_generate(model,
                     tokenizer,
                     prompts: Sequence[str],
                     max_new_tokens: int = 256,
                     batch_size: int = 16,
                     stop_strings: Sequence[str] = (),
                     progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
    """Greedy generation over a list of prompts.

    Decoder-only models must be LEFT-padded for generation: the continuation is
    produced after the final position, so right padding would make the model continue
    from pad tokens. The original tokenizer state is restored on exit.
    """
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    out: List[str] = []
    try:
        for bi, batch in enumerate(_chunk(list(prompts), batch_size)):
            enc = tokenizer(list(batch), return_tensors="pt", padding=True).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # With left padding every sequence starts generating at the same index.
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            for row in new_tokens:
                text = tokenizer.decode(row, skip_special_tokens=True)
                out.append(truncate_at_stop(text, stop_strings) if stop_strings else text)
            if progress:
                progress(len(out), len(prompts))
    finally:
        tokenizer.padding_side = prev_side
    return out


@torch.no_grad()
def batched_continuation_logprobs(model,
                                  tokenizer,
                                  pairs: Sequence[tuple[str, str]],
                                  batch_size: int = 16) -> List[float]:
    """Total log-probability of each continuation given its prompt.

    Scoring uses RIGHT padding, which is safe because the continuation tokens are
    indexed per sequence from that sequence's own prompt length rather than from a
    shared offset. Padded positions are excluded by the attention mask and by the
    per-sequence slice, never by a global one.
    """
    device = next(model.parameters()).device
    prev_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    totals: List[float] = []
    try:
        for batch in _chunk(list(pairs), batch_size):
            prompts = [p for p, _ in batch]
            fulls = [p + c for p, c in batch]
            p_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"])
                      for p in prompts]
            enc = tokenizer(fulls, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).logits
            logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = enc["input_ids"][:, 1:]
            token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            mask = enc["attention_mask"][:, 1:].bool()
            for i, p_len in enumerate(p_lens):
                # token at index t is predicted by logits[t-1]; continuation tokens
                # start at absolute index p_len, i.e. at p_len-1 in the shifted view
                sel = token_lp[i, p_len - 1:]
                sel_mask = mask[i, p_len - 1:]
                totals.append(float(sel[sel_mask].sum().item()))
    finally:
        tokenizer.padding_side = prev_side
    return totals


# ──────────────────────────────────────────────────────────────── evaluations

@dataclass
class EvalResult:
    benchmark: str
    accuracy: float
    n_eval: int
    correct: int
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {"benchmark": self.benchmark, "accuracy": self.accuracy,
                "n_eval": self.n_eval, "correct": self.correct}


def evaluate_multiple_choice(model, tokenizer, examples: Sequence[Dict[str, Any]],
                             benchmark: str, batch_size: int = 16,
                             progress=None) -> EvalResult:
    """`examples` carry `prompt`, `choices` (list of label strings) and `gold` index.

    All options of all examples in a chunk are scored in one batch, which is where
    the speed-up over the original comes from: the original ran one forward pass per
    (example, option).
    """
    pairs, owners = [], []
    for idx, ex in enumerate(examples):
        for lab in ex["choices"]:
            pairs.append((ex["prompt"], " " + lab))
            owners.append(idx)
    scores = batched_continuation_logprobs(model, tokenizer, pairs, batch_size)

    by_ex: Dict[int, List[float]] = {}
    for owner, s in zip(owners, scores):
        by_ex.setdefault(owner, []).append(s)

    rows, correct = [], 0
    for idx, ex in enumerate(examples):
        pred = int(max(range(len(ex["choices"])), key=lambda j: by_ex[idx][j]))
        ok = int(pred == ex["gold"])
        correct += ok
        rows.append({"benchmark": benchmark, "example_id": ex.get("example_id", idx),
                     "correct": ok, "pred_index": pred, "gold_index": ex["gold"],
                     "score_vector": [float(x) for x in by_ex[idx]]})
        if progress:
            progress(idx + 1, len(examples))
    n = len(examples)
    return EvalResult(benchmark, correct / max(n, 1), n, correct, rows)


def evaluate_gsm8k(model, tokenizer, examples: Sequence[Dict[str, Any]],
                   max_new_tokens: int = 256, batch_size: int = 16,
                   progress=None) -> EvalResult:
    """`examples` carry `prompt` and `gold` (the raw GSM8K answer field)."""
    completions = batched_generate(
        model, tokenizer, [ex["prompt"] for ex in examples],
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        stop_strings=GSM8K_STOP_STRINGS, progress=progress)

    rows, correct = [], 0
    for ex, comp in zip(examples, completions):
        pred = extract_gsm8k_answer(comp)
        gold = gsm8k_gold(ex["gold"]) if "####" in str(ex["gold"]) else normalize_number(str(ex["gold"]))
        ok = int(pred is not None and pred == gold)
        correct += ok
        rows.append({"benchmark": "gsm8k", "example_id": ex.get("example_id"),
                     "correct": ok, "pred_answer": pred, "gold_answer": gold,
                     "raw_completion": comp})
    n = len(examples)
    return EvalResult("gsm8k", correct / max(n, 1), n, correct, rows)


# HumanEval stop sequences from Chen et al. 2021.
HUMANEVAL_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```", "\n@")


def _run_humaneval_program(prompt: str, completion: str, test: str,
                           entry_point: str, timeout: float = 10.0) -> bool:
    program = prompt + completion + "\n\n" + test + "\n" + f"check({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def evaluate_humaneval(model, tokenizer, problems: Sequence[Dict[str, Any]],
                       max_new_tokens: int = 384, batch_size: int = 8,
                       timeout: float = 10.0, progress=None) -> EvalResult:
    """`problems` carry `prompt`, `test`, `entry_point`, `task_id`.

    Executes the completions directly rather than through `evaluate.load("code_eval")`,
    which is what silently failed in the original run. Generated code is run in a
    subprocess with a timeout; run this only on problems you trust.
    """
    completions = batched_generate(
        model, tokenizer, [p["prompt"] for p in problems],
        max_new_tokens=max_new_tokens, batch_size=batch_size,
        stop_strings=HUMANEVAL_STOPS, progress=progress)

    rows, correct = [], 0
    for prob, comp in zip(problems, completions):
        ok = int(_run_humaneval_program(prob["prompt"], comp, prob["test"],
                                        prob["entry_point"], timeout))
        correct += ok
        rows.append({"benchmark": "humaneval", "example_id": prob.get("example_id"),
                     "task_id": prob["task_id"], "correct": ok, "completion": comp})
    n = len(problems)
    return EvalResult("humaneval", correct / max(n, 1), n, correct, rows)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────── self-tests

def _self_test() -> int:
    import numpy as np
    failures: List[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("\n1. GSM8K extraction, the defect being corrected")
    hallucinated = (" Her total is $<<0.55*45=24.75>>24.75\n#### 447.5\n\n"
                    "You are a careful mathematician. Solve the problem step by step "
                    "and end with '#### <final answer>'.\n\nQuestion: A car travels\n"
                    "Answer: #### 200")
    check("takes the first ####, not the last", extract_gsm8k_answer(hallucinated) == "447.5",
          f"got {extract_gsm8k_answer(hallucinated)}")
    check("original last-#### behaviour would have differed",
          normalize_number(_HASH_NUM.findall(hallucinated)[-1]) == "200")
    check("clean completion unaffected", extract_gsm8k_answer("reasoning\n#### 18") == "18")
    check("comma normalisation", extract_gsm8k_answer("#### 1,234") == "1234")
    check("integer-valued float normalises", extract_gsm8k_answer("#### 18.0") == "18")
    check("no #### falls back to last number", extract_gsm8k_answer("the answer is 42") == "42")
    check("no digits at all returns None", extract_gsm8k_answer("no numbers here") is None)
    check("gold parser takes the final ####", gsm8k_gold("steps\n#### 72") == "72")
    # "\nQuestion:" matches at index 1 and "Question:" at index 2; the earliest wins.
    check("truncation cuts at the earliest of several markers",
          truncate_at_stop("a\nQuestion: b") == "a",
          repr(truncate_at_stop("a\nQuestion: b")))

    print("\n2. Batched scoring and generation against a synthetic model")

    class ToyTokenizer:
        """Character-level tokenizer with a real pad token, enough to exercise the
        padding and per-sequence indexing logic offline."""
        pad_token, eos_token = "\x00", "\x01"
        pad_token_id, eos_token_id = 0, 1
        padding_side = "right"

        def __call__(self, text, return_tensors=None, padding=False,
                     add_special_tokens=True):
            texts = [text] if isinstance(text, str) else list(text)
            # distinct characters must get distinct ids, otherwise two answer
            # options collide and the argmax test is decided by tie-break order
            ids = [[2 + (ord(c) % 62) for c in t] for t in texts]
            if padding:
                w = max(len(i) for i in ids)
                if self.padding_side == "left":
                    mask = [[0] * (w - len(i)) + [1] * len(i) for i in ids]
                    ids = [[self.pad_token_id] * (w - len(i)) + i for i in ids]
                else:
                    mask = [[1] * len(i) + [0] * (w - len(i)) for i in ids]
                    ids = [i + [self.pad_token_id] * (w - len(i)) for i in ids]
            else:
                mask = [[1] * len(i) for i in ids]
            if return_tensors == "pt":
                return _Enc({"input_ids": torch.tensor(ids),
                             "attention_mask": torch.tensor(mask)})
            return {"input_ids": ids[0] if isinstance(text, str) else ids,
                    "attention_mask": mask[0] if isinstance(text, str) else mask}

        def decode(self, row, skip_special_tokens=True):
            return "".join(chr(int(t)) for t in row
                           if not (skip_special_tokens and int(t) in (0, 1)))

    class _Enc(dict):
        def to(self, device):
            return self

    class ToyModel(torch.nn.Module):
        """Deterministic scorer: token id t gets logit t/100 regardless of context,
        so the total log-prob of a continuation is a known function of its tokens."""
        V = 64

        def __init__(self):
            super().__init__()
            self._p = torch.nn.Parameter(torch.zeros(1))

        def parameters(self, recurse=True):
            return iter([self._p])

        def forward(self, input_ids=None, attention_mask=None, **kw):
            b, t = input_ids.shape
            base = torch.arange(self.V, dtype=torch.float32) / 100.0
            return type("O", (), {"logits": base.view(1, 1, -1).expand(b, t, -1).clone()})()

    tok, toy = ToyTokenizer(), ToyModel()

    # Padding must not change a sequence's own score.
    solo = batched_continuation_logprobs(toy, tok, [("abc", "de")], batch_size=1)
    mixed = batched_continuation_logprobs(
        toy, tok, [("abc", "de"), ("abcdefghij", "kl")], batch_size=2)
    check("right padding does not contaminate the short sequence",
          abs(solo[0] - mixed[0]) < 1e-5, f"{solo[0]:.6f} vs {mixed[0]:.6f}")

    # Batch size must not change results.
    pairs = [("abc", "de"), ("abcdefgh", "ij"), ("a", "bcdef"), ("abcd", "e")]
    b1 = batched_continuation_logprobs(toy, tok, pairs, batch_size=1)
    b4 = batched_continuation_logprobs(toy, tok, pairs, batch_size=4)
    check("scores are invariant to batch size",
          all(abs(x - y) < 1e-5 for x, y in zip(b1, b4)),
          f"{[round(x,4) for x in b1]} vs {[round(y,4) for y in b4]}")

    # Continuation length must be respected: a longer continuation sums more terms.
    lens = batched_continuation_logprobs(toy, tok, [("ab", "c"), ("ab", "cd")], batch_size=2)
    check("longer continuation accumulates more log-prob terms", lens[1] < lens[0])

    # Multiple choice picks the argmax per example and keeps options grouped correctly.
    examples = [{"prompt": "q1", "choices": ["A", "B", "C"], "gold": 2, "example_id": 0},
                {"prompt": "q2", "choices": ["A", "B"], "gold": 0, "example_id": 1}]
    res = evaluate_multiple_choice(toy, tok, examples, "toy", batch_size=3)
    check("options are regrouped to the right example",
          [len(r["score_vector"]) for r in res.rows] == [3, 2],
          str([len(r["score_vector"]) for r in res.rows]))
    check("argmax picks the highest-scoring option",
          res.rows[0]["pred_index"] == 2 and res.rows[1]["pred_index"] == 1)
    check("accuracy counts only exact matches", res.correct == 1 and res.n_eval == 2)

    print("\n3. HumanEval execution, the metric that never ran")
    good = _run_humaneval_program("def f(x):\n", "    return x + 1\n",
                                  "def check(f):\n    assert f(1) == 2\n", "f")
    bad = _run_humaneval_program("def f(x):\n", "    return x + 2\n",
                                 "def check(f):\n    assert f(1) == 2\n", "f")
    loop = _run_humaneval_program("def f(x):\n", "    while True:\n        pass\n",
                                  "def check(f):\n    assert f(1) == 2\n", "f", timeout=3.0)
    check("a correct completion passes", good is True)
    check("an incorrect completion fails", bad is False)
    check("an infinite loop is killed by the timeout, not hung", loop is False)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    print(__doc__)
