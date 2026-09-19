"""Resolve data directories whether the scripts run from the repository or from the
working copy that contains it.

In the repository the data sits at `<root>/results/...`. In the development tree the
repository is itself a subdirectory, so the same data sits at `<root>/github/results/...`.
Hardcoding either one breaks the other, and the one that breaks is the clone, which is
the copy a reviewer runs.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

_CANDIDATE_PREFIXES: Sequence[str] = ("", "github")


def find_dir(*parts: str, start: Optional[str] = None) -> str:
    """First existing directory matching `<root>/[github/]<parts>`, else the repo-style path.

    The search walks upwards from `start` so a script can be invoked from anywhere.
    """
    here = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    tried = []
    for _ in range(4):
        for prefix in _CANDIDATE_PREFIXES:
            cand = os.path.join(here, prefix, *parts) if prefix else os.path.join(here, *parts)
            tried.append(cand)
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    # Nothing found: hand back the repository-style path so the error names that one.
    return tried[0]


def results_dir(*parts: str, start: Optional[str] = None) -> str:
    return find_dir("results", *parts, start=start)


if __name__ == "__main__":
    for name in ("discovery", "eval_details", "eval_indices"):
        p = results_dir(name)
        print(f"{name:14s} {'OK  ' if os.path.isdir(p) else 'MISS'}  {p}")
