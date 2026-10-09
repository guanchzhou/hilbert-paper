#!/usr/bin/env python3
"""Cell 2 of addendum 3: smooth the top cosine hits over the link graph, then keep
the higher of the smoothed score and the original cosine. Settings fixed before the
run: the top 20 notes are the seeds, alpha is 0.15, twenty iterations,
p <- alpha s + (1 - alpha) W p on the same undirected link graph as idea 9,
then rank by max(p, cosine). Passes when the 95% interval for the recall difference
sits above zero and the original top note stays in the top 10 on every question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench, boot_ci, CACHE  # noqa: E402

SEED = 20261008
TOP = 20
ALPHA = 0.15
ITERS = 20


def main() -> None:
    b = Bench()
    assert b.n == 817
    edges = json.loads((CACHE / "edges.json").read_text())
    n = len(b.pages)
    A = np.zeros((n, n), dtype=np.float32)
    for a, c in edges:
        if a in b.page_index and c in b.page_index:
            i, j = b.page_index[a], b.page_index[c]
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1)
    W = np.divide(A, deg[:, None], out=np.zeros_like(A), where=deg[:, None] > 0)
    cosine, _ = b.dense()
    order = b.order(cosine)
    s = np.zeros_like(cosine)
    for qi in range(b.n):
        cols = order[qi, :TOP]
        s[qi, cols] = cosine[qi, cols]
    p = s.copy()
    for step in range(ITERS):
        p = ALPHA * s + (1 - ALPHA) * (p @ W)
        print("iter", step + 1, flush=True)
    final = np.maximum(p, cosine)
    base = b.per_question(b.lists(cosine))["R@10"]
    final_lists = b.lists(final)
    new = b.per_question(final_lists)["R@10"]
    diff = new - base
    lo, hi = boot_ci(diff, seed=SEED)
    top_in = 0
    for qi in range(b.n):
        kept = b.pages[int(order[qi, 0])]
        if kept in final_lists[qi]:
            top_in += 1
    out = {
        "cell": 2,
        "name": "seed clamp",
        "arxiv": "2603.24925",
        "n": int(b.n),
        "settings": {"seeds": TOP, "alpha": ALPHA, "iterations": ITERS, "graph": "undirected links"},
        "edges": int(A.sum() / 2),
        "R@10": {"clamped": float(new.mean()), "best_chunk": float(base.mean())},
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
        "interval_above_zero": bool(lo > 0),
        "original_top_retained_in_10": top_in,
        "original_top_lost": int(b.n - top_in),
        "passes": bool(lo > 0 and top_in == b.n),
        "seed": SEED,
    }
    (HERE / "cell2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
