#!/usr/bin/env python3
"""Cell 3 of addendum 3: forward-push PageRank from the same seeds as the factorial
PageRank, on directed links. Settings fixed before the run. The existing method is
the factorial one: top 20 seeds, alpha 0.3, beta 0.3, 60 power iterations, dangling
mass restarted, fused with the cosine scores, on the undirected link graph. The one
change is a forward push (residual threshold 1e-4) on the directed links. The cell
passes when the 95% interval for the recall difference is not entirely below zero
and the push is faster per question.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))
from common import Bench, boot_ci, CACHE  # noqa: E402
from run_measure import pg_env  # noqa: E402

SEED = 20261008
TOP, ALPHA, BETA, ITERS, EPS = 20, 0.3, 0.3, 60, 1e-4


def load_directed(pages: dict[str, int]) -> list[tuple[int, int]]:
    sys.path.insert(0, str(HERE.parent / "efficiency"))
    from run import load_links  # noqa: E402
    env, pw = pg_env()
    pairs = load_links(env, pw)
    return [(pages[a], pages[c]) for a, c in pairs if a in pages and c in pages and a != c]


def fuse(score: np.ndarray, pi: np.ndarray) -> np.ndarray:
    fin = np.isfinite(score)
    lo = np.where(fin, score, np.inf).min(axis=1, keepdims=True)
    base = np.where(fin, score, lo)
    base_mm = (base - base.min(1, keepdims=True)) / np.maximum(base.max(1, keepdims=True) - base.min(1, keepdims=True), 1e-12)
    ppr_mm = (pi - pi.min(1, keepdims=True)) / np.maximum(pi.max(1, keepdims=True) - pi.min(1, keepdims=True), 1e-12)
    fused = (1 - BETA) * base_mm + BETA * ppr_mm
    out = np.where(fin | (pi > 0), fused, -np.inf)
    return out


def seeds_of(score: np.ndarray, order: np.ndarray) -> np.ndarray:
    e = np.zeros_like(score)
    for qi in range(len(score)):
        cols = order[qi, :TOP]
        cols = cols[np.isfinite(score[qi, cols])]
        if len(cols) == 0:
            continue
        w = score[qi, cols] - score[qi, cols].min() + 1e-6
        e[qi, cols] = w / w.sum()
    return e


def power(e: np.ndarray, P: np.ndarray, dangling: np.ndarray) -> np.ndarray:
    pi = e.copy()
    for _ in range(ITERS):
        lost = pi[:, dangling].sum(axis=1, keepdims=True)
        pi = ALPHA * e + (1 - ALPHA) * (pi @ P + lost * e)
    return pi


def forward_push(e: np.ndarray, outs: list[list[int]]) -> np.ndarray:
    n = e.shape[1]
    deg = np.array([len(v) for v in outs], dtype=np.int32)
    pi = np.zeros_like(e)
    for qi in range(len(e)):
        r = e[qi].copy()
        p = np.zeros(n, dtype=np.float32)
        guard = 0
        while True:
            u = int(np.argmax(r))
            threshold = EPS * max(int(deg[u]), 1)
            if r[u] <= threshold or guard > n * 20:
                break
            mass = float(r[u])
            r[u] = 0.0
            p[u] += ALPHA * mass
            if deg[u] == 0:
                r += (1 - ALPHA) * mass * e[qi]
            else:
                share = (1 - ALPHA) * mass / deg[u]
                for v in outs[u]:
                    r[v] += share
            guard += 1
        pi[qi] = p
        if qi % 100 == 0:
            print("push", qi, flush=True)
    return pi


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
    P = np.divide(A, deg[:, None], out=np.zeros_like(A), where=deg[:, None] > 0)
    cosine, _ = b.dense()
    order = b.order(cosine)
    e = seeds_of(cosine, order)
    t0 = time.perf_counter()
    pi = power(e, P, deg == 0)
    power_s = (time.perf_counter() - t0) / b.n
    existing = fuse(cosine, pi)
    directed = load_directed(b.page_index)
    outs: list[list[int]] = [[] for _ in range(n)]
    for i, j in directed:
        outs[i].append(j)
    t1 = time.perf_counter()
    pushed = forward_push(e, outs)
    push_s = (time.perf_counter() - t1) / b.n
    base = b.per_question(b.lists(existing))["R@10"]
    new = b.per_question(b.lists(fuse(cosine, pushed)))["R@10"]
    diff = new - base
    lo, hi = boot_ci(diff, seed=SEED)
    out = {
        "cell": 3,
        "name": "local forward push",
        "arxiv": "1908.10583",
        "n": int(b.n),
        "settings": {"seeds": TOP, "alpha": ALPHA, "beta": BETA, "power_iterations": ITERS,
                     "push_epsilon": EPS, "existing_graph": "undirected", "push_graph": "directed"},
        "directed_links": len(directed),
        "R@10": {"forward_push": float(new.mean()), "existing_pagerank": float(base.mean())},
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
        "seconds_per_question": {"forward_push": push_s, "existing_pagerank": power_s},
        "passes": bool(hi >= 0 and push_s < power_s),
        "seed": SEED,
    }
    (HERE / "cell3.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
