#!/usr/bin/env python3
"""Idea 9: personalised PageRank on the note link graph, seeded by dense scores (gate passed)."""

import json
from pathlib import Path

import numpy as np

from common import CACHE, Bench, halves, wilcoxon, write_result

SEEDS = 20
ITERS = 60


def minmax(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def main() -> None:
    b = Bench()
    edges = json.loads((CACHE / "edges.json").read_text())
    N = len(b.pages)
    A = np.zeros((N, N), dtype=np.float32)
    for a, c in edges:
        if a in b.page_index and c in b.page_index:
            i, j = b.page_index[a], b.page_index[c]
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1)
    P = np.divide(A, deg[:, None], out=np.zeros_like(A), where=deg[:, None] > 0)
    dangling = deg == 0
    s, _ = b.dense()
    base = b.per_question(b.lists(s))
    order = b.order(s)[:, :SEEDS]
    E = np.zeros_like(s)
    for qi in range(b.n):
        w = s[qi, order[qi]] - s[qi, order[qi]].min() + 1e-6
        E[qi, order[qi]] = w / w.sum()
    dense_mm = minmax(s)
    grid = {}
    for alpha in (0.15, 0.3, 0.5):
        pi = E.copy()
        for _ in range(ITERS):
            lost = pi[:, dangling].sum(axis=1, keepdims=True)
            pi = alpha * E + (1 - alpha) * (pi @ P + lost * E)
        ppr_mm = minmax(pi)
        for beta in (0.1, 0.2, 0.3):
            final = (1 - beta) * dense_mm + beta * ppr_mm
            pq = b.per_question(b.lists(final))
            key = f"alpha={alpha},beta={beta}"
            grid[key] = {"alpha": alpha, "beta": beta, "metrics": halves(b, pq), "pq": pq}
            print(key, {h: round(v["R@10"], 4) for h, v in grid[key]["metrics"].items()}, flush=True)
    chosen = max(grid, key=lambda g: (grid[g]["metrics"]["tune"]["R@10"], -grid[g]["beta"]))
    pq = grid[chosen]["pq"]
    c = b.confirm
    t = wilcoxon(pq["R@10"][c], base["R@10"][c])
    t["MRR_diff"] = float((pq["MRR"][c] - base["MRR"][c]).mean())
    t["MRR_ci95"] = wilcoxon(pq["MRR"][c], base["MRR"][c])["ci95"]
    t["nDCG_diff"] = float((pq["nDCG@10"][c] - base["nDCG@10"][c]).mean())
    t["threshold"] = 0.02
    t["guard"] = "MRR difference >= 0"
    t["guard_pass"] = t["MRR_diff"] >= 0
    write_result("9-ppr", {
        "idea": 9, "name": "link-graph expansion: personalised PageRank",
        "graph": {"nodes": N, "undirected_edges": int(A.sum() / 2), "isolated": int(dangling.sum())},
        "restart": f"dense scores of the top {SEEDS} notes, min-shifted, normalised; dangling mass restarts",
        "fusion": "(1 - beta) * minmax(dense) + beta * minmax(PPR), per question",
        "selection": "best tune-half R@10; ties to smaller beta",
        "chosen": chosen,
        "grid": {g: v["metrics"] for g, v in grid.items()},
        "tune_half_test": wilcoxon(pq["R@10"][b.tune], base["R@10"][b.tune]),
        "primary": {"H9 PPR": t},
        "per_question": {"confirm_idx": c, "dense_R@10": base["R@10"], "chosen_R@10": pq["R@10"],
                         "dense_MRR": base["MRR"], "chosen_MRR": pq["MRR"]},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"], t["MRR_diff"])


if __name__ == "__main__":
    main()
