#!/usr/bin/env python3
"""Idea 5: dense pseudo-relevance feedback (Rocchio), tuned on the tune half, tested on confirm."""

from pathlib import Path

import numpy as np

from common import Bench, halves, unit, wilcoxon, write_result


def main() -> None:
    b = Bench()
    sim = b.chunk_sim(b.Qn)
    base_s, _ = b.note_scores(sim)
    base = b.per_question(b.lists(base_s))
    top_chunks = np.argsort(-sim, axis=1)[:, :5]
    grid = {}
    for a in (0.4, 0.6, 0.8):
        for k in (3, 5):
            fb = b.Cn[top_chunks[:, :k]].mean(axis=1)
            q2 = unit(a * b.Qn + (1 - a) * fb)
            s, _ = b.dense(q2)
            pq = b.per_question(b.lists(s))
            grid[f"a={a},k={k}"] = {"a": a, "k": k, "metrics": halves(b, pq), "pq": pq}
            print(a, k, {h: round(v["R@10"], 4) for h, v in grid[f"a={a},k={k}"]["metrics"].items()}, flush=True)
    chosen = max(grid, key=lambda g: (grid[g]["metrics"]["tune"]["R@10"], -grid[g]["a"]))
    pq = grid[chosen]["pq"]
    c = b.confirm
    t = wilcoxon(pq["R@10"][c], base["R@10"][c])
    t["MRR_diff"] = float((pq["MRR"][c] - base["MRR"][c]).mean())
    t["MRR_ci95"] = wilcoxon(pq["MRR"][c], base["MRR"][c])["ci95"]
    t["nDCG_diff"] = float((pq["nDCG@10"][c] - base["nDCG@10"][c]).mean())
    t["threshold"] = 0.02
    t["guard"] = "MRR difference >= -0.01"
    t["guard_pass"] = t["MRR_diff"] >= -0.01
    tune_t = wilcoxon(pq["R@10"][b.tune], base["R@10"][b.tune])
    write_result("5-rocchio", {
        "idea": 5, "name": "dense pseudo-relevance feedback (Rocchio)",
        "selection": "best tune-half R@10 over a in {0.4,0.6,0.8}, k in {3,5}; ties to smaller a",
        "chosen": chosen,
        "grid": {g: v["metrics"] for g, v in grid.items()},
        "conditions": {"dense": halves(b, base), "chosen": grid[chosen]["metrics"]},
        "tune_half_test": tune_t,
        "primary": {"H5 Rocchio PRF": t},
        "per_question": {"confirm_idx": c, "dense_R@10": base["R@10"], "chosen_R@10": pq["R@10"],
                         "dense_MRR": base["MRR"], "chosen_MRR": pq["MRR"]},
    }, Path(__file__))
    print(chosen, t["diff"], t["ci95"], t["p"], t["MRR_diff"])


if __name__ == "__main__":
    main()
