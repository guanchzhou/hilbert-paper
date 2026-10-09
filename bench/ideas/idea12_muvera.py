#!/usr/bin/env python3
"""Idea 12: MUVERA fixed-dimensional encodings with hk1 level-1 sign buckets (256 SimHash cells)."""

from pathlib import Path

import numpy as np

from common import CACHE, Bench, halves, holm, unit, wilcoxon, write_result

POP = np.array([bin(i).count("1") for i in range(256)])


def buckets(cells: np.ndarray) -> np.ndarray:
    bits = (cells >= 128).astype(np.int64)
    return (bits * (1 << np.arange(7, -1, -1))).sum(axis=1)


def main() -> None:
    b = Bench()
    h = np.load(CACHE / "hk1-default.npz")
    cb, qb = buckets(h["ccell"]), buckets(h["qcell"])
    scores = np.full((b.n, len(b.pages)), -np.inf, dtype=np.float32)
    for bq in np.unique(qb):
        ham = POP[cb ^ bq]
        min_per_note = np.minimum.reduceat(ham, b.starts)
        sel = (ham == min_per_note[b.page_of_row]).astype(np.float32)
        sums = np.add.reduceat(b.Cn * sel[:, None], b.starts, axis=0)
        cnt = np.add.reduceat(sel, b.starts)
        block = sums / cnt[:, None]
        qs = np.flatnonzero(qb == bq)
        scores[qs] = b.Qn[qs] @ block.T
    muvera = b.per_question(b.lists(scores))
    best_s, _ = b.dense()
    best = b.per_question(b.lists(best_s))
    means = unit(np.add.reduceat(b.Cn, b.starts, axis=0))
    mean = b.per_question(b.lists(b.Qn @ means.T))
    nchunks = b.ends - b.starts
    subset = np.array([qi for qi in range(b.n)
                       if any(r in b.page_index and nchunks[b.page_index[r]] >= 5 for r in b.rels[qi])])
    t = wilcoxon(muvera["R@10"][subset], best["R@10"][subset])
    t["threshold"] = 0.03
    t["MRR_diff"] = float((muvera["MRR"][subset] - best["MRR"][subset]).mean())
    sec = wilcoxon(muvera["R@10"][subset], mean["R@10"][subset], alternative="two-sided")
    sec_mean_vs_best = wilcoxon(mean["R@10"][subset], best["R@10"][subset], alternative="two-sided")
    occ = np.bincount(cb, minlength=256)
    rel_in_bucket = []
    for qi in subset:
        rows = [r for s in b.rels[qi] if s in b.page_index
                for r in range(b.starts[b.page_index[s]], b.ends[b.page_index[s]])]
        rel_in_bucket.append(bool((cb[rows] == qb[qi]).any()))

    def sub(arrs):
        return {k: float(v[subset].mean()) for k, v in arrs.items()}

    write_result("12-muvera", {
        "idea": 12, "name": "MUVERA FDE with hk1 level-1 buckets",
        "subset_rule": "questions with at least one relevant live note that has >= 5 chunks",
        "subset_size": int(len(subset)),
        "buckets": {"nonempty": int((occ > 0).sum()), "max": int(occ.max()),
                    "share_subset_questions_with_a_relevant_chunk_in_query_bucket": float(np.mean(rel_in_bucket))},
        "conditions_subset": {"best_chunk": sub(best), "mean_of_chunks": sub(mean), "muvera": sub(muvera)},
        "conditions_all_817": {"best_chunk": halves(b, best), "mean_of_chunks": halves(b, mean),
                               "muvera": halves(b, muvera)},
        "primary": {"H12 MUVERA vs best chunk": t},
        "secondary": {"MUVERA vs mean of chunks (two-sided)": sec,
                      "mean of chunks vs best chunk (two-sided)": sec_mean_vs_best},
        "per_question": {"subset_idx": subset, "best_R@10": best["R@10"], "mean_R@10": mean["R@10"],
                         "muvera_R@10": muvera["R@10"]},
    }, Path(__file__))
    print(len(subset), sub(best), sub(mean), sub(muvera), t["p"])


if __name__ == "__main__":
    main()
