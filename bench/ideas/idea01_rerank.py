#!/usr/bin/env python3
"""Idea 1: rerank the dense top 50 notes (winning chunk text) with qwen3-reranker-0.6b over local MLX."""

import json
import time
from pathlib import Path

import numpy as np

from common import CACHE, RERANK_URL, Bench, halves, holm, post, wilcoxon, write_result

DEPTH = 50
CONDITIONS = {
    "default": "Given a web search query, retrieve relevant passages that answer the query",
    "corpus": ("Given a question about the author's personal notes on software, infrastructure, research and "
               "genealogy, retrieve the note passage that answers it"),
}


def run(b: Bench, order: np.ndarray, win: np.ndarray, cond: str) -> dict:
    path = CACHE / f"rerank-{cond}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for qi in range(b.n):
        if str(qi) in done:
            continue
        docs = [b.texts[win[qi, p]] for p in order[qi]]
        t0 = time.perf_counter()
        out = post(RERANK_URL, {"model": "qwen3-reranker-0.6b", "query": b.queries[qi], "documents": docs,
                                "instruction": CONDITIONS[cond]})
        dt = time.perf_counter() - t0
        scores = [0.0] * len(docs)
        for r in out["results"]:
            scores[r["index"]] = r["relevance_score"]
        done[str(qi)] = {"scores": scores, "seconds": dt}
        if qi % 50 == 0:
            path.write_text(json.dumps(done))
            print(cond, qi, round(dt, 2), flush=True)
    path.write_text(json.dumps(done))
    return done


def main() -> None:
    b = Bench()
    s, win = b.dense()
    order = b.order(s)[:, :DEPTH]
    base = b.per_question(b.lists(s))
    fr = b.first_rank(s)
    tests, conds, per_q, extra = {}, {"dense": halves(b, base)}, {"dense_R@10": base["R@10"]}, {}
    for cond, hyp in (("default", "H1a"), ("corpus", "H1b")):
        res = run(b, order, win, cond)
        lists = []
        for qi in range(b.n):
            sc = np.array(res[str(qi)]["scores"])
            o = np.lexsort((np.arange(DEPTH), -sc))[:10]
            lists.append([b.pages[order[qi, j]] for j in o])
        pq = b.per_question(lists)
        lat = np.array([res[str(qi)]["seconds"] for qi in range(b.n)])
        t = wilcoxon(pq["R@10"], base["R@10"])
        t["MRR_diff"] = float((pq["MRR"] - base["MRR"]).mean())
        t["MRR_ci95"] = wilcoxon(pq["MRR"], base["MRR"])["ci95"]
        t["nDCG_diff"] = float((pq["nDCG@10"] - base["nDCG@10"]).mean())
        t["threshold"] = 0.03
        tests[f"{hyp} rerank {cond} instruction"] = t
        conds[f"rerank-{cond}"] = halves(b, pq)
        per_q[f"rerank_{cond}_R@10"] = pq["R@10"]
        extra[cond] = {
            "near_misses_moved_into_top10": int(((fr > 10) & (fr <= DEPTH) & (pq["hit@10"] == 1)).sum()),
            "near_miss_pool_rank_11_50": int(((fr > 10) & (fr <= DEPTH)).sum()),
            "top10_hits_lost": int(((base["hit@10"] == 1) & (pq["hit@10"] == 0)).sum()),
            "latency_seconds": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                                "mean": float(lat.mean())},
        }
        print(cond, t["diff"], t["ci95"], t["p"], extra[cond], flush=True)
    within = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm_within_idea"] = within[k]
    write_result("1-rerank", {
        "idea": 1, "name": "cross-encoder rerank of the dense top 50",
        "set": "all 817 dev questions (no selection)", "depth": DEPTH,
        "document": "winning chunk text of each note; server truncates at 1,024 tokens",
        "instructions": CONDITIONS,
        "conditions": conds, "primary": tests, "secondary": extra,
        "per_question": per_q,
    }, Path(__file__))


if __name__ == "__main__":
    main()
