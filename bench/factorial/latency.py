#!/usr/bin/env python3
"""Per-query latency of each pipeline component, measured once (pre-registration 1.4).

CPU components are timed in-process on single queries. The query embedding (MLX) is measured with
`--embed` once the GPU is free. Reranker cost per pair comes from the cached idea 1 calls.
"""

import argparse
import json
import time

import numpy as np

from pipeline import CACHE, HERE, IDEAS_CACHE, PPR_ALPHA, PPR_ITERS, PRUNE_CAP, PRUNE_DEPTH, RRF_K, Data, unit
from common import EMBED_URL, SEED, post

REPS = 200


def per_query(fn, n: int = REPS) -> float:
    fn(0)
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args()
    path = HERE / "level1-latency.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    d = Data()
    b = d.b
    rng = np.random.default_rng(SEED)
    qs = rng.permutation(b.n)[:REPS]
    if a.embed:
        ms = []
        for qi in qs[:30]:
            t0 = time.perf_counter()
            post(EMBED_URL, {"model": "qwen3-embedding-8k", "input": "Instruct: Given a web search query, retrieve "
                             "relevant passages that answer the query\nQuery:" + b.queries[int(qi)]})
            ms.append((time.perf_counter() - t0) * 1000)
        res["embed_query_ms"] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "n": 30}
        path.write_text(json.dumps(res, indent=1) + "\n")
        print(res["embed_query_ms"])
        return
    C, NV = b.Cn, d.note_vec["off"]

    def dense(i):
        s = b.Qn[qs[i]] @ C.T
        np.maximum.reduceat(s, b.starts)
    t_dense = per_query(dense)

    def mean_notes(i):
        b.Qn[qs[i]] @ NV.T
    t_mean = per_query(mean_notes)
    mu = C.mean(axis=0)

    def centre(i):
        unit(b.Qn[qs[i]] - mu)
    t_centre = per_query(centre)

    def rocchio(i):
        s = b.Qn[qs[i]] @ C.T
        top = np.argpartition(-s, 5)[:5]
        unit(0.6 * b.Qn[qs[i]] + 0.4 * C[top].mean(axis=0))
    t_rocchio_extra = per_query(rocchio) - t_dense  # top-5 and update; the second pass is counted by candidates

    lex_top = d.topn(np.where(d.lex[qs] > 0, d.lex[qs], -np.inf)[:, b.starts], 50)
    den_top = d.topn((b.Qn[qs] @ NV.T).astype(np.float64), 50)

    def rrf(i):
        sc = {}
        for lst in (lex_top[i], den_top[i]):
            for r, p in enumerate(lst):
                if p >= 0:
                    sc[p] = sc.get(p, 0.0) + 1.0 / (RRF_K + r + 1)
        sorted(sc.items(), key=lambda kv: -kv[1])
    t_rrf = per_query(rrf)

    E = np.zeros(d.N)
    E[:20] = 1 / 20

    def ppr(i):
        pi = E.copy()
        for _ in range(PPR_ITERS):
            lost = pi[d.dangling].sum()
            pi = PPR_ALPHA * E + (1 - PPR_ALPHA) * (pi @ d.P + lost * E)
    t_ppr = per_query(ppr, 50)

    hk = np.load(CACHE / "hk1-mask.npz")
    t_hk1 = (float(hk["key_seconds_total"]) + float(hk["scan_seconds_total"])) / b.n * 1000
    pm = np.load(CACHE / "partition-mask.npz")
    t_part = float(pm["route_seconds_per_query"]) * 1000

    from stage2 import Sentences
    sent = Sentences(d)
    s0, win = b.dense()
    order = b.order(s0)[:, :PRUNE_DEPTH]

    def prune(i):
        qi = int(qs[i])
        for p in order[qi]:
            r = int(win[qi, p])
            sims = sent.vec[r] @ b.Qn[qi]
            sorted(np.argsort(-sims)[:PRUNE_CAP])
    t_prune = per_query(prune, 50)

    lex = json.loads((CACHE / "lexical-latency.json").read_text())
    sec = np.array([v["seconds"] for v in json.loads((IDEAS_CACHE / "rerank-default.json").read_text()).values()])
    sf = CACHE / "sentence-fill.json"
    res.update({
        "dense_ms_full_5157_chunks": t_dense,
        "dense_ms_per_vector": t_dense / d.R,
        "mean_notes_ms_1223_vectors": t_mean,
        "centre_ms": t_centre,
        "rocchio_extra_ms": max(t_rocchio_extra, 0.0),
        "rrf_ms": t_rrf,
        "ppr_ms": t_ppr,
        "hk1_ms": t_hk1,
        "partitions_route_ms": t_part,
        "lexical_sql_ms": lex["median_ms"],
        "rerank_ms_per_pair": float(np.median(sec / 50)) * 1000,
        "rerank_ms_50_median": float(np.median(sec)) * 1000,
        "prune_ms_precomputed_sentences": t_prune,
        "prune_embed_note": "on-the-fly sentence embedding cost per query, if sentence vectors are not precomputed",
        "sentence_fill": json.loads(sf.read_text()) if sf.exists() else None,
        "composition": ("latency = embed (if a dense part or a filter is present) + lexical SQL (if lexical part) "
                        "+ dense_ms_per_vector * vectors compared + centre + rocchio_extra + RRF (hybrid) + PPR "
                        "+ filter (hk1 key and scan, or partition routing) + rerank_ms_per_pair * pairs "
                        "+ prune (sentence-pruned packs only)"),
    })
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
