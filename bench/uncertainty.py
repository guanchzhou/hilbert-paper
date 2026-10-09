#!/usr/bin/env python3
"""Bootstrap intervals over queries, and a spread that re-embeds the queries."""

import json
import urllib.request
from pathlib import Path

import numpy as np

from metrics import estimate_tokens, recall_at_k
from run_measure import BUDGET, QRELS, keyword_rankings, load_corpus, pg_env, vector_rankings

BENCH = Path(__file__).resolve().parent
B = 10000


def embed(texts: list[str]) -> np.ndarray:
    body = json.dumps({"model": "qwen3-embedding-8k", "input": texts}).encode()
    req = urllib.request.Request("http://127.0.0.1:11436/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)["data"]
    data.sort(key=lambda d: d["index"])
    return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def ci(values: np.ndarray, rng) -> dict:
    n = len(values)
    idx = rng.integers(0, n, size=(B, n))
    means = values[idx].mean(axis=1)
    return {"mean": float(values.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def main() -> None:
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    queries = [q["query"] for q in qrels]
    rels = [set(q["relevant"]) for q in qrels]
    vectors = np.load(BENCH / "query-vectors.npy")
    qn = (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    truth, _ids, slugs, sources, texts, indexes, matrix = load_corpus(env, password)
    vpage, _, vchunk, _, winners, by_page, pos_of = vector_rankings(qn, matrix, slugs, sources, texts, indexes)
    kpage, _, kchunk, _ = keyword_rankings(queries, env, password)

    per = {name: np.array([recall_at_k(lst[i], rels[i], 10) for i in range(len(qrels))])
           for name, lst in {"keyword-page": kpage, "keyword-chunk": kchunk,
                             "vector-page": vpage, "vector-chunk": vchunk}.items()}

    from run_measure import join_chunks, section_span, window_span

    survive = {u: np.zeros(len(qrels)) for u in ("chunk", "window", "section", "page")}
    for qi, rows in enumerate(winners):
        for unit in survive:
            used, ok = 0, False
            for row in rows:
                slug, hit_i = pos_of[row]
                chunks = by_page[slug]
                text = {
                    "chunk": lambda: chunks[hit_i]["text"],
                    "window": lambda: join_chunks(chunks, window_span(chunks, hit_i)),
                    "section": lambda: join_chunks(chunks, section_span(chunks, hit_i)),
                    "page": lambda: truth.get(slug, ""),
                }[unit]()
                cost = estimate_tokens(text)
                if used + cost <= BUDGET:
                    used += cost
                    ok = ok or slug in rels[qi]
            survive[unit][qi] = 1.0 if ok else 0.0

    rng = np.random.default_rng(7)
    out = {
        "bootstrap_resamples": B,
        "recall_at_10": {k: ci(v, rng) for k, v in per.items()},
        "survival": {k: ci(v, rng) for k, v in survive.items()},
        "paired_differences": {
            "vector-chunk minus vector-page": ci(per["vector-chunk"] - per["vector-page"], rng),
            "vector-chunk minus keyword-page": ci(per["vector-chunk"] - per["keyword-page"], rng),
            "chunk pack minus page pack": ci(survive["chunk"] - survive["page"], rng),
            "chunk pack minus section pack": ci(survive["chunk"] - survive["section"], rng),
        },
    }
    print(json.dumps(out, indent=2), flush=True)

    prefix = json.loads((BENCH / "manifest.json").read_text())["query_prefix"]
    pick = list(range(0, 800, 40))[:20]
    chunk_norm = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    repeats = []
    max_drift = 0.0
    for rep in range(5):
        fresh = embed([prefix + queries[i] for i in pick])
        drift = float(np.max(np.abs(fresh - vectors[pick])))
        max_drift = max(max_drift, drift)
        fn = fresh / np.maximum(np.linalg.norm(fresh, axis=1, keepdims=True), 1e-12)
        sims = fn @ chunk_norm.T
        rec = []
        for j, qi in enumerate(pick):
            order = np.argsort(-sims[j])
            seen, pages = set(), []
            for r in order:
                s = slugs[r]
                if s not in seen:
                    seen.add(s)
                    pages.append(s)
                if len(pages) == 10:
                    break
            rec.append(recall_at_k(pages, rels[qi], 10))
        repeats.append(float(np.mean(rec)))
        print("re-embedded repeat", rep, repeats[-1], "max abs drift", drift, flush=True)
    out["reembedded_spread"] = {
        "queries": 20,
        "repeats": 5,
        "R@10": repeats,
        "max_abs_component_drift_vs_saved_vectors": max_drift,
        "rule": "each repeat embeds the 20 queries again through the live service, then ranks by cosine over every chunk",
    }
    (BENCH / "uncertainty.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
