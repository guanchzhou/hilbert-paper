#!/usr/bin/env python3
"""Per-query hits for the four retrieval cells. Development set only."""

import json
from pathlib import Path

import numpy as np

from run_measure import QRELS, keyword_rankings, load_corpus, pg_env, vector_rankings

BENCH = Path(__file__).resolve().parent


def first_rank(hits: list[str], rel: set[str]) -> int | None:
    for i, h in enumerate(hits):
        if h in rel:
            return i + 1
    return None


def main() -> None:
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    queries = [q["query"] for q in qrels]
    vectors = np.load(BENCH / "query-vectors.npy")
    qnorm = (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    _truth, _ids, slugs, sources, texts, indexes, matrix = load_corpus(env, password)
    vpage, _, vchunk, _, _, _, _ = vector_rankings(qnorm, matrix, slugs, sources, texts, indexes)
    kpage, _, kchunk, _ = keyword_rankings(queries, env, password)
    cells = {"keyword-page": kpage, "keyword-chunk": kchunk, "vector-page": vpage, "vector-chunk": vchunk}

    ranks = {name: [first_rank(lst[i], set(q["relevant"])) for i, q in enumerate(qrels)] for name, lst in cells.items()}

    def found(name: str) -> list[bool]:
        return [r is not None for r in ranks[name]]

    def overlap(a: str, b: str) -> dict:
        fa, fb = found(a), found(b)
        return {
            "both": sum(x and y for x, y in zip(fa, fb)),
            f"only_{a}": sum(x and not y for x, y in zip(fa, fb)),
            f"only_{b}": sum(y and not x for x, y in zip(fa, fb)),
            "neither": sum(not x and not y for x, y in zip(fa, fb)),
        }

    hit_at = {
        name: [round(sum(1 for r in rs if r is not None and r <= k) / len(rs), 4) for k in range(1, 11)]
        for name, rs in ranks.items()
    }
    rank_hist = {
        name: {str(k): sum(1 for r in rs if r == k) for k in range(1, 11)} | {"miss": sum(1 for r in rs if r is None)}
        for name, rs in ranks.items()
    }
    empty = {name: sum(1 for lst in cells[name] if not lst) for name in cells}
    out = {
        "n": len(qrels),
        "hit_at_k": hit_at,
        "first_rank_histogram": rank_hist,
        "empty_result_lists": empty,
        "overlap_keyword_page_vs_vector_chunk": overlap("keyword-page", "vector-chunk"),
        "overlap_vector_page_vs_vector_chunk": overlap("vector-page", "vector-chunk"),
        "overlap_keyword_page_vs_keyword_chunk": overlap("keyword-page", "keyword-chunk"),
        "definition": "hit means at least one relevant page in the top k; first rank is the rank of the first relevant page",
    }
    (BENCH / "per-query.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in out if k.startswith("overlap") or k == "empty_result_lists"}, indent=2))


if __name__ == "__main__":
    main()
