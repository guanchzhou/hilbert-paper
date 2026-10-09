#!/usr/bin/env python3
"""Idea 9 diagnostic: are deep dense misses (rank > 50) within 1-2 links of a dense top-10 note?

Reads the links table read-only. Writes the note graph (slug pairs) to cache/ for the PPR step.
"""

import json
from pathlib import Path

import numpy as np

from common import CACHE, Bench, write_result
from run_measure import pg_env, psql


def load_edges(b: Bench) -> list[tuple[str, str]]:
    env, password = pg_env()
    out = psql("SELECT f.slug, t.slug FROM links l JOIN pages f ON f.id = l.from_page_id "
               "JOIN pages t ON t.id = l.to_page_id WHERE f.deleted_at IS NULL AND t.deleted_at IS NULL",
               env, password)
    edges = set()
    for line in out.splitlines():
        a, _, c = line.partition("|")
        if a and c and a != c:
            edges.add(tuple(sorted((a, c))))
    return sorted(edges)


def main() -> None:
    b = Bench()
    edges = load_edges(b)
    (CACHE / "edges.json").write_text(json.dumps(edges))
    adj: dict[str, set[str]] = {}
    for a, c in edges:
        adj.setdefault(a, set()).add(c)
        adj.setdefault(c, set()).add(a)
    s, _ = b.dense()
    fr = b.first_rank(s)
    lists = b.lists(s)
    hit = np.array([any(x in b.rels[i] for x in lists[i]) for i in range(b.n)])
    deep = np.flatnonzero((~hit) & (fr > 50))
    one = two = 0
    rows, reach1, reach2 = [], [], []
    for qi in deep:
        top = set(lists[qi])
        n1 = set().union(*(adj.get(x, set()) for x in top))
        n2 = n1 | set().union(*(adj.get(x, set()) for x in n1))
        reach1.append(len(n1 - top) / len(b.pages))
        reach2.append(len(n2 - top) / len(b.pages))
        rel = [r for r in b.rels[qi] if r in b.page_index]
        d = 1 if any(r in n1 for r in rel) else (2 if any(r in n2 for r in rel) else 0)
        one += d == 1
        two += d in (1, 2)
        rows.append({"q": int(qi), "first_rank": int(fr[qi]), "link_distance": d,
                     "relevant_degree": max(len(adj.get(r, ())) for r in rel)})
    nodes_with_edges = sum(1 for p in b.pages if p in adj)
    share = two / len(deep) if len(deep) else 0.0
    write_result("9-links-diagnostic", {
        "idea": 9, "name": "link-graph expansion: diagnostic",
        "graph": {"undirected_edges": len(edges), "notes_with_links": nodes_with_edges, "notes": len(b.pages),
                  "median_degree": float(np.median([len(adj.get(p, ())) for p in b.pages]))},
        "deep_misses": int(len(deep)),
        "within_1_link": int(one), "within_2_links": int(two),
        "share_within_2_links": share,
        "base_rate": {"mean_share_of_all_notes_within_1_link_of_top10": float(np.mean(reach1)),
                      "mean_share_of_all_notes_within_2_links_of_top10": float(np.mean(reach2)),
                      "note": "chance that an arbitrary note is in the same neighbourhood; added after the gate was defined"},
        "gate": "full PPR runs only if share_within_2_links >= 0.30",
        "gate_pass": share >= 0.30,
        "per_question": rows,
    }, Path(__file__))
    print(len(edges), "edges; deep misses", len(deep), "within1", one, "within2", two, "share", share)


if __name__ == "__main__":
    main()
