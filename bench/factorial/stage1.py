#!/usr/bin/env python3
"""Compute the 102 pre-rerank rankings, then count reranker pairs and sentence rows still needed."""

import json
import time

import numpy as np

from pipeline import CACHE, DEPTH, IDEAS_CACHE, PRUNE_DEPTH, Data, rankings, sentences


def cached_pairs(d: Data) -> dict:
    """(question, chunk id) -> score from the idea 1 cache (dense top 50, reference pipeline)."""
    b = d.b
    s, win = b.dense()
    order = b.order(s)[:, :DEPTH]
    res = json.loads((IDEAS_CACHE / "rerank-default.json").read_text())
    out = {}
    for qi in range(b.n):
        for j, p in enumerate(order[qi]):
            out[f"{qi}:{b.ids[int(win[qi, p])]}"] = res[str(qi)]["scores"][j]
    return out


def main() -> None:
    t0 = time.perf_counter()
    d = Data()
    b = d.b
    print("data", round(time.perf_counter() - t0, 1), flush=True)
    pres = [c for c in rankings() if c.f3 == "off"]
    store = {}
    for i, cfg in enumerate(pres):
        r = d.pre_ranking(cfg)
        store[cfg.key] = r
        if i % 10 == 0:
            print(i, cfg.key, round(time.perf_counter() - t0, 1), flush=True)
    np.savez_compressed(CACHE / "pre-rankings.npz",
                        keys=np.array(list(store)),
                        top=np.stack([v["top"] for v in store.values()]).astype(np.int16),
                        rows=np.stack([v["rows"] for v in store.values()]).astype(np.int32),
                        cand=np.stack([v["cand"] for v in store.values()]).astype(np.int32),
                        cand_dense=np.stack([v["cand_dense"] for v in store.values()]).astype(np.int32),
                        cand_lex=np.stack([v["cand_lex"] for v in store.values()]).astype(np.int32))
    have = cached_pairs(d)
    path = CACHE / "rerank-pairs.json"
    extra = json.loads(path.read_text()) if path.exists() else {}
    need = {}
    for v in store.values():
        for qi in range(b.n):
            for row in v["rows"][qi]:
                if row < 0:
                    continue
                k = f"{qi}:{b.ids[int(row)]}"
                if k not in have and k not in extra:
                    need.setdefault(qi, set()).add(int(row))
    n_need = sum(len(x) for x in need.values())
    print("cached pairs", len(have), "extra cached", len(extra), "new pairs", n_need, "questions", len(need), flush=True)
    sec = np.array([json.loads((IDEAS_CACHE / "rerank-default.json").read_text())[str(q)]["seconds"]
                    for q in range(b.n)])
    per_pair = float(np.median(sec / 50))
    print("per-pair s", per_pair, "projected h", n_need * per_pair / 3600, flush=True)
    (CACHE / "rerank-need.json").write_text(json.dumps({str(k): sorted(v) for k, v in need.items()}))

    # sentence rows: winners within the top 30 of any final ranking lie within the pre-rerank top 50
    rows_needed = set()
    for v in store.values():
        rows_needed.update(int(r) for r in v["rows"][:, :DEPTH].ravel() if r >= 0)
    s, win = b.dense()
    order = b.order(s)[:, :PRUNE_DEPTH]
    have_rows = {int(win[qi, p]) for qi in range(b.n) for p in order[qi]}
    new_rows = sorted(rows_needed - have_rows)
    n_sent = sum(len(sentences(b.texts[r])) for r in new_rows)
    print("sentence rows needed", len(rows_needed), "new rows", len(new_rows), "new sentences", n_sent, flush=True)
    (CACHE / "sentence-need.json").write_text(json.dumps({"new_rows": new_rows, "new_sentences": n_sent}))


if __name__ == "__main__":
    main()
