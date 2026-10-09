#!/usr/bin/env python3
"""Evaluate all 1,020 cells per question: ranking metrics per ranking, pack survival and tokens per cell.

A question is complete for a cell when every reranker pair and every sentence vector the cell
needs is available; incomplete entries are NaN and the analysis uses the questions complete in
every cell (pre-registration 1.5).
"""

import json
import time

import numpy as np

from pipeline import (BUDGET, CACHE, DEPTH, F8, IDEAS_CACHE, PRUNE_CAP, PRUNE_DEPTH, Data, rankings,
                      estimate_tokens, rerank_order, sentences)
from stage1 import cached_pairs


class Sentences:
    def __init__(self, d: Data) -> None:
        b = d.b
        s, win = b.dense()
        order = b.order(s)[:, :PRUNE_DEPTH]
        rows = sorted({int(win[qi, p]) for qi in range(b.n) for p in order[qi]})
        V = np.load(IDEAS_CACHE / "sentence-vectors.npz")["V"]
        self.vec, self.text = {}, {}
        pos = 0
        for r in rows:
            sent = sentences(b.texts[r])
            self.vec[r] = V[pos:pos + len(sent)]
            self.text[r] = sent
            pos += len(sent)
        assert pos == len(V)
        path = CACHE / "sentence-vectors-new.npz"
        if path.exists():
            z = np.load(path)
            # NpzFile decompresses on every z["V"]; a slice of it would keep a full copy alive.
            V_new = z["V"]
            pos = 0
            for r, n in zip(z["rows"], z["counts"]):
                sent = sentences(b.texts[int(r)])
                assert len(sent) == n
                self.vec[int(r)] = V_new[pos:pos + n]
                self.text[int(r)] = sent
                pos += n
        for r in self.vec:
            v = self.vec[r]
            self.vec[r] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        self.memo = {}
        self.Qn = b.Qn

    def tokens(self, qi: int, r: int) -> int | None:
        key = (qi, r)
        if key in self.memo:
            return self.memo[key]
        if r not in self.vec:
            return None
        sims = self.vec[r] @ self.Qn[qi]
        keep = sorted(np.argsort(-sims)[:PRUNE_CAP])
        t = estimate_tokens("\n".join(self.text[r][j] for j in keep))
        self.memo[key] = t
        return t


def pack(slugs: list[str], costs: list[int], rel: set) -> tuple[float, int]:
    used, ok = 0, False
    for s, c in zip(slugs, costs):
        if used + c <= BUDGET:
            used += c
            ok = ok or s in rel
    return (1.0 if ok else 0.0), used


def main() -> None:
    t0 = time.perf_counter()
    d = Data()
    b = d.b
    z = np.load(CACHE / "pre-rankings.npz")
    pre = {k: (z["top"][i].astype(np.int64), z["rows"][i].astype(np.int64), z["cand"][i]) for i, k in enumerate(z["keys"])}
    split = {k: (z["cand_dense"][i], z["cand_lex"][i]) for i, k in enumerate(z["keys"])}
    scores = cached_pairs(d)
    path = CACHE / "rerank-pairs.json"
    if path.exists():
        scores.update({k: v["score"] for k, v in json.loads(path.read_text()).items()})
    sent = Sentences(d)
    cfgs = rankings()
    R = len(cfgs)
    out = {m: np.full((R, b.n), np.nan) for m in ("R@10", "MRR", "nDCG@10", "hit@10", "cand", "cand_dense", "cand_lex", "pairs")}
    surv = np.full((R, len(F8), b.n), np.nan)
    tok = np.full((R, len(F8), b.n), np.nan)
    for ci, cfg in enumerate(cfgs):
        top, rows, cand = pre[cfg.pre().key]
        lists, keep = [], np.ones(b.n, dtype=bool)
        finals = []
        for qi in range(b.n):
            t, r = top[qi], rows[qi]
            if cfg.f3 == "on":
                n = int((t >= 0).sum())
                sc = [scores.get(f"{qi}:{b.ids[int(x)]}") for x in r[:n]]
                if any(v is None for v in sc):
                    keep[qi] = False
                    finals.append(None)
                    lists.append([])
                    continue
                o = rerank_order(np.arange(DEPTH), np.array(sc + [0.0] * (DEPTH - n)))
                o = np.concatenate([o[:n], np.arange(n, DEPTH)])
                t, r = t[o], r[o]
            finals.append((t, r))
            lists.append([b.pages[int(x)] for x in t[:10] if x >= 0])
        pq = b.per_question(lists)
        for m in ("R@10", "MRR", "nDCG@10", "hit@10"):
            out[m][ci] = np.where(keep, pq[m], np.nan)
        out["cand"][ci] = cand
        out["cand_dense"][ci], out["cand_lex"][ci] = split[cfg.pre().key]
        out["pairs"][ci] = (top >= 0).sum(1) if cfg.f3 == "on" else 0
        for qi in range(b.n):
            if finals[qi] is None:
                continue
            t, r = finals[qi]
            ok = t >= 0
            t10, r10 = t[:10][ok[:10]], r[:10][ok[:10]]
            slugs10 = [b.pages[int(x)] for x in t10]
            rel = b.rels[qi]
            for ui, u in enumerate(F8):
                if u == "pruned":
                    t30, r30 = t[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]], r[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]]
                    costs = [sent.tokens(qi, int(x)) for x in r30]
                    if any(c is None for c in costs):
                        continue
                    surv[ci, ui, qi], tok[ci, ui, qi] = pack([b.pages[int(x)] for x in t30], costs, rel)
                    continue
                if u == "chunk":
                    costs = d.tok_chunk[r10]
                elif u == "window":
                    costs = d.tok_window[r10]
                elif u == "section":
                    costs = d.tok_section[r10]
                else:
                    costs = d.tok_page[t10]
                surv[ci, ui, qi], tok[ci, ui, qi] = pack(slugs10, [int(c) for c in costs], rel)
        if ci % 20 == 0:
            print(ci, cfg.key, round(float(np.nanmean(out["R@10"][ci])), 4), int(keep.sum()),
                  round(time.perf_counter() - t0, 1), flush=True)
    np.savez_compressed(CACHE / "level1-perq.npz", keys=np.array([c.key for c in cfgs]),
                        units=np.array(F8), surv=surv, tok=tok, **{k.replace("@", "_"): v for k, v in out.items()})
    print("done", round(time.perf_counter() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
