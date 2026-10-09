#!/usr/bin/env python3
"""Efficiency of hk1 as a candidate index, measured per token.

Value: whether a relevant note reaches the model (survival in the 6,000-token chunk pack) and
recall at 10. Cost: tokens processed (the text of the candidate chunks) and tokens delivered (the
pack the model reads). Conditions: cosine over every chunk, cosine over the keyed chunks, hk1
level 1 with 1, 4, 8 and 16 ranges, and for each hk1 setting a random subset of the keyed chunks of
the same size for the same question (three seeds). Candidates are rescored by cosine; the pack walks
the top 10 notes and keeps each winning chunk that fits the remaining budget (the rule of
section-rule.json). Exploratory; the pre-registered factorial study measures the same pack per
filter. Writes hk1-efficiency.json.
"""

import json
import statistics
from pathlib import Path

import numpy as np

import hilbert_measure as hm

BENCH = Path(__file__).resolve().parent
BUDGET = 6000
K = 10
SETTINGS = [1, 4, 8, 16]
SEEDS = [0, 1, 2]


def evaluate(q: np.ndarray, idx: np.ndarray, En: np.ndarray, slugs: list[str], tokens: np.ndarray,
             relevant: set[str]) -> dict:
    if idx.size == 0:
        return {"recall": 0.0, "survival": False, "delivered": 0, "processed": 0, "kept_relevant": 0}
    order = idx[np.argsort(-(En[idx] @ q))]
    pages, winners = [], []
    for j in order:
        s = slugs[j]
        if s not in pages:
            pages.append(s)
            winners.append(j)
            if len(pages) == K:
                break
    kept, used = [], 0
    for s, j in zip(pages, winners):
        if used + tokens[j] <= BUDGET:
            used += int(tokens[j])
            kept.append(s)
    return {"recall": len(set(pages) & relevant) / len(relevant), "survival": any(s in relevant for s in kept),
            "delivered": used, "processed": int(tokens[idx].sum()),
            "kept_relevant": sum(1 for s in kept if s in relevant)}


def summarize(name: str, rows: list[dict], candidates: list[int]) -> dict:
    n = len(rows)
    delivered = sum(r["delivered"] for r in rows) / n
    processed = sum(r["processed"] for r in rows) / n
    survival = sum(r["survival"] for r in rows) / n
    recall = sum(r["recall"] for r in rows) / n
    kept = sum(r["kept_relevant"] for r in rows) / n
    return {
        "condition": name, "questions": n, "mean_candidates": statistics.mean(candidates),
        "recall_at_10": recall, "survival": survival,
        "tokens_processed_mean": processed, "tokens_delivered_mean": delivered,
        "survival_per_1k_delivered": survival / delivered * 1000 if delivered else 0.0,
        "relevant_kept_per_1k_delivered": kept / delivered * 1000 if delivered else 0.0,
        "recall_per_100k_processed": recall / processed * 1e5 if processed else 0.0,
        "survival_per_100k_processed": survival / processed * 1e5 if processed else 0.0,
    }


def main() -> None:
    qrels = json.loads(hm.DEV.read_text())
    rels = [set(q["relevant"]) for q in qrels]
    Q = np.load(BENCH / "query-vectors.npy").astype(np.float32)
    Qn = hm.unit(Q)
    E, slugs, _ids, tokens, markers = hm.load_chunks()
    En = hm.unit(E)
    has = np.asarray([m.startswith(hm.PREFIX) for m in markers])
    key_arr = np.zeros(len(markers), dtype=np.uint64)
    for i, m in enumerate(markers):
        if has[i]:
            key_arr[i] = np.uint64(hm.key_int(m))
    all_idx = np.arange(len(slugs))
    keyed_idx = np.flatnonzero(has)

    out = []
    for name, idx in (("cosine, every chunk", all_idx), ("cosine, keyed chunks", keyed_idx)):
        rows = [evaluate(Qn[i], idx, En, slugs, tokens, rels[i]) for i in range(len(Qn))]
        out.append(summarize(name, rows, [idx.size] * len(Qn)))
        print(out[-1]["condition"], round(out[-1]["survival"], 3), round(out[-1]["tokens_delivered_mean"]), flush=True)
    for ranges in SETTINGS:
        spans = hm.probe_ranges(Q, 1, ranges)
        cand = [np.flatnonzero(hm.mask_counts(key_arr, has, sp)) for sp in spans]
        sizes = [c.size for c in cand]
        rows = [evaluate(Qn[i], cand[i], En, slugs, tokens, rels[i]) for i in range(len(Qn))]
        out.append(summarize(f"hk1 level 1, {ranges} ranges", rows, sizes))
        print(out[-1]["condition"], round(out[-1]["survival"], 3), round(out[-1]["tokens_delivered_mean"]), flush=True)
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            picks = [rng.choice(keyed_idx, size=s, replace=False) if s else np.array([], dtype=int) for s in sizes]
            rows = [evaluate(Qn[i], picks[i], En, slugs, tokens, rels[i]) for i in range(len(Qn))]
            per_seed.append(summarize(f"random, same size as hk1 {ranges} ranges", rows, sizes))
        mean = {k: (statistics.mean(d[k] for d in per_seed) if isinstance(per_seed[0][k], float) else per_seed[0][k])
                for k in per_seed[0]}
        mean["seeds"] = SEEDS
        out.append(mean)
        print(mean["condition"], round(mean["survival"], 3), round(mean["tokens_delivered_mean"]), flush=True)
    doc = {"budget": BUDGET, "k": K, "pack": "top 10 notes by best chunk, winning chunk kept if it fits",
           "keyed_chunks": int(keyed_idx.size), "chunks": int(len(slugs)),
           "note": "Exploratory, computed after the hypotheses were tested; the factorial study measures the same pack per filter.",
           "conditions": out}
    (BENCH / "hk1-efficiency.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
