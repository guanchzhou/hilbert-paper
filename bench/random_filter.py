#!/usr/bin/env python3
"""Recall of a uniform random candidate subset, the baseline for an hk1 range."""

import json
from pathlib import Path

import numpy as np

from metrics import summarize
from run_measure import QRELS, load_corpus, pg_env

BENCH = Path(__file__).resolve().parent
SIZES = [24, 100, 188, 329, 1000, 2500]
SEEDS = [0, 1, 2]


def main() -> None:
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    vectors = np.load(BENCH / "query-vectors.npy")
    qn = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    _truth, ids, slugs, _src, _texts, _idx, matrix = load_corpus(env, password)
    keyed = {int(k) for k in json.loads((BENCH / "chunk-keys.json").read_text())["keys"]}
    rows = np.array([i for i, cid in enumerate(ids) if cid in keyed])
    En = matrix[rows] / np.maximum(np.linalg.norm(matrix[rows], axis=1, keepdims=True), 1e-12)
    labels = [slugs[i] for i in rows]
    sim = qn.astype(np.float32) @ En.T
    out = []
    for size in SIZES:
        recalls = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            lists = []
            for qi in range(len(qrels)):
                pick = rng.choice(len(rows), size=size, replace=False)
                order = pick[np.argsort(-sim[qi, pick])]
                seen, pages = set(), []
                for j in order:
                    s = labels[j]
                    if s in seen:
                        continue
                    seen.add(s)
                    pages.append(s)
                    if len(pages) == 10:
                        break
                lists.append(pages)
            recalls.append(summarize(lists, qrels)["R@10"])
        out.append({"candidates": size, "R@10_mean": float(np.mean(recalls)), "R@10_seeds": recalls})
        print(size, round(float(np.mean(recalls)), 4), flush=True)
    (BENCH / "random-filter.json").write_text(json.dumps({
        "keyed_chunks": len(rows),
        "rule": "uniform random subset of keyed chunks per query, cosine rescore, best chunk wins the page",
        "seeds": SEEDS,
        "cells": out,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
