#!/usr/bin/env python3
"""Score the uncached (question, chunk) reranker pairs, question by question in a fixed random order.

Same server, model, instruction and request format as idea 1. Stops starting new questions at the
deadline (pre-registration 1.5). Resumable: scores are cached by (question index, chunk id).
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from pipeline import CACHE, IDEAS_CACHE, Bench
from common import RERANK_URL, SEED, post

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
BATCH = 50


def score(query: str, docs: list[str]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    out = post(RERANK_URL, {"model": "qwen3-reranker-0.6b", "query": query, "documents": docs,
                            "instruction": INSTRUCTION})
    s = [0.0] * len(docs)
    for r in out["results"]:
        s[r["index"]] = r["relevance_score"]
    return s, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="05:00")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="re-score question 0's cached top 50 and compare")
    a = ap.parse_args()
    b = Bench()
    if a.check:
        s, win = b.dense()
        order = b.order(s)[:, :50]
        cached = json.loads((IDEAS_CACHE / "rerank-default.json").read_text())["0"]["scores"]
        docs = [b.texts[int(win[0, p])] for p in order[0]]
        new, dt = score(b.queries[0], docs)
        diff = np.abs(np.array(new) - np.array(cached))
        res = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "seconds": dt}
        (CACHE / "rerank-consistency.json").write_text(json.dumps(res))
        print("consistency", res, flush=True)
        return
    need = {int(k): v for k, v in json.loads((CACHE / "rerank-need.json").read_text()).items()}
    path = CACHE / "rerank-pairs.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    calls_path = CACHE / "rerank-calls.jsonl"
    hh, mm = map(int, a.deadline.split(":"))
    now = datetime.now()
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline < now:
        deadline += timedelta(days=1)
    perm = np.random.default_rng(SEED).permutation(b.n)
    todo = [int(q) for q in perm if int(q) in need and any(f"{q}:{b.ids[r]}" not in done for r in need[int(q)])]
    print(len(todo), "questions to score; deadline", deadline, flush=True)
    lock = threading.Lock()
    state = {"n": 0, "t0": time.perf_counter(), "pairs": 0}

    def work(qi: int) -> None:
        if datetime.now() >= deadline:
            return
        rows = [r for r in need[qi] if f"{qi}:{b.ids[r]}" not in done]
        got = {}
        for s in range(0, len(rows), BATCH):
            part = rows[s:s + BATCH]
            sc, dt = score(b.queries[qi], [b.texts[r] for r in part])
            for r, v in zip(part, sc):
                got[f"{qi}:{b.ids[r]}"] = {"score": v}
            with lock:
                with open(calls_path, "a") as fh:
                    fh.write(json.dumps({"q": qi, "n": len(part), "seconds": dt}) + "\n")
        with lock:
            done.update(got)
            state["n"] += 1
            state["pairs"] += len(rows)
            if state["n"] % 10 == 0:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(done))
                tmp.replace(path)
                el = time.perf_counter() - state["t0"]
                print(f"{state['n']}/{len(todo)} q, {state['pairs']} pairs, {state['pairs'] / el:.1f} pairs/s, "
                      f"{datetime.now():%H:%M:%S}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(done))
    tmp.replace(path)
    complete = [q for q in need if all(f"{q}:{b.ids[r]}" in done for r in need[q])]
    print("complete questions", len(complete), "of", len(need), flush=True)


if __name__ == "__main__":
    main()
