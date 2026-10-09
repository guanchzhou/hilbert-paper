#!/usr/bin/env python3
"""Idea 10: HyDE / query2doc with the local qwen3.8. Generated passages stay in private/."""

import json
import time
from pathlib import Path

import numpy as np

from common import CACHE, OLLAMA_URL, PRIVATE, Bench, embed, halves, post, unit, wilcoxon, write_result

MODEL = "qwen3.8:latest"
WEIGHTS = (0.0, 0.25, 0.5, 0.75)
PROMPT = ("Write a short passage, three to five sentences, from a personal technical or research note that would "
          "answer the following search query. Write the passage directly, as the note would state it, without "
          "hedging.\n\nQuery: {q}")


def generate(q: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = post(OLLAMA_URL, {"model": MODEL, "stream": False, "think": False,
                            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 200},
                            "messages": [{"role": "user", "content": PROMPT.format(q=q)}]}, timeout=1200)
    return out["message"]["content"].strip(), time.perf_counter() - t0


def main() -> None:
    b = Bench()
    PRIVATE.mkdir(exist_ok=True)
    path = PRIVATE / "hyde.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for qi in range(b.n):
        if str(qi) in done:
            continue
        text, dt = generate(b.queries[qi])
        done[str(qi)] = {"text": text, "seconds": dt}
        if qi % 20 == 0:
            path.write_text(json.dumps(done))
            print(qi, round(dt, 2), flush=True)
    path.write_text(json.dumps(done))
    vpath = CACHE / "hyde-vectors.npy"
    if not vpath.exists():
        np.save(vpath, embed([done[str(qi)]["text"] for qi in range(b.n)]))
    H = unit(np.load(vpath))
    embed_times = []
    for qi in range(30):
        t0 = time.perf_counter()
        embed([done[str(qi)]["text"]])
        embed_times.append(time.perf_counter() - t0)
    s, _ = b.dense()
    base = b.per_question(b.lists(s))
    grid = {}
    for w in WEIGHTS:
        s2, _ = b.dense(unit(w * b.Qn + (1 - w) * H))
        pq = b.per_question(b.lists(s2))
        grid[w] = {"metrics": halves(b, pq), "pq": pq}
        print(w, {h: round(v["R@10"], 4) for h, v in grid[w]["metrics"].items()}, flush=True)
    chosen = max(WEIGHTS, key=lambda w: (grid[w]["metrics"]["tune"]["R@10"], w))
    pq = grid[chosen]["pq"]
    c = b.confirm
    t = wilcoxon(pq["R@10"][c], base["R@10"][c])
    t["MRR_diff"] = float((pq["MRR"][c] - base["MRR"][c]).mean())
    t["nDCG_diff"] = float((pq["nDCG@10"][c] - base["nDCG@10"][c]).mean())
    t["threshold"] = 0.03
    gen = np.array([done[str(qi)]["seconds"] for qi in range(b.n)])
    write_result("10-hyde", {
        "idea": 10, "name": "HyDE / query2doc with local qwen3.8",
        "model": MODEL, "fusion": "normalise(w * q + (1 - w) * h); h embedded as a document, no instruction",
        "selection": "best tune-half R@10 over w in {0, 0.25, 0.5, 0.75}; ties to larger w",
        "chosen_w": chosen,
        "grid": {str(w): v["metrics"] for w, v in grid.items()},
        "latency_seconds": {"generation_median": float(np.median(gen)), "generation_p95": float(np.percentile(gen, 95)),
                            "embedding_median": float(np.median(embed_times)),
                            "added_median": float(np.median(gen) + np.median(embed_times))},
        "tune_half_test": wilcoxon(pq["R@10"][b.tune], base["R@10"][b.tune]),
        "primary": {"H10 HyDE": t},
        "per_question": {"confirm_idx": c, "dense_R@10": base["R@10"], "chosen_R@10": pq["R@10"],
                         "generation_seconds": gen},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"])


if __name__ == "__main__":
    main()
