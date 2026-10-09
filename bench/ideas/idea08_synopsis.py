#!/usr/bin/env python3
"""Idea 8: per-chunk synopsis (contextual retrieval) on an in-memory scratch copy of the corpus.

Step 1 times synopsis generation on 20 chunks and projects the full run. If the projection exceeds
10 hours the idea is recorded as not run within budget. Synopsis text stays in private/.
"""

import json
import time
from pathlib import Path

import numpy as np

from common import OLLAMA_URL, PRIVATE, SEED, Bench, post, write_result

MODEL = "qwen3.8:latest"
BUDGET_HOURS = 10.0
SAMPLE = 20
PROMPT = ("<document>\n{doc}\n</document>\nHere is the chunk we want to situate within the whole document\n"
          "<chunk>\n{chunk}\n</chunk>\nPlease give a short succinct context to situate this chunk within the overall "
          "document for the purposes of improving search retrieval of the chunk. Answer only with the succinct "
          "context and nothing else.")


def synopsis(doc: str, chunk: str) -> tuple[str, float, dict]:
    t0 = time.perf_counter()
    out = post(OLLAMA_URL, {"model": MODEL, "stream": False, "think": False,
                            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 120},
                            "messages": [{"role": "user", "content": PROMPT.format(doc=doc, chunk=chunk)}]},
               timeout=1200)
    meta = {k: out.get(k) for k in ("prompt_eval_count", "eval_count", "total_duration", "load_duration")}
    return out["message"]["content"].strip(), time.perf_counter() - t0, meta


def main() -> None:
    b = Bench()
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(len(b.texts), size=SAMPLE, replace=False).tolist())
    PRIVATE.mkdir(exist_ok=True)
    warm = synopsis(b.truth[b.slugs[sample[0]]][:2000], b.texts[sample[0]])
    times, metas = [], []
    for r in sample:
        text, dt, meta = synopsis(b.truth[b.slugs[r]][:2000], b.texts[r])
        times.append(dt)
        metas.append(meta)
        print(r, round(dt, 2), meta["prompt_eval_count"], meta["eval_count"], flush=True)
    per_chunk = float(np.mean(times))
    projected_h = per_chunk * len(b.texts) / 3600
    within = projected_h <= BUDGET_HOURS
    write_result("8-synopsis", {
        "idea": 8, "name": "per-chunk synopsis (contextual retrieval)",
        "status": "timed" if within else "not run within budget",
        "timing": {"sample_chunks": SAMPLE, "model": MODEL, "num_predict": 120, "warmup_seconds": warm[1],
                   "seconds_per_chunk_mean": per_chunk, "seconds_per_chunk_median": float(np.median(times)),
                   "seconds_per_chunk_p95": float(np.percentile(times, 95)),
                   "prompt_tokens_mean": float(np.mean([m["prompt_eval_count"] or 0 for m in metas])),
                   "output_tokens_mean": float(np.mean([m["eval_count"] or 0 for m in metas])),
                   "chunks": len(b.texts), "projected_hours": projected_h, "budget_hours": BUDGET_HOURS},
        "primary": {"H8 synopsis": {"p": 1.0, "diff": None, "note": "not run within budget" if not within else "pending"}},
        "limitation": "dev questions were written by an LLM from the notes; LLM synopses may share their vocabulary",
    }, Path(__file__))
    print("per chunk", per_chunk, "projected hours", projected_h, "within budget", within)


if __name__ == "__main__":
    main()
