#!/usr/bin/env python3
"""zig-pq (product quantization, flat and IVF) as a candidate filter, next to RaBitQ and hk1.

For each development question, `zig-rabitq search` (one-bit codes, 4-bit query) returns the N chunks
with the smallest estimated distance on the unit vectors of the frozen snapshot; the candidates are
rescored by exact cosine, notes take their best candidate chunk, and recall at 10 is computed. N is
swept; exhaustive cosine is the comparator. Writes pq-sweep.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH / "ideas"))
from common import Bench  # noqa: E402

PQ = Path.home() / "Development/zig-pq/zig-out/bin/zig-pq"
SWEEP = (10, 25, 50, 100, 200, 400, 800, 1600)
CONFIGS = ({"m": 64}, {"m": 128}, {"m": 128, "nlist": 64, "nprobe": 8})
REPO = Path.home() / "Development/hilbert-paper/bench"


def main() -> None:
    t0 = time.time()
    b = Bench()
    data = [[round(float(x), 7) for x in row] for row in b.Cn]
    queries = [[round(float(x), 7) for x in row] for row in b.Qn]
    sim = b.Qn @ b.Cn.T
    exhaustive = float(b.per_question(b.lists(b.dense()[0]))["R@10"].mean())
    # Share of each question's exact top-10 chunks found among the first N estimates.
    exact10 = np.argsort(-sim, axis=1)[:, :10]
    rows = []
    search_s = {}
    for cfg in CONFIGS:
      name = f"m={cfg['m']}" + (f",nlist={cfg['nlist']},nprobe={cfg['nprobe']}" if "nlist" in cfg else "")
      t1 = time.time()
      res = json.loads(subprocess.run([str(PQ), "search"], input=json.dumps({"data": data, "queries": queries,
                                      "k": max(SWEEP), "nbits": 8, "iters": 20, **cfg}), capture_output=True, text=True,
                                      check=True).stdout)["results"]
      search_s[name] = time.time() - t1
      res = [r + [r[-1]] * (max(SWEEP) - len(r)) for r in res]
      top = np.array(res)
      for n in SWEEP:
        mask = np.zeros(sim.shape, dtype=bool)
        np.put_along_axis(mask, top[:, :n], True, axis=1)
        scores, _ = b.note_scores(sim, mask)
        r10 = float(b.per_question(b.lists(scores, finite_only=True))["R@10"].mean())
        chunk_recall = float(np.mean([len(set(exact10[q]) & set(top[q, :n])) / 10 for q in range(b.n)]))
        rows.append({"config": name, "bytes_per_vector": cfg["m"], "candidates": n, "share_of_chunks": n / len(b.ids),
                     "R@10_after_rescoring": r10, "exact_top10_chunks_found": chunk_recall})
        print(rows[-1], flush=True)
    out = {"chunks": int(len(b.ids)), "questions": b.n, "nbits": 8, "iters": 20, "configs": list(CONFIGS),
           "exhaustive_R@10": exhaustive, "search_seconds_all_questions_including_index_build": search_s,
           "rows": rows, "seconds": time.time() - t0}
    (BENCH / "pq-sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    REPO.joinpath("pq-sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    REPO.joinpath("pq_measure.py").write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
