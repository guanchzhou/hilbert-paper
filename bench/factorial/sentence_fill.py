#!/usr/bin/env python3
"""Embed the sentences of winning chunks not covered by the idea 2 cache (documents, no instruction)."""

import json
import time
from datetime import datetime

import numpy as np

from pipeline import CACHE, Bench, sentences
from common import embed


def main() -> None:
    b = Bench()
    rows = json.loads((CACHE / "sentence-need.json").read_text())["new_rows"]
    path = CACHE / "sentence-vectors-new.npz"
    flat, counts = [], []
    for r in rows:
        s = sentences(b.texts[r])
        flat.extend(s)
        counts.append(len(s))
    t0 = time.perf_counter()
    V = np.zeros((len(flat), 1024), dtype=np.float32)
    step = 2048
    filled = 0
    for s in range(0, len(flat), step):
        if datetime.now().hour == 6 and datetime.now().minute < 30:
            print("deadline reached", flush=True)
            break
        V[s:s + step] = embed(flat[s:s + step], batch=64)
        filled = min(s + step, len(flat))
        el = time.perf_counter() - t0
        print(f"{filled}/{len(flat)} sentences, {el:.0f} s", flush=True)
    ends = np.cumsum(counts)
    k = int((ends <= filled).sum())
    rows, counts, V = rows[:k], counts[:k], V[:int(ends[k - 1]) if k else 0]
    np.savez(path, rows=np.array(rows), counts=np.array(counts), V=V)
    (CACHE / "sentence-fill.json").write_text(json.dumps({"rows": len(rows), "sentences": len(flat),
                                                           "seconds": time.perf_counter() - t0}))
    print("done", len(flat), round(time.perf_counter() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
