#!/usr/bin/env python3
"""hk2 (multi-table angular LSH in zig-hilbert) against the paper's prediction.

`lsh-qa.json` predicts that, after mean-centring, about 71 independent 8-bit sign keys are needed
for a question to share a bucket with its best relevant chunk 90 percent of the time. This builds up
to 128 hk2 tables of 8 bits with `zig-hilbert hk2` on the centred vectors of the frozen snapshot and
measures, for L tables (and with query-directed multi-probe):
  - the share of question-answer pairs that collide in at least one table, against the prediction
    1 - (1 - (1 - θ/π)^8)^L averaged over the pairs' own angles;
  - the candidate set a query would scan (chunks sharing a bucket);
  - recall at 10 after rescoring the candidates by exact cosine, against exhaustive cosine.
Development questions only. Writes hk2-sweep.json.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH / "ideas"))
from common import Bench  # noqa: E402

HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
TABLES, BITS, PROBES = 128, 8, 4
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 71, 96, 128)
REPO = Path.home() / "Development/hilbert-paper/bench"


def hk2(vectors: np.ndarray, probes: int) -> tuple[np.ndarray, np.ndarray | None]:
    payload = "".join(json.dumps([round(float(x), 7) for x in v]) + "\n" for v in vectors)
    cmd = [str(HILBERT), "hk2", "--tables", str(TABLES), "--bits", str(BITS)]
    if probes > 1:
        cmd += ["--probes", str(probes)]
    out = subprocess.run(cmd, input=payload, capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [json.loads(line) for line in out]
    keys = np.array([[int(k, 16) for k in r["keys"]] for r in rows], dtype=np.int64)
    if probes <= 1:
        return keys, None
    pr = np.full((len(rows), TABLES, probes), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        for t, ps in enumerate(r["probes"]):
            pr[i, t, :len(ps)] = [int(p, 16) for p in ps]
    return keys, pr


def main() -> None:
    t0 = time.time()
    b = Bench()
    sim = b.Qn @ b.Cn.T
    rows_of = {}
    for i, s in enumerate(b.slugs):
        rows_of.setdefault(s, []).append(i)
    pairs = []
    for qi in range(b.n):
        cand = [r for s in b.rels[qi] for r in rows_of.get(s, [])]
        if cand:
            pairs.append((qi, max(cand, key=lambda r: sim[qi, r])))
    pq = np.array([p[0] for p in pairs])
    pc = np.array([p[1] for p in pairs])
    mu = b.Cn.mean(axis=0)
    C, Q = b.Cn - mu, b.Qn - mu
    un = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    cos = np.einsum("ij,ij->i", un(Q)[pq], un(C)[pc])
    p_table = (1 - np.arccos(np.clip(cos, -1, 1)) / math.pi) ** BITS

    kc, _ = hk2(C, 1)
    kq, prq = hk2(Q, PROBES)
    t_keys = time.time() - t0

    exhaustive = b.per_question(b.lists(b.dense()[0]))["R@10"].mean()
    out = {"tables_max": TABLES, "bits": BITS, "probes_per_table": PROBES, "pairs": len(pairs),
           "chunks": int(len(b.ids)), "questions": b.n, "exhaustive_R@10": float(exhaustive),
           "prediction_source": "per pair: 1 - (1 - (1 - θ/π)^8)^L with θ the centred question-chunk angle",
           "seconds_keys": t_keys, "rows": []}
    for probes in (1, PROBES):
        cand = np.zeros((b.n, len(b.ids)), dtype=bool)
        for t in range(TABLES):
            if probes == 1:
                cand |= kc[None, :, t] == kq[:, t][:, None]
            else:
                for p in range(probes):
                    cand |= kc[None, :, t] == prq[:, t, p][:, None]
            L = t + 1
            if L in CHECKPOINTS:
                hit = cand[pq, pc]
                pred = 1 - (1 - p_table) ** (L * (1 if probes == 1 else 1))
                scores, _ = b.note_scores(sim, cand)
                r10 = b.per_question(b.lists(scores, finite_only=True))["R@10"].mean()
                size = cand.sum(axis=1)
                row = {"tables": L, "probes": probes, "pair_collision_observed": float(hit.mean()),
                       "candidates_median": float(np.median(size)), "candidates_mean_share": float(size.mean() / len(b.ids)),
                       "R@10_after_rescoring": float(r10)}
                if probes == 1:
                    row["pair_collision_predicted"] = float(pred.mean())
                out["rows"].append(row)
                print(row, flush=True)
    first = next((r for r in out["rows"] if r["probes"] == 1 and r["pair_collision_observed"] >= 0.9), None)
    out["tables_for_0.9_observed"] = first["tables"] if first else None
    out["seconds"] = time.time() - t0
    (BENCH / "hk2-sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    REPO.joinpath("hk2-sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    REPO.joinpath("hk2_measure.py").write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
