#!/usr/bin/env python3
"""Question-to-answer collision rates under raw and centred keys, and the LSH table count they imply."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from run_measure import QRELS, load_corpus, pg_env

BENCH = Path(__file__).resolve().parent
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"


def top_bits(vectors: np.ndarray) -> np.ndarray:
    payload = "".join(json.dumps({"id": i, "embedding": v.tolist()}) + "\n" for i, v in enumerate(vectors))
    out = subprocess.run([str(HILBERT), "key", "--format", "jsonl"], input=payload, capture_output=True,
                         text=True, check=True).stdout
    return np.array([json.loads(line)["cell"] for line in out.splitlines()]) >= 128


def main() -> None:
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    _t, ids, slugs, _s, _x, _i, M = load_corpus(env, password)
    un = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Mn, Qn = un(M), un(np.load(BENCH / "query-vectors.npy"))
    rows = {}
    for i, s in enumerate(slugs):
        rows.setdefault(s, []).append(i)
    sim = Qn @ Mn.T
    pairs = []
    for qi, q in enumerate(qrels):
        cand = [r for s in q["relevant"] for r in rows.get(s, [])]
        if cand:
            pairs.append((qi, max(cand, key=lambda r: sim[qi, r])))
    qi = np.array([p[0] for p in pairs])
    ci = np.array([p[1] for p in pairs])
    mu = Mn.mean(axis=0)
    out = {"pairs": len(pairs), "target_recall": 0.9}
    for name, (C, Q) in {"raw": (Mn, Qn), "centred": (Mn - mu, Qn - mu)}.items():
        tc, tq = top_bits(C), top_bits(Q)
        same = (tc[ci] == tq[qi]).all(axis=1)
        cos = np.einsum("ij,ij->i", un(Q)[qi], un(C)[ci])
        p8 = (1 - np.arccos(np.clip(cos, -1, 1)) / math.pi) ** 8
        med = float(np.median(cos))
        p_med = (1 - math.acos(med) / math.pi) ** 8
        tables = math.log(1 - 0.9) / math.log(1 - p_med)
        out[name] = {"cos_quartiles": [float(x) for x in np.percentile(cos, [25, 50, 75])],
                     "same_cell_observed": float(same.mean()), "same_cell_predicted": float(p8.mean()),
                     "p_at_median_cos": p_med, "independent_8bit_tables_for_0.9": tables}
        print(name, out[name], flush=True)
    (BENCH / "lsh-qa.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
