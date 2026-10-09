#!/usr/bin/env python3
"""S3, deviation D4: exploratory matched-cost test, cross-polytope (centred, 128 tables, no probes)
against hyperplane (centred, 32 tables, no probes). Writes lsh-matched.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE))
from common import Bench  # noqa: E402
from lsh_measure import keys, two_sided  # noqa: E402


def r10(b: Bench, sim: np.ndarray, kc: np.ndarray, kq: np.ndarray, L: int) -> tuple[np.ndarray, float]:
    cand = np.zeros((b.n, len(b.ids)), dtype=bool)
    for t in range(L):
        cand |= kc[None, :, t] == kq[:, t][:, None]
    scores, _ = b.note_scores(sim, cand)
    return b.per_question(b.lists(scores, finite_only=True))["R@10"], float(cand.sum(axis=1).mean() / len(b.ids))


def main() -> None:
    b = Bench()
    sim = b.Qn @ b.Cn.T
    mu = b.Cn.mean(axis=0)
    C, Q = b.Cn - mu, b.Qn - mu
    cp, cp_share = r10(b, sim, keys(C, "cross-polytope", 1, 0)[:, :, 0], keys(Q, "cross-polytope", 1, 0)[:, :, 0], 128)
    hp, hp_share = r10(b, sim, keys(C, "hyperplane", 8, 0)[:, :, 0], keys(Q, "hyperplane", 8, 0)[:, :, 0], 32)
    out = {"deviation": "D4", "label": "exploratory",
           "cross_polytope": {"tables": 128, "probes": 0, "scanned_mean_share": cp_share, "R@10": float(cp.mean())},
           "hyperplane": {"tables": 32, "probes": 0, "scanned_mean_share": hp_share, "R@10": float(hp.mean())},
           "test": two_sided(cp, hp)}
    (HERE / "lsh-matched.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("cross_polytope", "hyperplane")}), out["test"]["diff"], out["test"]["ci95"], out["test"]["p"])


if __name__ == "__main__":
    main()
