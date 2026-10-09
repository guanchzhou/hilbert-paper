#!/usr/bin/env python3
"""S10: rebuilding the level-1 key at ingest, as pre-registered in Addendum 5.

Arms A (eight random +-1 axes, logistic u, the hk1 construction), B (top eight principal axes of
the centred chunk vectors, empirical u) and C (B rotated by iterative quantization, empirical u).
A cell is the eight bits u >= 0.5. A query reads its own cell, then cells reached by flipping bit
subsets in increasing order of the summed |u - 0.5| of the flipped axes, whole cells, until the
scan reaches the budget. Candidates are ranked by exact cosine of the original chunk vectors.
Arm D stops if s10-gate.json did not pass. Writes s10.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench, holm, wilcoxon  # noqa: E402

SEED = 20261009
BITS = 8
PRIMARY_SCAN = 330
SHARES = (0.025, 0.05, 0.10, 0.20)
ITQ_ITERS = 50
MARGIN = 0.03
MASKS = np.arange(1 << BITS)
MASK_BITS = ((MASKS[:, None] >> np.arange(BITS)) & 1).astype(bool)


class Key:
    def __init__(self, axes: np.ndarray, empirical: bool, Xc: np.ndarray) -> None:
        self.axes, self.empirical = axes, empirical
        z = self.project(Xc)
        self.sorted = np.sort(z, axis=0)

    def project(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.axes) / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def u(self, X: np.ndarray) -> np.ndarray:
        z = self.project(X)
        if not self.empirical:
            return 1.0 / (1.0 + np.exp(-1.702 * z))
        n = len(self.sorted)
        return np.stack([np.searchsorted(self.sorted[:, j], z[:, j], side="right") / n for j in range(z.shape[1])], axis=1)


def cells(u: np.ndarray) -> np.ndarray:
    return ((u >= 0.5).astype(np.int64) << np.arange(BITS)).sum(axis=1)


def itq(Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    R, _ = np.linalg.qr(rng.standard_normal((Z.shape[1], Z.shape[1])))
    for _ in range(ITQ_ITERS):
        B = np.sign(Z @ R)
        B[B == 0] = 1
        U, _, Vt = np.linalg.svd(Z.T @ B)
        R = U @ Vt
    return R


def candidate_masks(b: Bench, qu: np.ndarray, ccell: np.ndarray, budget: int) -> np.ndarray:
    counts = np.bincount(ccell, minlength=1 << BITS)
    qcell = cells(qu)
    dist = np.abs(qu - 0.5)
    out = np.zeros((len(qu), len(ccell)), dtype=bool)
    for i in range(len(qu)):
        cost = (MASK_BITS * dist[i]).sum(axis=1)
        order = MASKS[np.lexsort((MASKS, cost))]
        visit = qcell[i] ^ order
        cum = np.cumsum(counts[visit])
        take = visit[: int(np.searchsorted(cum, budget)) + 1]
        out[i] = np.isin(ccell, take)
    return out


def recall(b: Bench, idx: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    sim = b.chunk_sim(b.Qn[idx])
    scores, _ = b.note_scores(sim, mask)
    return b.per_question(b.lists(scores, finite_only=mask is not None), idx)["R@10"]


def best_relevant_chunk(b: Bench) -> tuple[np.ndarray, np.ndarray]:
    sim = b.chunk_sim(b.Qn)
    rows = {}
    for i, s in enumerate(b.slugs):
        rows.setdefault(s, []).append(i)
    qi, ci = [], []
    for q in range(b.n):
        cand = [r for s in b.rels[q] for r in rows.get(s, [])]
        if cand:
            qi.append(q)
            ci.append(max(cand, key=lambda r: sim[q, r]))
    return np.array(qi), np.array(ci)


def main() -> None:
    b = Bench()
    rng = np.random.default_rng(SEED)
    mu = b.Cn.mean(axis=0)
    Xc, Qc = b.Cn - mu, b.Qn - mu
    A_axes = rng.choice([-1.0, 1.0], size=(Xc.shape[1], BITS))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:BITS].T
    R = itq((Xc / np.linalg.norm(Xc, axis=1, keepdims=True)) @ V, rng)
    keys = {"A_random_axes": Key(A_axes, False, Xc), "B_corpus_axes": Key(V, True, Xc), "C_itq": Key(V @ R, True, Xc)}

    gate = json.loads((HERE / "s10-gate.json").read_text())
    conf = b.confirm
    exhaustive = recall(b, conf, None)
    qi, ci = best_relevant_chunk(b)
    budgets = {"primary_330": PRIMARY_SCAN, **{f"{int(s * 1000) / 10}%": int(round(s * len(b.slugs))) for s in SHARES}}

    arms, per_q = {}, {}
    for name, key in keys.items():
        cu, qu = key.u(Xc), key.u(Qc[conf])
        ccell = cells(cu)
        occ = np.bincount(ccell, minlength=1 << BITS)
        rec = {}
        for label, budget in budgets.items():
            m = candidate_masks(b, qu, ccell, budget)
            r = recall(b, conf, m)
            rec[label] = {"budget": budget, "R@10": float(r.mean()), "scan_median": float(np.median(m.sum(axis=1)))}
            if label == "primary_330":
                per_q[name] = r
        pu, pc = key.u(Qc[qi]), cu[ci]
        same = cells(pu) == cells(pc)
        cos = np.einsum("ij,ij->i", Qc[qi] / np.linalg.norm(Qc[qi], axis=1, keepdims=True),
                        Xc[ci] / np.linalg.norm(Xc[ci], axis=1, keepdims=True))
        theory = (1 - np.arccos(np.clip(cos, -1, 1)) / math.pi) ** BITS
        arms[name] = {"filter": rec, "qa_same_cell_observed": float(same.mean()), "qa_same_cell_theory": float(theory.mean()),
                      "qa_pairs": int(len(qi)), "occupied_cells": int((occ > 0).sum()), "largest_cell": int(occ.max()),
                      "axis_balance": [float(x) for x in (cu >= 0.5).mean(axis=0)]}
        print(name, {k: round(v["R@10"], 4) for k, v in rec.items()}, "qa", round(arms[name]["qa_same_cell_observed"], 4),
              "theory", round(arms[name]["qa_same_cell_theory"], 4), flush=True)

    vs_a = {f"{n} vs A_random_axes": wilcoxon(per_q[n], per_q["A_random_axes"], alternative="two-sided")
            for n in ("B_corpus_axes", "C_itq")}
    for t in vs_a.values():
        t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
    for k, p in holm({k: t["p"] for k, t in vs_a.items()}).items():
        vs_a[k]["p_holm"] = p
    noninf = {f"{n} non-inferior to exhaustive": wilcoxon(per_q[n], exhaustive, alternative="greater", margin=MARGIN)
              for n in keys}
    for k, p in holm({k: t["p"] for k, t in noninf.items()}).items():
        noninf[k]["p_holm"] = p
    out = {"preregistration": "preregistration.md#addendum-5", "seed": SEED, "chunks": len(b.slugs),
           "confirmation_questions": int(len(conf)), "exhaustive_R@10": float(exhaustive.mean()),
           "itq_iterations": ITQ_ITERS, "margin": MARGIN, "cell_rule": "bit = u >= 0.5 on each of eight axes",
           "arm_D": {"run": bool(gate["passes"]), "gate": "s10-gate.json",
                     "reason": None if gate["passes"] else "gate failed: generated questions farther from the question than the chunk, and over the time limit"},
           "arms": arms, "tests_vs_A": vs_a, "non_inferiority": noninf}
    (HERE / "s10.json").write_text(json.dumps(out, indent=1) + "\n")
    print("exhaustive", round(float(exhaustive.mean()), 4))
    for k, t in {**vs_a, **noninf}.items():
        print(k, round(t["diff"], 4), "p_holm", t["p_holm"])


if __name__ == "__main__":
    main()
