#!/usr/bin/env python3
"""Idea 6: Hilbert-ordered balanced partitions with centroid routing, plus SOAR secondary assignment.

Python prototype; zig-hilbert is used only for the stored hk1 keys. Compared at equal candidate
counts with exhaustive cosine, hk1 probes (hilbert-sweep.json), spherical k-means IVF, and random subsets.
"""

import json
from pathlib import Path

import numpy as np

from common import BENCH, SEED, Bench, halves, key_int, unit, wilcoxon, write_result

LAMBDA = 1.0
ITERS = 25
GRID_C = {64: [1, 2, 3, 4, 6, 8, 10, 12, 16, 20], 128: [1, 2, 4, 6, 8, 12, 16, 20, 25, 32, 40]}


def kmeans(X: np.ndarray, m: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cent = [X[rng.integers(len(X))]]
    d2 = 2 - 2 * (X @ cent[0])
    for _ in range(1, m):
        p = np.maximum(d2, 0) / np.maximum(d2, 0).sum()
        cent.append(X[rng.choice(len(X), p=p)])
        d2 = np.minimum(d2, 2 - 2 * (X @ cent[-1]))
    C = np.array(cent)
    for _ in range(ITERS):
        lab = (X @ C.T).argmax(axis=1)
        for j in range(m):
            sel = lab == j
            if sel.any():
                C[j] = X[sel].mean(axis=0)
        C = unit(C)
    return (X @ C.T).argmax(axis=1)


def soar_secondary(X: np.ndarray, reps: np.ndarray, primary: np.ndarray) -> np.ndarray:
    r = X - reps[primary]
    rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    # |x - c|^2 = |x|^2 - 2 x.c + |c|^2 ; ((x - c).r_hat)^2
    xc = X @ reps.T
    dist = (X * X).sum(1)[:, None] - 2 * xc + (reps * reps).sum(1)[None, :]
    proj = (X * rn).sum(1)[:, None] - rn @ reps.T
    loss = dist + LAMBDA * proj ** 2
    loss[np.arange(len(X)), primary] = np.inf
    return loss.argmin(axis=1)


class Index:
    def __init__(self, name: str, X: np.ndarray, rows: np.ndarray, labels: np.ndarray, m: int, soar: bool):
        self.name, self.m = name, m
        self.reps = unit(np.array([X[labels == j].mean(axis=0) for j in range(m)]))
        members = [list(rows[labels == j]) for j in range(m)]
        if soar:
            sec = soar_secondary(X, self.reps, labels)
            for i, j in enumerate(sec):
                members[j].append(rows[i])
        self.members = [np.array(x, dtype=np.int64) for x in members]
        self.sizes = np.array([len(x) for x in self.members])

    def route(self, Qn: np.ndarray, c: int, n_rows: int):
        top = np.argsort(-(Qn @ self.reps.T), axis=1)[:, :c]
        mask = np.zeros((len(Qn), n_rows), dtype=bool)
        cand = np.zeros(len(Qn), dtype=np.int64)
        for qi in range(len(Qn)):
            for j in top[qi]:
                mask[qi, self.members[j]] = True
            cand[qi] = self.sizes[top[qi]].sum()
        return mask, cand


def evaluate(b: Bench, sim: np.ndarray, mask: np.ndarray) -> dict:
    s, _ = b.note_scores(sim, mask)
    return b.per_question(b.lists(s, finite_only=True))


def main() -> None:
    b = Bench()
    keys = {int(k): key_int(v) for k, v in json.loads((BENCH / "chunk-keys.json").read_text())["keys"].items()}
    rows = np.array([i for i, c in enumerate(b.ids) if c in keys])
    n_keyed = len(rows)
    limit = int(0.2 * n_keyed)
    order = rows[np.lexsort((np.array([b.ids[i] for i in rows]), np.array([keys[b.ids[i]] for i in rows], dtype=np.uint64)))]
    X = b.Cn[order]
    sim = b.chunk_sim(b.Qn)
    keyed_mask = np.zeros((b.n, len(b.ids)), dtype=bool)
    keyed_mask[:, rows] = True
    exhaustive = evaluate(b, sim, keyed_mask)
    print("exhaustive keyed", exhaustive["R@10"].mean(), flush=True)

    curves, per_cfg = {}, {}
    for m in (64, 128):
        hil_labels = np.repeat(np.arange(m), [len(x) for x in np.array_split(np.arange(n_keyed), m)])
        km_labels = kmeans(X, m)
        for name, labels, soar in (("hilbert", hil_labels, False), ("hilbert+soar", hil_labels, True),
                                   ("kmeans", km_labels, False)):
            idx = Index(name, X, order, labels, m, soar)
            for c in GRID_C[m]:
                mask, cand = idx.route(b.Qn, c, len(b.ids))
                pq = evaluate(b, sim, mask)
                key = f"{name}/M{m}/C{c}"
                curves[key] = {"method": name, "M": m, "C": c, "partition_sizes_min_max": [int(idx.sizes.min()), int(idx.sizes.max())],
                               "median_candidates": {"all": float(np.median(cand)), "tune": float(np.median(cand[b.tune])),
                                                     "confirm": float(np.median(cand[b.confirm]))},
                               "mean_candidates": float(cand.mean()), "metrics": halves(b, pq)}
                per_cfg[key] = (pq, cand)
                print(key, round(float(np.median(cand))), round(pq["R@10"].mean(), 4), flush=True)

    rng = np.random.default_rng(SEED)
    random_curve = {}
    for size in (254, 508, 762, 1015):
        mask = np.zeros((b.n, len(b.ids)), dtype=bool)
        for qi in range(b.n):
            mask[qi, rng.choice(rows, size=size, replace=False)] = True
        random_curve[str(size)] = halves(b, evaluate(b, sim, mask))
    sweep = json.loads((BENCH / "hilbert-sweep.json").read_text())
    hk1 = [{"level": c["level"], "ranges": c["ranges"], "median_candidates": c["median_candidates"], "R@10": c["R@10"]}
           for c in sweep["cells"]]

    eligible = [k for k, v in curves.items() if v["method"] != "kmeans" and v["median_candidates"]["tune"] <= limit]
    chosen = max(eligible, key=lambda k: (curves[k]["metrics"]["tune"]["R@10"], -curves[k]["median_candidates"]["tune"]))
    pq, cand = per_cfg[chosen]
    c = b.confirm
    t = wilcoxon(pq["R@10"][c], exhaustive["R@10"][c], margin=0.03)
    t["median_candidates_confirm"] = float(np.median(cand[c]))
    t["candidate_limit"] = limit
    t["candidate_rule_pass"] = bool(np.median(cand[c]) <= limit)
    t["threshold"] = "diff >= -0.03 with median candidates <= 20% of keyed chunks"
    tune_t = wilcoxon(pq["R@10"][b.tune], exhaustive["R@10"][b.tune], margin=0.03)

    # Secondary: Hilbert partitions vs k-means IVF at the same M and C (paired, two-sided, all 817).
    versus = {}
    for m in (64, 128):
        for cc in GRID_C[m]:
            h = per_cfg[f"hilbert/M{m}/C{cc}"][0]["R@10"]
            k = per_cfg[f"kmeans/M{m}/C{cc}"][0]["R@10"]
            versus[f"M{m}/C{cc}"] = {"hilbert_minus_kmeans": float((h - k).mean())}
    write_result("6-partitions", {
        "idea": 6, "name": "Hilbert-ordered balanced partitions with centroid routing (+SOAR)",
        "keyed_chunks": n_keyed, "candidate_limit_20pct": limit, "soar_lambda": LAMBDA,
        "exhaustive_keyed": halves(b, exhaustive),
        "selection": "highest tune-half R@10 among Hilbert configs (plain or SOAR) with tune median candidates <= limit",
        "chosen": chosen,
        "curves": curves, "random_subsets": random_curve, "hk1_probes_from_sweep": hk1,
        "hilbert_vs_kmeans_same_M_C": versus,
        "tune_half_test": tune_t,
        "primary": {"H6 Hilbert partitions non-inferior to exhaustive": t},
        "per_question": {"confirm_idx": c, "exhaustive_R@10": exhaustive["R@10"], "chosen_R@10": pq["R@10"],
                         "chosen_candidates": cand},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"], t["median_candidates_confirm"])


if __name__ == "__main__":
    main()
