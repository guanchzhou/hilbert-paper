#!/usr/bin/env python3
"""Intervention: key mean-centred vectors, then repeat the LSH check and the hk1 filter."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from metrics import recall_at_k
from run_measure import QRELS, load_corpus, pg_env

BENCH = Path(__file__).resolve().parent
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
SETTINGS = [(1, 1), (1, 4), (1, 8), (1, 16), (2, 1), (2, 4), (2, 8), (2, 16)]


def run(vectors: np.ndarray, extra: list[str]) -> list[dict]:
    payload = "".join(json.dumps({"id": i, "embedding": v.tolist()}) + "\n" for i, v in enumerate(vectors))
    out = subprocess.run([str(HILBERT), "key", *extra, "--format", "jsonl"], input=payload,
                         capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def key_int(s: str) -> int:
    return int(s.rsplit(":", 1)[1], 16)


def main() -> None:
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    rels = [set(q["relevant"]) for q in qrels]
    _t, ids, slugs, _s, _x, _i, M = load_corpus(env, password)
    Q = np.load(BENCH / "query-vectors.npy")
    un = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Mn, Qn = un(M), un(Q)
    sim = Qn @ Mn.T
    mu_c = Mn.mean(axis=0)
    mu_q = Qn.mean(axis=0)
    variants = {
        "raw": (Mn, Qn),
        "centred-corpus-mean": (Mn - mu_c, Qn - mu_c),
        "centred-separate-means": (Mn - mu_c, Qn - mu_q),
    }
    rng = np.random.default_rng(3)
    a = rng.integers(0, len(ids), 200000)
    b = rng.integers(0, len(ids), 200000)
    keep = a != b
    a, b = a[keep], b[keep]
    out = {"mean_unit_vector_norm_chunks": float(np.linalg.norm(mu_c)),
           "mean_unit_vector_norm_questions": float(np.linalg.norm(mu_q)), "variants": {}}
    for name, (C, Qv) in variants.items():
        rows = run(C, [])
        cells = np.array([r["cell"] for r in rows])
        keys = np.array([key_int(r["marker"]) for r in rows], dtype=np.uint64)
        top = cells >= 128
        Cn = un(C)
        cos = np.einsum("ij,ij->i", Cn[a], Cn[b])
        pred_axis = 1 - np.arccos(np.clip(cos, -1, 1)) / math.pi
        agree = top[a] == top[b]
        same8 = agree.all(axis=1)
        occ = np.bincount(np.packbits(top, axis=1, bitorder="big")[:, 0], minlength=256)
        bins = []
        for lo in np.round(np.arange(-0.4, 1.0, 0.1), 1):
            m = (cos >= lo) & (cos < lo + 0.1)
            if m.sum() >= 50:
                bins.append({"lo": float(lo), "n": int(m.sum()), "axis_agree": float(agree[m].mean()),
                             "axis_pred": float(pred_axis[m].mean()), "cell_obs": float(same8[m].mean()),
                             "cell_pred": float((pred_axis[m] ** 8).mean())})
        filt = {}
        for level, ranges in SETTINGS:
            probes = run(Qv, ["--probe", str(level), "--ranges", str(ranges)])
            recs, sizes = [], []
            for p in probes:
                qi = int(p["id"])
                mask = np.zeros(len(ids), dtype=bool)
                for lo_s, hi_s in p["ranges"]:
                    mask |= (keys >= np.uint64(key_int(lo_s))) & (keys <= np.uint64(key_int(hi_s)))
                cand = np.flatnonzero(mask)
                sizes.append(len(cand))
                order = cand[np.argsort(-sim[qi, cand])]
                seen, pages = set(), []
                for j in order:
                    if slugs[j] not in seen:
                        seen.add(slugs[j])
                        pages.append(slugs[j])
                    if len(pages) == 10:
                        break
                recs.append(recall_at_k(pages, rels[qi], 10))
            filt[f"L{level}-R{ranges}"] = {"level": level, "ranges": ranges, "recall": float(np.mean(recs)),
                                           "median_candidates": float(np.median(sizes)),
                                           "mean_candidates": float(np.mean(sizes))}
            print(name, level, ranges, round(float(np.mean(recs)), 4), float(np.median(sizes)), flush=True)
        out["variants"][name] = {
            "axis_top_share": [float(x) for x in top.mean(axis=0)],
            "pair_axis_agree": float(agree.mean()), "pair_axis_pred": float(pred_axis.mean()),
            "pair_cell_obs": float(same8.mean()), "pair_cell_pred": float((pred_axis ** 8).mean()),
            "occupancy_nonempty": int((occ > 0).sum()), "occupancy_max": int(occ.max()),
            "occupancy_sorted": [int(x) for x in np.sort(occ)[::-1]],
            "calibration": bins, "filter": filt,
        }
    (BENCH / "centering.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
