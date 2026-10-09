#!/usr/bin/env python3
"""Idea 7: many randomised Hilbert orders (zig-hilbert with distinct seeds) with position windows."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import BENCH, CACHE, HILBERT, Bench, halves, key_int, wilcoxon, write_result

BASE_SEED = 0x9E3779B97F4A7C15
TREES = 64
LS = (8, 16, 32, 64)
K1S = (4, 8, 16, 32, 64)


def keys_for_seed(payload: str, i: int) -> np.ndarray:
    seed = format((BASE_SEED + i) % (1 << 64), "016x")
    out = subprocess.run([str(HILBERT), "key", "--seed", seed, "--format", "jsonl"], input=payload,
                         capture_output=True, text=True, check=True).stdout
    return np.array([key_int(json.loads(line)["marker"]) for line in out.splitlines()], dtype=np.uint64)


def main() -> None:
    b = Bench()
    stored = json.loads((BENCH / "chunk-keys.json").read_text())["keys"]
    rows = np.array([i for i, c in enumerate(b.ids) if str(c) in stored])
    limit = int(0.2 * len(rows))
    mu = b.Cn.mean(axis=0)
    V = np.vstack([b.Cn[rows] - mu, b.Qn - mu])
    cache = CACHE / "forest-keys.npy"
    if cache.exists():
        allkeys = np.load(cache)
    else:
        payload = "".join(json.dumps({"id": i, "embedding": [round(float(x), 7) for x in v]}) + "\n"
                          for i, v in enumerate(V))
        with ThreadPoolExecutor(max_workers=4) as ex:
            allkeys = np.array(list(ex.map(lambda i: keys_for_seed(payload, i), range(TREES))))
        np.save(cache, allkeys)
    ck, qk = allkeys[:, :len(rows)], allkeys[:, len(rows):]
    sorted_pos = [np.lexsort((rows, ck[t])) for t in range(TREES)]
    sorted_keys = [ck[t][sorted_pos[t]] for t in range(TREES)]
    qpos = np.array([np.searchsorted(sorted_keys[t], qk[t]) for t in range(TREES)])  # trees x queries

    sim = b.chunk_sim(b.Qn)
    keyed = np.zeros(len(b.ids), dtype=bool)
    keyed[rows] = True
    base_s, _ = b.note_scores(sim, np.broadcast_to(keyed, sim.shape))
    exhaustive = b.per_question(b.lists(base_s, finite_only=True))
    best_rel = np.full(b.n, -1)
    for qi in range(b.n):
        cand = [r for s in b.rels[qi] if s in b.page_index
                for r in range(b.starts[b.page_index[s]], b.ends[b.page_index[s]]) if keyed[r]]
        if cand:
            best_rel[qi] = max(cand, key=lambda r: sim[qi, r])

    curves, per_cfg = {}, {}
    for k1 in K1S:
        mask = np.zeros((b.n, len(b.ids)), dtype=bool)
        done = 0
        for L in LS:
            for t in range(done, L):
                n_rows = len(rows)
                for qi in range(b.n):
                    lo = max(0, qpos[t, qi] - k1)
                    hi = min(n_rows, qpos[t, qi] + k1)
                    mask[qi, rows[sorted_pos[t][lo:hi]]] = True
            done = L
            cand = mask.sum(axis=1)
            s, _ = b.note_scores(sim, mask)
            pq = b.per_question(b.lists(s, finite_only=True))
            has = best_rel >= 0
            covered = np.array([mask[qi, best_rel[qi]] for qi in np.flatnonzero(has)])
            key = f"L{L}/k{k1}"
            curves[key] = {"L": L, "k1": k1,
                           "median_candidates": {"all": float(np.median(cand)), "tune": float(np.median(cand[b.tune])),
                                                 "confirm": float(np.median(cand[b.confirm]))},
                           "best_relevant_chunk_in_candidates": float(covered.mean()),
                           "metrics": halves(b, pq)}
            per_cfg[key] = (pq, cand)
            print(key, int(np.median(cand)), round(pq["R@10"].mean(), 4), round(float(covered.mean()), 3), flush=True)

    eligible = [k for k, v in curves.items() if v["median_candidates"]["tune"] <= limit]
    chosen = max(eligible, key=lambda k: (curves[k]["metrics"]["tune"]["R@10"], -curves[k]["median_candidates"]["tune"]))
    pq, cand = per_cfg[chosen]
    c = b.confirm
    t = wilcoxon(pq["R@10"][c], exhaustive["R@10"][c], margin=0.03)
    t["median_candidates_confirm"] = float(np.median(cand[c]))
    t["candidate_limit"] = limit
    t["candidate_rule_pass"] = bool(np.median(cand[c]) <= limit)
    t["threshold"] = "diff >= -0.03 with median candidates <= 20% of keyed chunks"
    tune_t = wilcoxon(pq["R@10"][b.tune], exhaustive["R@10"][b.tune], margin=0.03)
    lsh = json.loads((BENCH / "lsh-qa.json").read_text())
    reach90 = sorted((v["L"], v["k1"]) for v in curves.values() if v["best_relevant_chunk_in_candidates"] >= 0.9)
    write_result("7-forest", {
        "idea": 7, "name": "Hilbert forest: randomised orders with position windows",
        "keyed_chunks": int(len(rows)), "candidate_limit_20pct": limit,
        "orders": f"zig-hilbert key --seed (0x9e3779b97f4a7c15 + i), i = 0..{TREES - 1}, on mean-centred unit vectors",
        "exhaustive_keyed": halves(b, exhaustive),
        "selection": "highest tune-half R@10 among (L, k1) with tune median candidates <= limit",
        "chosen": chosen, "curves": curves,
        "lsh_prediction": {"independent_8bit_tables_for_0.9_centred": lsh["centred"]["independent_8bit_tables_for_0.9"],
                           "settings_reaching_0.9_best_chunk_coverage": reach90},
        "tune_half_test": tune_t,
        "primary": {"H7 Hilbert forest non-inferior to exhaustive": t},
        "per_question": {"confirm_idx": c, "exhaustive_R@10": exhaustive["R@10"], "chosen_R@10": pq["R@10"],
                         "chosen_candidates": cand},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"], t["median_candidates_confirm"])


if __name__ == "__main__":
    main()
