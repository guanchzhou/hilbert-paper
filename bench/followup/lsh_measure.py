#!/usr/bin/env python3
"""S3: cross-polytope against hyperplane LSH, raw, centred and whitened, as pre-registered.

Keys come from `zig-lsh keys` (128 tables; questions with 4 probe keys per table). For L tables and
0, 1 or 4 extra probes per table: the share of questions whose best relevant chunk collides in at
least one table, the mean share of chunks scanned, and R@10 after exact rescoring on the original
vectors. Built like bench/hk2_measure.py, whose curve is the reference. Writes lsh-sweep.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench, wilcoxon  # noqa: E402

LSH = Path.home() / "Development/zig-lsh/zig-out/bin/zig-lsh"
TABLES, MAX_PROBES = 128, 4
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128)
FAMILIES = {"hyperplane": 8, "cross-polytope": 1}
SEED = "9e3779b97f4a7c15"


def keys(X: np.ndarray, family: str, k: int, probes: int) -> np.ndarray:
    payload = json.dumps({"data": np.round(X.astype(np.float64), 7).tolist(), "family": family, "k": k,
                          "tables": TABLES, "seed": SEED, "probes": probes})
    out = subprocess.run([str(LSH), "keys"], input=payload, capture_output=True, text=True, check=True).stdout
    rows = [json.loads(line) for line in out.splitlines()]
    if probes == 0:
        return np.array([[int(h, 16) for h in r["keys"]] for r in rows], dtype=np.uint64)[:, :, None]
    arr = np.zeros((len(rows), TABLES, probes + 1), dtype=np.uint64)
    for i, r in enumerate(rows):
        for t, ks in enumerate(r["probes"]):
            vals = [int(h, 16) for h in ks]
            vals += [vals[0]] * (probes + 1 - len(vals))
            arr[i, t] = vals
    return arr


def two_sided(x: np.ndarray, y: np.ndarray) -> dict:
    t = wilcoxon(x, y, alternative="two-sided")
    t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
    return t


def main() -> None:
    t0 = time.time()
    b = Bench()
    sim = b.Qn @ b.Cn.T
    rows_of = {}
    for i, s in enumerate(b.slugs):
        rows_of.setdefault(s, []).append(i)
    pairs = [(qi, max(c, key=lambda r: sim[qi, r])) for qi in range(b.n)
             if (c := [r for s in b.rels[qi] for r in rows_of.get(s, [])])]
    pq = np.array([p[0] for p in pairs])
    pc = np.array([p[1] for p in pairs])

    mu = b.Cn.mean(axis=0)
    Cc, Qc = b.Cn - mu, b.Qn - mu
    lam, E = np.linalg.eigh(Cc.T @ Cc / len(Cc))
    W = E / np.sqrt(np.maximum(lam, 0) + 1e-6)
    Cw, Qw = Cc @ W, Qc @ W
    prep = {"raw": (b.Cn, b.Qn), "centred": (Cc, Qc), "whitened": (Cw, Qw)}

    un = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)  # noqa: E731
    exhaustive_raw = b.per_question(b.lists(b.dense()[0]))["R@10"]
    exhaustive_white = b.per_question(b.lists(b.note_scores(un(Qw) @ un(Cw).T)[0]))["R@10"]

    out = {"preregistration": "preregistration.md#s3", "tables_max": TABLES, "families": FAMILIES, "seed": SEED,
           "chunks": int(len(b.ids)), "questions": b.n, "pairs": len(pairs),
           "whitening": {"eps": 1e-6, "eigenvalue_min": float(lam.min()), "eigenvalue_max": float(lam.max()),
                         "fitted_on": "chunk vectors only"},
           "exhaustive": {"raw_R@10": float(exhaustive_raw.mean()), "whitened_R@10": float(exhaustive_white.mean()),
                          "test": two_sided(exhaustive_white, exhaustive_raw)},
           "rows": []}
    per_q = {}
    for family, k in FAMILIES.items():
        for pname, (C, Q) in prep.items():
            ts = time.time()
            kc = keys(C, family, k, 0)[:, :, 0]
            kq = keys(Q, family, k, MAX_PROBES)
            print(family, pname, "keys", round(time.time() - ts, 1), "s", flush=True)
            for probes in (0, 1, MAX_PROBES):
                cand = np.zeros((b.n, len(b.ids)), dtype=bool)
                for t in range(TABLES):
                    for p in range(probes + 1):
                        cand |= kc[None, :, t] == kq[:, t, p][:, None]
                    L = t + 1
                    if L not in CHECKPOINTS:
                        continue
                    scores, _ = b.note_scores(sim, cand)
                    r10 = b.per_question(b.lists(scores, finite_only=True))["R@10"]
                    size = cand.sum(axis=1)
                    per_q[(family, pname, probes, L)] = r10
                    row = {"family": family, "k": k, "preprocessing": pname, "tables": L, "probes": probes,
                           "pair_collision": float(cand[pq, pc].mean()), "scanned_mean_share": float(size.mean() / len(b.ids)),
                           "scanned_median": float(np.median(size)), "R@10_after_rescoring": float(r10.mean())}
                    out["rows"].append(row)
                    print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}, flush=True)

    def at_share(family: str, pname: str, share: float, probes: int = 0):
        for r in out["rows"]:
            if r["family"] == family and r["preprocessing"] == pname and r["probes"] == probes and r["scanned_mean_share"] >= share:
                return r
        return None

    comparisons = {}
    for share in (0.10, 0.20):
        a, h = at_share("cross-polytope", "centred", share), at_share("hyperplane", "centred", share)
        key = f"cross-polytope vs hyperplane, centred, {int(share * 100)}% scanned"
        if a and h:
            comparisons[key] = {"cross_polytope_tables": a["tables"], "hyperplane_tables": h["tables"],
                                "cross_polytope_share": a["scanned_mean_share"], "hyperplane_share": h["scanned_mean_share"],
                                **two_sided(per_q[("cross-polytope", "centred", 0, a["tables"])], per_q[("hyperplane", "centred", 0, h["tables"])])}
        else:
            comparisons[key] = {"not_reached": True}
        for family in FAMILIES:
            w, c = at_share(family, "whitened", share), at_share(family, "centred", share)
            key = f"{family}: whitened vs centred, {int(share * 100)}% scanned"
            if w and c:
                comparisons[key] = {"whitened_tables": w["tables"], "centred_tables": c["tables"],
                                    **two_sided(per_q[(family, "whitened", 0, w["tables"])], per_q[(family, "centred", 0, c["tables"])])}
            else:
                comparisons[key] = {"not_reached": True}
    out["comparisons"] = comparisons
    out["primary"] = "cross-polytope vs hyperplane, centred, 10% scanned"
    hk2 = json.loads((HERE.parent / "hk2-sweep.json").read_text())
    out["hk2_reference"] = [{k2: r[k2] for k2 in ("tables", "probes", "pair_collision_observed", "candidates_mean_share", "R@10_after_rescoring")}
                            for r in hk2["rows"]]
    out["seconds"] = time.time() - t0
    (HERE / "lsh-sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    for k2, v in comparisons.items():
        print(k2, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if x in ("diff", "p", "mean_new", "mean_comparator", "cross_polytope_tables", "hyperplane_tables", "not_reached")})
    print("exhaustive raw", round(out["exhaustive"]["raw_R@10"], 4), "whitened", round(out["exhaustive"]["whitened_R@10"], 4),
          "p", out["exhaustive"]["test"]["p"])


if __name__ == "__main__":
    main()
