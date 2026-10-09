#!/usr/bin/env python3
"""Level 2 design (pre-registration 2.2): three pairwise-covering strata, task rotation, rounds, repeats."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261005
MODELS = {"claude": {"large": "claude-opus-5-5", "small": "claude-haiku-4-5", "shared": "claude-sonnet-5-5"},
          "cursor": {"large": "gpt-5.6-sol-high", "small": "composer-2.5", "shared": "claude-sonnet-5-5-medium"}}
TIERS = ("large", "small", "shared")
BUDGETS = (2000, 6000, 24000)
CTX = (8192, 32768, 131072)
L9 = [(0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2), (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1), (2, 0, 2, 1),
      (2, 1, 0, 2), (2, 2, 1, 0)]


def covered(rows: list[dict], factors: list[str]) -> set:
    out = set()
    for r in rows:
        for f, g in itertools.combinations(factors, 2):
            out.add((f, r[f], g, r[g]))
    return out


def complete(rows: list[dict], levels: dict) -> list[dict]:
    """Greedily add rows until every pair of levels of every two factors is covered."""
    factors = list(levels)
    need = {(f, a, g, b) for f, g in itertools.combinations(factors, 2) for a in levels[f] for b in levels[g]}
    while True:
        missing = need - covered(rows, factors)
        if not missing:
            return rows
        best, gain = None, -1
        for cand in itertools.product(*[levels[f] for f in factors]):
            r = dict(zip(factors, cand))
            g = len(missing & covered([r], factors))
            if g > gain:
                best, gain = r, g
        rows.append(best)


def build(best_unit: str) -> dict:
    units = ("chunk", "page", best_unit)
    gb = []
    for i, (_, t, bu, u) in enumerate(L9):
        gb.append({"host": ("cursor", "claude")[i % 2], "tier": TIERS[t], "budget": BUDGETS[bu], "unit": units[u]})
    gb = complete(gb, {"host": ("cursor", "claude"), "tier": TIERS, "budget": BUDGETS, "unit": units})
    files = [{"host": h, "tier": t, "rtk": r} for h in ("cursor", "claude") for t in TIERS for r in ("off", "on")]
    local = [{"tool": "files", "num_ctx": c} for c in CTX] + \
            [{"tool": "gbrain", "num_ctx": c, "unit": u} for c in CTX for u in units]
    rows = []
    for r in gb:
        rows.append({"stratum": "cloud-gbrain", "host": r["host"], "tier": r["tier"],
                     "model": MODELS[r["host"]][r["tier"]], "condition": "gbrain", "budget": r["budget"],
                     "unit": r["unit"]})
    for r in files:
        rows.append({"stratum": "cloud-files", "host": r["host"], "tier": r["tier"],
                     "model": MODELS[r["host"]][r["tier"]], "condition": f"files-rtk-{r['rtk']}"})
    for r in local:
        rows.append({"stratum": "local", "host": "local", "tier": "local", "model": "qwen3.8:latest",
                     "tool": r["tool"], "num_ctx": r["num_ctx"], "unit": r.get("unit")})
    gb_k = 0
    for i, r in enumerate(rows):
        r["row"] = i
        if r["stratum"] == "cloud-gbrain":
            start = gb_k
            gb_k += 1
        elif r["stratum"] == "cloud-files":
            start = 3 * TIERS.index(r["tier"])
        else:
            start = 5
        r["tasks"] = [f"t{(start + j) % 10:02d}" for j in range(5)]
    runs = []
    rng = np.random.default_rng(SEED)
    for j in range(5):
        for i in rng.permutation(len(rows)):
            r = rows[int(i)]
            runs.append({"run_id": f"r{r['row']:02d}-{r['tasks'][j]}-a", "row": r["row"], "round": j,
                         "task": r["tasks"][j], "rep": 0})
    for rep in (1, 2):
        for r in rows:
            if r["tier"] == "shared":
                for t in r["tasks"][:2]:
                    runs.append({"run_id": f"r{r['row']:02d}-{t}-{'abc'[rep]}", "row": r["row"], "round": 5 + rep,
                                 "task": t, "rep": rep})
    cloud = sum(1 for x in runs if rows[x["row"]]["stratum"] != "local")
    return {"best_unit": best_unit, "units": units, "rows": rows, "runs": runs,
            "n_runs": len(runs), "n_cloud_runs": cloud, "n_local_runs": len(runs) - cloud,
            "pair_coverage_cloud_gbrain_rows": len(gb)}


if __name__ == "__main__":
    d = build(sys.argv[1])
    (HERE / "level2-design.json").write_text(json.dumps(d, indent=1) + "\n")
    print(d["n_runs"], "runs;", d["n_cloud_runs"], "cloud;", d["n_local_runs"], "local; gbrain rows",
          d["pair_coverage_cloud_gbrain_rows"])
