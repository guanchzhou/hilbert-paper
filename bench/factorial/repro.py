#!/usr/bin/env python3
"""Pre-registered reproduction checks (section 1.7)."""

import json

import numpy as np

from pipeline import CACHE, HERE, Bench  # noqa: F401

CHECKS = [
    ("dense R@10", "dense|best|off|off|off|off|none", "R_10", None, "all", 0.698391),
    ("lexical-OR chunk R@10", "lexical|best|off|off|off|off|none", "R_10", None, "all", 0.534),
    ("hybrid RRF R@10", "hybrid|best|off|off|off|off|none", "R_10", None, "all", 0.701),
    ("note-mean R@10", "dense|mean|off|off|off|off|none", "R_10", None, "all", 0.683),
    ("dense rerank R@10", "dense|best|on|off|off|off|none", "R_10", None, "all", 0.735),
    ("centering H4a R@10", "dense|best|off|off|on|off|none", "R_10", None, "all", 0.703034),
    ("Rocchio chosen R@10", "dense|best|off|off|off|on|none", "R_10", None, "all", None),
    ("PPR chosen R@10 (confirm)", "dense|best|off|on|off|off|none", "R_10", None, "confirm", 0.71201),
    ("hk1 L1x16 R@10", "dense|best|off|off|off|off|hk1", "R_10", None, "all", 0.304491),
    ("partitions chosen R@10", "dense|best|off|off|off|off|partitions", "R_10", None, "all", 0.645465),
    ("chunk pack survival", "dense|best|off|off|off|off|none", "surv", 0, "all", 0.755202),
    ("pruned pack survival", "dense|best|off|off|off|off|none", "surv", 4, "all", 0.867809),
]


def main() -> None:
    b = Bench()
    z = np.load(CACHE / "level1-perq.npz")
    keys = list(z["keys"])
    roc = json.loads((HERE.parent / "ideas" / "5-rocchio.json").read_text())
    out = []
    for name, key, metric, unit, half, ref in CHECKS:
        arr = z[metric][keys.index(key)]
        if unit is not None:
            arr = arr[unit]
        idx = {"all": np.arange(b.n), "confirm": b.confirm}[half]
        val = float(np.nanmean(arr[idx]))
        if ref is None:
            ref = roc["grid"][roc["chosen"]]["all"]["R@10"]
        ok = abs(val - ref) <= 0.005
        out.append({"check": name, "cell": key, "half": half, "value": round(val, 6), "reference": ref,
                    "n": int(np.isfinite(arr[idx]).sum()), "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {val:.4f} vs {ref:.4f} (n={int(np.isfinite(arr[idx]).sum())})")
    (HERE / "level1-repro.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
