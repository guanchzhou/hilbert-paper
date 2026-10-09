#!/usr/bin/env python3
"""S5: how many questions the reranker comparison needs, as pre-registered in preregistration.md.

Per-question R@10 differences of the best-recall configuration against the reference on the 817
development questions (factorial level 1), shifted to each effect size; a paired bootstrap power
curve with the two-sided Wilcoxon signed-rank test at 0.05. Writes power.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "factorial"))
sys.path.insert(0, str(HERE.parent / "ideas"))
from pipeline import CACHE  # noqa: E402

BEST, REF = "hybrid|mean|on|on|off|off|none", "dense|best|off|off|off|off|none"
EFFECTS = {"held-out (+0.016)": 0.016, "development (observed)": None}
SIZES = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000]
B, ALPHA, TARGET = 5000, 0.05, 0.80


def power(d: np.ndarray, n: int, rng: np.random.Generator, keep: float = 1.0) -> float:
    hits = 0
    for _ in range(B):
        s = d[rng.integers(0, len(d), n)]
        if keep < 1.0:
            s = np.where(rng.random(n) < keep, s, 0.0)
        nz = s[np.abs(s) > 1e-12]
        if len(nz) and stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue < ALPHA:
            hits += 1
    return hits / B


def main() -> None:
    z = np.load(CACHE / "level1-perq.npz")
    keys = [str(k) for k in z["keys"]]
    r = z["R_10"]
    d = r[keys.index(BEST)] - r[keys.index(REF)]
    rng = np.random.default_rng(20261008)
    def curve(sample, keep):
        pts = []
        for n in SIZES:
            p = power(sample, n, rng, keep)
            pts.append({"n": n, "power": p})
            print(n, keep, p, flush=True)
            if p >= 0.99:
                break
        return pts

    def need(pts):
        for a, b in zip(pts, pts[1:]):
            if a["power"] < TARGET <= b["power"]:
                return round(a["n"] + (TARGET - a["power"]) / (b["power"] - a["power"]) * (b["n"] - a["n"]))
        return pts[0]["n"] if pts[0]["power"] >= TARGET else None

    curves, invalid = {}, {}
    for name, eff in EFFECTS.items():
        e = float(d.mean()) if eff is None else eff
        pts = curve(d, min(1.0, e / float(d.mean())))
        curves[name] = {"effect": e, "method": "dilution (deviation D3)", "keep_probability": min(1.0, e / float(d.mean())),
                        "curve": pts, "n_for_80_percent": need(pts)}
    for name, eff in (("held-out (+0.016)", 0.016), ("development (+0.050)", 0.050)):
        shifted = d - d.mean() + eff
        pts = []
        for n in SIZES:
            p = power(shifted, n, rng)
            pts.append({"n": n, "power": p})
            print(name, n, p, flush=True)
            if p >= 0.99:
                break
        invalid[name] = {"effect": eff, "method": "pre-registered shift; invalid with tied differences (D3)", "curve": pts}
    out = {"preregistration": "preregistration.md#s5", "best": BEST, "reference": REF,
           "observed_development_diff": float(d.mean()), "sd_of_differences": float(d.std(ddof=1)),
           "nonzero_share": float((np.abs(d) > 1e-12).mean()), "n_development": int(len(d)),
           "bootstrap_resamples": B, "alpha": ALPHA, "test": "two-sided Wilcoxon signed-rank, paired bootstrap",
           "heldout_n": 817, "deviations": ["D3"], "curves": curves, "shifted_method_invalid": invalid}
    (HERE / "power.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["n_for_80_percent"] for k, v in curves.items()}), "observed", round(float(d.mean()), 4))


if __name__ == "__main__":
    main()
