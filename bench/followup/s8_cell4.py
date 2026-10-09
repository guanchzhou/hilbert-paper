#!/usr/bin/env python3
"""Cell 4 of addendum 3: down-weight notes that sit close to every query, then
rank again. Settings fixed before the run. A note's hubness is its mean cosine
to the tuning-half queries. That constant is subtracted from its cosine on every
question. The comparison is recall at 10 against best-chunk ranking, the best
ranker so far. The cell passes when the 95% interval for the difference sits
above zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench, boot_ci  # noqa: E402

SEED = 20261008


def main() -> None:
    b = Bench()
    assert b.n == 817
    cosine, _ = b.dense()
    hub = cosine[b.tune].mean(axis=0)
    adjusted = cosine - hub
    base = b.per_question(b.lists(cosine))["R@10"]
    new = b.per_question(b.lists(adjusted))["R@10"]
    diff = new - base
    lo, hi = boot_ci(diff, seed=SEED)
    out = {
        "cell": 4,
        "name": "hubness",
        "arxiv": "2508.02538",
        "n": int(b.n),
        "settings": {"hubness": "mean cosine to the tuning-half queries", "adjustment": "subtract"},
        "tune_n": int(len(b.tune)),
        "R@10": {"hubness": float(new.mean()), "best_chunk": float(base.mean())},
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
        "interval_above_zero": bool(lo > 0),
        "passes": bool(lo > 0),
        "seed": SEED,
    }
    (HERE / "cell4.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
