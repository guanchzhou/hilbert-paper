#!/usr/bin/env python3
"""Cell 17 of addendum 3. A query is marked when its top 10 notes do not form one
cluster. Spread is one minus the mean pairwise cosine of those notes' winning
chunk vectors. The threshold is the tuning-half decile that most raises the mark
rate on misses over the mark rate on hits. It is applied once on the confirmation
half. A miss is a top 10 that contains no relevant note. The cell passes when,
on the confirmation half, the mark falls more often on misses than on hits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench  # noqa: E402

SEED = 20261008


def main() -> None:
    b = Bench()
    assert b.n == 817
    scores, win = b.dense()
    order = b.order(scores)
    spread = np.empty(b.n)
    miss = np.empty(b.n, dtype=bool)
    for qi in range(b.n):
        rows = [int(win[qi, int(p)]) for p in order[qi, :10]]
        V = b.Cn[rows]
        gram = V @ V.T
        iu = np.triu_indices(10, 1)
        spread[qi] = 1.0 - float(gram[iu].mean())
        rel = b.rels[qi]
        miss[qi] = not any(b.pages[int(p)] in rel for p in order[qi, :10])
    candidates = np.unique(np.quantile(spread[b.tune], np.linspace(0, 1, 11)))
    chosen, best = float(candidates[0]), -2.0
    tune_rows = []
    for t in candidates:
        mark = spread >= t
        tune_miss, tune_hit = miss[b.tune], ~miss[b.tune]
        m_miss = float(mark[b.tune][tune_miss].mean()) if tune_miss.any() else 0.0
        m_hit = float(mark[b.tune][tune_hit].mean()) if tune_hit.any() else 0.0
        tune_rows.append({"threshold": float(t), "mark_on_miss": m_miss, "mark_on_hit": m_hit})
        if m_miss - m_hit > best:
            chosen, best = float(t), m_miss - m_hit
    mark = spread >= chosen
    conf_miss, conf_hit = miss[b.confirm], ~miss[b.confirm]
    rate_miss = float(mark[b.confirm][conf_miss].mean()) if conf_miss.any() else 0.0
    rate_hit = float(mark[b.confirm][conf_hit].mean()) if conf_hit.any() else 0.0
    # paired indicator is not defined across different questions; the difference of rates
    # is bootstrapped by resampling confirmation questions.
    flagged = mark[b.confirm].astype(float)
    is_miss = miss[b.confirm].astype(float)
    # bootstrap the difference of rates
    rng = np.random.default_rng(SEED)
    n = len(b.confirm)
    diffs = []
    idx = rng.integers(0, n, size=(10000, n))
    for draw in idx:
        f, m = flagged[draw], is_miss[draw]
        if m.sum() == 0 or (1 - m).sum() == 0:
            continue
        diffs.append(f[m == 1].mean() - f[m == 0].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = {
        "cell": 17,
        "name": "ambiguity flag",
        "arxiv": "2406.07990",
        "n_confirm": int(len(b.confirm)),
        "confirm_misses": int(miss[b.confirm].sum()),
        "threshold": chosen,
        "tune": tune_rows,
        "confirm_mark_on_miss": rate_miss,
        "confirm_mark_on_hit": rate_hit,
        "diff": rate_miss - rate_hit,
        "ci95": [float(lo), float(hi)],
        "passes": bool(rate_miss > rate_hit and lo > 0),
        "seed": SEED,
    }
    (HERE / "cell17.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("threshold", "confirm_misses", "confirm_mark_on_miss", "confirm_mark_on_hit", "diff", "ci95", "passes")}, indent=1))


if __name__ == "__main__":
    main()
