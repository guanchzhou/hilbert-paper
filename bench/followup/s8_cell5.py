#!/usr/bin/env python3
"""Cell 5 of addendum 3. Call the reranker only when the top two cosine scores are
close. The threshold is chosen on the tuning half and applied once on the
confirmation half. Reranker scores and their seconds are the ones already stored
for the dense top 50. A skipped question costs 0.13 s, the no-reranker figure
fixed on the canvas. The cell passes when confirmation MRR stays inside the
always-on spread (the paired-difference interval includes zero) and the mean
time is nearer 0.13 s than the always-on time.
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
DEPTH = 50
SKIP_SECONDS = 0.13


def lists_of(b, order, rerank, call) -> list[list[str]]:
    out = []
    for qi in range(b.n):
        top = order[qi, :DEPTH]
        if call[qi]:
            scores = np.array(rerank[str(qi)]["scores"], dtype=np.float64)
            n = min(DEPTH, len(scores))
            o = np.lexsort((np.arange(n), -scores[:n]))
            top = top[:n][o]
        out.append([b.pages[j] for j in top[:10]])
    return out


def main() -> None:
    b = Bench()
    assert b.n == 817
    rerank = json.loads((HERE.parent / "ideas" / "cache" / "rerank-default.json").read_text())
    cosine, _ = b.dense()
    order = b.order(cosine)
    gap = cosine[np.arange(b.n), order[:, 0]] - cosine[np.arange(b.n), order[:, 1]]
    call_cost = np.array([rerank[str(qi)]["seconds"] for qi in range(b.n)])
    always = np.ones(b.n, dtype=bool)
    always_mrr = b.per_question(lists_of(b, order, rerank, always))["MRR"]
    candidates = np.unique(np.quantile(gap[b.tune], np.linspace(0, 1, 11)))
    chosen, best_rate = None, 2.0
    tune_rows = []
    for t in candidates:
        call = gap <= t
        mrr = b.per_question(lists_of(b, order, rerank, call))["MRR"]
        lo, hi = boot_ci(mrr[b.tune] - always_mrr[b.tune], seed=SEED)
        rate = float(call[b.tune].mean())
        tune_rows.append({"threshold": float(t), "call_rate": rate, "ci95": [lo, hi]})
        if lo <= 0 <= hi and rate < best_rate:
            chosen, best_rate = float(t), rate
    if chosen is None:
        chosen = float(candidates.max())
    call = gap <= chosen
    mrr = b.per_question(lists_of(b, order, rerank, call))["MRR"]
    lo, hi = boot_ci(mrr[b.confirm] - always_mrr[b.confirm], seed=SEED)
    selective = np.where(call, call_cost, SKIP_SECONDS)
    always_time = float(call_cost.mean())
    selective_time = float(selective[b.confirm].mean())
    nearer = abs(selective_time - SKIP_SECONDS) < abs(selective_time - always_time)
    out = {
        "cell": 5,
        "name": "skip reranker",
        "arxiv": "2606.07923",
        "n_confirm": int(len(b.confirm)),
        "threshold": chosen,
        "threshold_rule": "smallest tuning-half call rate whose MRR interval includes zero",
        "tune": tune_rows,
        "confirm_call_rate": float(call[b.confirm].mean()),
        "MRR": {"selective": float(mrr[b.confirm].mean()), "always_on": float(always_mrr[b.confirm].mean())},
        "diff": float((mrr[b.confirm] - always_mrr[b.confirm]).mean()),
        "ci95": [lo, hi],
        "seconds": {"selective": selective_time, "always_on": always_time, "skipped": SKIP_SECONDS},
        "nearer_skip_than_always": bool(nearer),
        "passes": bool(lo <= 0 <= hi and nearer),
        "seed": SEED,
    }
    (HERE / "cell5.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("threshold", "confirm_call_rate", "MRR", "diff", "ci95", "seconds", "passes")}, indent=1))


if __name__ == "__main__":
    main()
