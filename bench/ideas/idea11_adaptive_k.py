#!/usr/bin/env python3
"""Idea 11: adaptive-k cut-off at the largest score gap within the dense top 30 (offline)."""

from pathlib import Path

import numpy as np

from common import Bench, halves, holm, wilcoxon, write_result


def main() -> None:
    b = Bench()
    s, win = b.dense()
    order = b.order(s)[:, :30]
    top = np.take_along_axis(s, order, axis=1)
    gaps = top[:, :-1] - top[:, 1:]
    cut = gaps.argmax(axis=1) + 1
    fixed_surv, fixed_tok = b.chunk_pack(s, win)
    ad_surv, ad_tok = np.zeros(b.n), np.zeros(b.n)
    for qi in range(b.n):
        units = [(b.pages[p], int(b.chunk_tokens[win[qi, p]])) for p in order[qi, :cut[qi]]]
        ad_surv[qi], ad_tok[qi] = b.pack(qi, units)
    t = wilcoxon(ad_surv, fixed_surv, margin=0.02)
    reduction = 1 - ad_tok.mean() / fixed_tok.mean()
    t["threshold"] = "survival diff >= -0.02 and token reduction >= 30%"
    t["token_reduction"] = float(reduction)
    t["tokens_adaptive_mean"] = float(ad_tok.mean())
    t["tokens_fixed_mean"] = float(fixed_tok.mean())
    t["token_reduction_ci95"] = None
    rng = np.random.default_rng(20261005)
    idx = rng.integers(0, b.n, size=(10000, b.n))
    red = 1 - ad_tok[idx].mean(axis=1) / fixed_tok[idx].mean(axis=1)
    t["token_reduction_ci95"] = [float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))]
    hist = np.bincount(cut, minlength=31)[1:].tolist()
    write_result("11-adaptive-k", {
        "idea": 11, "name": "adaptive-k cut-off at the largest score gap",
        "set": "all 817 dev questions (no selection)",
        "rule": "top-30 dense note scores; cut after the largest consecutive gap; pack winning chunks under 6,000 tokens",
        "cut_position_histogram_1_to_29": hist,
        "cut_position_median": float(np.median(cut)), "cut_position_mean": float(cut.mean()),
        "share_cut_at_1": float((cut == 1).mean()),
        "conditions": {"fixed_top10": halves(b, {"survival": fixed_surv, "tokens": fixed_tok}),
                       "adaptive": halves(b, {"survival": ad_surv, "tokens": ad_tok, "k": cut.astype(float)})},
        "primary": {"H11 adaptive-k": t},
        "per_question": {"fixed_survival": fixed_surv, "adaptive_survival": ad_surv,
                         "fixed_tokens": fixed_tok, "adaptive_tokens": ad_tok, "cut": cut},
    }, Path(__file__))
    print("cut median", np.median(cut), "share 1", (cut == 1).mean(), "surv", fixed_surv.mean(), ad_surv.mean(),
          "tokens", fixed_tok.mean(), ad_tok.mean(), "p", t["p"])


if __name__ == "__main__":
    main()
