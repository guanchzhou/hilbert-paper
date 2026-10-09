#!/usr/bin/env python3
"""Idea 4: remove the mean direction before dense ranking (four pre-registered variants)."""

from pathlib import Path

import numpy as np

from common import Bench, halves, holm, unit, wilcoxon, write_result


def project_off(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = u / np.linalg.norm(u)
    return x - np.outer(x @ u, u)


def main() -> None:
    b = Bench()
    base_s, _ = b.dense()
    base = b.per_question(b.lists(base_s))
    mu_c = b.Cn.mean(axis=0)
    mu_q = b.Qn.mean(axis=0)
    variants = {
        "H4a subtract shared mean": (b.Cn - mu_c, b.Qn - mu_c),
        "H4b subtract separate means": (b.Cn - mu_c, b.Qn - mu_q),
        "H4c project off shared direction": (project_off(b.Cn, mu_c), project_off(b.Qn, mu_c)),
        "H4d project off separate directions": (project_off(b.Cn, mu_c), project_off(b.Qn, mu_q)),
    }
    conds, tests, per_q = {}, {}, {"dense_R@10": base["R@10"], "dense_MRR": base["MRR"]}
    for name, (C, Q) in variants.items():
        s, _ = b.dense(unit(Q), unit(C))
        pq = b.per_question(b.lists(s))
        conds[name] = halves(b, pq)
        t = wilcoxon(pq["R@10"], base["R@10"])
        t["MRR_diff"] = float((pq["MRR"] - base["MRR"]).mean())
        t["nDCG_diff"] = float((pq["nDCG@10"] - base["nDCG@10"]).mean())
        t["threshold"] = 0.02
        tests[name] = t
        per_q[name.split()[0] + "_R@10"] = pq["R@10"]
        per_q[name.split()[0] + "_MRR"] = pq["MRR"]
        print(name, round(t["diff"], 4), t["ci95"], t["p"], flush=True)
    within = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm_within_idea"] = within[k]
    write_result("4-centering", {
        "idea": 4, "name": "remove the mean direction",
        "set": "all 817 dev questions (no selection)",
        "mean_unit_norm": {"chunks": float(np.linalg.norm(mu_c)), "questions": float(np.linalg.norm(mu_q))},
        "cosine_mu_c_mu_q": float(mu_c @ mu_q / np.linalg.norm(mu_c) / np.linalg.norm(mu_q)),
        "conditions": {"dense": halves(b, base), **conds},
        "primary": tests,
        "per_question": per_q,
    }, Path(__file__))


if __name__ == "__main__":
    main()
