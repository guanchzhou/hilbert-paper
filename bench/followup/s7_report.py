#!/usr/bin/env python3
"""S7 report: agreement of each decision model with the 27B judge and with the author's labels, its
speed, and the S1, S4 and S2 quantities recomputed with its labels (as pre-registered in
preregistration.md, addendum 2). Reads private/s7-items.json and private/s7-<model>.json; writes
decision-judges.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent / "factorial"))
from common import mcnemar, rounded, wilcoxon  # noqa: E402
from metrics import recall_at_k  # noqa: E402

PACKS = ("A", "B", "Bcut", "H", "R", "L")
THRESHOLD = 0.5
BOOT = 10000


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return None
    r = rankdata(p)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def agreement(p: np.ndarray, y: np.ndarray, seed: int) -> dict:
    yhat = (p >= THRESHOLD).astype(float)
    rng = np.random.default_rng(seed)
    ag, ka, au = [], [], []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        ag.append(float((yhat[i] == y[i]).mean()))
        ka.append(kappa(yhat[i], y[i]))
        a = auroc(p[i], y[i])
        if a is not None:
            au.append(a)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"n": int(len(y)), "reference_yes": float(y.mean()), "model_yes": float(yhat.mean()),
            "agreement": float((yhat == y).mean()), "agreement_ci95": ci(ag),
            "kappa": kappa(yhat, y), "kappa_ci95": ci(ka),
            "auroc": auroc(p, y), "auroc_ci95": ci(au) if au else None}


def mc(a: np.ndarray, b: np.ndarray) -> dict:
    t = mcnemar(a, b, alternative="two-sided")
    t["test"] = "exact McNemar (binomial on discordant pairs), two-sided; ci95 is a bootstrap interval of the mean difference"
    return t


def main() -> None:
    data = json.loads((HERE / "private" / "s7-items.json").read_text())
    items, lists = data["items"], data["s2_lists"]
    human_path = HERE / "private" / "human-labels.json"
    human = json.loads(human_path.read_text()) if human_path.exists() else {}
    label_items = json.loads((HERE / "private" / "label-items.json").read_text())["items"]
    human_by_item = {}
    for it in label_items:
        lab = human.get(it["id"])
        if lab not in ("yes", "no"):
            continue
        if it["kind"] == "pack":
            human_by_item[f"sufficiency|{it['qi']}-{it['cond']}"] = 1.0 if lab == "yes" else 0.0
        elif it["kind"] == "relevance":
            human_by_item[f"relevance|{it['qi']}|{it['slug']}"] = 1.0 if lab == "yes" else 0.0

    y27 = {f"{i['task']}|{i['id']}": 1.0 if i["judge27b"] == "yes" else 0.0 for i in items}
    out = {"preregistration": "preregistration.md#addendum-2", "threshold": THRESHOLD,
           "human_labels": {"sufficiency": sum(k.startswith("sufficiency") for k in human_by_item),
                            "relevance": sum(k.startswith("relevance") for k in human_by_item)},
           "judge27b_vs_human": {}, "models": {}}
    for task in ("sufficiency", "relevance"):
        keys = [k for k in human_by_item if k.startswith(task)]
        if keys:
            y = np.array([human_by_item[k] for k in keys])
            j = np.array([y27[k] for k in keys])
            out["judge27b_vs_human"][task] = {"n": len(keys), "agreement": float((j == y).mean()), "kappa": kappa(j, y)}

    for name in ("clef-flash", "jev-9b"):
        path = HERE / "private" / f"s7-{name}.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        m = {"items_judged": len(res)}
        for task in ("sufficiency", "relevance"):
            keys = [f"{i['task']}|{i['id']}" for i in items if i["task"] == task and f"{i['task']}|{i['id']}" in res]
            if not keys:
                continue
            p = np.array([res[k]["p_yes"] for k in keys])
            sec = np.array([res[k]["seconds"] for k in keys])
            tok = np.array([res[k]["input_tokens"] for k in keys])
            t = {"vs_judge27b": agreement(p, np.array([y27[k] for k in keys]), 20261008),
                 "seconds_median": float(np.median(sec)), "seconds_mean": float(sec.mean()),
                 "input_tokens_median": float(np.median(tok)), "mean_p_yes": float(p.mean())}
            if "truncated" in res[keys[0]]:
                t["share_truncated"] = float(np.mean([res[k]["truncated"] for k in keys]))
            hk = [k for k in keys if k in human_by_item]
            if hk:
                t["vs_human"] = agreement(np.array([res[k]["p_yes"] for k in hk]), np.array([human_by_item[k] for k in hk]), 20261009)
            m[task] = t

        suff = {i["id"]: i for i in items if i["task"] == "sufficiency"}
        qis = sorted({i["qi"] for i in suff.values()})
        if all(f"sufficiency|{qi}-{c}" in res for qi in qis for c in PACKS):
            yes = {c: np.array([float(res[f"sufficiency|{qi}-{c}"]["p_yes"] >= THRESHOLD) for qi in qis]) for c in PACKS}
            m["s1_s4_replication"] = {"yes": {c: float(yes[c].mean()) for c in PACKS},
                                      "mean_p_yes": {c: float(np.mean([res[f"sufficiency|{qi}-{c}"]["p_yes"] for qi in qis])) for c in PACKS},
                                      "A_vs_B": mc(yes["A"], yes["B"])}

        rel = [i for i in items if i["task"] == "relevance"]
        if all(f"relevance|{i['id']}" in res for i in rel):
            by_q: dict[str, list[str]] = {}
            n_unlab = {}
            for i in rel:
                n_unlab[str(i["qi"])] = n_unlab.get(str(i["qi"]), 0) + 1
                if res[f"relevance|{i['id']}"]["p_yes"] >= THRESHOLD:
                    by_q.setdefault(str(i["qi"]), []).append(i["slug"])
            qs = sorted(lists, key=int)
            ny = np.array([len(by_q.get(q, [])) for q in qs], dtype=float)
            nu = np.array([n_unlab.get(q, 0) for q in qs], dtype=float)
            rng = np.random.default_rng(20261008)
            ratios = [ny[i].sum() / max(1.0, nu[i].sum()) for i in (rng.integers(0, len(qs), len(qs)) for _ in range(BOOT))]
            rec = {}
            for cfg in ("reference", "best"):
                rec[cfg] = np.array([recall_at_k(lists[q][cfg], set(lists[q]["rels"]) | set(by_q.get(q, [])), 10) for q in qs])
            g = wilcoxon(rec["best"], rec["reference"], alternative="two-sided")
            g["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
            m["s2_replication"] = {"unlabelled_judged_relevant": float(ny.sum() / nu.sum()),
                                   "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                                   "R@10_extended": {k: float(v.mean()) for k, v in rec.items()},
                                   "reranker_gain_extended": g}
        out["models"][name] = m

    (HERE / "decision-judges.json").write_text(json.dumps(rounded(out), indent=1) + "\n")
    print(json.dumps(rounded({k: {t: m[t]["vs_judge27b"] for t in ("sufficiency", "relevance") if t in m}
                              for k, m in out["models"].items()}), indent=1))


if __name__ == "__main__":
    main()
