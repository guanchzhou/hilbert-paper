#!/usr/bin/env python3
"""S1b and the human checks, as pre-registered: prediction-powered inference for the answer-quality
check (Angelopoulos et al. 2023), judge-human agreement and Cohen's kappa for sufficiency and
relevance, and the author's check of 30 S6 gold lists. Reads private/human-labels.json and
private/label-items.json. Writes judge-ppi.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Z = 1.959963984540054


def kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe == 1 else (po - pe) / (1 - pe)


def ppi_mean(f_all: np.ndarray, f_lab: np.ndarray, y_lab: np.ndarray) -> dict:
    rect = y_lab - f_lab
    est = float(f_all.mean() + rect.mean())
    se = float(np.sqrt(f_all.var(ddof=1) / len(f_all) + rect.var(ddof=1) / len(rect)))
    return {"estimate": est, "ci95": [est - Z * se, est + Z * se], "se": se, "n_all": int(len(f_all)), "n_labelled": int(len(y_lab))}


def main() -> None:
    items = {it["id"]: it for it in json.loads((HERE / "private" / "label-items.json").read_text())["items"]}
    labels = json.loads((HERE / "private" / "human-labels.json").read_text())
    jp = json.loads((HERE.parent / "factorial" / "judge-packs.json").read_text())
    sample = jp["sample_idx"]
    f = {c: np.array(jp["per_question"][f"yes_{c}"], dtype=float) for c in ("A", "B")}
    idx = {qi: i for i, qi in enumerate(sample)}

    human = {}
    for lid, lab in labels.items():
        it = items[lid]
        if it["kind"] == "pack":
            human[(it["qi"], it["cond"])] = 1.0 if lab == "yes" else 0.0
    both = sorted({qi for (qi, _c) in human if (qi, "A") in human and (qi, "B") in human})
    out = {"preregistration": "preregistration.md#s1", "labels": len(labels), "pack_labels": len(human),
           "questions_with_both_packs": len(both)}
    if both:
        li = np.array([idx[qi] for qi in both])
        yA = np.array([human[(qi, "A")] for qi in both])
        yB = np.array([human[(qi, "B")] for qi in both])
        out["ppi"] = {"A": ppi_mean(f["A"], f["A"][li], yA), "B": ppi_mean(f["B"], f["B"][li], yB),
                      "A_minus_B": ppi_mean(f["A"] - f["B"], f["A"][li] - f["B"][li], yA - yB)}
        out["human_only"] = {"A": float(yA.mean()), "B": float(yB.mean()), "A_minus_B": float((yA - yB).mean()), "n": len(both)}
        out["judge_only"] = {"A": float(f["A"].mean()), "B": float(f["B"].mean()), "A_minus_B": float((f["A"] - f["B"]).mean()), "n": len(sample)}
        jl = np.concatenate([f["A"][li], f["B"][li]])
        hl = np.concatenate([yA, yB])
        out["sufficiency_agreement"] = {"raw": float((jl == hl).mean()), "kappa": kappa(jl, hl), "n": int(len(hl)),
                                        "judge_yes_human_no": int(((jl == 1) & (hl == 0)).sum()),
                                        "judge_no_human_yes": int(((jl == 0) & (hl == 1)).sum())}

    rel = [(items[k], v) for k, v in labels.items() if items[k]["kind"] == "relevance"]
    if rel:
        j = np.array([1.0 if it["judge"] == "yes" else 0.0 for it, _ in rel])
        h = np.array([1.0 if v == "yes" else 0.0 for _, v in rel])
        out["relevance_agreement"] = {"raw": float((j == h).mean()), "kappa": kappa(j, h), "n": int(len(h)),
                                      "human_yes_when_judge_yes": float(h[j == 1].mean()) if (j == 1).any() else None,
                                      "human_yes_when_judge_no": float(h[j == 0].mean()) if (j == 0).any() else None}

    gold = [(items[k]["s6"], v) for k, v in labels.items() if items[k]["kind"] == "gold"]
    if gold:
        ok = {qid for qid, v in gold if v == "yes"}
        out["s6_gold_check"] = {"checked": len(gold), "correct_and_complete": len(ok), "share": len(ok) / len(gold)}
        comp = json.loads((HERE / "compositional.json").read_text())["per_question"]
        sent = json.loads((HERE / "compositional-sentences.json").read_text())["per_question"]
        sub = {}
        for name, rows, arms in (("chunk", comp, ("dense_question_top30", "dense_facets", "hk1_facets")),
                                 ("sentence", sent, ("sentence-dense", "sentence-hk1"))):
            keep = [r for r in rows if r["id"] in ok]
            sub[name] = {a: float(np.mean([r[a]["recall"] for r in keep])) if keep else None for a in arms}
        out["s6_on_accepted_gold"] = {"questions": len(ok), "set_recall": sub}

    (HERE / "judge-ppi.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "s6_on_accepted_gold"}, indent=1))


if __name__ == "__main__":
    main()
