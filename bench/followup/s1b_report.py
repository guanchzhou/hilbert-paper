#!/usr/bin/env python3
"""S1b, the S2 label check and the S6 gold check, from private/human-labels.json.

As pre-registered: the prediction-powered estimate is the judge's mean over all 150 questions plus
the mean of (human minus judge) over the 50 labelled ones, with the normal interval of Angelopoulos
et al. (2023); the difference uses the paired values. S2 agreement is reported per stratum. S6
results are repeated on the questions whose gold list the author marked correct. Writes
judge-human.json and gold-check.json. Reads no model and does not open qrels-test.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / "factorial"
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import holm, rounded, wilcoxon  # noqa: E402

Z95 = 1.959963984540054
ARMS = ("dense_question_top10", "dense_question_top30", "dense_facets", "hk1_facets", "links_join")


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def normal(x: np.ndarray) -> dict:
    se = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    m = float(x.mean())
    return {"estimate": m, "se": se, "ci95": [m - Z95 * se, m + Z95 * se], "n": int(len(x))}


def ppi(f_all: np.ndarray, y: np.ndarray, f: np.ndarray) -> dict:
    """Angelopoulos et al. (2023): mean of the judge on all questions, plus the mean rectifier."""
    rectifier = y - f
    estimate = float(f_all.mean() + rectifier.mean())
    se = float(np.sqrt(f_all.var(ddof=1) / len(f_all) + rectifier.var(ddof=1) / len(rectifier)))
    return {"estimate": estimate, "se": se, "ci95": [estimate - Z95 * se, estimate + Z95 * se],
            "rectifier": float(rectifier.mean()), "n_judge": int(len(f_all)), "n_human": int(len(y)),
            "interval": "normal, Var(judge)/N + Var(human - judge)/n"}


def agree(human: np.ndarray, judge: np.ndarray) -> dict:
    return {"n": int(len(human)), "agreement": float((human == judge).mean()), "kappa": kappa(human, judge),
            "human_yes": float(human.mean()), "judge_yes": float(judge.mean())}


def arm_summary(rows: list[dict]) -> dict:
    out = {}
    for arm in ARMS:
        found = np.array([r[arm]["recall"] * r["gold"] for r in rows])
        tokens = np.array([r[arm]["tokens"] for r in rows], dtype=float)
        out[arm] = {m: float(np.mean([r[arm][m] for r in rows])) for m in ("recall", "precision", "size", "examined", "tokens")}
        out[arm]["tokens_per_correct"] = float(tokens.sum() / found.sum()) if found.sum() else None
    return out


def two_sided(x: np.ndarray, y: np.ndarray) -> dict:
    t = wilcoxon(x, y, alternative="two-sided")
    t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
    return t


def main() -> None:
    labels = json.loads((HERE / "private" / "human-labels.json").read_text())
    data = json.loads((HERE / "private" / "label-items.json").read_text())
    items = data["items"]
    missing = [it["id"] for it in items if labels.get(it["id"]) not in ("yes", "no")]
    if missing:
        raise SystemExit(f"{len(missing)} items unlabelled")
    sample = [int(x) for x in json.loads((FACTORIAL / "judge-packs.json").read_text())["sample_idx"]]
    cached = json.loads((FACTORIAL / "private" / "judge-packs.json").read_text())

    packs = [it for it in items if it["kind"] == "pack"]
    by_q: dict[int, dict[str, tuple[float, float]]] = {}
    for it in packs:
        qi, cond = int(it["qi"]), it["cond"]
        human = 1.0 if labels[it["id"]] == "yes" else 0.0
        judge = 1.0 if cached[str(qi)][cond] == "yes" else 0.0
        by_q.setdefault(qi, {})[cond] = (human, judge)
    labelled = sorted(by_q)
    if len(labelled) != 50 or any(set(by_q[qi]) != {"A", "B"} for qi in labelled):
        raise SystemExit("S1b expects 50 questions with both packs")
    if any(qi not in sample for qi in labelled):
        raise SystemExit("a labelled question is outside the 150")

    def series(cond: str, which: int, qis: list[int]) -> np.ndarray:
        src = by_q if which == 0 and set(qis) <= set(labelled) else None
        if src is not None and set(qis) == set(labelled):
            return np.array([by_q[qi][cond][which] for qi in qis])
        if which == 1:
            return np.array([1.0 if cached[str(qi)][cond] == "yes" else 0.0 for qi in qis])
        return np.array([by_q[qi][cond][0] for qi in qis])

    hA, jA = series("A", 0, labelled), series("A", 1, labelled)
    hB, jB = series("B", 0, labelled), series("B", 1, labelled)
    J_A, J_B = series("A", 1, sample), series("B", 1, sample)
    human = np.concatenate([hA, hB])
    judge = np.concatenate([jA, jB])
    s1b = {"preregistration": "preregistration.md#s1", "n_questions_labelled": 50, "n_questions_judge": len(sample),
           "A": {"human": normal(hA), "judge_150": normal(J_A), "judge_50": normal(jA), "ppi": ppi(J_A, hA, jA)},
           "B": {"human": normal(hB), "judge_150": normal(J_B), "judge_50": normal(jB), "ppi": ppi(J_B, hB, jB)},
           "A_minus_B": {"human": normal(hA - hB), "judge_150": normal(J_A - J_B), "judge_50": normal(jA - jB),
                         "ppi": ppi(J_A - J_B, hA - hB, jA - jB)},
           "agreement": {"all": agree(human, judge), "A": agree(hA, jA), "B": agree(hB, jB)}}

    rel = [it for it in items if it["kind"] == "relevance"]
    s2 = {}
    for stratum in ("yes", "no"):
        part = [it for it in rel if it["judge"] == stratum]
        h = np.array([1.0 if labels[it["id"]] == "yes" else 0.0 for it in part])
        j = np.array([1.0 if stratum == "yes" else 0.0 for _ in part])
        s2[f"judge_{stratum}"] = {"n": len(part), "agreement": float((h == j).mean()), "human_yes": float(h.mean())}
    h = np.array([1.0 if labels[it["id"]] == "yes" else 0.0 for it in rel])
    j = np.array([1.0 if it["judge"] == "yes" else 0.0 for it in rel])
    s2["all"] = agree(h, j)
    out = {"s1b": s1b, "s2_agreement": s2}
    (HERE / "judge-human.json").write_text(json.dumps(rounded(out), indent=1) + "\n")

    comp = json.loads((HERE / "compositional.json").read_text())
    gold = {int(it["s6"]): labels[it["id"]] for it in items if it["kind"] == "gold"}
    yes_ids = {i for i, lab in gold.items() if lab == "yes"}
    rows = [r for r in comp["per_question"] if r["id"] in yes_ids]
    if len(gold) != 30:
        raise SystemExit(f"expected 30 gold checks, got {len(gold)}")
    gold_out = {"preregistration": "preregistration.md#s6", "checked": len(gold), "correct_and_complete": len(yes_ids),
                "share_yes": len(yes_ids) / len(gold), "marked_no": sorted(i for i, lab in gold.items() if lab == "no")}
    if rows:
        gold_out["n"] = len(rows)
        gold_out["summary"] = arm_summary(rows)
        R = {a: np.array([r[a]["recall"] for r in rows]) for a in ARMS}
        tests = {"hk1_facets vs dense_facets (primary)": two_sided(R["hk1_facets"], R["dense_facets"]),
                 "dense_facets vs dense_question_top30": two_sided(R["dense_facets"], R["dense_question_top30"]),
                 "hk1_facets vs dense_question_top30": two_sided(R["hk1_facets"], R["dense_question_top30"])}
        for k, p in holm({k: t["p"] for k, t in tests.items()}).items():
            tests[k]["p_holm"] = p
        gold_out["tests"] = tests
    (HERE / "gold-check.json").write_text(json.dumps(rounded(gold_out), indent=1) + "\n")
    print(json.dumps(rounded({"A": s1b["A"]["ppi"], "B": s1b["B"]["ppi"], "A_minus_B": s1b["A_minus_B"]["ppi"],
                              "agreement": s1b["agreement"]["all"], "s2": s2, "gold_share_yes": gold_out["share_yes"]}), indent=1))


if __name__ == "__main__":
    main()
