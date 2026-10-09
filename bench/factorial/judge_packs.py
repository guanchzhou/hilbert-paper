#!/usr/bin/env python3
"""Answer-quality check of sentence-pruned packs, as pre-registered in judge-preregistration.md.

Packs are rebuilt as stage2.py builds them; before any judgement, each pack's survival is checked
against stage2.py's stored per-question values. The judge is idea 2's, unchanged (local 27B).
Judge replies are cached in private/judge-packs.json (resumable). Writes judge-packs.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))

from pipeline import CACHE, F8, PRUNE_CAP, PRUNE_DEPTH, Cfg, Data  # noqa: E402
import stage2  # noqa: E402
from common import mcnemar, rounded, wilcoxon  # noqa: E402
from idea02_prune import JUDGE_MODEL, JUDGE_PROMPT, judge  # noqa: E402
from metrics import estimate_tokens  # noqa: E402

A = (Cfg("hybrid", "mean", "off", "on", "off", "on", "none"), "pruned")
B = (Cfg("dense", "best", "off", "off", "off", "off", "none"), "chunk")
N = 150
SEED = 20261007
BUDGET = 6000


def pack(units: list[tuple[str, str]], rel: set) -> tuple[str, float, int]:
    used, parts, ok = 0, [], False
    for slug, text in units:
        cost = estimate_tokens(text)
        if used + cost <= BUDGET:
            used += cost
            parts.append(text)
            ok = ok or slug in rel
    return "\n\n---\n\n".join(parts), (1.0 if ok else 0.0), used


def main() -> None:
    t0 = time.time()
    d = Data()
    b = d.b
    sent = stage2.Sentences(d)
    split = json.loads((HERE.parent / "ideas" / "split.json").read_text())
    rng = np.random.default_rng(SEED)
    sample = sorted(int(x) for x in rng.choice(np.array(split["confirm"]), size=N, replace=False))
    pre_a = d.pre_ranking(A[0].pre())
    pre_b = d.pre_ranking(B[0].pre())
    z = np.load(CACHE / "level1-perq.npz")
    keys = [str(k) for k in z["keys"]]

    def units_a(qi):
        t, r = pre_a["top"][qi], pre_a["rows"][qi]
        ok = t >= 0
        out = []
        for note, row in zip(t[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]], r[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]]):
            row = int(row)
            sims = sent.vec[row] @ b.Qn[qi]
            keep = sorted(np.argsort(-sims)[:PRUNE_CAP])
            out.append((b.pages[int(note)], "\n".join(sent.text[row][j] for j in keep)))
        return out

    def units_b(qi):
        t, r = pre_b["top"][qi], pre_b["rows"][qi]
        ok = t >= 0
        return [(b.pages[int(note)], b.texts[int(row)]) for note, row in zip(t[:10][ok[:10]], r[:10][ok[:10]])]

    packs = {}
    for name, (cfg, unit), units in (("A", A, units_a), ("B", B, units_b)):
        stored = z["surv"][keys.index(cfg.key), list(F8).index(unit)]
        for qi in sample:
            text, surv, tok = pack(units(qi), b.rels[qi])
            assert surv == stored[qi], (name, qi, surv, stored[qi])
            packs[(name, qi)] = (text, surv, tok)
    print("packs rebuilt and checked against stage2.py on", len(sample), "questions", flush=True)

    path = HERE / "private" / "judge-packs.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    order = {qi: ("A" if rng.random() < 0.5 else "B") for qi in sample}
    for qi in sample:
        if str(qi) in done:
            continue
        first = order[qi]
        out = {"first": first}
        for cond in (first, "B" if first == "A" else "A"):
            out[cond], out[cond + "_seconds"] = judge(b.queries[qi], packs[(cond, qi)][0])
        done[str(qi)] = out
        path.write_text(json.dumps(done))
        print("judge", qi, out, flush=True)
    yes = {c: np.array([done[str(q)][c] == "yes" for q in sample], dtype=float) for c in ("A", "B")}
    surv = {c: np.array([packs[(c, q)][1] for q in sample]) for c in ("A", "B")}
    tok = {c: np.array([packs[(c, q)][2] for q in sample], dtype=float) for c in ("A", "B")}
    ni = wilcoxon(yes["A"], yes["B"], margin=0.02)
    mc = mcnemar(yes["A"], yes["B"], alternative="two-sided")
    mc["test"] = "exact McNemar (binomial on discordant pairs), two-sided"
    secs = [done[str(q)][c + "_seconds"] for q in sample for c in ("A", "B")]
    out = {"preregistration": "judge-preregistration.md", "judge_model": JUDGE_MODEL, "prompt": JUDGE_PROMPT,
           "n": N, "sample_idx": sample, "conditions": {"A": f"{A[0].key} / {A[1]}", "B": f"{B[0].key} / {B[1]}"},
           "yes": {c: float(yes[c].mean()) for c in yes}, "survival": {c: float(surv[c].mean()) for c in surv},
           "tokens": {c: float(tok[c].mean()) for c in tok},
           "yes_given_survival": {c: float(yes[c][surv[c] == 1].mean()) if (surv[c] == 1).any() else None for c in yes},
           "non_inferiority_margin_0.02": ni, "mcnemar_two_sided": mc,
           "non_inferior": bool(ni["p"] < 0.05),
           "judge_seconds_median": float(np.median(secs)), "seconds": time.time() - t0,
           "per_question": {"yes_A": yes["A"], "yes_B": yes["B"], "survival_A": surv["A"], "survival_B": surv["B"]}}
    (HERE / "judge-packs.json").write_text(json.dumps(rounded(out), indent=1) + "\n")
    print(json.dumps(rounded({k: out[k] for k in ("yes", "survival", "tokens", "yes_given_survival", "non_inferior")})))
    print("NI", rounded(ni["diff"]), rounded(ni["ci95"]), rounded(ni["p"]), "McNemar", mc["new_only"], mc["comparator_only"], rounded(mc["p"]))


if __name__ == "__main__":
    main()
