#!/usr/bin/env python3
"""S1a and S4: judge B-cut (chunk pack cut to the sentence pack's length), H, R and L once each on
the 150 questions, with the idea 2 judge unchanged, in a seeded random order over all 600 calls.
Judgements are cached in private/judge-followup.json (resumable). Writes judge-length.json and
compress-packs.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from packs import Packs  # noqa: E402

sys.path.insert(0, str(HERE.parent / "ideas"))
from common import mcnemar, rounded, wilcoxon  # noqa: E402
from idea02_prune import JUDGE_MODEL, JUDGE_PROMPT, judge  # noqa: E402

CACHE = HERE / "private" / "judge-followup.json"
S4 = HERE / "private" / "s4-packs.json"


def mc(a: np.ndarray, b: np.ndarray) -> dict:
    t = mcnemar(a, b, alternative="two-sided")
    t["test"] = "exact McNemar (binomial on discordant pairs), two-sided; ci95 is a bootstrap interval of the mean difference"
    return t


def main() -> None:
    t0 = time.time()
    p = Packs()
    p.check()
    s4 = json.loads(S4.read_text())
    texts = {}
    for qi in p.sample:
        a_tok = p.a(qi)[2]
        texts[(qi, "Bcut")] = p.b_pack(qi, a_tok)
        for c in ("H", "R", "L"):
            e = s4[str(qi)][c]
            texts[(qi, c)] = (e["text"], e.get("survival"), e["tokens"])
    rng = np.random.default_rng(20261008)
    jobs = [(qi, c) for qi in p.sample for c in ("Bcut", "H", "R", "L")]
    order = [jobs[i] for i in rng.permutation(len(jobs))]
    done = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for qi, c in order:
        key = f"{qi}-{c}"
        if key in done:
            continue
        ans, secs = judge(p.b.queries[qi], texts[(qi, c)][0])
        done[key] = {"answer": ans, "seconds": secs}
        CACHE.write_text(json.dumps(done))
        print("judge", key, ans, round(secs, 1), len(done), "of", len(jobs), flush=True)

    yes = {c: np.array([1.0 if done[f"{qi}-{c}"]["answer"] == "yes" else 0.0 for qi in p.sample]) for c in ("Bcut", "H", "R", "L")}
    yes["A"] = np.array([p.judged(qi, "A") for qi in p.sample])
    yes["B"] = np.array([p.judged(qi, "B") for qi in p.sample])
    tok = {c: float(np.mean([texts[(qi, c)][2] for qi in p.sample])) for c in ("Bcut", "H", "R", "L")}
    surv = {c: float(np.mean([texts[(qi, c)][1] for qi in p.sample])) for c in ("Bcut", "H", "R")}
    secs = [v["seconds"] for v in done.values()]

    s1a = {"preregistration": "preregistration.md#s1", "n": len(p.sample), "yes": {c: float(yes[c].mean()) for c in ("A", "B", "Bcut")},
           "tokens": {"Bcut": tok["Bcut"]}, "survival": {"Bcut": surv["Bcut"]},
           "Bcut_vs_A": mc(yes["Bcut"], yes["A"]), "Bcut_vs_B": mc(yes["Bcut"], yes["B"])}
    bb, ba = s1a["Bcut_vs_B"]["p"] < 0.05, s1a["Bcut_vs_A"]["p"] < 0.05
    closer_a = abs(yes["Bcut"].mean() - yes["A"].mean()) < abs(yes["Bcut"].mean() - yes["B"].mean())
    s1a["reading"] = ("length explains the gap" if bb and closer_a else
                      "content explains the gap" if (not bb and ba) else "neither pre-registered reading applies")
    (HERE / "judge-length.json").write_text(json.dumps(rounded(s1a), indent=1) + "\n")

    s4o = {"preregistration": "preregistration.md#s4", "deviations": ["D5"], "judge_model": JUDGE_MODEL, "prompt": JUDGE_PROMPT,
           "n": len(p.sample), "yes": {c: float(yes[c].mean()) for c in ("A", "B", "H", "R", "L")},
           "tokens": {c: tok[c] for c in ("H", "R", "L")}, "survival": {c: surv[c] for c in ("H", "R")}, "tests": {}}
    for c in ("H", "R", "L"):
        s4o["tests"][f"{c} vs B, non-inferiority 0.02"] = wilcoxon(yes[c], yes["B"], margin=0.02)
        s4o["tests"][f"{c} vs B, non-inferiority 0.02"]["non_inferior"] = bool(s4o["tests"][f"{c} vs B, non-inferiority 0.02"]["p"] < 0.05)
        s4o["tests"][f"{c} vs A, McNemar"] = mc(yes[c], yes["A"])
    s4o["judge_seconds_median"] = float(np.median(secs))
    s4o["seconds"] = time.time() - t0
    s4o["per_question"] = {c: yes[c] for c in yes}
    (HERE / "compress-packs.json").write_text(json.dumps(rounded(s4o), indent=1) + "\n")
    print(json.dumps(rounded({"S1a": s1a["yes"], "reading": s1a["reading"], "S4": s4o["yes"], "tokens": s4o["tokens"]})))


if __name__ == "__main__":
    main()
