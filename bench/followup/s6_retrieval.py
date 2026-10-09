#!/usr/bin/env python3
"""S6 retrieval arms, as pre-registered in preregistration.md.

For each compositional question: (1) whole-question dense, top 10 and top 30 notes; (2) per-facet
dense, top 50 notes per title, intersected; (3) per-facet hk1, the title's level-1 probe with 16
ranges over the stored per-chunk keys, lifted to notes, intersected; (4) the links join, an upper
bound by construction. Set recall and precision against the link gold, notes examined, tokens.
Writes compositional.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))
from common import CACHE, Bench, embed, holm, unit, wilcoxon  # noqa: E402
from idea03_instruction import INSTRUCTIONS, prefix  # noqa: E402
from investigate import hilbert, key_int  # noqa: E402

FACET_K = 50
LEVEL, RANGES = 1, 16


def note_tokens(b: Bench, notes: set[str]) -> int:
    return int(sum(b.chunk_tokens[b.starts[b.page_index[s]]:b.ends[b.page_index[s]]].sum() for s in notes))


def scores_of(b: Bench, texts: list[str]) -> np.ndarray:
    V = unit(embed([prefix(INSTRUCTIONS["default"]) + t for t in texts]))
    return V, b.note_scores(b.chunk_sim(V))[0]


def main() -> None:
    b = Bench()
    spec = json.loads((HERE / "compositional-questions.json").read_text())
    qs = spec["questions"]
    _, sq = scores_of(b, [q["question"] for q in qs])
    VA, sa = scores_of(b, [q["title_a"] for q in qs])
    VB, sb = scores_of(b, [q["title_b"] for q in qs])

    h = np.load(CACHE / "hk1-default.npz")
    ckey, keyed = h["ckey"], h["keyed"]

    def regions(V: np.ndarray) -> list[set[str]]:
        out = []
        for r in sorted(hilbert(V, ["--probe", str(LEVEL), "--ranges", str(RANGES)]), key=lambda r: int(r["id"])):
            mask = np.zeros(len(ckey), dtype=bool)
            for lo, hi in r["ranges"]:
                mask |= (ckey >= np.uint64(key_int(lo))) & (ckey <= np.uint64(key_int(hi)))
            mask &= keyed
            out.append({b.pages[p] for p in np.unique(b.page_of_row[mask])})
        return out

    ra, rb = regions(VA), regions(VB)

    def top(scores: np.ndarray, k: int, drop: set[str]) -> list[str]:
        out = []
        for p in b.order(scores[None, :])[0]:
            s = b.pages[p]
            if s not in drop:
                out.append(s)
            if len(out) == k:
                break
        return out

    rows = []
    for i, q in enumerate(qs):
        drop = {q["a"], q["b"]}
        gold = set(q["gold"])
        ta50, tb50 = top(sa[i], FACET_K, drop), top(sb[i], FACET_K, drop)
        arms = {
            "dense_question_top10": (set(top(sq[i], 10, drop)), 10),
            "dense_question_top30": (set(top(sq[i], 30, drop)), 30),
            "dense_facets": (set(ta50) & set(tb50), len(set(ta50) | set(tb50))),
            "hk1_facets": ((ra[i] & rb[i]) - drop, len((ra[i] | rb[i]) - drop)),
        }
        row = {"id": q["id"], "gold": len(gold)}
        for name, (s, examined) in arms.items():
            hit = len(s & gold)
            row[name] = {"recall": hit / len(gold), "precision": hit / len(s) if s else 0.0, "size": len(s),
                         "examined": examined, "tokens": note_tokens(b, s)}
        row["hk1_region_coverage"] = {"a": len(ra[i] & gold) / len(gold), "b": len(rb[i] & gold) / len(gold),
                                      "size_a": len(ra[i] - drop), "size_b": len(rb[i] - drop)}
        row["links_join"] = {"recall": 1.0, "precision": 1.0, "size": len(gold),
                             "examined": q["linkers_a"] + q["linkers_b"], "tokens": note_tokens(b, gold)}
        rows.append(row)

    arms = ["dense_question_top10", "dense_question_top30", "dense_facets", "hk1_facets", "links_join"]
    summary = {a: {**{m: float(np.mean([r[a][m] for r in rows])) for m in ("recall", "precision", "size", "examined", "tokens")},
                   "tokens_median": float(np.median([r[a]["tokens"] for r in rows])),
                   "size_median": float(np.median([r[a]["size"] for r in rows])),
                   "empty_share": float(np.mean([r[a]["size"] == 0 for r in rows]))}
               for a in arms}
    R = {a: np.array([r[a]["recall"] for r in rows]) for a in arms}
    def two_sided(x: np.ndarray, y: np.ndarray) -> dict:
        t = wilcoxon(x, y, alternative="two-sided")
        t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
        return t

    tests = {
        "hk1_facets vs dense_facets (primary)": two_sided(R["hk1_facets"], R["dense_facets"]),
        "dense_facets vs dense_question_top30": two_sided(R["dense_facets"], R["dense_question_top30"]),
        "hk1_facets vs dense_question_top30": two_sided(R["hk1_facets"], R["dense_question_top30"]),
    }
    for k, p in holm({k: t["p"] for k, t in tests.items()}).items():
        tests[k]["p_holm"] = p
    out = {"preregistration": "preregistration.md#s6", "n": len(rows), "facet_k": FACET_K,
           "hk1_probe": {"level": LEVEL, "ranges": RANGES}, "summary": summary, "tests": tests,
           "hk1_region_coverage_mean": float(np.mean([(r["hk1_region_coverage"]["a"] + r["hk1_region_coverage"]["b"]) / 2 for r in rows])),
           "hk1_region_size_median": float(np.median([r["hk1_region_coverage"]["size_a"] for r in rows] + [r["hk1_region_coverage"]["size_b"] for r in rows])),
           "per_question": rows}
    (HERE / "compositional.json").write_text(json.dumps(out, indent=1) + "\n")
    for a in arms:
        print(a, {k: round(v, 3) for k, v in summary[a].items()})
    for k, t in tests.items():
        print(k, round(t["diff"], 3), "p", t["p"], "holm", t.get("p_holm"))


if __name__ == "__main__":
    main()
