#!/usr/bin/env python3
"""S9a: an explicit attribute (area) against the S6 ways of answering "notes in P about B".

A pair is an area P with at least 10 snapshot notes and a note B outside P that 3 to 15 notes in P
link to. Sixty pairs are drawn (an area in at most three pairs, B in at most two); the gold of
"Which notes in <area words> are about <title of B>?" is the notes in P linking to B. Arms: (1)
whole-question dense, top 10 and top 30; (2) dense top 50 of B's title kept if in P; (3) dense top
50 of B's title intersected with dense top 50 of the area words; (4) the level-1 hk1 region of B's
title kept if in P. Questions and names go to ../private/s9a-questions.json; s9a.json holds counts
and scores only. With --explore the area cap rises until 60 pairs are drawn (D13), written to
s9a-explore.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import CACHE, Bench, holm, wilcoxon  # noqa: E402
from investigate import hilbert, key_int  # noqa: E402
from s6_questions import TYPES, psql  # noqa: E402
from s6_retrieval import note_tokens, scores_of  # noqa: E402

SEED = 20261009
N_PAIRS = 60
MIN_AREA = 10
LINKERS_IN_P = (3, 15)
AREA_CAP, B_CAP = 3, 2
FACET_K = 50
LEVEL, RANGES = 1, 16
EXPLORE = "--explore" in sys.argv
PRIVATE = HERE.parent / "private" / ("s9a-questions-explore.json" if EXPLORE else "s9a-questions.json")
OUT = HERE / ("s9a-explore.json" if EXPLORE else "s9a.json")


def area(slug: str) -> str:
    parts = slug.removeprefix("obsidian/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else parts[0]


def area_words(a: str) -> str:
    return " ".join(a.replace("/", " ").replace("-", " ").replace("_", " ").split())


def build(b: Bench) -> tuple[list[dict], int, int, int]:
    snap = set(b.pages)
    members = defaultdict(set)
    for s in snap:
        members[area(s)].add(s)
    big = sorted(a for a, m in members.items() if len(m) >= MIN_AREA)
    types = ",".join("'" + t + "'" for t in TYPES)
    rows = psql(f"""
        select distinct fa.slug, ta.slug from links l
        join pages fa on fa.id = l.from_page_id join pages ta on ta.id = l.to_page_id
        where fa.source_id = 'default' and ta.source_id = 'default'
          and fa.deleted_at is null and ta.deleted_at is null
          and l.link_type in ({types}) and l.from_page_id <> l.to_page_id""")
    linkers = defaultdict(set)
    for f, t in rows:
        if f in snap and t in snap:
            linkers[t].add(f)
    titles = dict(psql("select slug, title from pages where source_id = 'default' and deleted_at is null"))

    pairs = []
    for p in big:
        for t in sorted(linkers):
            if area(t) == p:
                continue
            gold = linkers[t] & members[p]
            if LINKERS_IN_P[0] <= len(gold) <= LINKERS_IN_P[1]:
                pairs.append((p, t, sorted(gold)))
    area_cap = AREA_CAP
    while True:
        chosen = draw(pairs, members, titles, area_cap)
        if not EXPLORE or len(chosen) == N_PAIRS or area_cap >= N_PAIRS:
            break
        area_cap += 1
    PRIVATE.write_text(json.dumps({"seed": SEED, "qualifying_pairs": len(pairs), "areas_eligible": len(big),
                                   "area_cap": area_cap, "questions": chosen}, indent=1, ensure_ascii=False) + "\n")
    print(len(big), "areas with", MIN_AREA, "+ notes;", len(pairs), "qualifying pairs;", len(chosen), "drawn; area cap", area_cap)
    return chosen, len(pairs), len(big), area_cap


def draw(pairs: list, members: dict, titles: dict, area_cap: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    used_a, used_b = defaultdict(int), defaultdict(int)
    chosen = []
    for k in rng.permutation(len(pairs)):
        p, t, gold = pairs[k]
        if used_a[p] >= area_cap or used_b[t] >= B_CAP:
            continue
        used_a[p] += 1
        used_b[t] += 1
        tb = titles.get(t) or t
        chosen.append({"id": len(chosen), "area": p, "area_words": area_words(p), "b": t, "title_b": tb,
                       "question": f"Which notes in {area_words(p)} are about {tb}?", "gold": gold,
                       "area_size": len(members[p])})
        if len(chosen) == N_PAIRS:
            break
    return chosen


def main() -> None:
    b = Bench()
    qs, n_pairs, n_areas, area_cap = build(b)
    _, sq = scores_of(b, [q["question"] for q in qs])
    VB, sb = scores_of(b, [q["title_b"] for q in qs])
    _, sp = scores_of(b, [q["area_words"] for q in qs])

    h = np.load(CACHE / "hk1-default.npz")
    ckey, keyed = h["ckey"], h["keyed"]
    regions = []
    for r in sorted(hilbert(VB, ["--probe", str(LEVEL), "--ranges", str(RANGES)]), key=lambda r: int(r["id"])):
        mask = np.zeros(len(ckey), dtype=bool)
        for lo, hi in r["ranges"]:
            mask |= (ckey >= np.uint64(key_int(lo))) & (ckey <= np.uint64(key_int(hi)))
        mask &= keyed
        regions.append({b.pages[p] for p in np.unique(b.page_of_row[mask])})

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
        drop = {q["b"]}
        gold = set(q["gold"])
        in_p = {s for s in b.pages if area(s) == q["area"]}
        tb50, tp50 = top(sb[i], FACET_K, drop), top(sp[i], FACET_K, drop)
        reg = regions[i] - drop
        arms = {
            "dense_question_top10": (set(top(sq[i], 10, drop)), 10),
            "dense_question_top30": (set(top(sq[i], 30, drop)), 30),
            "dense_title_area_filter": (set(tb50) & in_p, FACET_K),
            "dense_title_and_area_words": (set(tb50) & set(tp50), len(set(tb50) | set(tp50))),
            "hk1_title_area_filter": (reg & in_p, len(reg)),
        }
        row = {"id": q["id"], "gold": len(gold), "area_size": q["area_size"]}
        for name, (s, examined) in arms.items():
            hit = len(s & gold)
            row[name] = {"recall": hit / len(gold), "precision": hit / len(s) if s else 0.0, "size": len(s),
                         "examined": examined, "tokens": note_tokens(b, s)}
        row["area_filter_upper_bound"] = {"recall": 1.0, "precision": len(gold) / len(in_p), "size": len(in_p)}
        rows.append(row)

    arms = ["dense_question_top10", "dense_question_top30", "dense_title_area_filter",
            "dense_title_and_area_words", "hk1_title_area_filter"]
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
        "dense_title_area_filter vs dense_title_and_area_words (primary)":
            two_sided(R["dense_title_area_filter"], R["dense_title_and_area_words"]),
        "dense_title_area_filter vs dense_question_top30":
            two_sided(R["dense_title_area_filter"], R["dense_question_top30"]),
        "hk1_title_area_filter vs dense_title_area_filter":
            two_sided(R["hk1_title_area_filter"], R["dense_title_area_filter"]),
    }
    for k, p in holm({k: t["p"] for k, t in tests.items()}).items():
        tests[k]["p_holm"] = p
    out = {"preregistration": "preregistration.md#addendum-4", "deviation": "deviations.md#d13",
           "exploratory": EXPLORE, "area_cap": area_cap, "seed": SEED, "n": len(rows),
           "qualifying_pairs": n_pairs, "areas_eligible": n_areas, "facet_k": FACET_K,
           "hk1_probe": {"level": LEVEL, "ranges": RANGES},
           "gold_size": {"median": float(np.median([r["gold"] for r in rows])),
                         "min": min(r["gold"] for r in rows), "max": max(r["gold"] for r in rows)},
           "area_size_median": float(np.median([r["area_size"] for r in rows])),
           "summary": summary, "tests": tests,
           "private": "note names, note text and question texts are kept out of the repository",
           "per_question": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for a in arms:
        print(a, {k: round(v, 3) for k, v in summary[a].items()})
    for k, t in tests.items():
        print(k, round(t["diff"], 3), "p", t["p"], "holm", t.get("p_holm"))


if __name__ == "__main__":
    main()
