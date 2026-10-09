#!/usr/bin/env python3
"""S6: compositional questions and their link-derived gold, as pre-registered in preregistration.md.

A pair (A, B) of snapshot pages qualifies when each has at most 60 distinct linking notes and they
share 3 to 15 linking notes. Sixty pairs are drawn (a page in at most two drawn pairs); the gold of
"Which notes are about both <A> and <B>?" is the set of notes linking to both. Writes
compositional-questions.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench  # noqa: E402

TYPES = ("mentions", "", "related_to", "see_also", "related", "topic", "relates_to")
SEED = 20261008
N_PAIRS = 60
MAX_DEGREE = 60
COMMON = (3, 15)


def psql(sql: str) -> list[list[str]]:
    url = json.loads(Path(os.path.expanduser("~/.gbrain/config.json")).read_text())["database_url"]
    out = subprocess.run(["/opt/homebrew/opt/postgresql@18/bin/psql", url, "-At", "-F", "\x1f", "-c", sql],
                         capture_output=True, text=True, check=True).stdout
    return [line.split("\x1f") for line in out.splitlines() if line]


def main() -> None:
    b = Bench()
    snap = set(b.pages)
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

    targets = sorted(t for t, s in linkers.items() if len(s) <= MAX_DEGREE)
    pairs = []
    for i, a in enumerate(targets):
        for bb in targets[i + 1:]:
            common = (linkers[a] & linkers[bb]) - {a, bb}
            if COMMON[0] <= len(common) <= COMMON[1]:
                pairs.append((a, bb, sorted(common)))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    used = defaultdict(int)
    chosen = []
    for k in order:
        a, bb, gold = pairs[k]
        if used[a] >= 2 or used[bb] >= 2:
            continue
        used[a] += 1
        used[bb] += 1
        ta, tb = titles.get(a) or a, titles.get(bb) or bb
        chosen.append({"id": len(chosen), "a": a, "b": bb, "title_a": ta, "title_b": tb,
                       "question": f"Which notes are about both {ta} and {tb}?", "gold": gold,
                       "linkers_a": len(linkers[a]), "linkers_b": len(linkers[bb])})
        if len(chosen) == N_PAIRS:
            break
    out = {"preregistration": "preregistration.md#s6", "seed": SEED, "link_types": list(TYPES),
           "max_degree": MAX_DEGREE, "common_range": list(COMMON), "qualifying_pairs": len(pairs),
           "snapshot_pages": len(snap), "n": len(chosen),
           "gold_size": {"median": float(np.median([len(c["gold"]) for c in chosen])),
                         "min": min(len(c["gold"]) for c in chosen), "max": max(len(c["gold"]) for c in chosen)},
           "questions": chosen}
    (HERE / "compositional-questions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(len(pairs), "qualifying pairs;", len(chosen), "drawn; gold", out["gold_size"])


if __name__ == "__main__":
    main()
