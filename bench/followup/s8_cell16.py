#!/usr/bin/env python3
"""Cell 16 of addendum 3. On the 60 two-page questions, the set is the union of the
top 10 notes for each facet title, by best-chunk cosine. The two target pages are
removed. The link join is not used. Set recall is against the link-derived gold.
The cell passes when mean set recall is above 0.479.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from common import Bench, embed, unit  # noqa: E402
from idea03_instruction import INSTRUCTIONS, prefix  # noqa: E402

PER_FACET = 10
BAR = 0.479


def top_notes(b: Bench, scores: np.ndarray, k: int, drop: set[str]) -> list[str]:
    out = []
    for p in b.order(scores[None, :])[0]:
        slug = b.pages[int(p)]
        if slug not in drop:
            out.append(slug)
        if len(out) == k:
            break
    return out


def main() -> None:
    b = Bench()
    spec = json.loads((HERE / "compositional-questions.json").read_text())
    qs = spec["questions"]
    assert len(qs) == 60
    published = json.loads((HERE / "compositional.json").read_text())
    baseline = float(published["summary"]["dense_facets"]["recall"])
    texts = [q["title_a"] for q in qs] + [q["title_b"] for q in qs]
    V = unit(embed([prefix(INSTRUCTIONS["default"]) + t for t in texts]))
    scores = b.note_scores(b.chunk_sim(V))[0]
    recalls = []
    rows = []
    for i, q in enumerate(qs):
        drop = {q["a"], q["b"]}
        gold = set(q["gold"])
        cover = set(top_notes(b, scores[i], PER_FACET, drop)) | set(top_notes(b, scores[i + 60], PER_FACET, drop))
        recall = len(cover & gold) / len(gold)
        recalls.append(recall)
        rows.append({"id": q["id"], "recall": recall, "size": len(cover), "gold": len(gold)})
    rec = np.array(recalls)
    out = {
        "cell": 16,
        "name": "cover every part",
        "arxiv": "2507.06838",
        "n": 60,
        "per_facet": PER_FACET,
        "set": "union of the top 10 notes for each facet title",
        "link_join_used": False,
        "set_recall": float(rec.mean()),
        "dense_facets_recall": baseline,
        "bar": BAR,
        "passes": bool(rec.mean() > BAR),
        "per_question": rows,
    }
    (HERE / "cell16.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("set_recall", "dense_facets_recall", "bar", "passes")}, indent=1))


if __name__ == "__main__":
    main()
