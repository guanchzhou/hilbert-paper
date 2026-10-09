#!/usr/bin/env python3
"""S7 items: the 900 packs and 1,059 relevance pairs already judged by the 27B judge, with their
texts and 27B labels, written to private/s7-items.json for s7_judge.py (as pre-registered in
preregistration.md, addendum 2). Also stores, for each S2 question, the two top-10 lists and the
original labels, so s7_report.py can recompute the S2 quantities with another judge's labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from packs import Packs  # noqa: E402

sys.path.insert(0, str(HERE.parent / "factorial"))
from metrics import estimate_tokens  # noqa: E402

PACKS = ("A", "B", "Bcut", "H", "R", "L")
NOTE_TOKENS = 6000


def main() -> None:
    p = Packs()
    p.check()
    b = p.b
    s4 = json.loads((HERE / "private" / "s4-packs.json").read_text())
    fu = json.loads((HERE / "private" / "judge-followup.json").read_text())
    items = []
    for qi in p.sample:
        texts = {"A": p.a(qi)[0], "B": p.b_pack(qi)[0], "Bcut": p.b_pack(qi, p.a(qi)[2])[0]}
        texts.update({c: s4[str(qi)][c]["text"] for c in ("H", "R", "L")})
        for c in PACKS:
            label = p.cached[str(qi)][c] if c in ("A", "B") else fu[f"{qi}-{c}"]["answer"]
            items.append({"task": "sufficiency", "id": f"{qi}-{c}", "qi": qi, "pack": c,
                          "question": b.queries[qi], "text": texts[c], "judge27b": label})

    def note_text(slug: str) -> str:
        i = b.page_index[slug]
        out, used = [], 0
        for row in range(b.starts[i], b.ends[i]):
            t = b.texts[row]
            if used + estimate_tokens(t) > NOTE_TOKENS:
                out.append(t[: max(0, (NOTE_TOKENS - used) * 4)])
                break
            out.append(t)
            used += estimate_tokens(t)
        return "\n\n".join(out)

    s2 = json.loads((HERE / "private" / "s2-judge.json").read_text())
    for key, v in s2.items():
        qi, slug = key.split("|", 1)
        items.append({"task": "relevance", "id": key, "qi": int(qi), "slug": slug,
                      "question": b.queries[int(qi)], "text": note_text(slug), "judge27b": v["answer"]})

    import s2_qrels
    qe = json.loads((HERE / "qrels-extended.json").read_text())
    lists = _s2_lists(p, [q["qi"] for q in qe["per_question"]])
    rng = np.random.default_rng(20261008)
    order = [items[i] for i in rng.permutation(len(items))]
    out = {"items": order, "s2_lists": lists, "s2_reference": s2_qrels.REF.key, "s2_best": s2_qrels.BEST.key}
    (HERE / "private" / "s7-items.json").write_text(json.dumps(out, ensure_ascii=False))
    print("items", len(order), "sufficiency", sum(i["task"] == "sufficiency" for i in order),
          "relevance", sum(i["task"] == "relevance" for i in order), "s2 questions", len(lists))


def _s2_lists(p: Packs, qis: list[int]) -> dict:
    """The top 10 of S2's two configurations, rebuilt as s2_qrels.py builds them."""
    import s2_qrels
    from pipeline import CACHE, DEPTH, rerank_order
    from stage1 import cached_pairs
    b = p.b
    z = np.load(CACHE / "pre-rankings.npz")
    pre = {k: (z["top"][i].astype(np.int64), z["rows"][i].astype(np.int64)) for i, k in enumerate(z["keys"])}
    scores = cached_pairs(p.d)
    path = CACHE / "rerank-pairs.json"
    if path.exists():
        scores.update({k: v["score"] for k, v in json.loads(path.read_text()).items()})

    def top10(cfg, qi: int) -> list[str]:
        t, r = pre[cfg.pre().key]
        t, r = t[qi], r[qi]
        if cfg.f3 == "on":
            n = int((t >= 0).sum())
            sc = [scores[f"{qi}:{b.ids[int(x)]}"] for x in r[:n]]
            o = rerank_order(np.arange(DEPTH), np.array(sc + [0.0] * (DEPTH - n)))
            t = t[np.concatenate([o[:n], np.arange(n, DEPTH)])]
        return [b.pages[int(x)] for x in t[:10] if x >= 0]

    return {str(qi): {"reference": top10(s2_qrels.REF, qi), "best": top10(s2_qrels.BEST, qi),
                      "rels": sorted(b.rels[qi])} for qi in qis}


if __name__ == "__main__":
    main()
