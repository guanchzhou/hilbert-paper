#!/usr/bin/env python3
"""Cell 9 of addendum 3. For each of the 150 sufficiency questions, the pack is the
winning sentence of each top note, in best-chunk order, with the markdown headings
open above that sentence. No new embedding. Tokens are counted here; the 27B judge
scores sufficiency afterwards.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "factorial"))
from metrics import estimate_tokens  # noqa: E402
from packs import Packs, pack  # noqa: E402
import stage2  # noqa: E402
from pipeline import sentences  # noqa: E402

HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+\S")
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
BAR = 0.02


def open_headings(page: str, at: int) -> list[str]:
    stack, fence = [], False
    for line in page[:at].splitlines():
        if FENCE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        m = HEADING.match(line)
        if not m:
            continue
        level = len(m.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, line.strip()))
    return [line for _, line in stack]


def unit_text(page: str, chunk: str, sentence: str) -> str:
    base = page.find(chunk)
    if base >= 0 and chunk.find(sentence) >= 0:
        at = base + chunk.find(sentence)
    else:
        at = page.find(sentence)
    heads = open_headings(page, at) if at >= 0 else []
    return "\n".join(heads + [sentence])


def main() -> None:
    packs = Packs()
    sent = stage2.Sentences(packs.d)
    rows, missing = [], 0
    texts = {}
    for qi in packs.sample:
        t, r = packs.pre_b["top"][qi], packs.pre_b["rows"][qi]
        ok = t >= 0
        units = []
        for note, row in zip(t[:10][ok[:10]], r[:10][ok[:10]]):
            row = int(row)
            slug, chunk = packs.b.pages[int(note)], packs.b.texts[row]
            if row in sent.vec:
                sims = sent.vec[row] @ packs.b.Qn[qi]
                sentence = sent.text[row][int(np.argmax(sims))]
            else:
                missing += 1
                parts = sentences(chunk)
                sentence = parts[0] if parts else chunk
            page = packs.b.truth.get(slug, "")
            units.append((slug, unit_text(page, chunk, sentence)))
        text, survival, tokens = pack(units, packs.b.rels[qi])
        full_tokens = packs.b_pack(qi)[2]
        rows.append({"qi": qi, "tokens": tokens, "full_tokens": full_tokens, "survival": survival})
        texts[str(qi)] = text
    (HERE / "private" / "s8-cell9-packs.json").write_text(json.dumps({"questions": texts}))
    tok = float(np.mean([r["tokens"] for r in rows]))
    full = float(np.mean([r["full_tokens"] for r in rows]))
    out = {
        "cell": 9, "name": "winning sentence with headings", "arxiv": "2606.27594", "n": len(rows),
        "missing_sentence_vectors": missing,
        "tokens": {"sentence_headings": tok, "full_chunk_pack": full},
        "fewer_tokens": bool(tok < full), "sufficiency": None, "sufficiency_bar": BAR,
        "per_question": rows,
    }
    (HERE / "cell9.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n", "missing_sentence_vectors", "tokens", "fewer_tokens")}, indent=1))


if __name__ == "__main__":
    main()
