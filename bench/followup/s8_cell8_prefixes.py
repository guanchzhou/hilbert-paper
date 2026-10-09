#!/usr/bin/env python3
"""Build the chunk prefixes for cell 8. A separate process so the checker does not
share memory with the corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from metrics import estimate_tokens  # noqa: E402
from packs import Packs  # noqa: E402

OUT = HERE / "private" / "s8-cell8-prefixes.json"


def main() -> None:
    packs = Packs()
    rows = []
    for qi in packs.sample:
        parts, used, prefixes = [], 0, []
        for _, text in packs.units_b(qi):
            cost = estimate_tokens(text)
            if used + cost > 6000:
                break
            used += cost
            parts.append(text)
            prefixes.append({"chunks": len(parts), "tokens": used, "text": "\n\n---\n\n".join(parts)})
        _, _, full_tokens = packs.b_pack(qi)
        rows.append({"qi": qi, "question": packs.b.queries[qi], "full_tokens": full_tokens, "prefixes": prefixes})
    OUT.write_text(json.dumps(rows))
    print("questions", len(rows), "prefixes", sum(len(r["prefixes"]) for r in rows))


if __name__ == "__main__":
    main()
