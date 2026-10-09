"""G23: do the federated Zig sources come back for Zig questions?

Runs the queries of queries.json, fixed before the sources were embedded, through gbrain's
`query` operation (the default federated search, top k) and records the source of each hit.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
GBRAIN = [os.path.expanduser("~/.bun/bin/gbrain"), "call", "query"]


def run(query: str, k: int) -> list[dict]:
    args = {"query": query, "limit": k, "autocut": False, "snippet_chars": 0, "return_unit": "chunk"}
    out = subprocess.run(GBRAIN + [json.dumps(args)], capture_output=True, text=True, check=True).stdout
    return json.loads(out[out.index("["):])


def main() -> None:
    spec = json.loads((HERE / "queries.json").read_text())
    k = spec["k"]
    rows = []
    for q in spec["queries"]:
        hits = run(q["query"], k)[:k]
        ranks = [i + 1 for i, h in enumerate(hits) if h["source_id"] == q["source"]]
        rows.append({
            **q,
            "passed": bool(ranks),
            "first_rank": ranks[0] if ranks else None,
            "hits_from_source": len(ranks),
            "top": [{"source": h["source_id"], "slug": h["slug"]} for h in hits],
        })
        print(f"{q['source']:14} {'pass' if ranks else 'FAIL'}  first rank {ranks[0] if ranks else '-'}  "
              f"{len(ranks)}/{k} from the source")
    result = {
        "registered_at": spec["registered_at"],
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": spec["rule"],
        "k": k,
        "passed": sum(r["passed"] for r in rows),
        "n": len(rows),
        "queries": rows,
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{result['passed']}/{result['n']} passed")


if __name__ == "__main__":
    main()
