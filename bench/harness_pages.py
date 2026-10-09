#!/usr/bin/env python3
"""Thirty page tasks: ripgrep in the vault versus gbrain search."""

import json
import re
import subprocess
import time
from pathlib import Path

VAULT = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/guanchzhou"
DEV = Path.home() / ".gbrain/eval/qrels-dev.json"
OUT = Path.home() / ".gbrain/eval/bench/harness-pages.json"


def main() -> None:
    qrels = json.loads(DEV.read_text())
    picked = [qrels[i] for i in range(0, len(qrels), len(qrels) // 30)][:30]
    rows = []
    for q in picked:
        query = q["query"]
        rel = q["relevant"]
        t0 = time.perf_counter()
        rg = subprocess.run(
            ["rg", "-l", "-i", "-F", query, str(VAULT), "-g", "*.md"],
            capture_output=True, text=True,
        )
        vault_s = time.perf_counter() - t0
        files = [line for line in rg.stdout.splitlines() if line]
        vault_hit = any(any(slug.split("/")[-1] in path for slug in rel) for path in files)
        t0 = time.perf_counter()
        gs = subprocess.run(
            ["gbrain", "search", query, "--limit", "10", "--snippet-chars", "20"],
            capture_output=True, text=True,
        )
        gbrain_s = time.perf_counter() - t0
        slugs = re.findall(r"\] (\S+) --", gs.stdout)
        gbrain_hit = any(s in slugs[:10] or s.removeprefix("obsidian/") in slugs[:10] for s in rel)
        rows.append({
            "query": query,
            "vault_files": len(files),
            "vault_hit": vault_hit,
            "vault_seconds": round(vault_s, 3),
            "gbrain_hit": gbrain_hit,
            "gbrain_seconds": round(gbrain_s, 3),
        })
        print(len(rows), "vault", vault_hit, round(vault_s, 2), "gbrain", gbrain_hit, round(gbrain_s, 2), flush=True)
    summary = {
        "n": len(rows),
        "vault_hit_rate": round(sum(r["vault_hit"] for r in rows) / len(rows), 4),
        "gbrain_hit_rate": round(sum(r["gbrain_hit"] for r in rows) / len(rows), 4),
        "vault_seconds_mean": round(sum(r["vault_seconds"] for r in rows) / len(rows), 3),
        "gbrain_seconds_mean": round(sum(r["gbrain_seconds"] for r in rows) / len(rows), 3),
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
