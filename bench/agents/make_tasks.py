#!/usr/bin/env python3
"""Pick pilot tasks from the development set and export a read-only copy of the corpus."""

import json
import os
import random
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_measure import pg_env, psql  # noqa: E402

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
N_TASKS = 10
SEED = 2026


def main() -> None:
    env, password = pg_env()
    import csv
    import io
    out = psql("COPY (SELECT slug, compiled_truth FROM pages WHERE deleted_at IS NULL ORDER BY slug) TO STDOUT WITH (FORMAT csv)",
               env, password)
    pages = {row[0]: row[1] for row in csv.reader(io.StringIO(out))}
    if KB.exists():
        for root, dirs, files in os.walk(KB):
            os.chmod(root, 0o755)
            for f in files:
                os.chmod(Path(root) / f, 0o644)
    for slug, text in pages.items():
        path = KB / f"{slug}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for root, dirs, files in os.walk(KB):
        for f in files:
            os.chmod(Path(root) / f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    qrels = json.loads((Path.home() / ".gbrain/eval/qrels-dev.json").read_text())
    pool = [q for q in qrels if len(q["relevant"]) == 1 and q["relevant"][0] in pages
            and 400 <= len(pages[q["relevant"][0]]) <= 12000]
    rng = random.Random(SEED)
    picked = rng.sample(pool, N_TASKS)
    tasks = [{"id": f"t{i:02d}", "query": q["query"], "relevant": q["relevant"][0],
              "reference": pages[q["relevant"][0]]} for i, q in enumerate(picked)]
    (HERE / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    print("pages", len(pages), "pool", len(pool), "tasks", len(tasks))


if __name__ == "__main__":
    main()
