#!/usr/bin/env python3
"""27B sufficiency on the packs of cells 8 and 9. The comparison rate is the chunk
pack's yes-rate from the factorial sufficiency check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
from idea02_prune import judge  # noqa: E402

PRIVATE = HERE / "private"
BASE = json.loads((HERE.parent / "factorial" / "judge-packs.json").read_text())["yes"]["B"]
MARGIN = 0.02


def score(items: list[dict], cache_path: Path, limit: list[int]) -> None:
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for n, item in enumerate(items, 1):
        key = str(item["qi"])
        if key in done:
            continue
        if limit[0] <= 0:
            return
        ans, secs = judge(item["question"], item["text"])
        done[key] = {"answer": ans, "seconds": secs}
        cache_path.write_text(json.dumps(done))
        limit[0] -= 1
        print(n, key, ans, round(secs, 1), flush=True)


def finish(path: Path, cache_path: Path, n_expected: int) -> None:
    if not cache_path.exists():
        return
    done = json.loads(cache_path.read_text())
    if len(done) < n_expected:
        return
    out = json.loads(path.read_text())
    answers = list(done.values())
    yes = sum(a["answer"] == "yes" for a in answers) / len(answers)
    out["sufficiency"] = {"judge": "qwen3.8:latest", "yes": yes, "chunk_pack": BASE, "diff": yes - BASE}
    out["within_0.02"] = abs(yes - BASE) <= MARGIN
    out["passes"] = bool(out["fewer_tokens"] and out["within_0.02"])
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(path.name, json.dumps(out["sufficiency"]), "passes", out["passes"])


def main() -> None:
    limit = [int(sys.argv[1])] if len(sys.argv) > 1 else [10**9]
    questions = {str(r["qi"]): r["question"] for r in json.loads((PRIVATE / "s8-cell8-prefixes.json").read_text())}
    stopped = json.loads((PRIVATE / "s8-cell8-jev.json").read_text())
    cell9 = json.loads((PRIVATE / "s8-cell9-packs.json").read_text())["questions"]
    score([{"qi": qi, "question": questions[qi], "text": row["text"]} for qi, row in stopped.items()],
          PRIVATE / "s8-cell8-yes.json", limit)
    score([{"qi": qi, "question": questions[qi], "text": text} for qi, text in cell9.items()],
          PRIVATE / "s8-cell9-yes.json", limit)
    finish(HERE / "cell8.json", PRIVATE / "s8-cell8-yes.json", len(stopped))
    finish(HERE / "cell9.json", PRIVATE / "s8-cell9-yes.json", len(cell9))


if __name__ == "__main__":
    main()
