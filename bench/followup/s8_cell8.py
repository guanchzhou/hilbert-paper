#!/usr/bin/env python3
"""Cell 8 checker. Reads the prefixes already built and asks JEV-9B, in chunk order,
whether the context can answer. Stops at the first prefix with p(yes) >= 0.5.
Does not load the corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s7_judge import jev  # noqa: E402
from huggingface_hub import snapshot_download

BAR = 0.8
PRIVATE = HERE / "private"


def main() -> None:
    specs = json.loads((PRIVATE / "s8-cell8-prefixes.json").read_text())
    path = Path(snapshot_download(
        "autotrust/JEV-9B", revision="b63f651ce8ed64481d3f5e73ecdb05f740042f01",
        local_files_only=True, ignore_patterns=["adapter_vllm/*", "reports/*", "vl/*", "code/*"]))
    p_yes = jev(path)
    cache_path = PRIVATE / "s8-cell8-jev.json"
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = []
    for n, spec in enumerate(specs, 1):
        key = str(spec["qi"])
        if key in done:
            rows.append({k: done[key][k] for k in ("qi", "full_tokens", "stopped_tokens", "stopped_chunks", "p_yes", "never_yes", "calls")})
            continue
        stopped = None
        calls = 0
        r = {"p_yes": 0.0}
        if not spec["prefixes"]:
            done[key] = {"qi": spec["qi"], "full_tokens": spec["full_tokens"], "stopped_tokens": 0,
                         "stopped_chunks": 0, "p_yes": 0.0, "never_yes": True, "calls": 0, "text": ""}
            rows.append({k: done[key][k] for k in ("qi", "full_tokens", "stopped_tokens", "stopped_chunks", "p_yes", "never_yes", "calls")})
            continue
        for prefix in spec["prefixes"]:
            r = p_yes({"task": "sufficiency", "question": spec["question"], "text": prefix["text"]})
            torch.mps.empty_cache()
            calls += 1
            print("q", n, "chunks", prefix["chunks"], "p", round(r["p_yes"], 3), flush=True)
            if r["p_yes"] >= 0.5:
                stopped = {**prefix, "p_yes": r["p_yes"], "never_yes": False}
                break
        if stopped is None:
            last = spec["prefixes"][-1]
            stopped = {**last, "p_yes": r["p_yes"], "never_yes": True}
        row = {"qi": spec["qi"], "full_tokens": spec["full_tokens"], "stopped_tokens": stopped["tokens"],
               "stopped_chunks": stopped["chunks"], "p_yes": stopped["p_yes"], "never_yes": stopped["never_yes"],
               "calls": calls}
        done[key] = {**row, "text": stopped["text"]}
        cache_path.write_text(json.dumps(done))
        rows.append(row)
    full = sum(r["full_tokens"] for r in rows)
    stopped_tok = sum(r["stopped_tokens"] for r in rows)
    out = {
        "cell": 8, "name": "stop early", "arxiv": "2506.05167", "checker": "JEV-9B", "n": len(rows),
        "tokens": {"stopped": stopped_tok / len(rows), "full_chunk_pack": full / len(rows)},
        "token_ratio": stopped_tok / full, "token_bar": BAR, "fewer_tokens": bool(stopped_tok / full <= BAR),
        "never_yes": sum(1 for r in rows if r["never_yes"]), "sufficiency": None, "per_question": rows,
    }
    (HERE / "cell8.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n", "tokens", "token_ratio", "fewer_tokens", "never_yes")}, indent=1))


if __name__ == "__main__":
    main()
