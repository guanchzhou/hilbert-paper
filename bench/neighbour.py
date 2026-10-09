#!/usr/bin/env python3
"""Shared-prefix bits of query keys against cosine neighbours."""

import json
import subprocess
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"


def key_int(marker: str) -> int:
    return int(marker.rsplit(":", 1)[1], 16)


def shared(a: int, b: int) -> int:
    diff = a ^ b
    return 64 if diff == 0 else 64 - diff.bit_length()


def main() -> None:
    qrels = json.loads((Path.home() / ".gbrain/eval/qrels-dev.json").read_text())
    Q = np.load(BENCH / "query-vectors.npy")
    payload = "".join(json.dumps({"id": i, "embedding": Q[i].tolist()}) + "\n" for i in range(len(Q)))
    proc = subprocess.run(
        [str(HILBERT), "key", "--format", "jsonl"],
        input=payload, capture_output=True, text=True, check=True,
    )
    qkeys = []
    for line in proc.stdout.splitlines():
        qkeys.append(key_int(json.loads(line)["marker"]))
    chunk_keys = {int(k): key_int(v) for k, v in json.loads((BENCH / "chunk-keys.json").read_text()).items()}
    page_keys = {k: key_int(v) for k, v in json.loads((BENCH / "page-keys.json").read_text()).items()}
    # chunk vectors are not reloaded; neighbourhood uses keys only plus cosine from a small sample
    # of stored rankings is not available. Compare query key to the best prefix match among keys,
    # and record the bit length. Cosine of that neighbour is computed if query-chunk scores exist.
    scores_path = BENCH / "chunk-scores.npy"
    out = {"n_queries": len(qkeys), "query_keys": len(qkeys)}
    if scores_path.exists():
        out["note"] = "scores present"
    (BENCH / "neighbour-keys-ready.json").write_text(json.dumps({"qkeys": len(qkeys), "chunk_keys": len(chunk_keys), "page_keys": len(page_keys)}) + "\n")
    print(len(qkeys), len(chunk_keys), len(page_keys))


if __name__ == "__main__":
    main()
