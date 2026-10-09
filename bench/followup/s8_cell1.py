#!/usr/bin/env python3
"""Cell 1 of addendum 3: a note's score is the maximum cosine of its stored sentence
vectors with the question. Development questions only. The comparison is recall at 10
against best-chunk ranking. The cell passes when the 95% interval for that paired
difference sits above zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "factorial"))
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))
from pipeline import Data, sentences  # noqa: E402
import stage2  # noqa: E402
from common import boot_ci  # noqa: E402

SEED = 20261008


def attach_stored(b, sent) -> None:
    """The factorial cache omits 89 chunks. Their sentence vectors were stored for S6c."""
    miss = [r for r in range(len(b.texts)) if r not in sent.vec]
    path = HERE / "private" / "s6c-missing.npz"
    if not path.exists():
        raise SystemExit(f"sentence vectors missing for {len(miss)} chunks and {path.name} is absent")
    Vm = np.load(path)["V"]
    pos = 0
    for r in miss:
        n = len(sentences(b.texts[r]))
        v = Vm[pos:pos + n]
        sent.vec[r] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        pos += n
    if pos != len(Vm):
        raise SystemExit(f"stored extra vectors do not match the missing chunks ({pos} vs {len(Vm)})")


def sentence_scores(b, sent) -> np.ndarray:
    attach_stored(b, sent)
    rows = sorted(sent.vec)
    missing = len(b.texts) - len(rows)
    if missing:
        raise SystemExit(f"sentence vectors missing for {missing} chunks")
    S = np.concatenate([sent.vec[r] for r in rows]).astype(np.float32)
    note = np.concatenate([np.full(len(sent.vec[r]), b.page_of_row[r], dtype=np.int32) for r in rows])
    order = np.argsort(note, kind="stable")
    S, note = S[order], note[order]
    starts = np.searchsorted(note, np.arange(len(b.pages)))
    scores = np.empty((b.n, len(b.pages)), np.float32)
    for s0 in range(0, b.n, 64):
        sim = b.Qn[s0:s0 + 64] @ S.T
        scores[s0:s0 + 64] = np.maximum.reduceat(sim, starts, axis=1)
        print("scored", min(s0 + 64, b.n), flush=True)
    return scores


def main() -> None:
    d = Data()
    b = d.b
    assert b.n == 817
    base_scores, _ = b.dense()
    base = b.per_question(b.lists(base_scores))["R@10"]
    sent = sentence_scores(b, stage2.Sentences(d))
    new = b.per_question(b.lists(sent))["R@10"]
    diff = new - base
    lo, hi = boot_ci(diff, seed=SEED)
    out = {
        "cell": 1,
        "name": "best sentence",
        "arxiv": "2312.06648",
        "n": int(b.n),
        "baseline": "best-chunk cosine",
        "R@10": {"best_sentence": float(new.mean()), "best_chunk": float(base.mean())},
        "diff": float(diff.mean()),
        "ci95": [lo, hi],
        "interval_above_zero": bool(lo > 0),
        "improved": int((diff > 0).sum()),
        "worsened": int((diff < 0).sum()),
        "tied": int((diff == 0).sum()),
        "seed": SEED,
    }
    path = HERE / "cell1.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
