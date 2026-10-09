"""Packs of the answer-quality check, rebuilt exactly as factorial/judge_packs.py builds them.

A: the best-survival ranking with pruned sentences of the top 30 notes. B: the reference ranking
with the best chunk of each of the top 10 notes. Both within 6,000 tokens by the same greedy rule.
Also B-cut (S1a, B within A's token count) and H (S4, chunks for the top 10 notes, pruned
sentences for ranks 11 to 30, best-survival ranking).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / "factorial"
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(FACTORIAL))

from pipeline import CACHE, F8, PRUNE_CAP, PRUNE_DEPTH, Data  # noqa: E402
import stage2  # noqa: E402
from judge_packs import A, B, BUDGET  # noqa: E402
from metrics import estimate_tokens  # noqa: E402


def pack(units: list[tuple[str, str]], rel: set, budget: int = BUDGET) -> tuple[str, float, int]:
    used, parts, ok = 0, [], False
    for slug, text in units:
        cost = estimate_tokens(text)
        if used + cost <= budget:
            used += cost
            parts.append(text)
            ok = ok or slug in rel
    return "\n\n---\n\n".join(parts), (1.0 if ok else 0.0), used


class Packs:
    def __init__(self) -> None:
        self.d = Data()
        self.b = self.d.b
        self.sent = stage2.Sentences(self.d)
        jp = json.loads((FACTORIAL / "judge-packs.json").read_text())
        self.sample = list(jp["sample_idx"])
        self.cached = json.loads((FACTORIAL / "private" / "judge-packs.json").read_text())
        self.pre_a = self.d.pre_ranking(A[0].pre())
        self.pre_b = self.d.pre_ranking(B[0].pre())
        z = np.load(CACHE / "level1-perq.npz")
        self._keys = [str(k) for k in z["keys"]]
        self._surv = z["surv"]

    def _pruned(self, qi: int, note: int, row: int) -> str:
        sims = self.sent.vec[row] @ self.b.Qn[qi]
        keep = sorted(np.argsort(-sims)[:PRUNE_CAP])
        return "\n".join(self.sent.text[row][j] for j in keep)

    def units_a(self, qi: int) -> list[tuple[str, str]]:
        t, r = self.pre_a["top"][qi], self.pre_a["rows"][qi]
        ok = t >= 0
        return [(self.b.pages[int(n)], self._pruned(qi, int(n), int(w)))
                for n, w in zip(t[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]], r[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]])]

    def units_b(self, qi: int) -> list[tuple[str, str]]:
        t, r = self.pre_b["top"][qi], self.pre_b["rows"][qi]
        ok = t >= 0
        return [(self.b.pages[int(n)], self.b.texts[int(w)]) for n, w in zip(t[:10][ok[:10]], r[:10][ok[:10]])]

    def units_h(self, qi: int) -> list[tuple[str, str]]:
        t, r = self.pre_a["top"][qi], self.pre_a["rows"][qi]
        ok = t >= 0
        out = []
        for rank, (n, w) in enumerate(zip(t[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]], r[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]])):
            text = self.b.texts[int(w)] if rank < 10 else self._pruned(qi, int(n), int(w))
            out.append((self.b.pages[int(n)], text))
        return out

    def a(self, qi: int) -> tuple[str, float, int]:
        return pack(self.units_a(qi), self.b.rels[qi])

    def b_pack(self, qi: int, budget: int = BUDGET) -> tuple[str, float, int]:
        return pack(self.units_b(qi), self.b.rels[qi], budget)

    def check(self) -> None:
        """Survival of A and B on the 150 questions must equal stage2.py's stored values."""
        for (cfg, unit), fn in ((A, self.a), (B, self.b_pack)):
            stored = self._surv[self._keys.index(cfg.key), list(F8).index(unit)]
            for qi in self.sample:
                assert fn(qi)[1] == stored[qi], (cfg.key, qi)

    def judged(self, qi: int, cond: str) -> float:
        return 1.0 if self.cached[str(qi)][cond] == "yes" else 0.0
