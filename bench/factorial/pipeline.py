"""Level 1 retrieval pipeline: one function per factor, composed per configuration.

Order (pre-registered): F7 filter -> F1 scoring (F5, F6, F2 on the dense part) -> F4 PPR fusion
-> F3 rerank -> F8 packing.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))

from common import CACHE as IDEAS_CACHE, Bench, unit  # noqa: E402
from metrics import estimate_tokens  # noqa: E402
from run_measure import BUDGET, join_chunks, section_span, window_span  # noqa: E402

CACHE = HERE / "cache"
DEPTH = 50
RRF_K = 60
PPR_SEEDS, PPR_ALPHA, PPR_BETA, PPR_ITERS = 20, 0.3, 0.3, 60
ROCCHIO_A, ROCCHIO_K = 0.6, 5
PRUNE_DEPTH, PRUNE_CAP = 30, 5
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

F1 = ("dense", "lexical", "hybrid")
F2 = ("best", "mean")
F3 = ("off", "on")
F4 = ("off", "on")
F5 = ("off", "on")
F6 = ("off", "on")
F7 = ("none", "hk1", "partitions")
F8 = ("chunk", "window", "section", "page", "pruned")
NAMES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")
LEVELS = (F1, F2, F3, F4, F5, F6, F7, F8)


@dataclass(frozen=True)
class Cfg:
    f1: str
    f2: str
    f3: str
    f4: str
    f5: str
    f6: str
    f7: str

    @property
    def key(self) -> str:
        return "|".join((self.f1, self.f2, self.f3, self.f4, self.f5, self.f6, self.f7))

    def pre(self) -> "Cfg":
        """The same pipeline before the rerank stage."""
        return Cfg(self.f1, self.f2, "off", self.f4, self.f5, self.f6, self.f7)


def rankings() -> list[Cfg]:
    out = []
    for f1, f2, f3, f4, f5, f6, f7 in itertools.product(F1, F2, F3, F4, F5, F6, F7):
        if f1 == "lexical" and (f2, f5, f6) != ("best", "off", "off"):
            continue
        out.append(Cfg(f1, f2, f3, f4, f5, f6, f7))
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(text) if s and s.strip()]


class Data:
    def __init__(self) -> None:
        b = self.b = Bench()
        self.N = len(b.pages)
        self.R = len(b.ids)
        self.lex = np.load(CACHE / "lexical.npz")["S"]
        self.masks = {"none": None,
                      "hk1": np.load(CACHE / "hk1-mask.npz")["mask"],
                      "partitions": np.load(CACHE / "partition-mask.npz")["mask"]}
        pm = np.load(CACHE / "partition-mask.npz")
        self.part_cand = pm["cand"]
        mu = b.Cn.mean(axis=0)
        self.space = {"off": (b.Qn, b.Cn), "on": (unit(b.Qn - mu), unit(b.Cn - mu))}
        self.note_vec = {k: unit(np.add.reduceat(C, b.starts, axis=0)) for k, (_, C) in self.space.items()}
        edges = json.loads((IDEAS_CACHE / "edges.json").read_text())
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for x, y in edges:
            if x in b.page_index and y in b.page_index:
                i, j = b.page_index[x], b.page_index[y]
                A[i, j] = A[j, i] = 1.0
        deg = A.sum(axis=1)
        self.P = np.divide(A, deg[:, None], out=np.zeros_like(A), where=deg[:, None] > 0)
        self.dangling = deg == 0
        self.first_row = b.starts.copy()
        self._units()

    # ---- evidence units ------------------------------------------------
    def _units(self) -> None:
        b = self.b
        self.tok_chunk = b.chunk_tokens.astype(np.int64)
        self.tok_window = np.zeros(self.R, dtype=np.int64)
        self.tok_section = np.zeros(self.R, dtype=np.int64)
        for p, (a, e) in enumerate(zip(b.starts, b.ends)):
            chunks = [{"text": b.texts[r], "source": b.sources[r]} for r in range(a, e)]
            for i in range(e - a):
                self.tok_window[a + i] = estimate_tokens(join_chunks(chunks, window_span(chunks, i)))
                self.tok_section[a + i] = estimate_tokens(join_chunks(chunks, section_span(chunks, i)))
        self.tok_page = np.array([estimate_tokens(b.truth.get(s, "")) for s in b.pages], dtype=np.int64)

    # ---- F1 parts --------------------------------------------------------
    def dense_part(self, cfg: Cfg):
        b = self.b
        Q, C = self.space[cfg.f5]
        mask = self.masks[cfg.f7]
        sim = Q @ C.T
        if mask is not None:
            sim = np.where(mask, sim, -np.inf)
        n_vec = np.full(b.n, self.R if mask is None else 0, dtype=np.int64)
        if mask is not None:
            n_vec = mask.sum(1).astype(np.int64)
        if cfg.f7 == "partitions":
            n_vec = self.part_cand.astype(np.int64)
        chunk_count = n_vec.copy()
        if cfg.f6 == "on":
            top = np.argsort(-sim, axis=1)[:, :ROCCHIO_K]
            Q = unit(ROCCHIO_A * Q + (1 - ROCCHIO_A) * C[top].mean(axis=1))
            sim = Q @ C.T
            if mask is not None:
                sim = np.where(mask, sim, -np.inf)
        best = np.maximum.reduceat(sim, b.starts, axis=1)
        win = np.empty(best.shape, dtype=np.int64)
        for p, (a, e) in enumerate(zip(b.starts, b.ends)):
            win[:, p] = a + sim[:, a:e].argmax(axis=1)
        if cfg.f2 == "mean":
            eligible = np.isfinite(best)
            score = np.where(eligible, Q @ self.note_vec[cfg.f5].T, -np.inf)
            note_count = eligible.sum(1)
            n_vec = note_count + (chunk_count if cfg.f6 == "on" else 0)
        else:
            score = best
            if cfg.f6 == "on":
                n_vec = 2 * chunk_count
        win = np.where(np.isfinite(best), win, -1)
        return score.astype(np.float64), win, n_vec

    def lexical_part(self, cfg: Cfg):
        b = self.b
        S = self.lex
        mask = self.masks[cfg.f7]
        if mask is not None:
            S = np.where(mask, S, 0.0)
        n_lex = (S > 0).sum(1).astype(np.int64)
        best = np.maximum.reduceat(S, b.starts, axis=1).astype(np.float64)
        win = np.empty(best.shape, dtype=np.int64)
        for p, (a, e) in enumerate(zip(b.starts, b.ends)):
            win[:, p] = a + S[:, a:e].argmax(axis=1)
        score = np.where(best > 0, best, -np.inf)
        win = np.where(best > 0, win, -1)
        return score, win, n_lex

    def topn(self, score: np.ndarray, n: int) -> np.ndarray:
        """Top-n note indices per question (finite scores only), -1 padded."""
        o = self.b.order(score)[:, :n]
        fin = np.take_along_axis(np.isfinite(score), o, axis=1)
        return np.where(fin, o, -1)

    def first_stage(self, cfg: Cfg):
        zero = np.zeros(self.b.n, dtype=np.int64)
        if cfg.f1 == "dense":
            s, w, n = self.dense_part(cfg)
            return s, w, n, zero
        if cfg.f1 == "lexical":
            s, w, n = self.lexical_part(cfg)
            return s, w, zero, n
        ds, dw, dn = self.dense_part(cfg)
        ls, lw, ln = self.lexical_part(cfg)
        rrf = np.zeros_like(ds)
        for s in (ds, ls):
            top = self.topn(s, DEPTH)
            for r in range(DEPTH):
                col = top[:, r]
                ok = col >= 0
                rrf[np.nonzero(ok)[0], col[ok]] += 1.0 / (RRF_K + r + 1)
        score = np.where(rrf > 0, rrf, -np.inf)
        win = np.where(dw >= 0, dw, lw)
        return score, win, dn, ln

    # ---- F4 ----------------------------------------------------------------
    def ppr(self, score: np.ndarray) -> np.ndarray:
        b = self.b
        seeds = self.topn(score, PPR_SEEDS)
        E = np.zeros_like(score)
        has = np.zeros(b.n, dtype=bool)
        for qi in range(b.n):
            s_ids = seeds[qi][seeds[qi] >= 0]
            if len(s_ids) == 0:
                continue
            w = score[qi, s_ids] - score[qi, s_ids].min() + 1e-6
            E[qi, s_ids] = w / w.sum()
            has[qi] = True
        pi = E.copy()
        for _ in range(PPR_ITERS):
            lost = pi[:, self.dangling].sum(axis=1, keepdims=True)
            pi = PPR_ALPHA * E + (1 - PPR_ALPHA) * (pi @ self.P + lost * E)
        fin = np.isfinite(score)
        lo_f = np.where(fin, score, np.inf).min(axis=1, keepdims=True)
        base = np.where(fin, score, lo_f)
        base_mm = (base - base.min(1, keepdims=True)) / np.maximum(base.max(1, keepdims=True) - base.min(1, keepdims=True), 1e-12)
        ppr_mm = (pi - pi.min(1, keepdims=True)) / np.maximum(pi.max(1, keepdims=True) - pi.min(1, keepdims=True), 1e-12)
        fused = (1 - PPR_BETA) * base_mm + PPR_BETA * ppr_mm
        alive = fin | (pi > 0)
        out = np.where(alive, fused, -np.inf)
        out[~has] = score[~has]
        return out

    # ---- pre-rerank ranking ------------------------------------------------
    def pre_ranking(self, cfg: Cfg) -> dict:
        """Ranking before rerank: top-50 notes, their winning rows, candidates scored."""
        assert cfg.f3 == "off"
        score, win, n_dense, n_lex = self.first_stage(cfg)
        if cfg.f4 == "on":
            score = self.ppr(score)
        top = self.topn(score, DEPTH)
        rows = np.where(top >= 0, np.take_along_axis(win, np.maximum(top, 0), axis=1), -1)
        rows = np.where((top >= 0) & (rows < 0), self.first_row[np.maximum(top, 0)], rows)
        return {"top": top, "rows": rows, "cand": n_dense + n_lex, "cand_dense": n_dense, "cand_lex": n_lex}


def rerank_order(top: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Re-order one question's top-50 by reranker score, ties by previous position."""
    n = int((top >= 0).sum())
    o = np.lexsort((np.arange(n), -scores[:n]))
    return np.concatenate([top[:n][o], top[n:]])
