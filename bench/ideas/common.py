"""Shared helpers for the pre-registered retrieval ideas. Development set only.

The corpus is read read-only from gbrain and cached under cache/ (private, never committed).
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

IDEAS = Path(__file__).resolve().parent
BENCH = IDEAS.parent
sys.path.insert(0, str(BENCH))

from metrics import estimate_tokens, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from run_measure import BUDGET, QRELS, load_corpus, pg_env  # noqa: E402

REPO = Path.home() / "Development/hilbert-paper/bench/ideas"
CACHE = IDEAS / "cache"
PRIVATE = IDEAS / "private"
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
EMBED_URL = "http://127.0.0.1:11436/v1/embeddings"
RERANK_URL = "http://127.0.0.1:11437/v1/rerank"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
SEED = 20261005
BOOT = 10000
K = 10


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


class Bench:
    def __init__(self) -> None:
        self.qrels = json.loads(QRELS.read_text())
        assert len(self.qrels) == 817
        self.queries = [q["query"] for q in self.qrels]
        self.rels = [set(q["relevant"]) for q in self.qrels]
        self.n = len(self.qrels)
        self.Q = np.load(BENCH / "query-vectors.npy")
        self.Qn = unit(self.Q)
        CACHE.mkdir(exist_ok=True)
        path = CACHE / "corpus.pkl"
        if path.exists():
            c = pickle.loads(path.read_bytes())
        else:
            env, password = pg_env()
            truth, ids, slugs, sources, texts, indexes, matrix = load_corpus(env, password)
            c = dict(truth=truth, ids=ids, slugs=slugs, sources=sources, texts=texts, indexes=indexes, matrix=matrix)
            path.write_bytes(pickle.dumps(c))
        self.truth = c["truth"]
        self.ids = c["ids"]
        self.slugs = c["slugs"]
        self.sources = c["sources"]
        self.texts = c["texts"]
        self.indexes = c["indexes"]
        self.M = c["matrix"]
        self.Cn = unit(self.M)
        starts, pages = [], []
        for i, s in enumerate(self.slugs):
            if i == 0 or s != self.slugs[i - 1]:
                starts.append(i)
                pages.append(s)
        self.starts = np.array(starts)
        self.ends = np.append(self.starts[1:], len(self.slugs))
        self.pages = pages
        self.page_of_row = np.repeat(np.arange(len(pages)), self.ends - self.starts)
        self.page_index = {s: i for i, s in enumerate(pages)}
        self.slug_rank = np.argsort(np.argsort(np.array(pages, dtype=object)))  # alphabetical tie-break
        self.chunk_tokens = np.array([estimate_tokens(t) for t in self.texts])
        split = json.loads((IDEAS / "split.json").read_text())
        self.tune = np.array(split["tune"])
        self.confirm = np.array(split["confirm"])

    # ---- ranking -------------------------------------------------------
    def chunk_sim(self, Qn: np.ndarray, Cn: np.ndarray | None = None) -> np.ndarray:
        return Qn @ (self.Cn if Cn is None else Cn).T

    def note_scores(self, sim: np.ndarray, mask: np.ndarray | None = None):
        """Best chunk wins: (scores, winning row) per note. Masked rows score -inf."""
        if mask is not None:
            sim = np.where(mask, sim, -np.inf)
        best = np.maximum.reduceat(sim, self.starts, axis=1)
        win = np.empty(best.shape, dtype=np.int64)
        for p, (a, b) in enumerate(zip(self.starts, self.ends)):
            win[:, p] = a + sim[:, a:b].argmax(axis=1)
        return best, win

    def order(self, scores: np.ndarray) -> np.ndarray:
        """Note indices by descending score, slug ascending on ties (as run_measure.topk_from_scores)."""
        return np.lexsort((np.broadcast_to(self.slug_rank, scores.shape), -scores), axis=-1)

    def lists(self, scores: np.ndarray, k: int = K, finite_only: bool = False) -> list[list[str]]:
        o = self.order(scores)[:, :k]
        out = []
        for qi in range(len(o)):
            row = o[qi]
            if finite_only:
                row = row[np.isfinite(scores[qi, row])]
            out.append([self.pages[j] for j in row])
        return out

    def dense(self, Qn: np.ndarray | None = None, Cn: np.ndarray | None = None):
        sim = self.chunk_sim(self.Qn if Qn is None else Qn, Cn)
        return self.note_scores(sim)

    # ---- metrics -------------------------------------------------------
    def per_question(self, lists: list[list[str]], idx=None) -> dict[str, np.ndarray]:
        idx = range(self.n) if idx is None else idx
        r, m, g, h = [], [], [], []
        for qi, hits in zip(idx, lists):
            rel = self.rels[qi]
            top = hits[:K]
            r.append(recall_at_k(top, rel, K))
            m.append(mrr(top, rel))
            g.append(ndcg_at_k(top, rel, K))
            h.append(1.0 if any(s in rel for s in top) else 0.0)
        return {"R@10": np.array(r), "MRR": np.array(m), "nDCG@10": np.array(g), "hit@10": np.array(h)}

    def first_rank(self, scores: np.ndarray) -> np.ndarray:
        """1-based rank of the best-ranked relevant live note per question (0 if none live)."""
        o = self.order(scores)
        pos = np.empty_like(o)
        np.put_along_axis(pos, o, np.arange(o.shape[1])[None, :].repeat(len(o), 0), axis=1)
        out = np.zeros(len(o), dtype=np.int64)
        for qi in range(len(o)):
            r = [pos[qi, self.page_index[s]] + 1 for s in self.rels[qi] if s in self.page_index]
            out[qi] = min(r) if r else 0
        return out

    def pack(self, qi: int, units: list[tuple[str, int]]) -> tuple[float, int]:
        """units: (slug, tokens) in rank order. Returns (survival, delivered tokens)."""
        used, ok = 0, False
        for slug, cost in units:
            if used + cost <= BUDGET:
                used += cost
                ok = ok or slug in self.rels[qi]
        return (1.0 if ok else 0.0), used

    def chunk_pack(self, scores: np.ndarray, win: np.ndarray, depth: int = K):
        o = self.order(scores)[:, :depth]
        surv, tok = np.zeros(self.n), np.zeros(self.n)
        for qi in range(self.n):
            units = [(self.pages[p], int(self.chunk_tokens[win[qi, p]])) for p in o[qi]]
            surv[qi], tok[qi] = self.pack(qi, units)
        return surv, tok


# ---- statistics --------------------------------------------------------
def boot_ci(d: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    b = d[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "greater", margin: float = 0.0) -> dict:
    d = a - b
    lo, hi = boot_ci(d)
    shifted = d + margin
    nz = shifted[np.abs(shifted) > 1e-12]
    if len(nz):
        p = float(stats.wilcoxon(nz, alternative=alternative, zero_method="wilcox").pvalue)
    else:
        p = 1.0
    test = "one-sided Wilcoxon signed-rank on non-zero paired differences"
    out = {}
    if margin:
        test = f"non-inferiority: one-sided Wilcoxon signed-rank on (new - comparator + {margin})"
        p_t = float(stats.ttest_1samp(shifted, 0.0, alternative="greater").pvalue) if shifted.std() > 0 else 1.0
        out = {"p_wilcoxon_preregistered": p, "p_t_shifted": p_t,
               "p_note": "deviation D1: p = max(Wilcoxon, paired t on shifted differences)"}
        p = max(p, p_t)
    return {"mean_new": float(a.mean()), "mean_comparator": float(b.mean()), "diff": float(d.mean()),
            "ci95": [lo, hi], "p": p, "test": test, "alternative": alternative, "n": int(len(d)),
            "n_nonzero": int(len(nz)), "improved": int((d > 0).sum()), "worsened": int((d < 0).sum()), **out}


def mcnemar(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> dict:
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    p = float(stats.binomtest(a_only, n, 0.5, alternative=alternative).pvalue) if n else 1.0
    lo, hi = boot_ci(a - b)
    return {"mean_new": float(a.mean()), "mean_comparator": float(b.mean()), "diff": float((a - b).mean()),
            "ci95": [lo, hi], "p": p, "test": "exact McNemar (binomial on discordant pairs), one-sided",
            "new_only": a_only, "comparator_only": b_only, "n": int(len(a))}


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def halves(b: Bench, arrs: dict[str, np.ndarray]) -> dict:
    return {"all": {k: float(v.mean()) for k, v in arrs.items()},
            "tune": {k: float(v[b.tune].mean()) for k, v in arrs.items()},
            "confirm": {k: float(v[b.confirm].mean()) for k, v in arrs.items()}}


def rounded(x):
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    if isinstance(x, np.ndarray):
        return rounded(x.tolist())
    if isinstance(x, (np.floating,)):
        return round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def write_result(name: str, obj: dict, script: Path | None = None) -> Path:
    path = IDEAS / f"{name}.json"
    path.write_text(json.dumps(rounded(obj), indent=1, ensure_ascii=False) + "\n")
    REPO.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, REPO / path.name)
    if script is not None:
        shutil.copy(script, REPO / script.name)
    shutil.copy(Path(__file__), REPO / "common.py")
    return path


# ---- external services ---------------------------------------------------
def post(url: str, payload: dict, timeout: int = 600) -> dict:
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed: {last}")


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    for s in range(0, len(texts), batch):
        data = post(EMBED_URL, {"model": "qwen3-embedding-8k", "input": texts[s:s + batch]})
        for item in data["data"]:
            out[s + int(item["index"])] = item["embedding"]
    return out


def hilbert_rows(vectors: np.ndarray, extra: list[str]) -> list[dict]:
    payload = "".join(json.dumps({"id": i, "embedding": [round(float(x), 7) for x in v]}) + "\n"
                      for i, v in enumerate(vectors))
    proc = subprocess.run([str(HILBERT), "key", *extra, "--format", "jsonl"], input=payload,
                          capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines()]


def key_int(marker: str) -> int:
    return int(marker.rsplit(":", 1)[1], 16)
