#!/usr/bin/env python3
"""Per-token efficiency study, M1-M7 and M9, as pre-registered in preregistration.md, with the
clarifications in deviations.md. Development questions only; gbrain Postgres is only read.

Writes m1.json ... m9.json and summary.json next to this file and copies them, with this script,
to ~/Development/hilbert-paper/bench/efficiency/.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "ideas"))
sys.path.insert(0, str(BENCH))

from common import boot_ci, holm, mcnemar, rounded, unit, wilcoxon  # noqa: E402
from metrics import estimate_tokens, recall_at_k  # noqa: E402
from run_measure import QRELS, load_corpus, pg_env, psql  # noqa: E402

REPO = Path.home() / "Development/hilbert-paper/bench/efficiency"
QRELS_SHA = "87b6f6a2f0b87c8d55f8bf902e00099be366a363326295732e281a7b26a5f338"
BUDGETS = (1000, 2000, 3000, 4000, 6000)
DEPTH = 10
POOL = 50
THRESHOLD = 0.02


# ---- data -------------------------------------------------------------------
def fingerprint(env, pw) -> dict:
    row = psql(
        "SELECT (SELECT count(*) FROM pages WHERE deleted_at IS NULL), "
        "(SELECT count(*) FROM content_chunks cc JOIN pages p ON p.id = cc.page_id "
        " WHERE p.deleted_at IS NULL AND cc.embedding IS NOT NULL), "
        "(SELECT count(*) FROM links), (SELECT max(updated_at) FROM pages), "
        "(SELECT max(embedded_at) FROM content_chunks)", env, pw).strip()
    notes, chunks, links, page_ts, chunk_ts = row.split("|")
    return {"live_notes": int(notes), "embedded_chunks": int(chunks), "links": int(links),
            "pages_updated_at_max": page_ts, "chunks_embedded_at_max": chunk_ts}


def load_links(env, pw) -> list[tuple[str, str]]:
    out = psql("SELECT f.slug, t.slug FROM links l JOIN pages f ON f.id = l.from_page_id "
               "JOIN pages t ON t.id = l.to_page_id WHERE f.deleted_at IS NULL AND t.deleted_at IS NULL",
               env, pw)
    pairs = set()
    for line in out.splitlines():
        a, _, c = line.partition("|")
        if a and c and a != c:
            pairs.add((a, c))
    return sorted(pairs)


class Data:
    def __init__(self) -> None:
        raw = QRELS.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == QRELS_SHA
        qrels = json.loads(raw)
        assert len(qrels) == 817
        self.rels = [set(q["relevant"]) for q in qrels]
        self.n = len(qrels)
        self.Qn = unit(np.load(BENCH / "query-vectors.npy"))
        env, pw = pg_env()
        self.fp_start = fingerprint(env, pw)
        _truth, _ids, slugs, _sources, texts, _idx, matrix = load_corpus(env, pw)
        links = load_links(env, pw)
        self.env, self.pw = env, pw
        self.Cn = unit(matrix)
        starts = [i for i, s in enumerate(slugs) if i == 0 or s != slugs[i - 1]]
        self.starts = np.array(starts)
        self.ends = np.append(self.starts[1:], len(slugs))
        self.pages = [slugs[i] for i in starts]
        self.slug_rank = np.argsort(np.argsort(np.array(self.pages, dtype=object)))
        self.N = len(self.pages)
        self.page_index = {s: i for i, s in enumerate(self.pages)}
        self.tokens = np.array([max(estimate_tokens(t), 1) for t in texts])
        sim = self.Qn @ self.Cn.T
        self.S = np.maximum.reduceat(sim, self.starts, axis=1)
        self.win = np.empty(self.S.shape, dtype=np.int64)
        for p, (a, b) in enumerate(zip(self.starts, self.ends)):
            self.win[:, p] = a + sim[:, a:b].argmax(axis=1)
        self.relmask = np.zeros((self.n, self.N), dtype=bool)
        for qi, rel in enumerate(self.rels):
            for s in rel:
                if s in self.page_index:
                    self.relmask[qi, self.page_index[s]] = True
        self.directed = [(self.page_index[a], self.page_index[c]) for a, c in links
                         if a in self.page_index and c in self.page_index]
        split = json.loads((BENCH / "ideas" / "split.json").read_text())
        self.tune = np.array(split["tune"])
        self.confirm = np.array(split["confirm"])
        self.base_order = self.order(self.S)
        self.corpus = {"notes_with_chunks": self.N, "chunks": int(matrix.shape[0]),
                       "directed_links_between_them": len(self.directed), **self.fp_start}
        print("corpus", self.corpus, flush=True)

    def order(self, scores: np.ndarray) -> np.ndarray:
        """Descending score; ties by slug (as ideas/common.py Bench.order)."""
        return np.lexsort((np.broadcast_to(self.slug_rank, scores.shape), -scores), axis=-1)

    def unit_of(self, qi: int, note: int) -> tuple[int, int, bool]:
        row = int(self.win[qi, note])
        return note, int(self.tokens[row]), bool(self.relmask[qi, note])


# ---- packing and outcomes ------------------------------------------------------
def walk(costs: list[int], budget: int) -> list[int]:
    """Baseline rule: indices kept, in order."""
    used, kept = 0, []
    for i, c in enumerate(costs):
        if used + c <= budget:
            used += c
            kept.append(i)
    return kept


class Outcome:
    """Per-question survival and tokens for every budget, plus relevant notes at 6,000."""

    def __init__(self, n: int) -> None:
        self.surv = np.zeros((n, len(BUDGETS)))
        self.tok = np.zeros((n, len(BUDGETS)))
        self.rel6k = np.zeros(n)

    def set(self, qi: int, bi: int, rel: list[bool], cost: list[int]) -> None:
        self.surv[qi, bi] = 1.0 if any(rel) else 0.0
        self.tok[qi, bi] = sum(cost)
        if BUDGETS[bi] == 6000:
            self.rel6k[qi] = sum(rel)

    @property
    def sbar(self) -> np.ndarray:
        return self.surv.mean(axis=1)


def pack_order(d: Data, notes_by_q, qs) -> Outcome:
    """Walk the first DEPTH notes of each question's order with the baseline rule."""
    out = Outcome(d.n)
    for qi in qs:
        units = [d.unit_of(qi, int(p)) for p in notes_by_q[qi][:DEPTH]]
        cost = [u[1] for u in units]
        for bi, b in enumerate(BUDGETS):
            kept = walk(cost, b)
            out.set(qi, bi, [units[i][2] for i in kept], [cost[i] for i in kept])
    return out


def r10(d: Data, order: np.ndarray, qs) -> np.ndarray:
    out = np.zeros(d.n)
    for qi in qs:
        out[qi] = recall_at_k([d.pages[p] for p in order[qi, :DEPTH]], d.rels[qi], DEPTH)
    return out


def qset(d: Data, qs):
    return range(d.n) if qs is None else qs


# ---- M1: isotonic calibration and value density -----------------------------------
def fit_isotonic(x: np.ndarray, y: np.ndarray):
    xs, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    ys = np.bincount(inv, weights=y) / w
    fit = optimize.isotonic_regression(ys, weights=w, increasing=True).x
    keep = np.r_[True, np.diff(fit) != 0]
    return xs[keep], fit[keep]


def predict_isotonic(model, x: np.ndarray) -> np.ndarray:
    starts, vals = model
    i = np.searchsorted(starts, x, side="right") - 1
    return vals[np.clip(i, 0, len(vals) - 1)]


def m1(d: Data, model, pool: int, qs=None) -> Outcome:
    out = Outcome(d.n)
    for qi in qset(d, qs):
        notes = d.base_order[qi, :pool]
        units = [d.unit_of(qi, int(p)) for p in notes]
        prob = predict_isotonic(model, d.S[qi, notes])
        cost = np.array([u[1] for u in units])
        order = np.argsort(-(prob / cost), kind="stable")
        costs = [int(cost[i]) for i in order]
        for bi, b in enumerate(BUDGETS):
            kept = [order[i] for i in walk(costs, b)]
            out.set(qi, bi, [units[i][2] for i in kept], [int(cost[i]) for i in kept])
    return out


# ---- M2: budgeted submodular coverage ------------------------------------------
def m2_select(rel: np.ndarray, sim: np.ndarray, cost: np.ndarray, budget: int, lam: float, p: float) -> list[int]:
    n = len(rel)
    simp = np.maximum(sim, 0)
    cover = np.zeros(n)
    taken = np.zeros(n, dtype=bool)
    used, chosen = 0, []
    scaled = cost.astype(float) ** p
    while True:
        fits = (~taken) & (used + cost <= budget)
        if not fits.any():
            break
        gain = lam * rel + (rel[:, None] * np.maximum(simp - cover[:, None], 0)).sum(axis=0)
        gain = np.where(fits, gain, -np.inf)
        ratio = np.where(gain > 0, gain / scaled, -np.inf)
        j = int(np.argmax(ratio))
        if not np.isfinite(ratio[j]):
            break
        taken[j] = True
        chosen.append(j)
        used += int(cost[j])
        cover = np.maximum(cover, simp[:, j])

    def value(sel):
        if not sel:
            return 0.0
        return float((rel * simp[:, sel].max(axis=1)).sum() + lam * rel[sel].sum())

    best_single, best_v = None, -np.inf
    for j in np.flatnonzero(cost <= budget):
        v = value([int(j)])
        if v > best_v:
            best_single, best_v = int(j), v
    if best_single is not None and best_v > value(chosen):
        return [best_single]
    return chosen


def m2(d: Data, lam: float, p: float, qs=None) -> Outcome:
    out = Outcome(d.n)
    for qi in qset(d, qs):
        notes = d.base_order[qi, :POOL]
        units = [d.unit_of(qi, int(n)) for n in notes]
        rows = d.win[qi, notes]
        U = d.Cn[rows].astype(float)
        sim = U @ U.T
        rel = d.S[qi, notes].astype(float)
        cost = np.array([u[1] for u in units])
        for bi, b in enumerate(BUDGETS):
            sel = m2_select(rel, sim, cost, b, lam, p)
            out.set(qi, bi, [units[i][2] for i in sel], [int(cost[i]) for i in sel])
    return out


# ---- M3: maximal marginal relevance -----------------------------------------------
def mmr_order(rel: np.ndarray, sim: np.ndarray, lam: float, k: int) -> list[int]:
    n = len(rel)
    chosen = [int(np.argmax(rel))]
    maxsim = sim[:, chosen[0]].copy()
    taken = np.zeros(n, dtype=bool)
    taken[chosen[0]] = True
    while len(chosen) < min(k, n):
        score = np.where(taken, -np.inf, lam * rel - (1 - lam) * maxsim)
        j = int(np.argmax(score))
        chosen.append(j)
        taken[j] = True
        maxsim = np.maximum(maxsim, sim[:, j])
    return chosen


def reorder_pool(d: Data, picker, qs=None) -> Outcome:
    orders = {}
    for qi in qset(d, qs):
        notes = d.base_order[qi, :POOL]
        rows = d.win[qi, notes]
        U = d.Cn[rows].astype(float)
        rel = d.S[qi, notes].astype(float)
        idx = picker(rel, U @ U.T)
        orders[qi] = [int(notes[i]) for i in idx]
    return pack_order(d, orders, qset(d, qs))


# ---- M4: determinantal point process ---------------------------------------------
def dpp_order(rel: np.ndarray, K: np.ndarray, alpha: float, k: int, eps: float = 1e-12) -> list[int]:
    q = np.exp(alpha * (rel - rel.max()))
    L = q[:, None] * K * q[None, :]
    n = len(rel)
    c = np.zeros((k, n))
    d2 = np.diag(L).copy()
    chosen = [int(np.argmax(d2))]
    while len(chosen) < min(k, n):
        j = chosen[-1]
        m = len(chosen) - 1
        e = (L[j] - c[:m, j] @ c[:m]) / np.sqrt(d2[j])
        c[m] = e
        d2 = d2 - e ** 2
        d2[chosen] = -np.inf
        nxt = int(np.argmax(d2))
        if d2[nxt] < eps * q.max() ** 2:
            break
        chosen.append(nxt)
    rest = [i for i in range(n) if i not in chosen]
    return chosen + rest


# ---- M5-M7: diffusion on graphs -----------------------------------------------------
def undirected(d: Data) -> np.ndarray:
    W = np.zeros((d.N, d.N))
    for i, j in d.directed:
        W[i, j] = W[j, i] = 1.0
    return W


def sym_laplacian(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deg = W.sum(axis=1)
    inv = np.where(deg > 0, 1 / np.sqrt(np.where(deg > 0, deg, 1)), 0)
    S = inv[:, None] * W * inv[None, :]
    return np.diag((deg > 0).astype(float)) - S, S


def rerank(d: Data, scores: np.ndarray, qs=None) -> tuple[Outcome, np.ndarray]:
    order = d.order(scores)
    return pack_order(d, order, qset(d, qs)), order


# ---- statistics and selection --------------------------------------------------------
def summary_of(o: Outcome, qs) -> dict:
    return {"S_bar": float(o.sbar[qs].mean()),
            "survival": {str(b): float(o.surv[qs, i].mean()) for i, b in enumerate(BUDGETS)},
            "tokens": {str(b): float(o.tok[qs, i].mean()) for i, b in enumerate(BUDGETS)},
            "relevant_notes_6000": float(o.rel6k[qs].mean())}


def per_budget_tests(o: Outcome, base: Outcome, qs) -> dict:
    tests = {str(b): mcnemar(o.surv[qs, i], base.surv[qs, i]) for i, b in enumerate(BUDGETS)}
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm_within_idea"] = adj[k]
    return tests


def choose(grid: dict) -> str:
    best = None
    for key, g in grid.items():
        if best is None or g["tune"]["S_bar"] > grid[best]["tune"]["S_bar"]:
            best = key
    return best


def run_grid(d: Data, base: Outcome, name: str, settings: list[tuple[str, dict]], make, extra=None) -> dict:
    grid = {}
    for key, params in settings:
        o = make(**params)
        grid[key] = {"params": params, "tune": summary_of(o, d.tune), "confirm": summary_of(o, d.confirm), "_o": o}
        print(name, key, round(grid[key]["tune"]["S_bar"], 4), flush=True)
    chosen = choose(grid)
    params = grid[chosen]["params"]
    t0 = time.perf_counter()
    o = make(**params, qs=d.confirm)
    per_q = (time.perf_counter() - t0) / len(d.confirm)
    full = grid[chosen]["_o"]
    assert np.array_equal(o.surv[d.confirm], full.surv[d.confirm])
    c = d.confirm
    primary = wilcoxon(full.sbar[c], base.sbar[c])
    primary["threshold"] = THRESHOLD
    primary["criterion_met"] = primary["diff"] >= THRESHOLD
    doc = {"chosen": chosen, "params": params,
           "grid": {k: {"params": v["params"], "tune": v["tune"], "confirm": v["confirm"]} for k, v in grid.items()},
           "baseline": {"tune": summary_of(base, d.tune), "confirm": summary_of(base, d.confirm)},
           "tune_half_test": wilcoxon(full.sbar[d.tune], base.sbar[d.tune]),
           "primary": primary,
           "per_budget_confirm": per_budget_tests(full, base, c),
           "seconds_per_question": per_q,
           "per_question_confirm": {"idx": c, "S_bar": full.sbar[c], "S_bar_baseline": base.sbar[c]}}
    if extra:
        doc.update(extra(chosen, grid))
    return doc


def main() -> None:
    t_start = time.time()
    d = Data()
    results, onetime = {}, {}

    base = pack_order(d, d.base_order, range(d.n))
    base_all = summary_of(base, np.arange(d.n))
    print("baseline", base_all, flush=True)

    # M1
    t0 = time.perf_counter()
    xs, ys = [], []
    for qi in d.tune:
        notes = d.base_order[qi, :POOL]
        xs.append(d.S[qi, notes])
        ys.append(d.relmask[qi, notes].astype(float))
    model = fit_isotonic(np.concatenate(xs).astype(float), np.concatenate(ys))
    onetime["M1 isotonic fit"] = time.perf_counter() - t0
    results["M1"] = run_grid(d, base, "M1", [(f"pool={k}", {"pool": k}) for k in (10, 20, 50)],
                             lambda pool, qs=None: m1(d, model, pool, qs),
                             lambda c, g: {"isotonic": {"blocks": len(model[0]), "starts": model[0], "values": model[1]}})

    # M2
    results["M2"] = run_grid(d, base, "M2", [(f"lambda={lam},p={p}", {"lam": lam, "p": p})
                                             for lam in (0, 0.5, 1, 2) for p in (0.5, 1)],
                             lambda lam, p, qs=None: m2(d, lam, p, qs))

    # M3
    results["M3"] = run_grid(d, base, "M3", [(f"lambda={lam}", {"lam": lam}) for lam in (0.5, 0.6, 0.7, 0.8, 0.9)],
                             lambda lam, qs=None: reorder_pool(d, lambda r, s: mmr_order(r, s, lam, DEPTH), qs))

    # M4
    results["M4"] = run_grid(d, base, "M4", [(f"alpha={a}", {"alpha": a}) for a in (2, 5, 10, 20)],
                             lambda alpha, qs=None: reorder_pool(d, lambda r, K: dpp_order(r, K, alpha, DEPTH), qs))

    # graphs
    W = undirected(d)
    L_sym, S_adj = sym_laplacian(W)
    graph = {"nodes": d.N, "undirected_edges": int(W.sum() / 2), "directed_links": len(d.directed),
             "isolated": int((W.sum(axis=1) == 0).sum())}
    base_r10 = r10(d, d.base_order, range(d.n))

    def ranking_extra(make_scores):
        def extra(chosen, grid):
            sc = make_scores(**grid[chosen]["params"])
            o = d.order(sc)
            rr = r10(d, o, range(d.n))
            c = d.confirm
            return {"graph": graph, "R@10": {"tune": float(rr[d.tune].mean()), "confirm": float(rr[c].mean()),
                                             "baseline_tune": float(base_r10[d.tune].mean()),
                                             "baseline_confirm": float(base_r10[c].mean()),
                                             "confirm_diff_ci95": list(boot_ci(rr[c] - base_r10[c]))}}
        return extra

    # M5
    t0 = time.perf_counter()
    lam_e, V = linalg.eigh(L_sym)
    onetime["M5 eigendecomposition"] = time.perf_counter() - t0
    SV = d.S.astype(float) @ V
    heat_cache = {}

    def m5_scores(t, beta):
        if t not in heat_cache:
            heat_cache[t] = (SV * np.exp(-t * lam_e)) @ V.T
        return (1 - beta) * d.S + beta * heat_cache[t]

    results["M5"] = run_grid(d, base, "M5", [(f"t={t},beta={b}", {"t": t, "beta": b})
                                             for t in (0.5, 1, 2, 4) for b in (0.1, 0.3, 0.5)],
                             lambda t, beta, qs=None: rerank(d, m5_scores(t, beta), qs)[0],
                             ranking_extra(m5_scores))

    # M6
    Nn = np.zeros((d.N, d.Cn.shape[1]))
    for p, (a, b) in enumerate(zip(d.starts, d.ends)):
        Nn[p] = d.Cn[a:b].mean(axis=0)
    Nn = unit(Nn).astype(float)
    cos_nn = Nn @ Nn.T
    np.fill_diagonal(cos_nn, -np.inf)
    qn = d.Qn.astype(float) @ Nn.T
    Y = np.zeros((d.n, d.N))
    for qi in range(d.n):
        top = d.base_order[qi, :POOL]
        Y[qi, top] = np.maximum(0, qn[qi, top])
    s_lo = d.S.min(axis=1, keepdims=True)
    s_hi = d.S.max(axis=1, keepdims=True)
    manifold_cache = {}

    def m6_F(k, a):
        if (k, a) not in manifold_cache:
            t0 = time.perf_counter()
            Wk = np.zeros((d.N, d.N))
            nbr = np.argsort(-cos_nn, axis=1, kind="stable")[:, :k]
            for i in range(d.N):
                for j in nbr[i]:
                    if cos_nn[i, j] > 0:
                        Wk[i, j] = max(Wk[i, j], cos_nn[i, j])
                        Wk[j, i] = max(Wk[j, i], cos_nn[i, j])
            _, Sk = sym_laplacian(Wk)
            F = linalg.solve(np.eye(d.N) - a * Sk, Y.T, assume_a="pos").T
            manifold_cache[(k, a)] = F
            onetime[f"M6 graph and solve k={k} a={a}"] = time.perf_counter() - t0
        return manifold_cache[(k, a)]

    def m6_scores(k, a, beta):
        F = m6_F(k, a)
        f_lo = F.min(axis=1, keepdims=True)
        f_hi = F.max(axis=1, keepdims=True)
        span = f_hi - f_lo
        scaled = np.where(span > 0, s_lo + (F - f_lo) / np.where(span > 0, span, 1) * (s_hi - s_lo), s_lo)
        return (1 - beta) * d.S + beta * scaled

    results["M6"] = run_grid(d, base, "M6", [(f"k={k},a={a},beta={b}", {"k": k, "a": a, "beta": b})
                                             for k in (10, 20) for a in (0.5, 0.8, 0.95) for b in (0.3, 0.5)],
                             lambda k, a, beta, qs=None: rerank(d, m6_scores(k, a, beta), qs)[0],
                             ranking_extra(m6_scores))

    # M7
    A = np.zeros((d.N, d.N))
    for i, j in d.directed:
        A[i, j] = 1.0
    L_adv = np.diag(A.sum(axis=1)) - A.T
    expm_cache = {}

    def m7_scores(t, gamma, beta):
        if (t, gamma) not in expm_cache:
            t0 = time.perf_counter()
            E = linalg.expm(-t * (gamma * L_adv + (1 - gamma) * L_sym))
            expm_cache[(t, gamma)] = d.S.astype(float) @ E.T
            onetime[f"M7 expm t={t} gamma={gamma}"] = time.perf_counter() - t0
        return (1 - beta) * d.S + beta * expm_cache[(t, gamma)]

    results["M7"] = run_grid(d, base, "M7", [(f"t={t},gamma={g},beta={b}", {"t": t, "gamma": g, "beta": b})
                                             for t in (0.5, 1, 2) for g in (0.25, 0.5, 0.75) for b in (0.1, 0.3)],
                             lambda t, gamma, beta, qs=None: rerank(d, m7_scores(t, gamma, beta), qs)[0],
                             ranking_extra(m7_scores))

    # M9
    b6 = BUDGETS.index(6000)
    packs = []
    for qi in range(d.n):
        units = [d.unit_of(qi, int(p)) for p in d.base_order[qi, :DEPTH]]
        kept = walk([u[1] for u in units], 6000)
        packs.append([(float(d.S[qi, units[i][0]]), units[i][1], units[i][2]) for i in kept])
    lambdas = [round(1 - i / 1000, 3) for i in range(1001)] + [-np.inf]

    def at(lmb, qi):
        units = [u for u in packs[qi] if u[0] >= lmb]
        return (0.0 if any(u[2] for u in units) else 1.0), sum(u[1] for u in units)

    n_t = len(d.tune)
    miss_tune = float(1 - base.surv[d.tune, b6].mean())
    alpha = miss_tune + 0.02
    chosen_l = None
    for lmb in lambdas:
        risk = float(np.mean([at(lmb, qi)[0] for qi in d.tune]))
        if n_t / (n_t + 1) * risk + 1 / (n_t + 1) <= alpha:
            chosen_l = lmb
            break
    t0 = time.perf_counter()
    loss = np.zeros(d.n)
    tok = np.zeros(d.n)
    for qi in range(d.n):
        loss[qi], tok[qi] = at(chosen_l, qi)
    per_q9 = (time.perf_counter() - t0) / d.n
    c = d.confirm
    base_tok = base.tok[:, b6]
    surv9 = 1 - loss
    primary9 = wilcoxon(base_tok[c], tok[c])
    primary9["test"] = "one-sided Wilcoxon signed-rank on non-zero (baseline tokens - M9 tokens)"
    fall = (base_tok[c].mean() - tok[c].mean()) / base_tok[c].mean()
    primary9["relative_fall"] = float(fall)
    primary9["criterion_met"] = bool(fall >= 0.20)
    ni = wilcoxon(surv9[c], base.surv[c, b6], margin=0.02)
    results["M9"] = {
        "alpha_target": alpha, "tune_baseline_miss_6000": miss_tune, "lambda_hat": chosen_l,
        "lambda_grid": "1.000 to 0.000 in steps of 0.001, then -inf",
        "realised_miss": {"tune": float(loss[d.tune].mean()), "confirm": float(loss[c].mean())},
        "tokens": {"tune": float(tok[d.tune].mean()), "confirm": float(tok[c].mean()),
                   "baseline_tune": float(base_tok[d.tune].mean()), "baseline_confirm": float(base_tok[c].mean())},
        "survival": {"tune": float(surv9[d.tune].mean()), "confirm": float(surv9[c].mean()),
                     "baseline_tune": float(base.surv[d.tune, b6].mean()),
                     "baseline_confirm": float(base.surv[c, b6].mean())},
        "primary": primary9, "non_inferiority": ni, "seconds_per_question": per_q9,
        "per_question_confirm": {"idx": c, "tokens": tok[c], "tokens_baseline": base_tok[c],
                                 "survival": surv9[c], "survival_baseline": base.surv[c, b6]},
    }
    print("M9", chosen_l, results["M9"]["tokens"], results["M9"]["survival"], flush=True)

    fp_end = fingerprint(d.env, d.pw)
    if fp_end != d.fp_start:
        print("CORPUS CHANGED during the run; discard and repeat", d.fp_start, fp_end, flush=True)
        sys.exit(3)

    family = {k: v["primary"]["p"] for k, v in results.items()}
    adj = holm(family)
    rows = {}
    for k, v in results.items():
        crit = v["primary"]["criterion_met"] and (k != "M9" or v["non_inferiority"]["p"] < 0.05)
        v["primary"]["p_holm"] = adj[k]
        v["accepted"] = bool(crit and adj[k] < 0.05)
        rows[k] = {"chosen": v.get("chosen", v.get("lambda_hat")), "diff": v["primary"]["diff"],
                   "ci95": v["primary"]["ci95"], "p": v["primary"]["p"], "p_holm": adj[k],
                   "criterion_met": v["primary"]["criterion_met"], "accepted": v["accepted"]}
    meta = {"corpus": d.corpus, "corpus_end": fp_end, "qrels_sha256": QRELS_SHA, "budgets": BUDGETS,
            "split": "ideas/split.json", "seed_bootstrap": 20261005, "one_time_seconds": onetime,
            "run_seconds": time.time() - t_start}
    REPO.mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        doc = {"idea": k, **v, "meta": meta}
        path = HERE / f"{k.lower()}.json"
        path.write_text(json.dumps(rounded(doc), indent=1) + "\n")
        shutil.copy(path, REPO / path.name)
    summary = {"family": rows, "holm": "step-down over the eight primary p-values, alpha 0.05",
               "baseline_all": base_all, "meta": meta}
    (HERE / "summary.json").write_text(json.dumps(rounded(summary), indent=1) + "\n")
    shutil.copy(HERE / "summary.json", REPO / "summary.json")
    shutil.copy(Path(__file__), REPO / "run.py")
    print(json.dumps(rounded(rows), indent=1))


if __name__ == "__main__":
    main()
