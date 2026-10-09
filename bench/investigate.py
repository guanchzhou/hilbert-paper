#!/usr/bin/env python3
"""Hypothesis tests, mechanism analyses, and failure taxonomy. Development set only."""

import csv
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from metrics import estimate_tokens, mrr, ndcg_at_k, recall_at_k
from run_measure import (BUDGET, QRELS, join_chunks, load_corpus, pg_env, psql_script,
                         section_span, vector_rankings, window_span)

BENCH = Path(__file__).resolve().parent
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
DEPTH = 50
RRF_K = 60
SEED = 11


def keyword(queries, env, password, mode: str, unit: str, k: int = DEPTH):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for i, q in enumerate(queries):
        w.writerow([i, q])
    tsq = ("websearch_to_tsquery('english', query)" if mode == "and"
           else "replace(websearch_to_tsquery('english', query)::text, ' & ', ' | ')::tsquery")
    if unit == "page":
        body = f"""
CREATE TEMP TABLE page_tv AS SELECT slug, to_tsvector('english', compiled_truth) AS tv FROM pages WHERE deleted_at IS NULL;
CREATE INDEX ON page_tv USING gin (tv);
\\copy (WITH qq AS (SELECT idx, {tsq} AS tsq FROM bench_q) SELECT qq.idx, h.slug FROM qq CROSS JOIN LATERAL (SELECT slug, ts_rank_cd(tv, qq.tsq) AS s FROM page_tv WHERE qq.tsq::text <> '' AND tv @@ qq.tsq ORDER BY s DESC, slug LIMIT {k}) h ORDER BY qq.idx, h.s DESC, h.slug) TO STDOUT WITH (FORMAT csv)
"""
    else:
        body = f"""
\\copy (WITH qq AS (SELECT idx, {tsq} AS tsq FROM bench_q) SELECT qq.idx, h.slug FROM qq CROSS JOIN LATERAL (SELECT p.slug, max(ts_rank_cd(cc.search_vector, qq.tsq)) AS s FROM content_chunks cc JOIN pages p ON p.id = cc.page_id WHERE p.deleted_at IS NULL AND qq.tsq::text <> '' AND cc.search_vector @@ qq.tsq GROUP BY p.slug ORDER BY s DESC, p.slug LIMIT {k}) h ORDER BY qq.idx, h.s DESC, h.slug) TO STDOUT WITH (FORMAT csv)
"""
    script = f"""SET statement_timeout = 0;
SET client_min_messages = warning;
CREATE TEMP TABLE bench_q (idx int PRIMARY KEY, query text);
COPY bench_q (idx, query) FROM STDIN WITH (FORMAT csv);
{buf.getvalue()}\\.
{body}"""
    out = psql_script(script, env, password)
    lists = [[] for _ in queries]
    for row in csv.reader(io.StringIO(out)):
        if len(row) == 2 and row[0].isdigit():
            lists[int(row[0])].append(row[1])
    return lists


def rrf(a: list[str], b: list[str], k: int = RRF_K) -> list[str]:
    score = {}
    for lst in (a, b):
        for r, s in enumerate(lst):
            score[s] = score.get(s, 0.0) + 1.0 / (k + r + 1)
    return [s for s, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:10]


def key_int(marker: str) -> int:
    return int(marker.rsplit(":", 1)[1], 16)


def shared_bits(a: int, b: int) -> int:
    d = a ^ b
    return 64 if d == 0 else 64 - d.bit_length()


def hilbert(vectors: np.ndarray, extra: list[str]) -> list[dict]:
    payload = "".join(json.dumps({"id": i, "embedding": v.tolist()}) + "\n" for i, v in enumerate(vectors))
    proc = subprocess.run([str(HILBERT), "key", *extra, "--format", "jsonl"], input=payload,
                          capture_output=True, text=True, check=True)
    return [json.loads(line) for line in proc.stdout.splitlines()]


def lsh_p(cos: np.ndarray, bits: int = 8) -> np.ndarray:
    theta = np.arccos(np.clip(cos, -1.0, 1.0))
    return (1.0 - theta / math.pi) ** bits


def paired(a: np.ndarray, b: np.ndarray, rng, alternative: str) -> dict:
    d = a - b
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    nz = d[d != 0]
    if len(nz):
        w = stats.wilcoxon(nz, alternative=alternative, zero_method="wilcox")
        ranks = stats.rankdata(np.abs(nz))
        w_pos = ranks[nz > 0].sum()
        w_neg = ranks[nz < 0].sum()
        rbc = float((w_pos - w_neg) / (w_pos + w_neg))
        p = float(w.pvalue)
    else:
        p, rbc = 1.0, 0.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "test": "Wilcoxon signed-rank on non-zero paired differences", "alternative": alternative,
            "p": p, "rank_biserial": rbc, "n_nonzero": int(len(nz))}


def mcnemar(a: np.ndarray, b: np.ndarray, rng, alternative: str) -> dict:
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    p = float(stats.binomtest(b01, n, 0.5, alternative=alternative).pvalue) if n else 1.0
    d = a - b
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    boot = d[idx].mean(axis=1)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
            "test": "exact McNemar (binomial on discordant pairs)", "alternative": alternative,
            "discordant_a_only": b01, "discordant_b_only": b10, "p": p,
            "odds_ratio": (b01 / b10) if b10 else None}


def holm(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    env, password = pg_env()
    qrels = json.loads(QRELS.read_text())
    queries = [q["query"] for q in qrels]
    rels = [set(q["relevant"]) for q in qrels]
    n = len(qrels)
    vectors = np.load(BENCH / "query-vectors.npy")
    qn = (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    truth, ids, slugs, sources, texts, indexes, matrix = load_corpus(env, password)
    cn = (matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    starts, page_slugs = [], []
    for i, s in enumerate(slugs):
        if i == 0 or s != slugs[i - 1]:
            starts.append(i)
            page_slugs.append(s)
    starts_a = np.array(starts)
    sim = qn @ cn.T
    page_best = np.maximum.reduceat(sim, starts_a, axis=1)
    full_rank = np.argsort(-page_best, axis=1)
    vchunk50 = [[page_slugs[j] for j in full_rank[i, :DEPTH]] for i in range(n)]
    vchunk = [l[:10] for l in vchunk50]
    vpage, _, _vc, _, winners, by_page, pos_of = vector_rankings(qn, matrix, slugs, sources, texts, indexes)

    print("keyword runs", flush=True)
    kw = {(m, u): keyword(queries, env, password, m, u) for m in ("and", "or") for u in ("page", "chunk")}
    hybrid = [rrf(kw[("or", "page")][i], vchunk50[i]) for i in range(n)]
    hybrid_chunk = [rrf(kw[("or", "chunk")][i], vchunk50[i]) for i in range(n)]

    lists = {
        "keyword-and-page": [l[:10] for l in kw[("and", "page")]],
        "keyword-and-chunk": [l[:10] for l in kw[("and", "chunk")]],
        "keyword-or-page": [l[:10] for l in kw[("or", "page")]],
        "keyword-or-chunk": [l[:10] for l in kw[("or", "chunk")]],
        "vector-page": vpage,
        "vector-chunk": vchunk,
        "hybrid-rrf": hybrid,
        "hybrid-rrf-chunk": hybrid_chunk,
    }

    print("hk1 probes", flush=True)
    keyed = {int(k): key_int(v) for k, v in json.loads((BENCH / "chunk-keys.json").read_text())["keys"].items()}
    ckey = np.array([keyed.get(c, 0) for c in ids], dtype=np.uint64)
    has_key = np.array([c in keyed for c in ids])
    for level, ranges in ((1, 1), (1, 16)):
        rows = hilbert(vectors, ["--probe", str(level), "--ranges", str(ranges)])
        out = []
        for r in rows:
            qi = int(r["id"])
            mask = np.zeros(len(ids), dtype=bool)
            for lo, hi in r["ranges"]:
                mask |= (ckey >= np.uint64(key_int(lo))) & (ckey <= np.uint64(key_int(hi)))
            mask &= has_key
            cand = np.flatnonzero(mask)
            order = cand[np.argsort(-sim[qi, cand])]
            seen, pages = set(), []
            for j in order:
                if slugs[j] not in seen:
                    seen.add(slugs[j])
                    pages.append(slugs[j])
                if len(pages) == 10:
                    break
            out.append(pages)
        lists[f"hk1-L{level}-R{ranges}"] = out

    per = {}
    for name, ls in lists.items():
        per[name] = {
            "recall": np.array([recall_at_k(ls[i], rels[i], 10) for i in range(n)]),
            "mrr": np.array([mrr(ls[i], rels[i]) for i in range(n)]),
            "ndcg": np.array([ndcg_at_k(ls[i], rels[i], 10) for i in range(n)]),
        }

    survive = {u: np.zeros(n) for u in ("chunk", "window", "section", "page")}
    for qi, rows in enumerate(winners):
        for unit in survive:
            used, ok = 0, False
            for row in rows:
                slug, hit_i = pos_of[row]
                ch = by_page[slug]
                if unit == "chunk":
                    text = ch[hit_i]["text"]
                elif unit == "window":
                    text = join_chunks(ch, window_span(ch, hit_i))
                elif unit == "section":
                    text = join_chunks(ch, section_span(ch, hit_i))
                else:
                    text = truth.get(slug, "")
                cost = estimate_tokens(text)
                if used + cost <= BUDGET:
                    used += cost
                    ok = ok or slug in rels[qi]
            survive[unit][qi] = 1.0 if ok else 0.0

    R = {k: v["recall"] for k, v in per.items()}
    confirm = {
        "H1 vector-chunk > keyword-and-page": paired(R["vector-chunk"], R["keyword-and-page"], rng, "greater"),
        "H2 vector-chunk != vector-page": paired(R["vector-chunk"], R["vector-page"], rng, "two-sided"),
        "H3 chunk pack > page pack (survival)": mcnemar(survive["chunk"], survive["page"], rng, "greater"),
        "H4 hk1 L1x16 non-inferior to cosine": paired(R["hk1-L1-R16"], R["vector-chunk"], rng, "greater"),
    }
    h4 = confirm["H4 hk1 L1x16 non-inferior to cosine"]
    shifted = R["hk1-L1-R16"] - R["vector-chunk"] + 0.03
    nz = shifted[shifted != 0]
    h4["noninferiority_margin"] = 0.03
    # With many zero paired differences the shift turns ties into small positive values, which can make the
    # shifted Wilcoxon reject even for a clearly worse method; the paired t-test on the shifted values does not.
    h4["p_wilcoxon_shifted"] = float(stats.wilcoxon(nz, alternative="greater").pvalue)
    h4["p_t_shifted"] = float(stats.ttest_1samp(shifted, 0.0, alternative="greater").pvalue)
    h4["p"] = max(h4["p_wilcoxon_shifted"], h4["p_t_shifted"])
    h4["test"] = ("non-inferiority, H0: hk1 worse than cosine by more than 0.03; p is the larger of a one-sided "
                  "Wilcoxon signed-rank test and a one-sided paired t-test on (hk1 minus cosine plus 0.03)")
    adj = holm({k: v["p"] for k, v in confirm.items()})
    for k in confirm:
        confirm[k]["p_holm"] = adj[k]

    explore = {
        "E1 keyword OR > keyword AND (page)": paired(R["keyword-or-page"], R["keyword-and-page"], rng, "greater"),
        "E2 hybrid RRF vs vector-chunk": paired(R["hybrid-rrf"], R["vector-chunk"], rng, "two-sided"),
        "E3 chunk pack vs section pack": mcnemar(survive["chunk"], survive["section"], rng, "two-sided"),
        "E4 keyword OR chunk vs vector-chunk": paired(R["keyword-or-chunk"], R["vector-chunk"], rng, "two-sided"),
        "E5 hybrid RRF (chunk keyword) vs vector-chunk": paired(R["hybrid-rrf-chunk"], R["vector-chunk"], rng, "two-sided"),
    }
    adj = holm({k: v["p"] for k, v in explore.items()})
    for k in explore:
        explore[k]["p_holm"] = adj[k]

    print("lsh model", flush=True)
    qkeys = [key_int(r["marker"]) for r in hilbert(vectors, [])]
    page_rows = {}
    for i, s in enumerate(slugs):
        page_rows.setdefault(s, []).append(i)
    qa = []
    for qi in range(n):
        cand = [r for s in rels[qi] for r in page_rows.get(s, []) if has_key[r]]
        if not cand:
            continue
        j = max(cand, key=lambda r: sim[qi, r])
        c = float(sim[qi, j])
        qa.append({"cos": c, "shared": shared_bits(qkeys[qi], int(ckey[j]))})
    qa_cos = np.array([x["cos"] for x in qa])
    qa_same = np.array([x["shared"] >= 8 for x in qa])

    krows = np.flatnonzero(has_key)
    a = rng.choice(krows, size=300000)
    b = rng.choice(krows, size=300000)
    keep = a != b
    a, b = a[keep], b[keep]
    pc = np.einsum("ij,ij->i", cn[a], cn[b])
    ka, kb = ckey[a], ckey[b]
    same = ((ka ^ kb) >> np.uint64(56)) == 0
    csim = cn[krows] @ cn[krows].T
    np.fill_diagonal(csim, -2)
    nn = csim.argmax(axis=1)
    nn_cos = csim[np.arange(len(krows)), nn]
    nn_same = ((ckey[krows] ^ ckey[krows[nn]]) >> np.uint64(56)) == 0
    del csim

    def calib(cos, obs, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (cos >= lo) & (cos < hi)
            if m.sum() < 30:
                continue
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "observed": float(obs[m].mean()), "predicted": float(lsh_p(cos[m]).mean())})
        return out

    edges = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    top = (ckey[krows] >> np.uint64(56)).astype(int)
    occ = np.bincount(top, minlength=256)
    lsh = {
        "model": "P(same level-1 cell) = (1 - arccos(cos)/pi)^8, angular LSH with eight sign projections",
        "random_pairs": {"n": int(len(pc)), "observed": float(same.mean()), "predicted": float(lsh_p(pc).mean()),
                         "calibration": calib(pc, same, edges)},
        "nearest_neighbour_pairs": {"n": int(len(nn_cos)), "observed": float(nn_same.mean()),
                                    "predicted": float(lsh_p(nn_cos).mean()), "median_cos": float(np.median(nn_cos)),
                                    "calibration": calib(nn_cos, nn_same, edges)},
        "question_answer_pairs": {"n": len(qa), "observed": float(qa_same.mean()),
                                  "predicted": float(lsh_p(qa_cos).mean()),
                                  "cos_quartiles": [float(x) for x in np.percentile(qa_cos, [25, 50, 75])],
                                  "calibration": calib(qa_cos, qa_same, edges)},
        "occupancy": {"cells": 256, "nonempty": int((occ > 0).sum()), "max": int(occ.max()),
                      "uniform_expectation": float(len(krows) / 256), "top10_share": float(np.sort(occ)[-10:].sum() / len(krows)),
                      "counts_sorted": [int(x) for x in np.sort(occ)[::-1]]},
    }

    print("failures", flush=True)
    live = set(page_slugs)
    pos_of_page = {s: i for i, s in enumerate(page_slugs)}
    tax = {"relevant_not_in_corpus": 0, "rank_11_20": 0, "rank_21_50": 0, "rank_51_100": 0, "rank_over_100": 0,
           "duplicate_in_top10": 0}
    misses = 0
    for qi in range(n):
        if any(s in rels[qi] for s in vchunk[qi]):
            continue
        misses += 1
        present = [s for s in rels[qi] if s in live]
        if not present:
            tax["relevant_not_in_corpus"] += 1
            continue
        alt = {("obsidian/" + s) if not s.startswith("obsidian/") else s.removeprefix("obsidian/") for s in present}
        if any(s in alt for s in vchunk[qi]):
            tax["duplicate_in_top10"] += 1
            continue
        ranks = np.argsort(full_rank[qi])
        best = min(int(ranks[pos_of_page[s]]) + 1 for s in present)
        if best <= 20:
            tax["rank_11_20"] += 1
        elif best <= 50:
            tax["rank_21_50"] += 1
        elif best <= 100:
            tax["rank_51_100"] += 1
        else:
            tax["rank_over_100"] += 1
    tax["misses"] = misses

    qwords = np.array([len(q.split()) for q in queries])
    relsize = np.array([len(r) for r in rels])
    ctok = np.array([estimate_tokens(t) for t in texts])
    cpp = np.diff(np.append(starts_a, len(slugs)))
    missing_rel = sum(1 for r in rels for s in r if s not in live)
    dataset = {
        "questions": n, "question_words": {"median": float(np.median(qwords)), "p25": float(np.percentile(qwords, 25)),
                                           "p75": float(np.percentile(qwords, 75)), "max": int(qwords.max()),
                                           "hist": np.bincount(qwords).tolist()},
        "relevant_per_question": {"mean": float(relsize.mean()), "hist": np.bincount(relsize).tolist()},
        "relevant_slugs_missing_from_corpus": missing_rel,
        "chunk_tokens": {"median": float(np.median(ctok)), "p90": float(np.percentile(ctok, 90)),
                         "mean": float(ctok.mean()), "max": int(ctok.max()), "values_sample": ctok[::5].tolist()},
        "chunks_per_page": {"median": float(np.median(cpp)), "p90": float(np.percentile(cpp, 90)),
                            "max": int(cpp.max()), "hist": np.bincount(np.minimum(cpp, 30)).tolist()},
        "pages": len(page_slugs), "chunks": len(slugs),
    }

    summary = {name: {m: float(v[m].mean()) for m in ("recall", "mrr", "ndcg")} for name, v in per.items()}
    out = {"seed": SEED, "rrf_k": RRF_K, "depth": DEPTH, "cells": summary,
           "survival": {u: float(v.mean()) for u, v in survive.items()},
           "confirmatory": confirm, "exploratory": explore, "lsh": lsh, "failures": tax, "dataset": dataset}
    (BENCH / "investigation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"cells": summary, "confirm": {k: (v["diff"], v["p_holm"]) for k, v in confirm.items()},
                      "explore": {k: (v["diff"], v["p_holm"]) for k, v in explore.items()},
                      "lsh": {k: (lsh[k]["observed"], lsh[k]["predicted"]) for k in
                              ("random_pairs", "nearest_neighbour_pairs", "question_answer_pairs")},
                      "occupancy": {k: lsh["occupancy"][k] for k in ("nonempty", "max", "top10_share")},
                      "failures": tax}, indent=1))


if __name__ == "__main__":
    main()
