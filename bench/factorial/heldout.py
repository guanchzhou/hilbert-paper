#!/usr/bin/env python3
"""Held-out confirmation of level 1 (plan C9): the best-recall and best-survival configurations of
tier B, each run once on the sealed held-out questions, with the reference configuration (dense
best-chunk scores, chunk pack) on the same questions for paired comparison.

  --questions dev   reproduces stage2.py on the development questions through this script's own
                    code path and stops on any difference; it never opens qrels-test.json.
  --questions test  the single held-out run. Refuses to start unless the dev check has passed for
                    the same configurations and this script's current contents.

Keyword scores are recomputed against the frozen corpus snapshot (ts_rank_cd over
setweight(to_tsvector('english', chunk_text), 'B'), gbrain's chunk search vector), in TEMP tables
only, so the live database is read but not changed. Writes heldout-level1-<questions>.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from pipeline import CACHE, DEPTH, F8, IDEAS_CACHE, PRUNE_DEPTH, Cfg, Data, rerank_order, sentences, unit
import stage2
from stage1 import cached_pairs
from common import BENCH, boot_ci, embed, holm, mcnemar, rounded
from prep import TSQ, bench_q
from rerank_fill import score as rerank_score
from run_measure import QRELS, pg_env, psql_script

HERE = Path(__file__).resolve().parent
TEST = QRELS.with_name("qrels-test.json")
REFERENCE = (Cfg("dense", "best", "off", "off", "off", "off", "none"), "chunk")
PASS = HERE / "private" / "heldout-dev-pass.json"


def script_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def winners() -> list[tuple[str, Cfg, str]]:
    lb = json.loads((HERE / os.environ.get("HELDOUT_LEADERBOARD", "level1-leaderboard.json")).read_text())
    out = []
    for axis, unit_default in (("best_recall", "chunk"), ("best_survival", None)):
        c = lb[axis]["top20"][0]["config"]
        cfg = Cfg(c["F1"], c["F2"], c["F3"], c["F4"], c["F5"], c["F6"], c["F7"])
        if cfg.f7 != "none":
            raise SystemExit(f"{axis} uses the {cfg.f7} filter, which this script does not rebuild")
        out.append((axis, cfg, c.get("F8", unit_default)))
    return out


def snapshot_lexical(d: Data, queries: list[str]) -> np.ndarray:
    b = d.b
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for i, t in enumerate(b.texts):
        w.writerow([i, t])
    script = bench_q(queries) + (
        "CREATE TEMP TABLE snap_raw (row int PRIMARY KEY, txt text);\n"
        f"COPY snap_raw (row, txt) FROM STDIN WITH (FORMAT csv);\n{buf.getvalue()}\\.\n"
        "CREATE TEMP TABLE snap AS SELECT row, setweight(to_tsvector('english', coalesce(txt, '')), 'B') AS sv FROM snap_raw;\n"
        f"\\copy (WITH qq AS (SELECT idx, {TSQ} AS tsq FROM bench_q) SELECT qq.idx, s.row, ts_rank_cd(s.sv, qq.tsq) "
        "FROM qq JOIN snap s ON qq.tsq::text <> '' AND s.sv @@ qq.tsq) TO STDOUT WITH (FORMAT csv)\n")
    env, password = pg_env()
    out = psql_script(script, env, password)
    S = np.zeros((len(queries), len(b.ids)), dtype=np.float32)
    for row in csv.reader(io.StringIO(out)):
        if len(row) == 3 and row[0].isdigit():
            S[int(row[0]), int(row[1])] = float(row[2])
    return S


def retarget(d: Data, queries: list[str], Q: np.ndarray, rels: list[set], lex: np.ndarray) -> None:
    b = d.b
    b.queries, b.Q, b.Qn, b.rels, b.n = queries, Q, unit(Q), rels, len(queries)
    mu = b.Cn.mean(axis=0)
    d.space = {"off": (b.Qn, b.Cn), "on": (unit(b.Qn - mu), unit(b.Cn - mu))}
    d.lex = lex
    d.masks = {"none": None}


def evaluate(d: Data, sent, cfg: Cfg, unit_name: str, scores: dict) -> dict:
    b = d.b
    pre = d.pre_ranking(cfg.pre())
    top_all, rows_all = pre["top"], pre["rows"]
    lists, finals = [], []
    for qi in range(b.n):
        t, r = top_all[qi], rows_all[qi]
        if cfg.f3 == "on":
            n = int((t >= 0).sum())
            sc = [scores.get(f"{qi}:{b.ids[int(x)]}") for x in r[:n]]
            if any(v is None for v in sc):
                finals.append(None)
                lists.append([])
                continue
            o = rerank_order(np.arange(DEPTH), np.array(sc + [0.0] * (DEPTH - n)))
            o = np.concatenate([o[:n], np.arange(n, DEPTH)])
            t, r = t[o], r[o]
        finals.append((t, r))
        lists.append([b.pages[int(x)] for x in t[:10] if x >= 0])
    pq = b.per_question(lists)
    missing = np.array([f is None for f in finals])
    for k in pq:
        pq[k] = np.where(missing, np.nan, pq[k])
    surv, tok = np.full(b.n, np.nan), np.full(b.n, np.nan)
    for qi, f in enumerate(finals):
        if f is None:
            continue
        t, r = f
        ok = t >= 0
        rel = b.rels[qi]
        if unit_name == "pruned":
            t30, r30 = t[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]], r[:PRUNE_DEPTH][ok[:PRUNE_DEPTH]]
            costs = [sent.tokens(qi, int(x)) for x in r30]
            if any(c is None for c in costs):
                continue
            surv[qi], tok[qi] = stage2.pack([b.pages[int(x)] for x in t30], costs, rel)
            continue
        t10, r10 = t[:10][ok[:10]], r[:10][ok[:10]]
        costs = {"chunk": d.tok_chunk[r10], "window": d.tok_window[r10], "section": d.tok_section[r10],
                 "page": d.tok_page[t10]}[unit_name]
        surv[qi], tok[qi] = stage2.pack([b.pages[int(x)] for x in t10], [int(c) for c in costs], rel)
    return {**pq, "survival": surv, "tokens": tok, "rows": rows_all, "top": top_all}


def needed_rerank(d: Data, cfgs: list[Cfg]) -> dict[str, list[int]]:
    b = d.b
    need = {}
    for cfg in cfgs:
        if cfg.f3 != "on":
            continue
        rows = d.pre_ranking(cfg.pre())["rows"]
        for qi in range(b.n):
            need.setdefault(qi, set()).update(int(x) for x in rows[qi] if x >= 0)
    return {qi: sorted(v) for qi, v in need.items()}


def fill_rerank(d: Data, need: dict, path: Path) -> dict:
    b = d.b
    done = json.loads(path.read_text()) if path.exists() else {}
    t0 = time.perf_counter()
    for i, (qi, rows) in enumerate(sorted(need.items())):
        todo = [r for r in rows if f"{qi}:{b.ids[r]}" not in done]
        for s in range(0, len(todo), 50):
            part = todo[s:s + 50]
            sc, _ = rerank_score(b.queries[qi], [b.texts[r] for r in part])
            for r, v in zip(part, sc):
                done[f"{qi}:{b.ids[r]}"] = v
        if i % 25 == 0:
            path.write_text(json.dumps(done))
            print(f"rerank {i}/{len(need)} questions, {time.perf_counter() - t0:.0f} s", flush=True)
    path.write_text(json.dumps(done))
    return done


def fill_sentences(d: Data, sent, cfgs_units) -> int:
    b = d.b
    rows = set()
    for cfg, unit_name, ev in cfgs_units:
        if unit_name == "pruned":
            for qi in range(b.n):
                t, r = ev["top"][qi], ev["rows"][qi]
                rows.update(int(x) for x in r[:DEPTH] if x >= 0)
    missing = sorted(r for r in rows if r not in sent.vec)
    flat, counts = [], []
    for r in missing:
        s = sentences(b.texts[r])
        flat.extend(s)
        counts.append(len(s))
    if flat:
        V = embed(flat, batch=64)
        pos = 0
        for r, n in zip(missing, counts):
            v = V[pos:pos + n]
            sent.vec[r] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            sent.text[r] = sentences(b.texts[r])
            pos += n
    return len(missing)


def paired(new: dict, ref: dict) -> dict:
    from scipy import stats
    d_r = new["R@10"] - ref["R@10"]
    nz = d_r[np.abs(d_r) > 1e-12]
    p_r = float(stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue) if len(nz) else 1.0
    lo, hi = boot_ci(d_r)
    surv = mcnemar(new["survival"], ref["survival"], alternative="two-sided")
    return {"R@10": {"diff": float(d_r.mean()), "ci95": [lo, hi], "p": p_r,
                     "test": "two-sided Wilcoxon signed-rank on non-zero differences", "n_nonzero": int(len(nz))},
            "survival": surv,
            "tokens_diff": float((new["tokens"] - ref["tokens"]).mean())}


def summary(ev: dict) -> dict:
    return {k: float(np.mean(ev[k])) for k in ("R@10", "MRR", "nDCG@10", "hit@10", "survival", "tokens")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", choices=["dev", "test"], required=True)
    a = ap.parse_args()
    t_start = time.time()
    picks = winners()
    runs = picks + [("reference", *REFERENCE)]
    d = Data()
    sent = stage2.Sentences(d)
    b = d.b
    manifest = json.loads((BENCH / "manifest.json").read_text())
    prefix = manifest["query_prefix"]

    if a.questions == "dev":
        # The saved dev vectors were embedded in batches of 32; MLX output depends on batch padding.
        fresh = embed([prefix + q for q in b.queries[:32]], batch=32)
        emb_diff = float(np.abs(fresh - b.Q[:32]).max())
        assert emb_diff == 0, f"query embedding path differs from the saved dev vectors: {emb_diff}"
        lex = snapshot_lexical(d, b.queries)
        cached = np.load(CACHE / "lexical.npz")["S"]
        both = (lex > 0) & (cached > 0)
        lex_check = {"pairs_new": int((lex > 0).sum()), "pairs_cached": int((cached > 0).sum()),
                     "pairs_both": int(both.sum()), "max_abs_diff_on_both": float(np.abs(lex[both] - cached[both]).max())}
        print("lexical check", lex_check, flush=True)
        retarget(d, b.queries, b.Q, b.rels, cached)
        scores = cached_pairs(d)
        scores.update({k: v["score"] for k, v in json.loads((CACHE / "rerank-pairs.json").read_text()).items()})
        z = np.load(CACHE / "level1-perq.npz")
        keys = [str(k) for k in z["keys"]]
        report = {"embedding_max_abs_diff_first_32": emb_diff, "lexical_check": lex_check, "runs": {}}
        for name, cfg, unit_name in runs:
            ev = evaluate(d, sent, cfg, unit_name, scores)
            ci = keys.index(cfg.key)
            ui = list(F8).index(unit_name)
            ref_r = z["R_10"][ci]
            ref_s = z["surv"][ci, ui]
            ok = np.isfinite(ref_r) & np.isfinite(ref_s)
            r_diff = float(np.abs(ev["R@10"][ok] - ref_r[ok]).max())
            s_diff = float(np.abs(ev["survival"][ok] - ref_s[ok]).max())
            report["runs"][name] = {"config": cfg.key, "unit": unit_name, "questions_compared": int(ok.sum()),
                                    "R@10_max_abs_diff_vs_stage2": r_diff, "survival_max_abs_diff_vs_stage2": s_diff}
            print(name, report["runs"][name], flush=True)
            assert r_diff == 0 and s_diff == 0, f"{name} does not reproduce stage2.py"
        PASS.write_text(json.dumps({"script_sha256": script_hash(), "configs": [r[1].key + "/" + r[2] for r in runs]}))
        (HERE / "heldout-level1-dev.json").write_text(json.dumps(rounded(report), indent=1) + "\n")
        print("dev check passed", flush=True)
        return

    ok = PASS.exists() and json.loads(PASS.read_text()) == {
        "script_sha256": script_hash(), "configs": [r[1].key + "/" + r[2] for r in runs]}
    if not ok:
        raise SystemExit("run --questions dev first; the dev check has not passed for this script and these configurations")
    raw = TEST.read_bytes()
    test_sha = hashlib.sha256(raw).hexdigest()
    qrels = json.loads(raw)
    queries = [q["query"] for q in qrels]
    rels = [set(q["relevant"]) for q in qrels]
    Q = embed([prefix + q for q in queries], batch=32)
    lex = snapshot_lexical(d, queries)
    retarget(d, queries, Q, rels, lex)
    need = needed_rerank(d, [r[1] for r in runs])
    scores = fill_rerank(d, need, CACHE / "heldout-rerank-pairs.json") if need else {}
    evs = {}
    pre_evs = []
    for name, cfg, unit_name in runs:
        if unit_name == "pruned":
            pre_evs.append((cfg, unit_name, evaluate(d, sent, cfg, "chunk", scores)))
    new_rows = fill_sentences(d, sent, pre_evs)
    sent.memo = {}
    sent.Qn = b.Qn
    for name, cfg, unit_name in runs:
        evs[name] = evaluate(d, sent, cfg, unit_name, scores)
    tests = {name: paired(evs[name], evs["reference"]) for name, _, _ in picks}
    family = {f"{n} {m}": tests[n][m]["p"] for n in tests for m in ("R@10", "survival")}
    adj = holm(family)
    for key, p in adj.items():
        n, m = key.rsplit(" ", 1)
        tests[n][m]["p_holm"] = p
    out = {"split": "qrels-test", "qrels_test_sha256": test_sha, "n": len(queries),
           "snapshot": {"chunks": len(b.ids), "notes": len(b.pages)},
           "runs": {name: {"config": cfg.key, "unit": unit_name, **summary(evs[name])} for name, cfg, unit_name in runs},
           "paired_vs_reference": tests, "holm": "over the four paired tests",
           "rerank_pairs_scored": sum(len(v) for v in need.values()), "sentence_rows_embedded": new_rows,
           "seconds": time.time() - t_start}
    (HERE / "heldout-level1.json").write_text(json.dumps(rounded(out), indent=1) + "\n")
    print(json.dumps(rounded(out["runs"]), indent=1))
    print(json.dumps(rounded(tests), indent=1))


if __name__ == "__main__":
    main()
