#!/usr/bin/env python3
"""ArangoDB 3.12 against pgvector on the same frozen vectors, in a throwaway instance.

The instance is a separate container on 127.0.0.1:38529. The genealogy instance is never contacted.
"""

import json
import os
import statistics
import time
import urllib.parse
from pathlib import Path

import numpy as np
import psycopg
import requests

BENCH = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BENCH_OUT", BENCH))
ARANGO = "http://127.0.0.1:38529"
DB = "knowledge_bench"
WARMUP = 20
HUB_CAP = 50
HOP_STARTS = 300
SEED = 5


def ms(xs):
    xs = sorted(xs)
    return {"median_ms": statistics.median(xs), "p95_ms": xs[int(0.95 * (len(xs) - 1))], "mean_ms": statistics.mean(xs)}


def main() -> None:
    pw = (BENCH / "arango" / ".pw").read_text().strip()
    s = requests.Session()
    s.auth = ("root", pw)
    ver = s.get(f"{ARANGO}/_api/version").json()
    dbs = s.get(f"{ARANGO}/_api/database").json()["result"]
    assert "familio" not in dbs, "this must be the throwaway instance"

    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    conn = psycopg.connect(cfg["database_url"], autocommit=True, prepare_threshold=None)
    cur = conn.cursor()

    qrels = json.loads((Path.home() / ".gbrain/eval/qrels-dev.json").read_text())
    rels = [set(q["relevant"]) for q in qrels]
    Q = np.load(BENCH / "query-vectors.npy").astype(np.float32)
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)

    cur.execute("""SELECT c.id, p.slug, c.embedding::text FROM content_chunks c JOIN pages p ON p.id = c.page_id
                   WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL ORDER BY c.id""")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    slug_of = {r[0]: r[1] for r in rows}
    M = np.array([json.loads(r[2]) for r in rows], dtype=np.float32)
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    live = set(ids)
    exact = np.argsort(-(Qn @ Mn.T), axis=1)[:, :10]
    exact_ids = [[ids[j] for j in row] for row in exact]
    print("live chunks", len(ids), flush=True)

    # ---------------------------------------------------------------- load
    if DB in dbs:
        s.delete(f"{ARANGO}/_api/database/{DB}").raise_for_status()
    s.post(f"{ARANGO}/_api/database", json={"name": DB}).raise_for_status()
    base = f"{ARANGO}/_db/{DB}"
    s.post(f"{base}/_api/collection", json={"name": "chunks"}).raise_for_status()
    t0 = time.perf_counter()
    for i in range(0, len(ids), 500):
        body = "\n".join(json.dumps({"_key": str(ids[j]), "slug": slug_of[ids[j]], "embedding": M[j].tolist()})
                         for j in range(i, min(i + 500, len(ids))))
        r = s.post(f"{base}/_api/import", params={"collection": "chunks", "type": "documents"}, data=body)
        r.raise_for_status()
        assert r.json()["errors"] == 0, r.json()
    load_s = time.perf_counter() - t0
    count = s.get(f"{base}/_api/collection/chunks/count").json()["count"]
    n_lists = int(round(len(ids) ** 0.5))
    t0 = time.perf_counter()
    r = s.post(f"{base}/_api/index", params={"collection": "chunks"}, json={
        "type": "vector", "name": "emb", "fields": ["embedding"],
        "params": {"metric": "cosine", "dimension": int(M.shape[1]), "nLists": n_lists}})
    r.raise_for_status()
    index_s = time.perf_counter() - t0
    (OUT / "arango-load.json").write_text(json.dumps({
        "server": ver, "instance": "throwaway container arango-bench on 127.0.0.1:38529, no persistent volume",
        "database": DB, "documents": count, "load_seconds": load_s, "vector_index": {"metric": "cosine",
        "dimension": int(M.shape[1]), "nLists": n_lists, "build_seconds": index_s},
        "genealogy_instance_contacted": False}, indent=2) + "\n")
    print("loaded", count, "in", round(load_s, 1), "s; index", round(index_s, 1), "s", flush=True)

    def aql(query, bind):
        r = s.post(f"{base}/_api/cursor", json={"query": query, "bindVars": bind, "batchSize": 1000})
        r.raise_for_status()
        return r.json()["result"]

    def run(name, fn, k=50):
        lat, ann, lists = [], [], []
        for qi in range(WARMUP):
            fn(Qn[qi], k)
        for qi in range(len(Qn)):
            t0 = time.perf_counter()
            got = fn(Qn[qi], k)
            lat.append((time.perf_counter() - t0) * 1000)
            got = [g for g in got if g in live]
            ann.append(len(set(got[:10]) & set(exact_ids[qi])) / 10)
            seen, pages = set(), []
            for g in got:
                sl = slug_of[g]
                if sl not in seen:
                    seen.add(sl)
                    pages.append(sl)
                if len(pages) == 10:
                    break
            lists.append(pages)
        rec = [len(set(lists[i]) & rels[i]) / len(rels[i]) for i in range(len(Qn))]
        out = {"setting": name, **ms(lat), "chunk_recall_vs_exact_at_10": float(np.mean(ann)),
               "page_recall_at_10": float(np.mean(rec))}
        print(out, flush=True)
        return out

    results = []
    for nprobe in (1, 4, 8, 16, n_lists):
        q = "FOR d IN chunks SORT APPROX_NEAR_COSINE(d.embedding, @q, {nProbe: @np}) DESC LIMIT @k RETURN TO_NUMBER(d._key)"
        results.append({"engine": "arangodb", "index": "vector (IVF)", "nProbe": nprobe,
                        **run(f"arango nProbe {nprobe}", lambda v, k, np_=nprobe: aql(q, {"q": v.tolist(), "np": np_, "k": k}))})
    q = "FOR d IN chunks LET sc = COSINE_SIMILARITY(d.embedding, @q) SORT sc DESC LIMIT @k RETURN TO_NUMBER(d._key)"
    results.append({"engine": "arangodb", "index": "none (full scan)", **run("arango exact", lambda v, k: aql(q, {"q": v.tolist(), "k": k}))})

    def pg(ef, exact_scan=False):
        def fn(v, k):
            lit = "[" + ",".join(f"{x:.7f}" for x in v.tolist()) + "]"
            with conn.transaction():
                c = conn.cursor()
                if exact_scan:
                    c.execute("SET LOCAL enable_indexscan = off")
                else:
                    c.execute("SET LOCAL enable_seqscan = off")
                    c.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
                c.execute("SELECT id FROM content_chunks ORDER BY embedding <=> %s::vector LIMIT %s", (lit, k))
                return [r[0] for r in c.fetchall()]
        return fn

    for ef in (50, 100, 200, 400):
        results.append({"engine": "pgvector", "index": "HNSW", "ef_search": ef, **run(f"pgvector ef_search {ef}", pg(ef))})
    results.append({"engine": "pgvector", "index": "none (full scan)", **run("pgvector exact", pg(0, True))})
    (OUT / "arango-vector.json").write_text(json.dumps({
        "queries": len(Qn), "warmup": WARMUP, "k_fetched": 50,
        "client": "one persistent connection per engine from Python; latency is client-observed",
        "pgvector_note": "HNSW rows force the index (enable_seqscan off); ef_search is at least the 50 rows fetched. At LIMIT 50 the planner otherwise prefers a sequential scan on this table. Automatic statement preparation is off so that each setting is planned afresh.",
        "page_recall_rule": "top 50 chunks collapsed to the first 10 distinct notes",
        "exact_numpy_page_recall_reference": 0.6984, "results": results}, indent=2) + "\n")

    # ---------------------------------------------------------------- one hop
    cur.execute("SELECT id FROM pages WHERE deleted_at IS NULL")
    pages = [r[0] for r in cur.fetchall()]
    pset = set(pages)
    cur.execute("SELECT from_page_id, to_page_id FROM links WHERE from_page_id <> to_page_id")
    edges = [(a, b) for a, b in cur.fetchall() if a in pset and b in pset]
    nbr = {p: set() for p in pages}
    for a, b in edges:
        nbr[a].add(b)
        nbr[b].add(a)
    deg = {p: len(v) for p, v in nbr.items()}
    hubs = sum(1 for d in deg.values() if d > HUB_CAP)
    s.post(f"{base}/_api/collection", json={"name": "pages"}).raise_for_status()
    s.post(f"{base}/_api/collection", json={"name": "links", "type": 3}).raise_for_status()
    s.post(f"{base}/_api/import", params={"collection": "pages", "type": "documents"},
           data="\n".join(json.dumps({"_key": str(p), "deg": deg[p]}) for p in pages)).raise_for_status()
    s.post(f"{base}/_api/import", params={"collection": "links", "type": "documents"},
           data="\n".join(json.dumps({"_from": f"pages/{a}", "_to": f"pages/{b}"}) for a, b in edges)).raise_for_status()
    cur.execute("CREATE TEMP TABLE bench_deg (id int PRIMARY KEY, deg int)")
    with cur.copy("COPY bench_deg (id, deg) FROM STDIN") as cp:
        for p in pages:
            cp.write_row((p, deg[p]))
    cur.execute("CREATE TEMP TABLE bench_links AS SELECT l.from_page_id a, l.to_page_id b FROM links l "
                "JOIN bench_deg x ON x.id = l.from_page_id JOIN bench_deg y ON y.id = l.to_page_id "
                "WHERE l.from_page_id <> l.to_page_id")
    cur.execute("CREATE INDEX ON bench_links (a)")
    cur.execute("CREATE INDEX ON bench_links (b)")
    rng = np.random.default_rng(SEED)
    linked = [p for p in pages if deg[p] > 0]
    starts = [int(x) for x in rng.choice(linked, size=min(HOP_STARTS, len(linked)), replace=False)]
    hop_aql = "FOR v IN 1..1 ANY CONCAT('pages/', @k) links FILTER v.deg <= @cap RETURN DISTINCT TO_NUMBER(v._key)"
    hop_sql = ("SELECT DISTINCT n FROM (SELECT b AS n FROM bench_links WHERE a = %s UNION ALL "
               "SELECT a FROM bench_links WHERE b = %s) x JOIN bench_deg d ON d.id = x.n WHERE d.deg <= %s")
    for p in starts[:WARMUP]:
        aql(hop_aql, {"k": str(p), "cap": HUB_CAP})
        cur.execute(hop_sql, (p, p, HUB_CAP))
        cur.fetchall()
    la, ls, jac = [], [], []
    for p in starts:
        t0 = time.perf_counter()
        A = set(aql(hop_aql, {"k": str(p), "cap": HUB_CAP}))
        la.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        cur.execute(hop_sql, (p, p, HUB_CAP))
        S = {r[0] for r in cur.fetchall()}
        ls.append((time.perf_counter() - t0) * 1000)
        truth = {n for n in nbr[p] if deg[n] <= HUB_CAP}
        assert S == truth, (p, len(S), len(truth))
        jac.append(len(A & S) / len(A | S) if (A | S) else 1.0)
    (OUT / "arango-hop.json").write_text(json.dumps({
        "pages": len(pages), "edges": len(edges), "hub_cap": HUB_CAP, "hubs_excluded": hubs,
        "starts": len(starts), "neighbour_set_jaccard_mean": float(np.mean(jac)),
        "identical_sets": int(sum(1 for j in jac if j == 1.0)),
        "arangodb_aql": ms(la), "postgres_sql": ms(ls)}, indent=2) + "\n")
    print("hop", float(np.mean(jac)), ms(la), ms(ls), flush=True)

    # ---------------------------------------------------------------- drop
    s.delete(f"{ARANGO}/_api/database/{DB}").raise_for_status()
    left = s.get(f"{ARANGO}/_api/database").json()["result"]
    (OUT / "arango-dropped.json").write_text(json.dumps({
        "database_dropped": DB not in left, "databases_left_on_throwaway_instance": left,
        "genealogy_instance_contacted": False}, indent=2) + "\n")
    print("dropped", DB not in left, flush=True)


if __name__ == "__main__":
    main()
