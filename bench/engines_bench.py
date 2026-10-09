#!/usr/bin/env python3
"""D12: the engines Module G left out, measured as Module G measured ArangoDB.

Each engine gets the live chunk vectors and note links (read-only from gbrain Postgres) in a
throwaway instance: SQLite with sqlite-vec (exact search; the extension has no approximate index),
libSQL with its native vector index, and MongoDB Community 8.2 with mongot (vector search) when
`--mongo URI` is given. pgvector is re-measured in the same session as the comparator. Measures,
over the 817 development questions: latency (client-observed), recall of the exact top-10 chunks,
and note-level recall at 10 (top 50 chunks collapsed to 10 distinct notes); then one-hop link
expansion from 300 seeded notes, checked against the true neighbour sets.

Run: uv run --python 3.12 --with numpy --with 'psycopg[binary]' --with sqlite-vec --with libsql \
       [--with pymongo] python engines_bench.py [--mongo URI]
Writes engines-d12.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import psycopg

BENCH = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BENCH_OUT", BENCH))
REPO = Path.home() / "Development/hilbert-paper/bench"
WARMUP = 20
HUB_CAP = 50
HOP_STARTS = 300
SEED = 5
K = 50


def ms(xs):
    xs = sorted(xs)
    return {"median_ms": statistics.median(xs), "p95_ms": xs[int(0.95 * (len(xs) - 1))], "mean_ms": statistics.mean(xs)}


class Data:
    def __init__(self) -> None:
        cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
        self.conn = psycopg.connect(cfg["database_url"], autocommit=True, prepare_threshold=None)
        cur = self.conn.cursor()
        qrels = json.loads((Path.home() / ".gbrain/eval/qrels-dev.json").read_text())
        self.rels = [set(q["relevant"]) for q in qrels]
        Q = np.load(BENCH / "query-vectors.npy").astype(np.float32)
        self.Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        cur.execute("""SELECT c.id, p.slug, c.embedding::text FROM content_chunks c JOIN pages p ON p.id = c.page_id
                       WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL ORDER BY c.id""")
        rows = cur.fetchall()
        self.ids = [r[0] for r in rows]
        self.slug_of = {r[0]: r[1] for r in rows}
        self.M = np.array([json.loads(r[2]) for r in rows], dtype=np.float32)
        Mn = self.M / np.linalg.norm(self.M, axis=1, keepdims=True)
        exact = np.argsort(-(self.Qn @ Mn.T), axis=1)[:, :10]
        self.exact_ids = [[self.ids[j] for j in row] for row in exact]
        self.live = set(self.ids)
        cur.execute("SELECT id FROM pages WHERE deleted_at IS NULL")
        self.pages = [r[0] for r in cur.fetchall()]
        pset = set(self.pages)
        cur.execute("SELECT from_page_id, to_page_id FROM links WHERE from_page_id <> to_page_id")
        self.edges = [(a, b) for a, b in cur.fetchall() if a in pset and b in pset]
        self.nbr = {p: set() for p in self.pages}
        for a, b in self.edges:
            self.nbr[a].add(b)
            self.nbr[b].add(a)
        self.deg = {p: len(v) for p, v in self.nbr.items()}
        rng = np.random.default_rng(SEED)
        linked = [p for p in self.pages if self.deg[p] > 0]
        self.starts = [int(x) for x in rng.choice(linked, size=min(HOP_STARTS, len(linked)), replace=False)]
        print("live chunks", len(self.ids), "pages", len(self.pages), "edges", len(self.edges), flush=True)

    def run(self, name, fn):
        lat, ann, lists = [], [], []
        for qi in range(WARMUP):
            fn(self.Qn[qi], K)
        for qi in range(len(self.Qn)):
            t0 = time.perf_counter()
            got = fn(self.Qn[qi], K)
            lat.append((time.perf_counter() - t0) * 1000)
            got = [g for g in got if g in self.live]
            ann.append(len(set(got[:10]) & set(self.exact_ids[qi])) / 10)
            seen, pages = set(), []
            for g in got:
                sl = self.slug_of[g]
                if sl not in seen:
                    seen.add(sl)
                    pages.append(sl)
                if len(pages) == 10:
                    break
            lists.append(pages)
        rec = [len(set(lists[i]) & self.rels[i]) / len(self.rels[i]) for i in range(len(self.Qn))]
        out = {"setting": name, **ms(lat), "chunk_recall_vs_exact_at_10": float(np.mean(ann)),
               "page_recall_at_10": float(np.mean(rec))}
        print(out, flush=True)
        return out

    def hop(self, name, fn):
        for p in self.starts[:WARMUP]:
            fn(p)
        lat, same = [], 0
        for p in self.starts:
            t0 = time.perf_counter()
            got = set(fn(p))
            lat.append((time.perf_counter() - t0) * 1000)
            truth = {n for n in self.nbr[p] if self.deg[n] <= HUB_CAP}
            same += got == truth
        out = {"engine": name, **ms(lat), "identical_sets": same, "starts": len(self.starts)}
        print("hop", out, flush=True)
        return out


HOP_SQL = ("SELECT DISTINCT n FROM (SELECT b AS n FROM links WHERE a = ? UNION ALL "
           "SELECT a FROM links WHERE b = ?) x JOIN pages d ON d.id = x.n WHERE d.deg <= ?")


def load_graph_sql(db, d: Data) -> None:
    db.execute("CREATE TABLE pages (id INTEGER PRIMARY KEY, deg INTEGER)")
    db.execute("CREATE TABLE links (a INTEGER, b INTEGER)")
    db.executemany("INSERT INTO pages VALUES (?, ?)", [(p, d.deg[p]) for p in d.pages])
    db.executemany("INSERT INTO links VALUES (?, ?)", d.edges)
    db.execute("CREATE INDEX links_a ON links (a)")
    db.execute("CREATE INDEX links_b ON links (b)")
    db.commit()


def sqlite_vec(d: Data, tmp: Path) -> dict:
    import sqlite_vec as sv
    db = sqlite3.connect(tmp / "bench.sqlite")
    db.enable_load_extension(True)
    sv.load(db)
    db.enable_load_extension(False)
    version = db.execute("SELECT vec_version(), sqlite_version()").fetchone()
    t0 = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{d.M.shape[1]}] distance_metric=cosine)")
    db.executemany("INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                   [(cid, sv.serialize_float32(v.tolist())) for cid, v in zip(d.ids, d.M)])
    db.commit()
    load_s = time.perf_counter() - t0

    def fn(v, k):
        return [r[0] for r in db.execute("SELECT rowid FROM vec_chunks WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                                         (sv.serialize_float32(v.tolist()), k))]
    res = [{"engine": "sqlite-vec", "index": "none (exact vec0 scan)", **d.run("sqlite-vec exact", fn)}]
    load_graph_sql(db, d)
    hop = d.hop("sqlite", lambda p: [r[0] for r in db.execute(HOP_SQL, (p, p, HUB_CAP))])
    db.close()
    return {"version": {"sqlite_vec": version[0], "sqlite": version[1]}, "load_seconds": load_s, "vector": res, "hop": hop}


def libsql(d: Data, tmp: Path) -> dict:
    import libsql as ls
    db = ls.connect(str(tmp / "bench.libsql"))
    version = db.execute("SELECT sqlite_version()").fetchone()[0]
    dim = d.M.shape[1]
    db.execute(f"CREATE TABLE chunks (id INTEGER PRIMARY KEY, emb F32_BLOB({dim}))")
    t0 = time.perf_counter()
    blob = lambda v: struct.pack(f"<{dim}f", *v.tolist())
    for i in range(0, len(d.ids), 500):
        db.executemany("INSERT INTO chunks VALUES (?, ?)", [(cid, blob(v)) for cid, v in zip(d.ids[i:i + 500], d.M[i:i + 500])])
    db.commit()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    db.execute("CREATE INDEX chunks_idx ON chunks (libsql_vector_idx(emb, 'metric=cosine'))")
    db.commit()
    index_s = time.perf_counter() - t0
    res = []

    def ann(v, k):
        return [r[0] for r in db.execute("SELECT id FROM vector_top_k('chunks_idx', ?, ?)", (blob(v), k)).fetchall()]

    def exact(v, k):
        return [r[0] for r in db.execute("SELECT id FROM chunks ORDER BY vector_distance_cos(emb, ?) LIMIT ?",
                                         (blob(v), k)).fetchall()]
    res.append({"engine": "libsql", "index": "libsql_vector_idx (DiskANN, defaults)", **d.run("libsql index", ann)})
    res.append({"engine": "libsql", "index": "none (full scan)", **d.run("libsql exact", exact)})
    load_graph_sql(db, d)
    hop = d.hop("libsql", lambda p: [r[0] for r in db.execute(HOP_SQL, (p, p, HUB_CAP)).fetchall()])
    return {"version": {"libsql_sqlite": version}, "load_seconds": load_s, "index_seconds": index_s, "vector": res, "hop": hop}


def pgvector(d: Data) -> dict:
    conn = d.conn

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
    res = [{"engine": "pgvector", "index": "HNSW", "ef_search": 100, **d.run("pgvector ef_search 100", pg(100))},
           {"engine": "pgvector", "index": "none (full scan)", **d.run("pgvector exact", pg(0, True))}]
    cur = conn.cursor()
    cur.execute("SELECT current_setting('server_version'), (SELECT extversion FROM pg_extension WHERE extname = 'vector')")
    ver = cur.fetchone()
    cur.execute("CREATE TEMP TABLE bench_deg (id int PRIMARY KEY, deg int)")
    with cur.copy("COPY bench_deg (id, deg) FROM STDIN") as cp:
        for p in d.pages:
            cp.write_row((p, d.deg[p]))
    cur.execute("CREATE TEMP TABLE bench_links AS SELECT l.from_page_id a, l.to_page_id b FROM links l "
                "JOIN bench_deg x ON x.id = l.from_page_id JOIN bench_deg y ON y.id = l.to_page_id "
                "WHERE l.from_page_id <> l.to_page_id")
    cur.execute("CREATE INDEX ON bench_links (a)")
    cur.execute("CREATE INDEX ON bench_links (b)")
    sql = ("SELECT DISTINCT n FROM (SELECT b AS n FROM bench_links WHERE a = %s UNION ALL "
           "SELECT a FROM bench_links WHERE b = %s) x JOIN bench_deg d ON d.id = x.n WHERE d.deg <= %s")

    def hop(p):
        cur.execute(sql, (p, p, HUB_CAP))
        return [r[0] for r in cur.fetchall()]
    return {"version": {"postgresql": ver[0], "pgvector": ver[1]}, "vector": res, "hop": d.hop("postgres", hop)}


def mongo(d: Data, uri: str) -> dict:
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel
    cli = MongoClient(uri)
    assert "familio" not in cli.list_database_names(), "this must be the throwaway instance"
    db = cli["knowledge_bench"]
    db.chunks.drop()
    t0 = time.perf_counter()
    db.chunks.insert_many([{"_id": cid, "embedding": v.tolist()} for cid, v in zip(d.ids, d.M)])
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    db.chunks.create_search_index(SearchIndexModel(name="emb", type="vectorSearch", definition={
        "fields": [{"type": "vector", "path": "embedding", "numDimensions": int(d.M.shape[1]), "similarity": "cosine"}]}))
    while True:
        idx = list(db.chunks.list_search_indexes("emb"))
        if idx and idx[0].get("queryable"):
            break
        time.sleep(1)
    index_s = time.perf_counter() - t0
    res = []
    for cand in (100, 200, 500):
        def fn(v, k, c=cand):
            return [x["_id"] for x in db.chunks.aggregate([{"$vectorSearch": {
                "index": "emb", "path": "embedding", "queryVector": v.tolist(), "numCandidates": c, "limit": k}},
                {"$project": {"_id": 1}}])]
        res.append({"engine": "mongodb", "index": "vectorSearch (HNSW, mongot)", "numCandidates": cand,
                    **d.run(f"mongodb numCandidates {cand}", fn)})

    def exact(v, k):
        return [x["_id"] for x in db.chunks.aggregate([{"$vectorSearch": {
            "index": "emb", "path": "embedding", "queryVector": v.tolist(), "exact": True, "limit": k}}, {"$project": {"_id": 1}}])]
    res.append({"engine": "mongodb", "index": "vectorSearch exact (ENN)", **d.run("mongodb exact", exact)})
    db.pages.drop()
    db.links.drop()
    db.pages.insert_many([{"_id": p, "deg": d.deg[p]} for p in d.pages])
    db.links.insert_many([{"a": a, "b": b} for a, b in d.edges])
    db.links.create_index("a")
    db.links.create_index("b")
    deg = {p: d.deg[p] for p in d.pages}

    def hop(p):
        pipe = [{"$match": {"$or": [{"a": p}, {"b": p}]}},
                {"$project": {"n": {"$cond": [{"$eq": ["$a", p]}, "$b", "$a"]}}},
                {"$lookup": {"from": "pages", "localField": "n", "foreignField": "_id", "as": "pg"}},
                {"$match": {"pg.deg": {"$lte": HUB_CAP}}}, {"$group": {"_id": "$n"}}]
        return [x["_id"] for x in db.links.aggregate(pipe)]
    hop_out = d.hop("mongodb", hop)
    info = cli.server_info()
    cli.drop_database("knowledge_bench")
    left = cli.list_database_names()
    return {"version": {"mongodb": info["version"]}, "load_seconds": load_s, "index_seconds": index_s,
            "vector": res, "hop": hop_out, "dropped": "knowledge_bench" not in left}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo", default=None)
    ap.add_argument("--only", default=None, choices=["mongo"])
    a = ap.parse_args()
    d = Data()
    path = OUT / "engines-d12.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    corpus = {"chunks": len(d.ids), "pages": len(d.pages), "edges": len(d.edges)}
    doc.update({"queries": len(d.Qn), "warmup": WARMUP, "k_fetched": K, "hub_cap": HUB_CAP,
                "client": "one connection per engine from Python; latency is client-observed",
                "page_recall_rule": "top 50 chunks collapsed to the first 10 distinct notes"})
    if a.only != "mongo":
        with tempfile.TemporaryDirectory() as tmp:
            doc["pgvector"] = {**pgvector(d), "corpus": corpus}
            doc["sqlite_vec"] = {**sqlite_vec(d, Path(tmp)), "corpus": corpus}
            doc["libsql"] = {**libsql(d, Path(tmp)), "corpus": corpus}
    if a.mongo:
        doc["mongodb"] = {**mongo(d, a.mongo), "corpus": corpus}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    REPO.joinpath("engines-d12.json").write_text(json.dumps(doc, indent=2) + "\n")
    REPO.joinpath("engines_bench.py").write_text(Path(__file__).read_text())


if __name__ == "__main__":
    main()
