"""Repeat the database-dependent measurements on the upgraded server (PostgreSQL 18.6, pgvector 0.8.7).

Writes versions.json, pgvector-live.json, keyword.json and hk1-scans.json into this directory.
The PostgreSQL 16 results one level up are not touched; compare.py reads both.
Read-only against the live gbrain database: SELECT statements and one temporary table per session.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

import hilbert_measure as hm  # noqa: E402
import run_measure as rm  # noqa: E402

PROBES = [(1, 1), (1, 4), (1, 8), (1, 16), (2, 8), (2, 16)]


def dump(name: str, obj) -> None:
    (HERE / name).write_text(json.dumps(obj, indent=2) + "\n")
    print(name, flush=True)


def connect():
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    return psycopg.connect(cfg["database_url"], autocommit=True, prepare_threshold=None)


def versions(cur) -> dict:
    cur.execute("SELECT current_setting('server_version'), (SELECT extversion FROM pg_extension WHERE extname = 'vector')")
    server, vector = cur.fetchone()
    cur.execute("SELECT count(*) FROM content_chunks c JOIN pages p ON p.id = c.page_id "
                "WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL")
    chunks = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM pages WHERE deleted_at IS NULL")
    pages = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM chunk_hilbert")
    keyed = cur.fetchone()[0]
    return {"server_version": server, "pgvector": vector, "pages_live": pages, "embedded_chunks": chunks,
            "chunk_hilbert_rows": keyed, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def pgvector_live(cur, Q: np.ndarray) -> dict:
    sql = "SELECT id FROM content_chunks ORDER BY embedding <=> %s::vector LIMIT 10"
    lit = ["[" + ",".join(f"{x:.7f}" for x in v.tolist()) + "]" for v in Q[:50]]
    cur.execute("SHOW hnsw.ef_search")
    ef = cur.fetchone()[0]
    cur.execute("EXPLAIN " + sql, (lit[0],))
    plan = "\n".join(r[0] for r in cur.fetchall())
    for v in lit[:3]:
        cur.execute(sql, (v,))
        cur.fetchall()
    samples = []
    for v in lit:
        t0 = time.perf_counter()
        cur.execute(sql, (v,))
        cur.fetchall()
        samples.append((time.perf_counter() - t0) * 1000)
    s = sorted(samples)
    return {"n_queries": len(samples), "warmup_queries": 3, "hnsw_ef_search": ef, "sql": sql,
            "index_scan": "Index Scan using idx_chunks_embedding" in plan, "plan_first_line": plan.splitlines()[0],
            "median_ms": statistics.median(s), "p95_ms": s[int(0.95 * (len(s) - 1))], "min_ms": s[0], "max_ms": s[-1],
            "client": "psycopg 3, one persistent connection, automatic preparation off"}


def keyword(env: dict, password: str) -> dict:
    qrels = json.loads(rm.QRELS.read_text())
    queries = [q["query"] for q in qrels]
    truth = rm.load_corpus(env, password)[0]
    kw_page, kw_page_s, kw_chunk, kw_chunk_s = rm.keyword_rankings(queries, env, password)
    return {"keyword-page": rm.cell_record("keyword-page", kw_page, qrels, kw_page_s, truth),
            "keyword-chunk": rm.cell_record("keyword-chunk", kw_chunk, qrels, kw_chunk_s, truth),
            "empty_result_lists": {"keyword-page": sum(1 for x in kw_page if not x),
                                   "keyword-chunk": sum(1 for x in kw_chunk if not x)}}


def hk1_scans(Q: np.ndarray) -> list[dict]:
    db = hm.Psql()
    out = []
    try:
        for level, ranges in PROBES:
            spans = hm.probe_ranges(Q, level, ranges)
            counts, seconds = hm.sql_scan_counts(db, spans)
            out.append({"level": level, "ranges": ranges, "n_queries": len(Q), "scan_seconds": seconds,
                        "median_candidates": statistics.median(counts), "mean_candidates": statistics.mean(counts)})
            print(f"hk1 L{level} x{ranges}: {seconds:.3f} s, median {statistics.median(counts)}", flush=True)
    finally:
        db.close()
    return out


def main() -> None:
    Q = np.load(BENCH / "query-vectors.npy").astype(np.float32)
    with connect() as conn, conn.cursor() as cur:
        dump("versions.json", versions(cur))
        dump("pgvector-live.json", pgvector_live(cur, Q))
    env, password = rm.pg_env()
    dump("keyword.json", keyword(env, password))
    dump("hk1-scans.json", {"probes": hk1_scans(Q)})


if __name__ == "__main__":
    main()
