"""200,000-vector pgvector timing inside one session, in a scratch database that is dropped afterwards.

Method of the original in-session run: the live chunk embeddings plus copies of randomly chosen live
embeddings with Gaussian noise (sigma 0.02 per component), an HNSW cosine index, 50 top-10 queries
timed over one persistent connection, and recall at 10 against an exact scan of the same 200,000 rows.
Writes scale-in-session.json into the directory given by BENCH_OUT (default: this directory).
"""

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
OUT = Path(os.environ.get("BENCH_OUT", HERE))
TARGET = 200_000
SIGMA = 0.02
SCRATCH = "scale_bench_session"


def url(db: str) -> str:
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    base = cfg["database_url"].rsplit("/", 1)[0]
    return f"{base}/{db}"


def lit(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def main() -> None:
    with psycopg.connect(url("postgres"), autocommit=True) as admin:
        live = np.array([json.loads(r[0]) for r in admin.execute(
            "SELECT c.embedding::text FROM content_chunks c JOIN pages p ON p.id = c.page_id "
            "WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL ORDER BY c.id").fetchall()], dtype=np.float32)
        live_count_before = len(live)
        admin.execute(f"DROP DATABASE IF EXISTS {SCRATCH}")
        admin.execute(f"CREATE DATABASE {SCRATCH}")
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(live), TARGET - len(live))
    M = np.vstack([live, live[pick] + rng.normal(0, SIGMA, (len(pick), live.shape[1])).astype(np.float32)])
    out = {"n_vectors": TARGET, "n_real_embeddings": len(live), "noise": f"gaussian sigma {SIGMA}"}
    try:
        with psycopg.connect(url(SCRATCH), autocommit=True, prepare_threshold=None) as conn:
            conn.execute("CREATE EXTENSION vector")
            conn.execute("CREATE TABLE chunks (id int PRIMARY KEY, embedding vector(1024))")
            with conn.cursor().copy("COPY chunks (id, embedding) FROM STDIN") as cp:
                for i, v in enumerate(M):
                    cp.write_row((i, lit(v)))
            conn.execute("SET maintenance_work_mem = '2GB'")
            t0 = time.perf_counter()
            conn.execute("CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
            out["index_build_s"] = round(time.perf_counter() - t0, 1)
            out["index_bytes"] = conn.execute("SELECT pg_relation_size('chunks_hnsw')").fetchone()[0]
            out["maintenance_work_mem"] = "2GB"
            out["max_parallel_maintenance_workers"] = conn.execute("SHOW max_parallel_maintenance_workers").fetchone()[0]
            out["hnsw_ef_search"] = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
            conn.execute("ANALYZE chunks")
            Q = np.load(BENCH / "query-vectors.npy").astype(np.float32)[:50]
            sql = "SELECT id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 10"
            for v in Q[:3]:
                conn.execute(sql, (lit(v),)).fetchall()
            times, got = [], []
            for v in Q:
                t0 = time.perf_counter()
                got.append([r[0] for r in conn.execute(sql, (lit(v),)).fetchall()])
                times.append((time.perf_counter() - t0) * 1000)
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        exact = np.argsort(-(Qn @ Mn.T), axis=1)[:, :10]
        s = sorted(times)
        out.update({"median_ms": round(statistics.median(s), 3), "p95_ms": round(s[int(0.95 * (len(s) - 1))], 3),
                    "exact_recall_at_10": round(float(np.mean([len(set(g) & set(e.tolist())) / 10
                                                               for g, e in zip(got, exact)])), 3),
                    "note": "Timed inside one Postgres session over psycopg. Scratch database dropped."})
    finally:
        with psycopg.connect(url("postgres"), autocommit=True) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {SCRATCH}")
            after = admin.execute("SELECT count(*) FROM content_chunks c JOIN pages p ON p.id = c.page_id "
                                  "WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL").fetchone()[0]
    out["dropped"] = True
    out["live_embedded_chunks_before_after"] = [live_count_before, after]
    (OUT / "scale-in-session.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
