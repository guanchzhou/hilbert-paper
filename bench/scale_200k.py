#!/usr/bin/env python3
"""Scratch 200,000-vector timing. Creates scale_bench and drops it."""

import json
import os
import statistics
import subprocess
import time
import urllib.parse
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
OUT = Path(os.environ.get("BENCH_OUT", BENCH))


def password() -> str:
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    return urllib.parse.urlparse(cfg["database_url"]).password or ""


def psql(db: str, sql: str, timeout: int = 600) -> str:
    env = {**os.environ, "PGPASSWORD": password()}
    proc = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", db, "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=False, capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"psql exit {proc.returncode}")
    return proc.stdout


def main() -> None:
    admin = "postgres"
    psql(admin, "DROP DATABASE IF EXISTS scale_bench")
    psql(admin, "CREATE DATABASE scale_bench")
    psql("scale_bench", "CREATE EXTENSION IF NOT EXISTS vector")
    psql("scale_bench", "CREATE TABLE chunks (id int PRIMARY KEY, embedding vector(1024), marker text)")
    print("copying live vectors", flush=True)
    blob = psql(admin, """
    SELECT c.embedding::text
    FROM content_chunks c
    JOIN pages p ON p.id = c.page_id
    WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL
    """)
    rows = [line for line in blob.splitlines() if line.startswith("[")]
    n_real = len(rows)
    print("real", n_real, flush=True)
    target = 200_000
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((target - n_real, 1024), dtype=np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    path = Path("/tmp/scale_bench_copy.tsv")
    with path.open("w") as fh:
        for i, line in enumerate(rows):
            key = f"{i:016x}"
            fh.write(f"{i}\t{line}\thk1:8:8:9e3779b97f4a7c15:{key}\n")
        for j, vec in enumerate(noise):
            i = n_real + j
            text = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
            key = f"{int.from_bytes(rng.bytes(8), 'big'):016x}"
            fh.write(f"{i}\t{text}\thk1:8:8:9e3779b97f4a7c15:{key}\n")
    print("loading", flush=True)
    env = {**os.environ, "PGPASSWORD": password()}
    subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "scale_bench", "-X", "-q", "-v", "ON_ERROR_STOP=1",
         "-c", "\\copy chunks (id, embedding, marker) FROM '/tmp/scale_bench_copy.tsv'"],
        check=True, env=env, timeout=900,
    )
    path.unlink(missing_ok=True)
    print("indexing", flush=True)
    psql(
        "scale_bench",
        "SET maintenance_work_mem = '2GB'; CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)",
        timeout=1800,
    )
    psql("scale_bench", 'CREATE INDEX chunks_marker ON chunks (marker COLLATE "C")')
    n = int(psql("scale_bench", "SELECT count(*) FROM chunks").strip())
    q = np.load(BENCH / "query-vectors.npy")[:20]
    times = []
    for vec in q:
        lit = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
        t0 = time.perf_counter()
        psql("scale_bench", f"SELECT id FROM chunks ORDER BY embedding <=> '{lit}'::vector LIMIT 10")
        times.append((time.perf_counter() - t0) * 1000)
    # one Hilbert range scan: first 4 hex nibbles
    t0 = time.perf_counter()
    scanned = int(psql(
        "scale_bench",
        "SELECT count(*) FROM chunks WHERE marker >= 'hk1:8:8:9e3779b97f4a7c15:3000000000000000' AND marker <= 'hk1:8:8:9e3779b97f4a7c15:30ffffffffffffff'",
    ).strip())
    hilbert_ms = (time.perf_counter() - t0) * 1000
    out = {
        "n": n,
        "real_vectors": n_real,
        "pgvector_median_ms": round(statistics.median(times), 3),
        "pgvector_p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 3),
        "hilbert_one_range_ms": round(hilbert_ms, 3),
        "hilbert_one_range_rows": scanned,
        "dropped": True,
    }
    psql(admin, "DROP DATABASE scale_bench")
    (OUT / ("scale-range-scan.json" if OUT != BENCH else "scale-200k.json")).write_text(json.dumps(out, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
