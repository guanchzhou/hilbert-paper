#!/usr/bin/env python3
"""Level 1 inputs that need no GPU: lexical-OR chunk scores (read-only SQL, TEMP tables only),
hk1 probe candidates, Hilbert+SOAR partition candidates, and the lexical per-query latency sample.

Outputs go to factorial/cache/ (private).
"""

import csv
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))

from common import BENCH, HILBERT, SEED, Bench, key_int  # noqa: E402
from run_measure import pg_env, psql_script  # noqa: E402

CACHE = HERE / "cache"
TSQ = "replace(websearch_to_tsquery('english', query)::text, ' & ', ' | ')::tsquery"


def bench_q(queries: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for i, q in enumerate(queries):
        w.writerow([i, q])
    return ("SET statement_timeout = 0;\nSET client_min_messages = warning;\n"
            "CREATE TEMP TABLE bench_q (idx int PRIMARY KEY, query text);\n"
            f"COPY bench_q (idx, query) FROM STDIN WITH (FORMAT csv);\n{buf.getvalue()}\\.\n")


def lexical_scores(b: Bench) -> None:
    path = CACHE / "lexical.npz"
    if path.exists():
        return
    env, password = pg_env()
    script = bench_q(b.queries) + (
        f"\\copy (WITH qq AS (SELECT idx, {TSQ} AS tsq FROM bench_q) "
        "SELECT qq.idx, cc.id, ts_rank_cd(cc.search_vector, qq.tsq) FROM qq JOIN content_chunks cc "
        "ON qq.tsq::text <> '' AND cc.search_vector @@ qq.tsq JOIN pages p ON p.id = cc.page_id "
        "WHERE p.deleted_at IS NULL) TO STDOUT WITH (FORMAT csv)\n")
    t0 = time.perf_counter()
    out = psql_script(script, env, password)
    print("lexical sql", round(time.perf_counter() - t0, 1), "s", flush=True)
    row_of = {cid: i for i, cid in enumerate(b.ids)}
    S = np.zeros((b.n, len(b.ids)), dtype=np.float32)
    extra = 0
    for row in csv.reader(io.StringIO(out)):
        if len(row) != 3 or not row[0].isdigit():
            continue
        r = row_of.get(int(row[1]))
        if r is None:
            extra += 1
            continue
        S[int(row[0]), r] = float(row[2])
    print("matched pairs", int((S > 0).sum()), "outside snapshot", extra, flush=True)
    np.savez_compressed(path, S=S, outside_snapshot=extra)


def lexical_latency(b: Bench) -> None:
    path = CACHE / "lexical-latency.json"
    if path.exists():
        return
    env, password = pg_env()
    sample = sorted(np.random.default_rng(SEED).choice(b.n, size=100, replace=False).tolist())
    lines = [bench_q(b.queries), "\\timing on"]
    for qi in sample:
        lines.append(
            f"SELECT p.slug, max(ts_rank_cd(cc.search_vector, q.tsq)) s FROM (SELECT {TSQ} AS tsq FROM bench_q "
            f"WHERE idx = {qi}) q, content_chunks cc JOIN pages p ON p.id = cc.page_id WHERE p.deleted_at IS NULL "
            "AND q.tsq::text <> '' AND cc.search_vector @@ q.tsq GROUP BY p.slug ORDER BY s DESC, p.slug LIMIT 50;")
    out = psql_script("\n".join(lines) + "\n", env, password)
    ms = [float(m) for m in re.findall(r"Time: ([0-9.]+) ms", out)]
    ms = ms[-len(sample):]
    assert len(ms) == len(sample), len(ms)
    path.write_text(json.dumps({"sample_idx": sample, "ms": ms, "median_ms": float(np.median(ms)),
                                "p95_ms": float(np.percentile(ms, 95)), "mean_ms": float(np.mean(ms))}))
    print("lexical latency median ms", np.median(ms), flush=True)


def hk1_candidates(b: Bench) -> None:
    path = CACHE / "hk1-mask.npz"
    if path.exists():
        return
    keys = json.loads((BENCH / "chunk-keys.json").read_text())["keys"]
    karr = np.array([key_int(keys[str(c)]) if str(c) in keys else 0 for c in b.ids], dtype=np.uint64)
    has = np.array([str(c) in keys for c in b.ids])
    payload = "".join(json.dumps({"id": i, "embedding": b.Q[i].astype(float).tolist()}) + "\n" for i in range(b.n))
    t0 = time.perf_counter()
    proc = subprocess.run([str(HILBERT), "key", "--dims", "8", "--bits", "8", "--seed", "9e3779b97f4a7c15",
                           "--probe", "1", "--ranges", "16", "--format", "jsonl"],
                          input=payload, capture_output=True, text=True, check=True)
    key_s = time.perf_counter() - t0
    ranges = [None] * b.n
    for line in proc.stdout.splitlines():
        o = json.loads(line)
        ranges[int(o["id"])] = o["ranges"]
    mask = np.zeros((b.n, len(b.ids)), dtype=bool)
    t1 = time.perf_counter()
    for qi in range(b.n):
        m = np.zeros(len(b.ids), dtype=bool)
        for lo, hi in ranges[qi]:
            m |= (karr >= np.uint64(key_int(lo))) & (karr <= np.uint64(key_int(hi)))
        mask[qi] = m & has
    scan_s = time.perf_counter() - t1
    cand = mask.sum(1)
    print("hk1 median candidates", np.median(cand), flush=True)
    np.savez_compressed(path, mask=mask, key_seconds_total=key_s, scan_seconds_total=scan_s)


def partition_candidates(b: Bench) -> None:
    path = CACHE / "partition-mask.npz"
    if path.exists():
        return
    import idea06_partitions as p6
    keys = {int(k): key_int(v) for k, v in json.loads((BENCH / "chunk-keys.json").read_text())["keys"].items()}
    rows = np.array([i for i, c in enumerate(b.ids) if c in keys])
    order = rows[np.lexsort((np.array([b.ids[i] for i in rows]),
                             np.array([keys[b.ids[i]] for i in rows], dtype=np.uint64)))]
    X = b.Cn[order]
    m = 128
    labels = np.repeat(np.arange(m), [len(x) for x in np.array_split(np.arange(len(rows)), m)])
    idx = p6.Index("hilbert+soar", X, order, labels, m, True)
    mask, cand = idx.route(b.Qn, 12, len(b.ids))
    t0 = time.perf_counter()
    for qi in range(b.n):
        top = np.argsort(-(b.Qn[qi] @ idx.reps.T))[:12]
        np.concatenate([idx.members[j] for j in top])
    route_s = (time.perf_counter() - t0) / b.n
    print("partition median candidates (with duplicates)", np.median(cand), flush=True)
    np.savez_compressed(path, mask=mask, cand=cand, route_seconds_per_query=route_s, n_reps=m)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    b = Bench()
    lexical_scores(b)
    lexical_latency(b)
    hk1_candidates(b)
    partition_candidates(b)


if __name__ == "__main__":
    main()
