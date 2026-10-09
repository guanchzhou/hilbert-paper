#!/usr/bin/env python3
"""Read-only dev-qrels retrieval grid for the personal gbrain.

Writes only under this directory. Does not re-embed the corpus, does not
change search.reranker.enabled, and does not open qrels-test.json.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

csv.field_size_limit(min(sys.maxsize, 50_000_000))

import numpy as np

from metrics import estimate_tokens, mrr, ndcg_at_k, precision_at_k, recall_at_k

BENCH = Path(__file__).resolve().parent
QRELS = Path("/Users/andreymaltsev/.gbrain/eval/qrels-dev.json")
CONFIG = Path.home() / ".gbrain" / "config.json"
EMBED_URL = "http://127.0.0.1:11436/v1/embeddings"
EMBED_MODEL = "qwen3-embedding-8k"
HILBERT_PREFIX = "hk1:8:8:9e3779b97f4a7c15:"
K = 10
BUDGET = 6000
DIM = 1024

HEADING_LINE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S")
FENCE_LINE = re.compile(r"^[ \t]{0,3}(?:```|~~~)")


def pg_env() -> tuple[dict[str, str], str]:
    cfg = json.loads(CONFIG.read_text())
    parsed = urlparse(cfg["database_url"])
    password = unquote(parsed.password or "")
    env = os.environ.copy()
    env["PGHOST"] = parsed.hostname or "127.0.0.1"
    env["PGPORT"] = str(parsed.port or 5432)
    env["PGUSER"] = unquote(parsed.username or "")
    env["PGPASSWORD"] = password
    env["PGDATABASE"] = (parsed.path or "/postgres").lstrip("/") or "postgres"
    env["PGCLIENTENCODING"] = "UTF8"
    return env, password


def redact(text: str, password: str) -> str:
    if password and password in text:
        return text.replace(password, "<redacted>")
    return text


def psql(sql: str, env: dict[str, str], password: str, stdin: str | None = None) -> str:
    proc = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-X", "-q", "-At", "-c", sql],
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(redact(proc.stderr[-2000:], password) or "psql failed")
    return proc.stdout


def psql_script(script: str, env: dict[str, str], password: str) -> str:
    proc = subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-X", "-q"],
        env=env,
        input=script,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(redact(proc.stderr[-4000:], password) or "psql script failed")
    return proc.stdout


def dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def has_markdown_heading(text: str, source: str) -> bool:
    if source == "fenced_code":
        return False
    in_fence = False
    for line in text.splitlines():
        if FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and HEADING_LINE.match(line):
            return True
    return False


def section_span(chunks: list[dict], hit_i: int) -> list[int]:
    """Hit chunk plus neighbors until a markdown heading."""
    hit = chunks[hit_i]
    if has_markdown_heading(hit["text"], hit["source"]):
        start = hit_i
    else:
        start = hit_i
        j = hit_i - 1
        while j >= 0:
            start = j
            if has_markdown_heading(chunks[j]["text"], chunks[j]["source"]):
                break
            j -= 1
    end = hit_i
    j = hit_i + 1
    while j < len(chunks):
        if has_markdown_heading(chunks[j]["text"], chunks[j]["source"]):
            break
        end = j
        j += 1
    return list(range(start, end + 1))


def window_span(chunks: list[dict], hit_i: int) -> list[int]:
    lo = max(0, hit_i - 1)
    hi = min(len(chunks) - 1, hit_i + 1)
    return list(range(lo, hi + 1))


def join_chunks(chunks: list[dict], idxs: list[int]) -> str:
    return "\n".join(chunks[i]["text"] for i in idxs)


def macro(lists: list[list[str]], qrels: list[dict]) -> dict[str, float]:
    n = len(qrels)
    p = r = m = g = 0.0
    for hits, row in zip(lists, qrels):
        rel = set(row["relevant"])
        p += precision_at_k(hits, rel, K)
        r += recall_at_k(hits, rel, K)
        m += mrr(hits, rel)
        g += ndcg_at_k(hits, rel, K)
    return {"P": p / n, "R": r / n, "MRR": m / n, "nDCG": g / n}


def topk_from_scores(scores: np.ndarray, labels: list[str], k: int = K) -> list[str]:
    label_arr = np.array(labels, dtype=object)
    order = np.lexsort((label_arr, -scores))
    out: list[str] = []
    seen: set[str] = set()
    for i in order:
        slug = labels[int(i)]
        if slug in seen:
            continue
        seen.add(slug)
        out.append(slug)
        if len(out) == k:
            break
    return out


def page_token_sum(hits: list[str], truth: dict[str, str]) -> int:
    return sum(estimate_tokens(truth.get(slug, "")) for slug in hits[:K])


def write_manifest(env: dict[str, str], password: str, qrels_sha: str, n_queries: int) -> dict:
    prefix_b64 = psql(
        "SELECT encode(convert_to(value,'UTF8'),'base64') FROM config WHERE key='embedding_query_prefix';",
        env,
        password,
    ).strip()
    import base64

    prefix = base64.b64decode(prefix_b64).decode("utf-8")
    reranker = psql(
        "SELECT value FROM config WHERE key='search.reranker.enabled';",
        env,
        password,
    ).strip()
    model = psql(
        "SELECT value FROM config WHERE key='embedding_model';",
        env,
        password,
    ).strip()
    dims = int(
        psql(
            "SELECT value FROM config WHERE key='embedding_dimensions';",
            env,
            password,
        ).strip()
    )
    counts = psql(
        """
        SELECT
          (SELECT count(*) FROM pages WHERE deleted_at IS NULL),
          (SELECT count(*) FROM content_chunks cc
             JOIN pages p ON p.id = cc.page_id WHERE p.deleted_at IS NULL),
          (SELECT count(*) FROM content_chunks),
          (SELECT count(*) FROM chunk_hilbert),
          (SELECT count(*) FROM chunk_hilbert WHERE marker LIKE 'hk1:8:8:9e3779b97f4a7c15:%'),
          (SELECT count(*) FROM content_chunks WHERE model = 'ollama:qwen3-embedding-8k' AND embedding IS NOT NULL),
          (SELECT vector_dims(embedding) FROM content_chunks WHERE embedding IS NOT NULL LIMIT 1);
        """,
        env,
        password,
    ).strip().split("|")
    pages_live, chunks_live, chunks_all, hilbert_n, hilbert_ok, embedded, dim_live = (
        int(x) for x in counts
    )
    if hilbert_n != hilbert_ok:
        raise RuntimeError(f"hilbert keys outside prefix: {hilbert_n - hilbert_ok}")
    if not prefix.endswith("Query:"):
        raise RuntimeError("embedding_query_prefix does not end with Query:")
    manifest = {
        "pages_live": pages_live,
        "chunk_count": chunks_live,
        "chunks_all": chunks_all,
        "embedding_model": model,
        "embedding_model_http": EMBED_MODEL,
        "dimensions": dims,
        "embedding_column_dimensions": dim_live,
        "embedded_chunks": embedded,
        "reranker_enabled": reranker == "true",
        "query_prefix": prefix,
        "query_prefix_bytes": len(prefix.encode("utf-8")),
        "qrels": "qrels-dev.json",
        "qrels_count": n_queries,
        "qrels_dev_sha256": qrels_sha,
        "chunk_hilbert_count": hilbert_n,
        "chunk_hilbert_prefix": HILBERT_PREFIX,
        "chunk_hilbert_all_keys_match_prefix": True,
    }
    dump(BENCH / "manifest.json", manifest)
    return manifest


def embed_one(text: str) -> np.ndarray:
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    return np.asarray(data["data"][0]["embedding"], dtype=np.float32)


def embed_queries(queries: list[str], prefix: str) -> np.ndarray:
    """Prepend the config prefix with no extra separator, matching gbrain embedQuery."""
    ids_path = BENCH / "query-ids.json"
    vec_path = BENCH / "query-vectors.npy"
    if vec_path.exists() and ids_path.exists():
        saved_ids = json.loads(ids_path.read_text())
        saved = np.load(vec_path)
        if saved_ids == queries and saved.shape == (len(queries), DIM) and saved.dtype == np.float32:
            fresh = embed_one(prefix + queries[0])
            if np.max(np.abs(fresh - saved[0])) < 1e-5:
                print(f"reuse {vec_path.name} {saved.shape}", flush=True)
                return saved
            print("saved query vectors do not match the exact prefix; re-embedding", flush=True)

    out = np.zeros((len(queries), DIM), dtype=np.float32)
    batch = 32
    done = 0
    t0 = time.perf_counter()
    while done < len(queries):
        chunk = queries[done : done + batch]
        payload = {
            "model": EMBED_MODEL,
            "input": [prefix + q for q in chunk],
        }
        body = json.dumps(payload).encode()
        last_err = None
        data = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    EMBED_URL, data=body, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"embed failed at {done}: {last_err}")
        for item in data["data"]:
            vec = np.asarray(item["embedding"], dtype=np.float32)
            if vec.shape != (DIM,):
                raise RuntimeError(f"dim {vec.shape} at row {done}")
            out[done + int(item["index"])] = vec
        done += len(chunk)
        if done % 64 == 0 or done == len(queries):
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"embedded {done}/{len(queries)} ({rate:.1f}/s)", flush=True)
    ids_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2) + "\n")
    np.save(vec_path, out)
    return out


def load_corpus(env: dict[str, str], password: str):
    print("loading corpus", flush=True)
    pages_sql = """
        COPY (
          SELECT id, slug, compiled_truth
          FROM pages
          WHERE deleted_at IS NULL
          ORDER BY id
        ) TO STDOUT WITH (FORMAT csv)
    """
    chunks_sql = """
        COPY (
          SELECT cc.id, cc.page_id, p.slug, cc.chunk_index, cc.chunk_source, cc.chunk_text, cc.embedding::text
          FROM content_chunks cc
          JOIN pages p ON p.id = cc.page_id
          WHERE p.deleted_at IS NULL AND cc.embedding IS NOT NULL
          ORDER BY p.slug, cc.chunk_index, cc.id
        ) TO STDOUT WITH (FORMAT csv)
    """
    page_csv = psql(pages_sql, env, password)
    chunk_csv = psql(chunks_sql, env, password)
    truth: dict[str, str] = {}
    for row in csv.reader(io.StringIO(page_csv)):
        _pid, slug, text = row
        truth[slug] = text if slug not in truth else truth[slug] + "\n" + text

    ids: list[int] = []
    slugs: list[str] = []
    sources: list[str] = []
    texts: list[str] = []
    indexes: list[int] = []
    vecs = []
    for row in csv.reader(io.StringIO(chunk_csv)):
        cid, _pid, slug, cidx, source, text, emb = row
        ids.append(int(cid))
        slugs.append(slug)
        sources.append(source)
        texts.append(text)
        indexes.append(int(cidx))
        vecs.append(np.fromstring(emb[1:-1], sep=",", dtype=np.float32))
    matrix = np.vstack(vecs)
    if matrix.shape[1] != DIM:
        raise RuntimeError(f"corpus dim {matrix.shape}")
    print(f"corpus pages={len(truth)} chunks={matrix.shape[0]}", flush=True)
    return truth, ids, slugs, sources, texts, indexes, matrix


def vector_rankings(
    queries_norm: np.ndarray,
    matrix: np.ndarray,
    slugs: list[str],
    sources: list[str],
    texts: list[str],
    indexes: list[int],
):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    chunk_norm = matrix / norms

    # Stable page order: first-seen slug, chunks already ordered by slug, index.
    page_slugs: list[str] = []
    slices: list[tuple[int, int]] = []
    start = 0
    for i, slug in enumerate(slugs):
        if i == 0 or slug != slugs[i - 1]:
            if i:
                slices.append((start, i))
                start = i
            page_slugs.append(slug)
    slices.append((start, len(slugs)))
    n_pages = len(page_slugs)

    means = np.zeros((n_pages, DIM), dtype=np.float64)
    for i, (a, b) in enumerate(slices):
        means[i] = matrix[a:b].mean(axis=0)
    mean_norm = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    mean_norm = mean_norm.astype(np.float32)

    t0 = time.perf_counter()
    page_sim = queries_norm @ mean_norm.T
    page_lists = [topk_from_scores(page_sim[i], page_slugs) for i in range(len(queries_norm))]
    page_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    chunk_sim = queries_norm @ chunk_norm.T
    chunk_lists: list[list[str]] = []
    winners: list[list[int]] = []
    for qi in range(len(queries_norm)):
        scores = np.empty(n_pages, dtype=np.float32)
        win = np.empty(n_pages, dtype=np.int32)
        row = chunk_sim[qi]
        for pi, (a, b) in enumerate(slices):
            sl = row[a:b]
            j = int(sl.argmax())
            scores[pi] = sl[j]
            win[pi] = a + j
        label_arr = np.array(page_slugs, dtype=object)
        order = np.lexsort((label_arr, -scores))[:K]
        chunk_lists.append([page_slugs[int(i)] for i in order])
        winners.append([int(win[int(i)]) for i in order])
    chunk_seconds = time.perf_counter() - t1

    by_page: dict[str, list[dict]] = {}
    pos_of: dict[int, tuple[str, int]] = {}
    for i, slug in enumerate(slugs):
        rec = {
            "text": texts[i],
            "source": sources[i],
            "chunk_index": indexes[i],
            "row": i,
        }
        bucket = by_page.setdefault(slug, [])
        pos_of[i] = (slug, len(bucket))
        bucket.append(rec)
    return page_lists, page_seconds, chunk_lists, chunk_seconds, winners, by_page, pos_of


def keyword_rankings(queries: list[str], env: dict[str, str], password: str):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for i, q in enumerate(queries):
        writer.writerow([i, q])
    csv_body = buf.getvalue()
    script = f"""
SET statement_timeout = 0;
CREATE TEMP TABLE bench_q (idx int PRIMARY KEY, query text);
COPY bench_q (idx, query) FROM STDIN WITH (FORMAT csv);
{csv_body}\\.
CREATE TEMP TABLE page_tv AS
SELECT id, slug, to_tsvector('english', compiled_truth) AS tv
FROM pages
WHERE deleted_at IS NULL;
CREATE INDEX page_tv_gin ON page_tv USING gin (tv);
ANALYZE page_tv;
ANALYZE bench_q;
\\copy (WITH qq AS (SELECT idx, websearch_to_tsquery('english', query) AS tsq FROM bench_q) SELECT qq.idx, h.slug FROM qq CROSS JOIN LATERAL (SELECT slug, ts_rank_cd(tv, qq.tsq) AS score FROM page_tv WHERE qq.tsq::text <> '' AND tv @@ qq.tsq ORDER BY score DESC, slug ASC LIMIT {K}) h ORDER BY qq.idx, h.score DESC, h.slug ASC) TO STDOUT WITH (FORMAT csv)
"""
    # The page and chunk cells are timed separately. Page includes the
    # one-time compiled_truth tsvector build, because that vector is not stored.
    t0 = time.perf_counter()
    page_out = psql_script(script, env, password)
    page_seconds = time.perf_counter() - t0

    script_chunk = f"""
SET statement_timeout = 0;
CREATE TEMP TABLE bench_q (idx int PRIMARY KEY, query text);
COPY bench_q (idx, query) FROM STDIN WITH (FORMAT csv);
{csv_body}\\.
\\copy (WITH qq AS (SELECT idx, websearch_to_tsquery('english', query) AS tsq FROM bench_q) SELECT qq.idx, h.slug FROM qq CROSS JOIN LATERAL (SELECT p.slug, max(ts_rank_cd(cc.search_vector, qq.tsq)) AS score FROM content_chunks cc JOIN pages p ON p.id = cc.page_id WHERE p.deleted_at IS NULL AND qq.tsq::text <> '' AND cc.search_vector @@ qq.tsq GROUP BY p.slug ORDER BY score DESC, p.slug ASC LIMIT {K}) h ORDER BY qq.idx, h.score DESC, h.slug ASC) TO STDOUT WITH (FORMAT csv)
"""
    t1 = time.perf_counter()
    chunk_out = psql_script(script_chunk, env, password)
    chunk_seconds = time.perf_counter() - t1

    def parse(text: str) -> list[list[str]]:
        lists: list[list[str]] = [[] for _ in queries]
        # psql may print a COPY count line; keep csv rows only.
        for row in csv.reader(io.StringIO(text)):
            if len(row) != 2:
                continue
            if not row[0].isdigit():
                continue
            lists[int(row[0])].append(row[1])
        return lists

    return parse(page_out), page_seconds, parse(chunk_out), chunk_seconds


def cell_record(name: str, lists: list[list[str]], qrels: list[dict], seconds: float, truth: dict[str, str]) -> dict:
    stats = macro(lists, qrels)
    tokens = [page_token_sum(hits, truth) for hits in lists]
    return {
        "cell": name,
        "k": K,
        "n": len(qrels),
        "P": round(stats["P"], 6),
        "R": round(stats["R"], 6),
        "MRR": round(stats["MRR"], 6),
        "nDCG": round(stats["nDCG"], 6),
        "seconds": round(seconds, 3),
        "tokens": round(sum(tokens) / len(tokens), 3),
        "tokens_total": int(sum(tokens)),
    }


def pack_units(qrels, winners, by_page, pos_of, truth):
    unit_names = ("chunk", "window", "section", "page")
    per = {name: [] for name in unit_names}
    for qi, win_rows in enumerate(winners):
        rel = set(qrels[qi]["relevant"])
        built = {name: [] for name in unit_names}
        for rank, row in enumerate(win_rows):
            slug, hit_i = pos_of[row]
            chunks = by_page[slug]
            texts = {
                "chunk": chunks[hit_i]["text"],
                "window": join_chunks(chunks, window_span(chunks, hit_i)),
                "section": join_chunks(chunks, section_span(chunks, hit_i)),
                "page": truth.get(slug, ""),
            }
            for name in unit_names:
                built[name].append(
                    {"slug": slug, "rank": rank, "tokens": estimate_tokens(texts[name]), "relevant": slug in rel}
                )
        for name in unit_names:
            kept_tokens = 0
            kept_slugs = []
            for item in built[name]:
                if kept_tokens + item["tokens"] <= BUDGET:
                    kept_tokens += item["tokens"]
                    kept_slugs.append(item["slug"])
            per[name].append(
                {
                    "tokens": kept_tokens,
                    "dropped": len(built[name]) - len(kept_slugs),
                    "offered": len(built[name]),
                    "success": any(slug in rel for slug in kept_slugs),
                }
            )
    out = {}
    n = len(qrels)
    for name in unit_names:
        rows = per[name]
        successes = [r for r in rows if r["success"]]
        delivered = sum(r["tokens"] for r in rows)
        record = {
            "unit": name,
            "budget": BUDGET,
            "source_ranking": "cell-vector-chunk",
            "queries": n,
            "tokens_delivered": round(delivered / n, 3),
            "tokens_delivered_total": int(delivered),
            "hits_dropped": round(sum(r["dropped"] for r in rows) / n, 3),
            "hits_dropped_total": int(sum(r["dropped"] for r in rows)),
            "relevant_in_pack": round(len(successes) / n, 6),
            "tokens_per_successful_query": (
                round(sum(r["tokens"] for r in successes) / len(successes), 3) if successes else None
            ),
            "successful_queries": len(successes),
        }
        out[name] = record
    return out


def write_section_rule() -> None:
    dump(
        BENCH / "section-rule.json",
        {
            "definition": "A section is the hit chunk plus neighboring chunks on the same page until a markdown heading.",
            "order": "Chunks on one page sorted by chunk_index.",
            "heading": "An ATX heading line matching ^[ \\t]{0,3}#{1,6}[ \\t]+\\S. Lines inside a ``` or ~~~ fence do not count. fenced_code chunks are never heading boundaries.",
            "left": "If the hit chunk contains a heading, it opens the section and earlier chunks are excluded. Otherwise include preceding chunks through the nearest chunk that contains a heading.",
            "right": "Include following chunks that contain no heading. Stop before the next heading chunk, without including it.",
            "window": "The hit chunk plus one neighbor on each side in chunk_index order.",
            "chunk": "The winning chunk_text only.",
            "page": "pages.compiled_truth for that slug.",
            "pack": "Walk the chunk-vector top 10 in rank order. Keep a unit when estimate_tokens of its text fits in the remaining 6000-token budget. A unit that does not fit is dropped and later units are still considered. Units are not truncated.",
            "tokens": "ceil((len-cjk)/4)+cjk via metrics.estimate_tokens.",
        },
    )


def main() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("漢字") == 2
    assert estimate_tokens("あいうえお") == 5
    assert abs(ndcg_at_k(["a", "b"], {"a"}, 10) - 1.0) < 1e-12
    assert abs(ndcg_at_k(["b", "a"], {"a"}, 10) - (1 / __import__("math").log2(3))) < 1e-12

    env, password = pg_env()
    raw = QRELS.read_bytes()
    qrels_sha = hashlib.sha256(raw).hexdigest()
    qrels = json.loads(raw)
    if len(qrels) != 817:
        raise RuntimeError(f"expected 817 dev queries, got {len(qrels)}")
    queries = [row["query"] for row in qrels]
    manifest = write_manifest(env, password, qrels_sha, len(queries))
    write_section_rule()
    print(
        f"manifest pages={manifest['pages_live']} chunks={manifest['chunk_count']} "
        f"hilbert={manifest['chunk_hilbert_count']} prefix_bytes={manifest['query_prefix_bytes']} "
        f"reranker={manifest['reranker_enabled']}",
        flush=True,
    )

    vectors = embed_queries(queries, manifest["query_prefix"])
    if vectors.shape != (817, DIM) or vectors.dtype != np.float32:
        raise RuntimeError(f"query vectors {vectors.shape} {vectors.dtype}")

    truth, _ids, slugs, sources, texts, indexes, matrix = load_corpus(env, password)
    qnorm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    page_lists, page_s, chunk_lists, chunk_s, winners, by_page, pos_of = vector_rankings(
        qnorm.astype(np.float32), matrix, slugs, sources, texts, indexes
    )
    missing = sorted({slug for row in qrels for slug in row["relevant"] if slug not in truth})
    print(f"qrel slugs missing from live pages: {len(missing)}", flush=True)

    kw_page, kw_page_s, kw_chunk, kw_chunk_s = keyword_rankings(queries, env, password)

    cells = {
        "keyword-page": cell_record("keyword-page", kw_page, qrels, kw_page_s, truth),
        "keyword-chunk": cell_record("keyword-chunk", kw_chunk, qrels, kw_chunk_s, truth),
        "vector-page": cell_record("vector-page", page_lists, qrels, page_s, truth),
        "vector-chunk": cell_record("vector-chunk", chunk_lists, qrels, chunk_s, truth),
    }
    names = {
        "keyword-page": "cell-keyword-page.json",
        "keyword-chunk": "cell-keyword-chunk.json",
        "vector-page": "cell-vector-page.json",
        "vector-chunk": "cell-vector-chunk.json",
    }
    for key, filename in names.items():
        dump(BENCH / filename, cells[key])
        print(filename, cells[key], flush=True)
    dump(
        BENCH / "retrieval-grid.json",
        {
            "k": K,
            "n": len(qrels),
            "qrels": "qrels-dev.json",
            "qrels_dev_sha256": qrels_sha,
            "keyword": "websearch_to_tsquery('english') AND, ts_rank_cd, page collapse by max score",
            "vector_page": "cosine vs mean of the page's chunk embeddings",
            "vector_chunk": "cosine vs every live chunk; best chunk wins the page",
            "tokens": "mean over queries of estimate_tokens(compiled_truth) summed over the top-10 pages",
            "cells": cells,
        },
    )

    packed = pack_units(qrels, winners, by_page, pos_of, truth)
    file_for = {
        "chunk": "unit-chunk.json",
        "window": "unit-window.json",
        "section": "unit-section.json",
        "page": "unit-page.json",
    }
    for key, filename in file_for.items():
        dump(BENCH / filename, packed[key])
        print(filename, packed[key], flush=True)
    dump(BENCH / "evidence-units.json", {"budget": BUDGET, "source_ranking": "cell-vector-chunk", "units": packed})
    print("done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise
