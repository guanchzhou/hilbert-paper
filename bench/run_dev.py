#!/usr/bin/env python3
"""Development-set cells for the knowledge-search bench. Does not open qrels-test."""

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from metrics import estimate_tokens, summarize

BENCH = Path(__file__).resolve().parent
DEV = Path.home() / ".gbrain/eval/qrels-dev.json"
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
PREFIX_KEY = "hk1:8:8:9e3779b97f4a7c15:"


def password() -> str:
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    return urllib.parse.urlparse(cfg["database_url"]).password or ""


def psql(sql: str) -> str:
    env = {**os.environ, "PGPASSWORD": password()}
    proc = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-c", sql],
        check=True, capture_output=True, text=True, env=env,
    )
    return proc.stdout


def psql_at(sql: str) -> str:
    env = {**os.environ, "PGPASSWORD": password()}
    proc = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-qAt", "-c", sql],
        check=True, capture_output=True, text=True, env=env,
    )
    return proc.stdout


def dump(name: str, obj) -> None:
    (BENCH / name).write_text(json.dumps(obj, indent=2) + "\n")


def embed_batch(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": "qwen3-embedding-8k", "input": texts}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11436/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    data["data"].sort(key=lambda d: d["index"])
    return [row["embedding"] for row in data["data"]]


def keyword_page(query: str) -> list[str]:
    q = query.replace("'", " ")
    sql = f"""
    SELECT p.slug FROM pages p
    WHERE p.deleted_at IS NULL
      AND to_tsvector('simple', coalesce(p.compiled_truth,'')) @@ plainto_tsquery('simple', '{q}')
    ORDER BY ts_rank(to_tsvector('simple', coalesce(p.compiled_truth,'')), plainto_tsquery('simple', '{q}')) DESC
    LIMIT 10
    """
    return [line for line in psql_at(sql).splitlines() if line]


def keyword_chunk(query: str) -> list[str]:
    q = query.replace("'", " ")
    sql = f"""
    SELECT p.slug
    FROM content_chunks c
    JOIN pages p ON p.id = c.page_id
    WHERE p.deleted_at IS NULL
      AND c.search_vector @@ plainto_tsquery('simple', '{q}')
    GROUP BY p.slug
    ORDER BY max(ts_rank(c.search_vector, plainto_tsquery('simple', '{q}'))) DESC
    LIMIT 10
    """
    return [line for line in psql_at(sql).splitlines() if line]


def main() -> None:
    qrels = json.loads(DEV.read_text())
    assert len(qrels) == 817, len(qrels)
    prefix = psql_at("SELECT value FROM config WHERE key = 'embedding_query_prefix'").splitlines()[0]
    # psql -At may turn the newline into a real newline and split the value.
    raw = psql_at("SELECT value FROM config WHERE key = 'embedding_query_prefix'")
    prefix = raw if raw.endswith("\n") else raw + "\n"

    pages = int(psql_at("SELECT count(*) FROM pages WHERE deleted_at IS NULL AND page_kind = 'markdown'").strip())
    chunks = int(psql_at("SELECT count(*) FROM content_chunks c JOIN pages p ON p.id=c.page_id WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL").strip())
    keys = int(psql_at("SELECT count(*) FROM chunk_hilbert WHERE marker LIKE 'hk1:8:8:9e3779b97f4a7c15:%'").strip())
    rerank = psql_at("SELECT value FROM config WHERE key = 'search.reranker.enabled'").strip()
    ver = subprocess.run([str(HILBERT), "version"], capture_output=True, text=True).stdout.strip()
    sample = psql_at("SELECT marker FROM chunk_hilbert LIMIT 1").strip()
    check = subprocess.run([str(HILBERT), "check", sample], capture_output=True, text=True)
    dump("manifest.json", {
        "pages": pages,
        "chunks_with_vectors": chunks,
        "chunk_hilbert_rows": keys,
        "model": "ollama:qwen3-embedding-8k",
        "dimensions": 1024,
        "reranker": rerank,
        "query_prefix": prefix,
        "qrels_dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(),
        "qrels_dev_n": 817,
        "zig_hilbert": ver,
        "hilbert_check_ok": check.returncode == 0,
    })
    print("manifest", pages, chunks, keys, "check", check.returncode, flush=True)

    npy = BENCH / "query-vectors.npy"
    if npy.exists():
        Q = np.load(npy)
    else:
        texts = [prefix + q["query"] for q in qrels]
        rows = []
        for i in range(0, len(texts), 32):
            rows.extend(embed_batch(texts[i:i + 32]))
            print("embedded", len(rows), flush=True)
        Q = np.asarray(rows, dtype=np.float32)
        np.save(npy, Q)
    (BENCH / "query-ids.json").write_text(json.dumps([q["query"] for q in qrels]))
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    qn[qn == 0] = 1
    Qn = Q / qn

    print("loading chunks", flush=True)
    blob = psql_at("""
    SELECT c.id::text || E'\\x1f' || p.slug || E'\\x1f' || c.chunk_index::text
      || E'\\x1f' || c.embedding::text || E'\\x1f'
      || replace(replace(c.chunk_text, E'\\n', ' '), E'\\x1f', ' ')
    FROM content_chunks c
    JOIN pages p ON p.id = c.page_id
    WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL
    """)
    ids, slugs, indexes, vecs, texts_c = [], [], [], [], []
    for line in blob.splitlines():
        cid, slug, idx, vec, text = line.split("\x1f", 4)
        ids.append(int(cid))
        slugs.append(slug)
        indexes.append(int(idx))
        vecs.append(json.loads(vec))
        texts_c.append(text)
    E = np.asarray(vecs, dtype=np.float32)
    en = np.linalg.norm(E, axis=1, keepdims=True)
    en[en == 0] = 1
    En = E / en
    print("chunks loaded", len(ids), flush=True)

    # page means
    from collections import defaultdict
    groups = defaultdict(list)
    for i, slug in enumerate(slugs):
        groups[slug].append(i)
    page_slugs = list(groups)
    means = np.stack([E[groups[s]].mean(axis=0) for s in page_slugs])
    mn = np.linalg.norm(means, axis=1, keepdims=True)
    mn[mn == 0] = 1
    Mn = means / mn

    def rank_cosine_chunks():
        lists = []
        winners = []
        t0 = time.perf_counter()
        for qi in range(len(Qn)):
            scores = En @ Qn[qi]
            order = np.argpartition(-scores, 80)[:80]
            order = order[np.argsort(-scores[order])]
            seen, pages, win = set(), [], None
            for j in order:
                s = slugs[j]
                if s in seen:
                    continue
                seen.add(s)
                pages.append(s)
                if win is None:
                    win = j
                if len(pages) >= 10:
                    break
            lists.append(pages)
            winners.append(win if win is not None else int(order[0]))
        return lists, winners, time.perf_counter() - t0

    def rank_cosine_pages():
        lists = []
        t0 = time.perf_counter()
        for qi in range(len(Qn)):
            scores = Mn @ Qn[qi]
            order = np.argsort(-scores)[:10]
            lists.append([page_slugs[int(j)] for j in order])
        return lists, time.perf_counter() - t0

    print("vector chunk", flush=True)
    vchunk, winners, t_chunk = rank_cosine_chunks()
    print("vector page", flush=True)
    vpage, t_page = rank_cosine_pages()
    print(summarize(vchunk, qrels), "seconds", round(t_chunk, 2), flush=True)

    print("keyword", flush=True)
    t0 = time.perf_counter()
    kpage = [keyword_page(q["query"]) for q in qrels]
    t_kpage = time.perf_counter() - t0
    t0 = time.perf_counter()
    kchunk = [keyword_chunk(q["query"]) for q in qrels]
    t_kchunk = time.perf_counter() - t0

    def page_tokens(slug_list: list[str]) -> int:
        total = 0
        for s in slug_list:
            if s not in groups:
                continue
            total += estimate_tokens(texts_c[groups[s][0]])
        return total

    def cell(lists, seconds):
        out = summarize(lists, qrels)
        out["seconds"] = round(seconds, 3)
        out["top10_text_tokens_mean"] = round(
            statistics.mean(page_tokens(h) for h in lists), 1
        )
        return out

    dump("cell-keyword-page.json", cell(kpage, t_kpage))
    dump("cell-keyword-chunk.json", cell(kchunk, t_kchunk))
    dump("cell-vector-page.json", cell(vpage, t_page))
    dump("cell-vector-chunk.json", cell(vchunk, t_chunk))
    dump("retrieval-grid.json", {
        "keyword_page": json.loads((BENCH / "cell-keyword-page.json").read_text()),
        "keyword_chunk": json.loads((BENCH / "cell-keyword-chunk.json").read_text()),
        "vector_page": json.loads((BENCH / "cell-vector-page.json").read_text()),
        "vector_chunk": json.loads((BENCH / "cell-vector-chunk.json").read_text()),
    })
    dump("section-rule.json", {
        "rule": "A section is the winning chunk plus later chunks of the same page until a chunk whose text starts with a markdown heading."
    })

    by_page = {s: sorted(ix, key=lambda i: indexes[i]) for s, ix in groups.items()}

    def pack(kind: str) -> dict:
        delivered = 0
        dropped = 0
        survived = 0
        for q, pages, win in zip(qrels, vchunk, winners):
            rel = set(q["relevant"])
            used = 0
            got = False
            for rank, slug in enumerate(pages):
                if kind == "chunk" and rank == 0:
                    piece = texts_c[win]
                elif kind == "window" and rank == 0:
                    idxs = by_page.get(slug, [])
                    here = indexes[win]
                    piece = " ".join(texts_c[i] for i in idxs if abs(indexes[i] - here) <= 1)
                elif kind == "section" and rank == 0:
                    idxs = by_page.get(slug, [])
                    start = next((n for n, i in enumerate(idxs) if i == win), 0)
                    parts = []
                    for i in idxs[start:]:
                        if parts and texts_c[i].lstrip().startswith("#"):
                            break
                        parts.append(texts_c[i])
                    piece = " ".join(parts)
                elif kind == "page":
                    idxs = by_page.get(slug, [])
                    piece = " ".join(texts_c[i] for i in idxs)
                else:
                    if kind != "page":
                        continue
                    piece = ""
                cost = estimate_tokens(piece)
                if used + cost > 6000:
                    dropped += 1
                    continue
                used += cost
                if slug in rel:
                    got = True
            delivered += used
            if got:
                survived += 1
        n = len(qrels)
        return {
            "unit": kind,
            "budget": 6000,
            "tokens_delivered_mean": round(delivered / n, 1),
            "hits_dropped": dropped,
            "relevant_page_survived": round(survived / n, 4),
            "tokens_per_success": round(delivered / survived, 1) if survived else None,
        }

    units = [pack(k) for k in ("chunk", "window", "section", "page")]
    for u in units:
        dump(f"unit-{u['unit']}.json", u)
    dump("evidence-units.json", {"units": units})
    print("units", units, flush=True)

    # keys
    page_keys = {}
    for line in psql_at("SELECT slug, frontmatter->>'hilbert' FROM pages WHERE deleted_at IS NULL AND frontmatter ? 'hilbert'").splitlines():
        slug, marker = line.split("|", 1) if "|" in line else line.split("\t", 1)
        page_keys[slug] = marker.strip()
    dump("page-keys.json", page_keys)
    chunk_keys = {}
    for line in psql_at("SELECT chunk_id::text || '|' || marker FROM chunk_hilbert").splitlines():
        cid, marker = line.split("|", 1)
        chunk_keys[int(cid)] = marker.strip()
    dump("chunk-keys.json", {str(k): v for k, v in chunk_keys.items()})

    def key_int(marker: str) -> int:
        return int(marker.rsplit(":", 1)[1], 16)

    def shared(a: int, b: int) -> int:
        diff = a ^ b
        if diff == 0:
            return 64
        return 64 - diff.bit_length()

    def neighbourhood(kind: str):
        # sample 80 queries: prefix bits of the true best cosine hit
        rows = []
        step = max(1, len(qrels) // 80)
        for qi in range(0, len(qrels), step):
            scores = (Mn if kind == "page" else En) @ Qn[qi]
            best = int(np.argmax(scores))
            if kind == "page":
                marker = page_keys.get(page_slugs[best])
            else:
                marker = chunk_keys.get(ids[best])
            if not marker:
                continue
            rows.append({"cosine": float(scores[best]), "shared_with_self": 64, "slug": page_slugs[best] if kind == "page" else slugs[best]})
        return {"n": len(rows), "note": "self-neighbour is 64 bits by construction; see probe files for retrieval", "sample": rows[:5]}

    dump("neighbour-page.json", neighbourhood("page"))
    dump("neighbour-chunk.json", neighbourhood("chunk"))

    # probes via zig-hilbert
    payload = "".join(
        json.dumps({"id": i, "embedding": Q[i].tolist()}) + "\n" for i in range(len(Q))
    )
    key_arr = np.array([key_int(chunk_keys[i]) if i in chunk_keys else 0 for i in ids], dtype=np.uint64)

    def probes(level: int, ranges: int):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(HILBERT), "key", "--probe", str(level), "--ranges", str(ranges), "--format", "jsonl"],
            input=payload, capture_output=True, text=True, check=True,
        )
        scan = time.perf_counter() - t0
        lists = []
        sizes = []
        for line in proc.stdout.splitlines():
            obj = json.loads(line)
            qi = int(obj["id"]) if not isinstance(obj["id"], int) else obj["id"]
            mask = np.zeros(len(ids), dtype=bool)
            for lo, hi in obj["ranges"]:
                a = np.uint64(key_int(lo))
                b = np.uint64(key_int(hi))
                mask |= (key_arr >= a) & (key_arr <= b)
            idx = np.flatnonzero(mask)
            sizes.append(int(idx.size))
            if idx.size == 0:
                lists.append([])
                continue
            scores = En[idx] @ Qn[qi]
            order = np.argsort(-scores)
            seen, pages = set(), []
            for j in order:
                s = slugs[idx[j]]
                if s in seen:
                    continue
                seen.add(s)
                pages.append(s)
                if len(pages) >= 10:
                    break
            lists.append(pages)
        # align lists to queries: zig-hilbert json id is the input id
        # stdout order matches input order
        out = summarize(lists, qrels)
        out.update({
            "level": level,
            "ranges": ranges,
            "candidates_median": statistics.median(sizes) if sizes else 0,
            "candidates_mean": statistics.mean(sizes) if sizes else 0,
            "scan_seconds": round(scan, 3),
        })
        return out

    sweep = []
    for level in (0, 1, 2, 3):
        for ranges in (1, 4, 8, 16):
            print("probe", level, ranges, flush=True)
            cellp = probes(level, ranges)
            dump(f"probe-{level}-{ranges}.json", cellp)
            sweep.append(cellp)
    cosine = summarize(vchunk, qrels)
    dump("hilbert-sweep.json", {
        "cosine": cosine,
        "cells": sweep,
        "label_not_index": all(c["R@10"] < cosine["R@10"] - 0.02 for c in sweep),
        "pilot_level1_8": next(c for c in sweep if c["level"] == 1 and c["ranges"] == 8),
        "pilot_level2_8": next(c for c in sweep if c["level"] == 2 and c["ranges"] == 8),
    })
    print("done", flush=True)


if __name__ == "__main__":
    main()
