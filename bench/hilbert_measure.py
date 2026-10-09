#!/usr/bin/env python3
"""Hilbert neighbourhood and probe sweep on the dev qrels.

Reads qrels-dev.json only. Does not write pages, content_chunks, or Arango.
"""

import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from metrics import summarize

BENCH = Path(__file__).resolve().parent
DEV = Path.home() / ".gbrain/eval/qrels-dev.json"
HILBERT = Path.home() / "Development/zig-hilbert/zig-out/bin/zig-hilbert"
PREFIX = "hk1:8:8:9e3779b97f4a7c15:"
PILOT = {(1, 8): 0.243, (2, 8): 0.005}
COSINE_RECALL = 0.698


def log(msg: str) -> None:
    print(msg, flush=True)


def db_password() -> str:
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    return urllib.parse.urlparse(cfg["database_url"]).password or ""


def psql_env() -> dict:
    return {**os.environ, "PGPASSWORD": db_password()}


def psql_out(args: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True, env=psql_env(), **kw)
    if proc.returncode != 0:
        err = proc.stderr.replace(db_password(), "***")
        raise RuntimeError(err[-800:])
    return proc


class Psql:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres",
             "-X", "-q", "-v", "ON_ERROR_STOP=1", "-At"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=psql_env(), bufsize=1,
        )

    def query(self, sql: str) -> list[str]:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(sql.strip().rstrip(";") + ";\n\\echo __END__\n")
        self.proc.stdin.flush()
        rows = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                err = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(err.replace(db_password(), "***")[-800:])
            if line.startswith("__END__"):
                return rows
            line = line.rstrip("\n")
            if line:
                rows.append(line)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def config_prefix() -> str:
    raw = psql_out([
        "psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-qAt",
        "-c", "SELECT encode(convert_to(value, 'UTF8'), 'hex') FROM config WHERE key = 'embedding_query_prefix'",
    ]).stdout.strip()
    return bytes.fromhex(raw).decode("utf-8")


def embed_batch(texts: list[str]) -> list[list[float]]:
    body = json.dumps({"model": "qwen3-embedding-8k", "input": texts}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11436/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
            data["data"].sort(key=lambda d: d["index"])
            return [row["embedding"] for row in data["data"]]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"embed failed: {last}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_queries(prefix: str) -> tuple[list[dict], np.ndarray]:
    qrels = json.loads(DEV.read_text())
    if len(qrels) != 817:
        raise SystemExit(f"qrels-dev has {len(qrels)} queries")
    ids_path = BENCH / "query-ids.json"
    npy_path = BENCH / "query-vectors.npy"
    ids = [q["query"] for q in qrels]
    ids_path.write_text(json.dumps(ids))
    have = npy_path.exists()
    Q = np.load(npy_path) if have else None
    probe = embed_batch([prefix + qrels[0]["query"]])[0]
    saved_cos = cosine(Q[0], np.asarray(probe, dtype=np.float32)) if Q is not None and Q.shape == (817, 1024) else 0.0
    log(f"saved query0 vs exact prefix cosine {saved_cos:.6f}")
    if saved_cos < 0.999:
        log("re-embedding 817 queries with the exact config prefix")
        rows: list[list[float]] = []
        texts = [prefix + q["query"] for q in qrels]
        for i in range(0, len(texts), 32):
            rows.extend(embed_batch(texts[i:i + 32]))
            log(f"embedded {len(rows)}")
        Q = np.asarray(rows, dtype=np.float32)
        if Q.shape != (817, 1024):
            raise SystemExit(f"bad embedding shape {Q.shape}")
        np.save(npy_path, Q)
        check = cosine(Q[0], np.asarray(probe, dtype=np.float32))
        log(f"rewritten query0 cosine {check:.6f}")
        if check < 0.999:
            raise SystemExit("rewritten vectors do not match the prefix")
    assert Q is not None
    return qrels, Q


def load_chunks() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str]]:
    sql = """
    COPY (
      SELECT c.id, p.slug, COALESCE(h.marker, ''), c.embedding::text, c.chunk_text
      FROM content_chunks c
      JOIN pages p ON p.id = c.page_id
      LEFT JOIN chunk_hilbert h
        ON h.chunk_id = c.id AND h.marker LIKE 'hk1:8:8:9e3779b97f4a7c15:%'
      WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL
    ) TO STDOUT WITH (FORMAT csv, FORCE_QUOTE *)
    """
    proc = psql_out(["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-q", "-c", sql])
    from metrics import estimate_tokens
    ids, slugs, tokens, markers, vecs = [], [], [], [], []
    for row in csv.reader(io.StringIO(proc.stdout)):
        cid, slug, marker, vec, text = row
        ids.append(int(cid))
        slugs.append(slug)
        tokens.append(estimate_tokens(text))
        markers.append(marker)
        vecs.append(np.fromstring(vec.strip()[1:-1], sep=",", dtype=np.float32))
    E = np.vstack(vecs)
    if E.shape[1] != 1024:
        raise SystemExit(f"bad chunk dim {E.shape}")
    log(f"chunks {E.shape[0]} keyed {sum(bool(m) for m in markers)}")
    return E, slugs, np.asarray(ids), np.asarray(tokens, dtype=np.int32), markers


def unit(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1
    return E / n


def key_int(marker: str) -> int:
    return int(marker.rsplit(":", 1)[1], 16)


def shared_bits(a: int, b: int) -> int:
    diff = a ^ b
    if diff == 0:
        return 64
    return 64 - diff.bit_length()


def shared_bits_u64(diff: np.ndarray) -> np.ndarray:
    out = np.empty(diff.shape, dtype=np.int16)
    zero = diff == np.uint64(0)
    out[zero] = 64
    d = diff[~zero]
    hi = (d >> np.uint64(32)).astype(np.uint32)
    lo = (d & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    def bl32(v: np.ndarray) -> np.ndarray:
        b = np.zeros(v.shape, dtype=np.int16)
        nz = v > 0
        b[nz] = np.floor(np.log2(v[nz].astype(np.float64))).astype(np.int16) + 1
        return b

    out[~zero] = 64 - np.where(hi > 0, bl32(hi) + 32, bl32(lo))
    return out


def hilbert_keys(vectors: np.ndarray) -> list[str]:
    payload = "".join(
        json.dumps({"embedding": row.astype(float).tolist()}) + "\n"
        for row in vectors
    )
    proc = subprocess.run(
        [str(HILBERT), "key", "--dims", "8", "--bits", "8", "--seed", "9e3779b97f4a7c15", "--format", "jsonl"],
        input=payload, capture_output=True, text=True, check=True,
    )
    markers = [json.loads(line)["marker"] for line in proc.stdout.splitlines() if line]
    if len(markers) != len(vectors):
        raise SystemExit("hilbert key count mismatch")
    return markers


def check_shared_against_binary(markers: list[str]) -> None:
    if len(markers) < 2:
        return
    a, b = markers[0], markers[1]
    proc = subprocess.run([str(HILBERT), "similar", a, b], capture_output=True, text=True, check=True)
    got = int(proc.stdout.strip())
    mine = shared_bits(key_int(a), key_int(b))
    if got != mine:
        raise SystemExit(f"shared bits {mine} != zig-hilbert {got}")
    log(f"shared-bits check {got}")


def rank_pages(scores: np.ndarray, page_index: np.ndarray, page_slugs: list[str], k: int = 10) -> list[str]:
    best = np.full(len(page_slugs), -np.inf, dtype=np.float64)
    np.maximum.at(best, page_index, scores)
    present = np.isfinite(best)
    if not present.any():
        return []
    idx = np.flatnonzero(present)
    take = min(k, idx.size)
    top = idx[np.argpartition(-best[idx], take - 1)[:take]]
    top = top[np.argsort(-best[top])]
    return [page_slugs[int(i)] for i in top]


def neighbourhood(name: str, En: np.ndarray, keys: np.ndarray) -> dict:
    n = En.shape[0]
    nn_bits = np.empty(n, dtype=np.int16)
    nn_cos = np.empty(n, dtype=np.float32)
    block = 256
    for i in range(0, n, block):
        j = min(i + block, n)
        sims = En[i:j] @ En.T
        for row, src in enumerate(range(i, j)):
            sims[row, src] = -2
        dest = np.argmax(sims, axis=1)
        nn_cos[i:j] = sims[np.arange(j - i), dest]
        diff = np.bitwise_xor(keys[i:j], keys[dest])
        nn_bits[i:j] = shared_bits_u64(diff)
    # all unordered pairs, binned
    counts = np.zeros(65, dtype=np.int64)
    sums = np.zeros(65, dtype=np.float64)
    samples: dict[int, list[float]] = defaultdict(list)
    rng = np.random.default_rng(1)
    pair_bits = []
    pair_cos = []
    for i in range(n - 1):
        cos = En[i + 1:] @ En[i]
        bits = shared_bits_u64(np.bitwise_xor(keys[i + 1:], np.uint64(keys[i]))).astype(np.int64)
        counts += np.bincount(bits, minlength=65)
        sums += np.bincount(bits, weights=cos.astype(np.float64), minlength=65)
        take = rng.random(cos.size) < 0.02
        if take.any():
            for b, cval in zip(bits[take].tolist(), cos[take].tolist()):
                if len(samples[b]) < 4000:
                    samples[b].append(float(cval))
        if len(pair_bits) < 2000:
            pick = int(rng.integers(0, cos.size))
            pair_bits.append(int(bits[pick]))
            pair_cos.append(float(cos[pick]))
    by_bits = []
    for b in range(65):
        if counts[b] == 0:
            continue
        vals = samples[b]
        by_bits.append({
            "bits": b,
            "n": int(counts[b]),
            "mean_cosine": sums[b] / counts[b],
            "median_cosine": float(statistics.median(vals)) if vals else None,
            "median_from_sample": bool(len(vals) < int(counts[b])),
        })
    return {
        "kind": name,
        "n_items": n,
        "n_pairs": int(counts.sum()),
        "nearest_neighbor": {
            "mean_shared_bits": float(nn_bits.mean()),
            "median_shared_bits": float(np.median(nn_bits)),
            "mean_cosine": float(nn_cos.mean()),
            "median_cosine": float(np.median(nn_cos)),
            "shared_bits_histogram": {str(i): int((nn_bits == i).sum()) for i in range(65) if (nn_bits == i).any()},
        },
        "by_shared_bits": by_bits,
        "sample_pairs": [{"bits": b, "cosine": c} for b, c in zip(pair_bits, pair_cos)],
    }


def probe_ranges(Q: np.ndarray, level: int, ranges: int) -> list[list[tuple[str, str]]]:
    payload = "".join(json.dumps({"id": i, "embedding": Q[i].astype(float).tolist()}) + "\n" for i in range(len(Q)))
    proc = subprocess.run(
        [str(HILBERT), "key", "--dims", "8", "--bits", "8", "--seed", "9e3779b97f4a7c15",
         "--probe", str(level), "--ranges", str(ranges), "--format", "jsonl"],
        input=payload, capture_output=True, text=True, check=True,
    )
    out: list[list[tuple[str, str]] | None] = [None] * len(Q)
    for line in proc.stdout.splitlines():
        obj = json.loads(line)
        qi = int(obj["id"])
        out[qi] = [(lo, hi) for lo, hi in obj["ranges"]]
    if any(r is None for r in out):
        raise SystemExit(f"missing probe rows level {level} ranges {ranges}")
    return out  # type: ignore[return-value]


def sql_scan_counts(db: Psql, ranges: list[list[tuple[str, str]]]) -> tuple[list[int], float]:
    counts = [0] * len(ranges)
    elapsed = 0.0
    batch = 25
    for start in range(0, len(ranges), batch):
        values = []
        for qi, spans in enumerate(ranges[start:start + batch], start):
            for lo, hi in spans:
                if not (lo.startswith(PREFIX) and hi.startswith(PREFIX)):
                    raise SystemExit("probe marker outside hk1")
                values.append(f"({qi},'{lo}'::text,'{hi}'::text)")
        sql = f"""
        SELECT qid, count(DISTINCT chunk_id)
        FROM (
          SELECT r.qid, h.chunk_id
          FROM (VALUES {",".join(values)}) AS r(qid, lo, hi)
          JOIN chunk_hilbert h
            ON h.marker COLLATE "C" >= r.lo AND h.marker COLLATE "C" <= r.hi
        ) s
        GROUP BY qid
        """
        t0 = time.perf_counter()
        rows = db.query(sql)
        elapsed += time.perf_counter() - t0
        for row in rows:
            qid_s, cnt_s = row.split("|", 1)
            counts[int(qid_s)] = int(cnt_s)
    return counts, elapsed


def mask_counts(key_arr: np.ndarray, has: np.ndarray, spans: list[tuple[str, str]]) -> np.ndarray:
    mask = np.zeros(key_arr.shape[0], dtype=bool)
    for lo, hi in spans:
        a = np.uint64(key_int(lo))
        b = np.uint64(key_int(hi))
        mask |= (key_arr >= a) & (key_arr <= b)
    mask &= has
    return mask


def page_codes(slugs: list[str]) -> tuple[list[str], np.ndarray]:
    page_slugs = sorted(set(slugs))
    index = {s: i for i, s in enumerate(page_slugs)}
    return page_slugs, np.asarray([index[s] for s in slugs], dtype=np.int32)


def run_dev_alive() -> bool:
    proc = subprocess.run(["pgrep", "-f", "python3 run_dev.py"], capture_output=True, text=True)
    return proc.returncode == 0


def dump(name: str, obj) -> None:
    (BENCH / name).write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    prefix = config_prefix()
    log(f"prefix bytes {len(prefix)} sha256 {hashlib.sha256(prefix.encode()).hexdigest()[:16]}")
    qrels, Q = load_queries(prefix)
    Qn = unit(Q)
    E, slugs, ids, tokens, markers = load_chunks()
    En = unit(E)
    has = np.asarray([m.startswith(PREFIX) for m in markers])
    key_arr = np.zeros(len(markers), dtype=np.uint64)
    for i, m in enumerate(markers):
        if has[i]:
            key_arr[i] = np.uint64(key_int(m))
    check_shared_against_binary([m for m in markers if m][:2])

    chunk_keys = {str(int(ids[i])): markers[i] for i in range(len(markers)) if has[i]}
    db = Psql()
    stored_rows = db.query(
        "SELECT slug || E'\\x1f' || (frontmatter->>'hilbert') FROM pages "
        "WHERE deleted_at IS NULL AND frontmatter ? 'hilbert' "
        "AND frontmatter->>'hilbert' LIKE 'hk1:8:8:9e3779b97f4a7c15:%'"
    )
    stored = {}
    for row in stored_rows:
        slug, marker = row.split("\x1f", 1)
        stored[slug] = marker
    log(f"frontmatter hilbert keys {len(stored)}")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, slug in enumerate(slugs):
        groups[slug].append(i)
    page_slugs = sorted(groups)
    means = np.stack([E[groups[s]].mean(axis=0) for s in page_slugs])
    log("keying page means")
    mean_markers = hilbert_keys(means)
    mean_map = dict(zip(page_slugs, mean_markers))
    matched = [s for s in stored if mean_map.get(s) == stored[s]]
    log(f"page-mean keys matching frontmatter {len(matched)}/{len(stored)}")
    for slug, marker in stored.items():
        mine = mean_map.get(slug)
        if not mine:
            log(f"no chunks for stored page {slug}")
            continue
        log(f"stored vs mean shared {shared_bits(key_int(marker), key_int(mine))} {slug}")
    # Also try the mean of unit vectors if the raw mean missed.
    if len(matched) != len(stored):
        unit_means = np.stack([En[groups[s]].mean(axis=0) for s in page_slugs])
        unit_markers = hilbert_keys(unit_means)
        unit_map = dict(zip(page_slugs, unit_markers))
        unit_matched = [s for s in stored if unit_map.get(s) == stored[s]]
        log(f"unit-mean keys matching frontmatter {len(unit_matched)}/{len(stored)}")
        if len(unit_matched) > len(matched):
            mean_map = unit_map
            means = unit_means
            matched = unit_matched

    page_keys_doc = {
        "source": "pages.frontmatter->>'hilbert'",
        "prefix": PREFIX,
        "stored_n": len(stored),
        "stored": stored,
        "page_mean_matches_stored": len(matched),
        "page_mean_n": len(mean_map),
        "page_mean": mean_map,
    }
    chunk_keys_doc = {"source": "chunk_hilbert", "prefix": PREFIX, "n": len(chunk_keys), "keys": chunk_keys}

    keyed_idx = np.flatnonzero(has)
    log(f"neighbour chunks {keyed_idx.size}")
    neighbour_chunk = neighbourhood("chunk", En[keyed_idx], key_arr[keyed_idx])
    page_key_arr = np.asarray([key_int(mean_map[s]) for s in page_slugs], dtype=np.uint64)
    page_En = unit(means)
    log(f"neighbour pages {len(page_slugs)}")
    neighbour_page = neighbourhood("page-mean", page_En, page_key_arr)
    neighbour_page["frontmatter_keys"] = len(stored)
    neighbour_page["frontmatter_reproduced"] = len(matched)

    page_names, page_index = page_codes(slugs)
    log("plain cosine")
    cosine_lists = []
    for qi in range(len(Qn)):
        cosine_lists.append(rank_pages(En @ Qn[qi], page_index, page_names))
    cosine_metrics = summarize(cosine_lists, qrels)
    cosine_metrics["candidates"] = int(En.shape[0])
    cosine_metrics["tokens"] = int(tokens.sum())
    log(f"cosine R@10 {cosine_metrics['R@10']:.4f} nDCG {cosine_metrics['nDCG@10']:.4f}")

    cells = []
    probe_docs = {}
    for level in (0, 1, 2, 3):
        for ranges in (1, 4, 8, 16):
            log(f"probe {level} {ranges}")
            spans = probe_ranges(Q, level, ranges)
            mem_counts = []
            lists = []
            tok = []
            for qi, sp in enumerate(spans):
                mask = mask_counts(key_arr, has, sp)
                idx = np.flatnonzero(mask)
                mem_counts.append(int(idx.size))
                tok.append(int(tokens[idx].sum()) if idx.size else 0)
                if idx.size == 0:
                    lists.append([])
                    continue
                lists.append(rank_pages(En[idx] @ Qn[qi], page_index[idx], page_names))
            sql_counts, scan_s = sql_scan_counts(db, spans)
            mismatches = sum(a != b for a, b in zip(mem_counts, sql_counts))
            if mismatches:
                raise SystemExit(
                    f"probe {level}-{ranges} count mismatches {mismatches} "
                    f"example mem {mem_counts[0]} sql {sql_counts[0]}"
                )
            metrics = summarize(lists, qrels)
            doc = {
                "level": level,
                "ranges": ranges,
                "n_queries": 817,
                "page_level": "best chunk wins",
                "rescore": "cosine",
                "median_candidates": statistics.median(mem_counts),
                "mean_candidates": statistics.mean(mem_counts),
                "R@10": metrics["R@10"],
                "MRR": metrics["MRR"],
                "nDCG@10": metrics["nDCG@10"],
                "P@10": metrics["P@10"],
                "tokens_median": statistics.median(tok),
                "tokens_mean": statistics.mean(tok),
                "tokens": "sum of characters/4 plus one token per CJK character, over candidate chunks",
                "scan_seconds": scan_s,
                "sql_count_mismatches": mismatches,
                "keyed_chunks": int(has.sum()),
            }
            if (level, ranges) in PILOT:
                doc["pilot_R@10"] = PILOT[(level, ranges)]
                doc["pilot_delta"] = doc["R@10"] - PILOT[(level, ranges)]
            probe_docs[(level, ranges)] = doc
            cells.append(doc)
            log(
                f"  cand {doc['median_candidates']:.0f}/{doc['mean_candidates']:.1f}"
                f" R {doc['R@10']:.4f} MRR {doc['MRR']:.4f} nDCG {doc['nDCG@10']:.4f}"
                f" tokens {doc['tokens_mean']:.0f} scan {doc['scan_seconds']:.2f}s"
            )

    def far_fewer(cell: dict) -> bool:
        return cell["median_candidates"] <= 0.2 * int(has.sum())

    # A cell matches when recall is within 0.03 of plain cosine (the measured
    # point, and the stated 0.698 when this run reproduces it) and the median
    # candidate list is at most a fifth of the keyed chunks.
    cosine_bar = cosine_metrics["R@10"]
    matching = []
    for cell in cells:
        close = cell["R@10"] >= cosine_bar - 0.03 and cell["R@10"] >= COSINE_RECALL - 0.03
        if close and far_fewer(cell):
            matching.append({"level": cell["level"], "ranges": cell["ranges"], "R@10": cell["R@10"],
                             "median_candidates": cell["median_candidates"]})
    decision = "index" if matching else "label"
    sweep = {
        "prefix": PREFIX,
        "query_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "query_prefix_bytes": len(prefix),
        "cosine": cosine_metrics,
        "stated_cosine_R@10": COSINE_RECALL,
        "pilot": {
            "probe-1-8": {
                "measured_R@10": probe_docs[(1, 8)]["R@10"],
                "pilot_R@10": 0.243,
                "delta": probe_docs[(1, 8)]["R@10"] - 0.243,
            },
            "probe-2-8": {
                "measured_R@10": probe_docs[(2, 8)]["R@10"],
                "pilot_R@10": 0.005,
                "delta": probe_docs[(2, 8)]["R@10"] - 0.005,
            },
        },
        "match_rule": "R@10 within 0.03 of this run's plain cosine and of 0.698, and median candidates at most 20% of keyed chunks",
        "matching_cells": matching,
        "decision": decision,
        "cells": cells,
    }

    def write_all() -> None:
        dump("page-keys.json", page_keys_doc)
        dump("chunk-keys.json", chunk_keys_doc)
        dump("neighbour-page.json", neighbour_page)
        dump("neighbour-chunk.json", neighbour_chunk)
        for (level, ranges), doc in probe_docs.items():
            dump(f"probe-{level}-{ranges}.json", doc)
        dump("hilbert-sweep.json", sweep)

    write_all()
    # run_dev.py writes a thinner version of these files. The last write after
    # it exits keeps this measurement.
    deadline = time.time() + 90 * 60
    while run_dev_alive() and time.time() < deadline:
        log("waiting for run_dev.py so it cannot overwrite the sweep")
        time.sleep(20)
    write_all()
    db.close()
    log(f"decision {decision} matching {len(matching)}")


if __name__ == "__main__":
    main()
