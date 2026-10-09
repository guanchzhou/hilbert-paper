#!/usr/bin/env python3
"""Five repeats of 20 dev queries, then one held-out pass of the winning cell."""

import json
import os
import statistics
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from metrics import summarize

BENCH = Path(__file__).resolve().parent


def password() -> str:
    cfg = json.loads((Path.home() / ".gbrain/config.json").read_text())
    return urllib.parse.urlparse(cfg["database_url"]).password or ""


def psql_at(sql: str) -> str:
    env = {**os.environ, "PGPASSWORD": password()}
    proc = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres", "-X", "-qAt", "-c", sql],
        check=True, capture_output=True, text=True, env=env,
    )
    return proc.stdout


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


def load_chunks():
    blob = psql_at("""
    SELECT p.slug || E'\\x1f' || c.embedding::text
    FROM content_chunks c
    JOIN pages p ON p.id = c.page_id
    WHERE p.deleted_at IS NULL AND c.embedding IS NOT NULL
    """)
    slugs, vecs = [], []
    for line in blob.splitlines():
        slug, vec = line.split("\x1f", 1)
        slugs.append(slug)
        vecs.append(json.loads(vec))
    E = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1
    return slugs, E / n


def rank(Qn, En, slugs):
    lists = []
    for qi in range(len(Qn)):
        scores = En @ Qn[qi]
        order = np.argpartition(-scores, 40)[:40]
        order = order[np.argsort(-scores[order])]
        seen, pages = set(), []
        for j in order:
            s = slugs[j]
            if s in seen:
                continue
            seen.add(s)
            pages.append(s)
            if len(pages) >= 10:
                break
        lists.append(pages)
    return lists


def main() -> None:
    prefix = json.loads((BENCH / "manifest.json").read_text())["query_prefix"]
    slugs, En = load_chunks()
    dev = json.loads((Path.home() / ".gbrain/eval/qrels-dev.json").read_text())
    Q = np.load(BENCH / "query-vectors.npy")
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    qn[qn == 0] = 1
    Qn = Q / qn
    idx = list(range(0, 800, 40))[:20]
    recalls = []
    for run in range(5):
        sub_q = [dev[i] for i in idx]
        lists = rank(Qn[idx], En, slugs)
        recalls.append(summarize(lists, sub_q)["R@10"])
        print("spread", run, recalls[-1], flush=True)
    (BENCH / "spread.json").write_text(json.dumps({
        "cell": "vector-chunk",
        "queries": 20,
        "repeats": 5,
        "R@10": recalls,
        "mean": statistics.mean(recalls),
        "stdev": statistics.pstdev(recalls),
    }, indent=2) + "\n")

    test = json.loads((Path.home() / ".gbrain/eval/qrels-test.json").read_text())
    rows = []
    texts = [prefix + q["query"] for q in test]
    for i in range(0, len(texts), 32):
        rows.extend(embed_batch(texts[i:i + 32]))
        print("heldout embedded", len(rows), flush=True)
    T = np.asarray(rows, dtype=np.float32)
    tn = np.linalg.norm(T, axis=1, keepdims=True)
    tn[tn == 0] = 1
    stats = summarize(rank(T / tn, En, slugs), test)
    stats["cell"] = "vector-chunk"
    stats["split"] = "qrels-test"
    (BENCH / "heldout.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("heldout", stats, flush=True)


if __name__ == "__main__":
    main()
