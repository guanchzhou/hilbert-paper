---
type: knowledge-concept
title: Knowledge search bench plan
source: obsidian
obsidian_path: brain/Knowledge/concepts/knowledge-search-bench-plan.md
tags:
  - obsidian-mirror
hilbert: hk1:8:8:9e3779b97f4a7c15:dc7466683c0ab5d3
---

# Knowledge search bench plan

A measurement paper in the genealogy style: every claim is a difference on a frozen task list. A database is replaced only when a cell shows Postgres failed that operation.

## Goal

For one personal brain, measure what changes two numbers: whether the right page is in the answer, and how many tokens the model reads to get it there.

## Frozen setup

- Corpus: live brain after the title-tier re-embed. 1,223 pages, 5,143 chunks. Embedding `ollama:qwen3-embedding-8k`, 1024 dimensions, served by the local MLX server.
- Dev tasks: `~/.gbrain/eval/qrels-dev.json`, 817 queries. Held-out `qrels-test.json` stays sealed until the final table.
- Agent tasks: 30 questions whose answer is a known page, plus a separate shell list for RTK (`git status`, `rg`, directory listings). RTK never sees file reads, so it is not scored on the page tasks.
- Same local chat model in every agent cell. Temperature fixed. Each agent cell repeated; report median and spread.
- Token count is gbrain's own heuristic: characters/4, one token per CJK character. Read `tokens_delivered`, not the truncated CLI line.
- Reranker off for retrieval cells. It is a latency factor, not a token factor. The existing dev result stays in the appendix: about +0.01 MRR and 0.13 s to 3.8 s.

## Numbers recorded in every cell

- Recall@10, MRR, nDCG@10.
- Tokens delivered, and whether the relevant page is inside that text.
- Tokens per successful answer.
- Candidates examined.
- Wall time.
- For RTK: bytes before the filter and bytes after.

A cell that uses fewer tokens and drops the page is a regression.

## Order

Do not run the full 64-cell cross first. One switch at a time. Cross two switches only when each moved a number on its own by more than the run-to-run spread.

### 1. Retrieval grid

Four cells, no agent, reranker off, 817 dev queries.

| | Whole page | Chunks |
|---|---|---|
| Keyword only | | |
| Keyword + vectors | | |

### 2. Evidence unit and tokens

Same ranking. Units: `chunk`, `window`, `section`, `page`. Budget 6,000 tokens. Record tokens delivered, hits dropped by the budget, and whether the relevant page survived.

### 3. Hilbert

Its own factor. Same 817 queries. Same token count.

Keys:

- None. Plain cosine, and current pgvector search.
- One key per note, from the mean of that note's chunk vectors. Already stored as the `hilbert` frontmatter property.
- One key per chunk, from that chunk's own vector. Table `chunk_hilbert`, 5,143 rows. Space 8×8, seed `9e3779b97f4a7c15`.

Measurements:

1. Neighbourhood. Longest shared prefix versus true cosine neighbours. Plot shared prefix bits against cosine.
2. Candidate filter. Probe levels 0, 1, 2, 3 and range counts 1, 4, 8, 16. For each: candidates, recall after cosine rescore, tokens those candidates would deliver, range-scan time.
3. Against Postgres. The same sweep at 5,143 chunks and at the 200,000-chunk scale. Compare with pgvector on recall, candidates, tokens, and latency.

Pilot points already on that curve, not the conclusion: level 1 with 8 ranges, R@10 0.243, median 195 candidates; level 2 with 8 ranges, R@10 0.005, median 1 candidate; plain cosine R@10 0.698, nDCG@10 0.534.

Figure: recall against candidates. Hilbert is the curve. Full cosine and current pgvector search are single points. Tokens marked on each point.

### 4. Agent grid

Hosts: clean Cursor, clean Claude Code. Then each with RTK only, gbrain only, and both. Thirty page tasks for gbrain. Shell tasks for RTK. Repeated runs.

### 5. Appendix, redrawn in the same format

- Title prefix: dev R@10 0.65, MRR 0.52, nDCG@10 0.52, against 0.654 / 0.518 / 0.527 without it. No gain.
- Reranker: about +0.01 MRR, query time 0.13 s to 3.8 s.

## Diagrams

- 2×2 of chunks × vectors: recall and tokens per hit.
- Four evidence units: tokens delivered against recall inside the budget.
- Hilbert curve: recall and tokens against candidates.
- Per host: clean, RTK, gbrain, both. Tokens in, and fraction of tasks whose page was used.
- Waterfall of switches that moved a number by more than the spread. Switches inside the spread are left off the figure.

## Database rule

Postgres stays unless a cell fails because of Postgres.

- Re-embedding, not query latency, was the limit at 200,000 chunks. `shared_buffers` at 128 MB is a configuration change.
- Hilbert justifies another key store only if a range scan matches cosine recall with far fewer candidates and pgvector cannot serve that scan as fast at 200,000 chunks. The vectors stay in Postgres.
- ArangoDB is the graph candidate, because the genealogy base already uses it for documents plus a graph. Run it only if a one-hop expansion is wrong or too slow in Postgres. Same pages, same edges, hubs capped, AQL against SQL.
- Turso only if a single-file replica is required and libSQL vectors match pgvector recall on this frozen set.
- A new database is built only for the operation Postgres failed, and only after that cell is measured.

The paper may conclude that Postgres was sufficient and the token change came from returning a chunk instead of a page.

## Topics

- [[obsidian/knowledge/topics/databases-and-storage]]

## See Also

- [[obsidian/knowledge/tools/gbrain-operations]]
- [[obsidian/projects/zig-hilbert/readme]]
- [[obsidian/knowledge/concepts/engineering-decisions]]


## Fifty steps

Each step writes one file under `~/.gbrain/eval/bench/` or checks one number. `qrels-test.json` is never opened. The live reranker flag is restored to true at the end. The agent grid is a later paper, not these steps.

1. Write `manifest.json`: live page count, chunk count, embedding model, dimension, reranker flag, query-prefix bytes.
2. Confirm `qrels-dev.json` has 817 queries and record the file hash.
3. Confirm `chunk_hilbert` has 5,143 rows and the key prefix `hk1:8:8:9e3779b97f4a7c15:`.
4. Confirm the zig-hilbert binary version and that `check` accepts one stored key.
5. Copy the shared metric functions into `metrics.py` and check them against the known cosine point R@10 0.698.
6. Embed the 817 dev queries once, with the live query prefix, and save `query-vectors.npy`.
7. Keyword, whole page: `tsvector` on `compiled_truth`. Write `cell-keyword-page.json`.
8. Keyword, chunks: `content_chunks.search_vector`, collapse to one page. Write `cell-keyword-chunk.json`.
9. Vectors, whole page: cosine against the mean of each page's chunk vectors. Write `cell-vector-page.json`.
10. Vectors, chunks: cosine against every chunk, best chunk wins the page. Write `cell-vector-chunk.json`.
11. For cells 7–10, add wall time and tokens of the top-10 page texts.
12. Merge cells 7–10 into `retrieval-grid.json`.
13. Define a section as the chunk plus neighbors until a markdown heading. Save `section-rule.json`.
14. Pack the chunk-vector ranking as `chunk` under 6,000 tokens. Write `unit-chunk.json`.
15. Pack the same ranking as `window`. Write `unit-window.json`.
16. Pack the same ranking as `section`. Write `unit-section.json`.
17. Pack the same ranking as `page`. Write `unit-page.json`.
18. For units 14–17, record tokens delivered, hits dropped by the budget, and whether the relevant page survived.
19. Merge units 14–17 into `evidence-units.json`. A unit that saves tokens and loses the page is a regression.
20. Load page-mean `hk1` keys from note frontmatter into `page-keys.json`.
21. Load per-chunk keys from `chunk_hilbert` into `chunk-keys.json`.
22. Neighbourhood, page keys: shared prefix bits against cosine. Write `neighbour-page.json`.
23. Neighbourhood, chunk keys. Write `neighbour-chunk.json`.
24–27. Probe level 0, range counts 1, 4, 8, 16. One JSON file each.
28–31. Probe level 1, range counts 1, 4, 8, 16. Step 30 checks the pilot R@10 0.243.
32–35. Probe level 2, range counts 1, 4, 8, 16. Step 34 checks the pilot R@10 0.005.
36–39. Probe level 3, range counts 1, 4, 8, 16.
40. Merge probes 24–39 into `hilbert-sweep.json`. Hilbert stays a label unless one cell matches cosine recall with far fewer candidates.
41. Time a pgvector top-10 on the live 5,143 chunks. Write `pgvector-5143.json`.
42. Build a scratch database of 200,000 vectors. Do not write it into the live brain.
43. Time pgvector and one Hilbert range scan at 200,000. Write `scale-200k.json`. Drop the scratch database.
44. Confirm ArangoDB 3.12 on port 30529. Create a new database. Do not open `familio`.
45. Load the 5,143 vectors. Write `arango-load.json`.
46. Arango vector recall and latency against step 41. Write `arango-vector.json`.
47. One-hop link expansion, hubs capped, in AQL and in SQL. Write `arango-hop.json`.
48. Drop the Arango scratch database. Confirm `familio` was not modified.
49. One canvas, four views: retrieval 2×2, evidence units, Hilbert curve including Arango, waterfall of switches that beat the spread.
50. Append every cell to this note and to gbrain-operations, reconcile, and make the PDF.
