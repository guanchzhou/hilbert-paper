# Pre-registration: twelve retrieval ideas on the personal knowledge-base bench

Written on 2026-10-05, before any of the experiments below were run. Every hypothesis,
metric, comparison, test and acceptance criterion is fixed here. Results are findings,
not decisions. The sealed held-out file `qrels-test.json` is not opened.

## 1. Common setup

**Data.** The 817 development questions in `qrels-dev.json` (SHA-256
`87b6f6a2f0b87c8d55f8bf902e00099be366a363326295732e281a7b26a5f338`), the live corpus read
read-only from gbrain Postgres (1,225 notes, about 5,155 embedded chunks, Qwen3-Embedding,
1,024 dimensions), and the saved query vectors `query-vectors.npy` (817 x 1,024, live prefix
`Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`
prepended with no separator, 91 bytes).

**Baseline ("dense").** Cosine between the unit query vector and every unit chunk vector; a
note scores the maximum over its chunks ("best chunk wins"); notes ranked by that score, ties
broken by slug. Reference values on all 817 questions: R@10 0.698, MRR 0.517, nDCG@10 0.538,
chunk-pack survival 0.755 at 6,000 tokens.

**Metrics.** Per question, on the top-10 note list: recall R@10, reciprocal rank MRR computed
on the top-10 list (as in the baseline), nDCG@10 with log2(rank+1) discount, and hit@10.
Functions from `metrics.py`. Tokens are `metrics.estimate_tokens`. A pack walks a ranked list
of units in rank order and keeps a unit if it fits in the remaining 6,000-token budget (units
are not truncated); survival means at least one relevant note is in the pack.

**Split.** `perm = numpy.random.default_rng(20261005).permutation(817)`; the tune half is the
sorted first 409 indices, the confirm half the sorted remaining 408. The split is saved as
`ideas/split.json` (question indices only). Any tunable parameter, or any choice among
variants, is chosen on the tune half only and the chosen setting is tested on the confirm
half. Both halves are reported. Ideas with no choice to make are tested on all 817
questions, and the two halves are reported as well.

**Statistics.** Effects are paired mean differences (new minus comparator) over questions.
95% confidence intervals are percentile bootstrap intervals over questions, 10,000
resamples, generator seed 20261005. Retrieval metrics are tested with the one-sided
Wilcoxon signed-rank test on non-zero paired differences (`zero_method="wilcox"`, as in
`investigate.py`). Binary survival is tested with the exact McNemar test (binomial on
discordant pairs). Non-inferiority with margin m is tested with the one-sided Wilcoxon
signed-rank test on (new minus comparator plus m), H0: new worse by more than m.

**Acceptance.** An idea is accepted when (a) the point estimate of its primary effect
reaches the threshold stated below, (b) its Holm-adjusted p-value in the family below is
under 0.05, and (c) any additional guard stated for that idea passes. Thresholds apply to
the point estimate, not to the confidence-interval bound.

**Family and Holm correction.** The exploratory family consists of the 16 primary
hypotheses H1a to H12 listed in section 3. Holm's step-down correction is applied over all
16. A hypothesis that is not run (budget, gated off, or infrastructure failure) enters the
correction with p = 1, so the number of hypotheses m stays 16. Within-idea Holm values are
reported as secondary information only.

**Resources.** Heavy GPU work (reranker, embedding calls, local qwen3.8 generation) runs
only when the agent-harness pilot is not running. No writes to the live gbrain database or
configuration; any SQL is read-only or uses TEMP tables. The live corpus is not re-embedded.
Private note text, synopses, generated hypothetical documents and judge transcripts stay in
`~/.gbrain/eval/bench/ideas/private/` and are never committed.

## 2. Ideas, hypotheses and criteria

### Idea 1. Cross-encoder rerank of the dense top 50

- Method. Take the dense top 50 distinct notes, each represented by the text of its winning
  chunk. Score (question, chunk text) pairs with qwen3-reranker-0.6b served by local MLX
  (`POST /v1/rerank`, document text truncated by the server at 1,024 tokens). The new top 10
  is the 10 highest reranker scores among those 50 notes.
- Conditions. (a) Default instruction ("Given a web search query, retrieve relevant passages
  that answer the query"). (b) Corpus instruction, fixed now: "Given a question about the
  author's personal notes on software, infrastructure, research and genealogy, retrieve the
  note passage that answers it".
- Hypotheses. H1a: rerank (a) R@10 > dense. H1b: rerank (b) R@10 > dense. No selection, so
  both are tested on all 817 questions.
- Secondary. MRR and nDCG@10 differences; count of questions whose relevant note moves from
  dense rank 11 to 50 into the top 10, and of questions that lose a top-10 hit; per-query
  rerank latency median and p95.
- Accept. R@10 difference at least +0.03, Holm p < 0.05.

### Idea 2. Sentence-pruned packing

- Method. For the dense top 30 notes, split each winning chunk into sentences (split after
  `.`, `!`, `?` followed by whitespace and at newlines; empty pieces dropped). Embed each
  sentence through local MLX as a document (no query instruction). Keep the s sentences
  with the highest cosine to the query vector, in original order; the pruned unit is their
  newline-joined text. Pack pruned units of the top-30 notes in rank order under the
  6,000-token budget.
- Free ceiling, computed first and descriptive only: the survival a pack would have if the
  pruned units of all top-30 notes fitted, which equals hit@30 of the dense ranking.
- Tunable. Sentence cap s in {2, 3, 5}, chosen on the tune half by survival.
- Hypothesis H2: pruned-pack survival > chunk-pack survival (existing rule: winning chunks of
  the dense top 10, same budget), exact McNemar, one-sided, on the confirm half.
- Construct-validity guard. On 100 confirm-half questions (sampled with seed 20261005), the
  local qwen3.8 judges, once per pack and blind to the condition (the two packs are judged
  in separate calls in random order, with identical prompts), whether the packed text
  contains the answer to the question (yes/no). The guard passes if the pruned "yes" rate is
  at least the unpruned "yes" rate minus 0.02. The guard is run only if H2 passes its
  threshold and Holm test, since otherwise acceptance fails anyway.
- Accept. Survival difference at least +0.03, Holm p < 0.05, and the guard passes.

### Idea 3. Corpus-specific query instruction

- Method. Re-embed only the 817 questions, keeping the exact live format
  `Instruct: <task>\nQuery:` + question. The corpus vectors are unchanged.
- Instructions, fixed now (at most three): (i) default, the live task "Given a web search
  query, retrieve relevant passages that answer the query" (existing vectors, checked by
  re-embedding one question); (ii) personal notes: "Given a question about the author's
  personal notes on software, infrastructure, research and genealogy, retrieve the note
  passage that answers it"; (iii) question-oriented: "Given a question, retrieve the passage
  of a note that answers the question".
- Tunable. The better of (ii) and (iii) by tune-half R@10.
- Hypothesis H3: chosen instruction R@10 > default R@10 on the confirm half.
- Accept. R@10 difference at least +0.02 on the confirm half, Holm p < 0.05.

### Idea 4. Removing the mean direction

- Method. Unit chunk vectors C, unit query vectors Q. Means: mu_c is the mean of unit chunk
  vectors; mu_q is the mean of the 817 unit dev query vectors (no labels used). Variants:
  (a) subtract shared mean: C - mu_c and Q - mu_c; (b) subtract separate means: C - mu_c
  and Q - mu_q; (c) project off the shared mean direction u = mu_c/|mu_c|: x - (x.u)u for
  both; (d) project off separate directions (u_c for chunks, u_q for queries). Vectors are
  re-normalised and ranked with the dense rule.
- Hypotheses H4a to H4d: each variant R@10 > dense, all 817 questions (no selection).
- Accept. R@10 difference at least +0.02, Holm p < 0.05 (per variant).

### Idea 5. Dense pseudo-relevance feedback (Rocchio)

- Method. q' = a*q + (1-a)*mean of the top-k chunk vectors of the initial chunk ranking
  (unit vectors), re-normalised, then the dense rule.
- Tunable. a in {0.4, 0.6, 0.8}, k in {3, 5}; the pair with the best tune-half R@10.
- Hypothesis H5: chosen setting R@10 > dense on the confirm half.
- Accept. R@10 difference at least +0.02 on the confirm half, Holm p < 0.05, and the MRR
  difference point estimate at least -0.01 (non-inferiority guard).

### Idea 6. Hilbert-ordered balanced partitions with centroid routing

- Method (Python prototype; the zig-hilbert repository is not modified). Sort the 5,079
  chunks that have hk1 keys by key; cut the order into M equal-size contiguous partitions,
  M in {64, 128}. Each partition's representative is the normalised mean of its unit
  vectors. Route a query to its top-C partitions by cosine to the representatives; score
  every chunk in them exactly; dense rule over the candidates. SOAR variant: each chunk is
  also added to the second partition c' (c' not its own) that minimises
  |x - c'|^2 + lambda * ((x - c').r_hat)^2 with r = x - c_primary, lambda = 1.
- Cost. Candidates are the chunk vectors scored, counting a chunk once per partition it is
  read from (so SOAR duplicates count), plus M representative comparisons reported
  separately.
- Comparators, at equal candidate counts (descriptive curves): exhaustive cosine over the
  keyed chunks; hk1 probes (`hilbert-sweep.json`); k-means IVF with the same M (numpy
  k-means, k-means++ initialisation, seed 20261005, 25 Lloyd iterations, spherical:
  normalised centroids), routed the same way; uniform random subsets.
- Tunable. Variant (plain or SOAR), M, and C, among settings with median candidates at most
  20% of keyed chunks (1,015). Chosen as the highest tune-half R@10.
- Hypothesis H6 (H4 rule): non-inferiority of the chosen setting to exhaustive cosine with
  margin 0.03 on the confirm half, and median candidates at most 20% on the confirm half.
- Secondary. Whether Hilbert partitions beat hk1 probes and k-means IVF at equal candidates.
- Accept. Confirm-half R@10 difference at least -0.03, Holm p < 0.05, candidate rule met.

### Idea 7. Many randomised Hilbert orders with position windows (Hilbert forest)

- Method. Centre the unit vectors by mu_c. For each of L independent orders, compute hk1
  keys with zig-hilbert using a different `--seed` (seed i = 0x9e3779b97f4a7c15 + i, so
  each order is a different random projection), sort the chunks by key, and take the k1
  chunks on each side of the query's position. The candidate set is the union over the L
  orders; exact cosine rescoring and the dense rule.
- Grid. L in {8, 16, 32, 64}; k1 in {4, 8, 16, 32, 64}.
- Tunable. (L, k1) with the best tune-half R@10 among settings with median candidates at most
  20% of keyed chunks.
- Hypothesis H7 (H4 rule): non-inferiority to exhaustive cosine with margin 0.03 on the
  confirm half, median candidates at most 20%.
- Secondary. Comparison with hk1 probes, idea 6, and the LSH table-count prediction in
  `lsh-qa.json` (about 71 independent 8-bit tables for 90% recall of question-answer pairs
  under centring): the L at which the best relevant chunk is in the candidate set for 90%
  of questions.
- Accept. As idea 6.

### Idea 8. Per-chunk synopsis (contextual retrieval)

- Method. On an in-memory scratch copy of the corpus (no database), the local qwen3.8
  writes a short synopsis that situates each chunk in its note (title plus up to 2,000
  characters of note text plus the chunk). Each chunk is embedded through MLX as synopsis
  plus newline plus chunk; dense rule unchanged; queries unchanged.
- Budget gate. Generation time is first measured on 20 chunks (seed 20261005). If the
  projected total for all chunks exceeds 10 hours, the idea is recorded as "not run within
  budget" with the measured per-chunk time and the projection.
- Hypothesis H8: synopsis-augmented dense R@10 > dense, all 817 questions.
- Limitation stated in advance. The dev questions were written by an LLM from the notes, so
  LLM-written synopses may share vocabulary with them and inflate the effect.
- Accept. R@10 difference at least +0.03, Holm p < 0.05.

### Idea 9. Link-graph expansion

- Diagnostic (all 817 questions, descriptive). Among dense misses whose best relevant note
  is ranked beyond 50 (about 70), the share whose relevant note is within one or two links
  (undirected, `links` table read-only) of any dense top-10 note.
- Gate. The full method runs only if that share is at least 30%.
- Method. Personalised PageRank on the undirected note link graph, restart distribution
  proportional to the dense scores of the top 20 notes (min-shifted), restart probability
  alpha; final score = (1 - beta)*dense + beta*PPR, each min-max normalised per query.
- Tunable. alpha in {0.15, 0.3, 0.5}, beta in {0.1, 0.2, 0.3}, chosen on tune-half R@10.
- Hypothesis H9: chosen setting R@10 > dense on the confirm half.
- Accept. R@10 difference at least +0.02, Holm p < 0.05, and MRR difference point estimate
  at least 0 (no loss). If gated off, H9 enters Holm with p = 1 and the idea is "not run".

### Idea 10. HyDE / query2doc

- Method. The local qwen3.8 (thinking disabled, no tools) writes a short hypothetical note
  passage answering each question. The passage is embedded as a document (no
  instruction). q' = normalise(w*q + (1 - w)*h), dense rule.
- Tunable. w in {0, 0.25, 0.5, 0.75}, chosen on tune-half R@10.
- Hypothesis H10: chosen w R@10 > dense on the confirm half.
- Secondary. Added latency per query (generation plus embedding), median and p95.
- Accept. R@10 difference at least +0.03, Holm p < 0.05.

### Idea 11. Adaptive-k cut-off

- Method (offline). On the dense note ranking, take the top 30 note scores, find the
  largest gap between consecutive scores, return the notes before it, and pack their
  winning chunks under the same budget and rule.
- Comparator. The fixed top-10 chunk pack.
- Hypothesis H11: survival non-inferiority with margin 0.02 (Wilcoxon on paired
  differences plus 0.02), all 817 questions.
- Secondary. Distribution of cut positions; mean delivered tokens of both packs.
- Accept. Mean delivered tokens at least 30% lower than the fixed pack, survival
  difference point estimate at least -0.02, Holm p < 0.05.

### Idea 12. MUVERA fixed-dimensional encodings with hk1 level-1 buckets

- Method. The partition is the hk1 level-1 cell, the 8 sign bits (cell coordinate >= 128 per
  axis) of the zig-hilbert projection: 256 SimHash buckets, computed with zig-hilbert for
  every chunk and query. A note's encoding in bucket b is the mean of its unit chunk vectors
  in b; an empty bucket is filled with the mean of the note's chunks at the minimum Hamming
  distance from b (MUVERA fill-empty rule). The query encoding is its unit vector in its own
  bucket (one vector, so the inner product reduces to the note's block for that bucket).
  One repetition, no final projection.
- Comparators. Best chunk (dense) and mean of chunks (normalised mean chunk vector).
- Subset. Questions with at least one relevant live note that has at least 5 chunks.
- Hypothesis H12: MUVERA R@10 > best-chunk R@10 on that subset (all dev questions in it, no
  tuning).
- Secondary. MUVERA versus mean of chunks.
- Accept. R@10 difference at least +0.03 on the subset, Holm p < 0.05.

### Not applicable

Late chunking is not tested: Qwen3-Embedding pools the last token, so per-chunk mean pooling
over a long-context pass is not defined for this model.

## 3. The family

| ID | Idea | Primary comparison | Test | Set | Threshold |
|---|---|---|---|---|---|
| H1a | rerank, default instruction | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.03 |
| H1b | rerank, corpus instruction | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.03 |
| H2 | sentence-pruned pack | survival vs chunk pack | exact McNemar one-sided | confirm | +0.03 and guard |
| H3 | query instruction | R@10 vs default | Wilcoxon one-sided | confirm | +0.02 |
| H4a | subtract shared mean | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.02 |
| H4b | subtract separate means | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.02 |
| H4c | project off shared direction | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.02 |
| H4d | project off separate directions | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.02 |
| H5 | Rocchio PRF | R@10 vs dense | Wilcoxon one-sided | confirm | +0.02, MRR >= -0.01 |
| H6 | Hilbert partitions | R@10 vs exhaustive | non-inferiority, margin 0.03 | confirm | >= -0.03, <= 20% scanned |
| H7 | Hilbert forest | R@10 vs exhaustive | non-inferiority, margin 0.03 | confirm | >= -0.03, <= 20% scanned |
| H8 | chunk synopsis | R@10 vs dense | Wilcoxon one-sided | all 817 | +0.03 |
| H9 | link-graph PPR | R@10 vs dense | Wilcoxon one-sided | confirm | +0.02, MRR >= 0 |
| H10 | HyDE | R@10 vs dense | Wilcoxon one-sided | confirm | +0.03 |
| H11 | adaptive-k | survival vs top-10 pack | non-inferiority, margin 0.02 | all 817 | -30% tokens, >= -0.02 |
| H12 | MUVERA | R@10 vs best chunk | Wilcoxon one-sided | subset >= 5 chunks | +0.03 |

## 4. Outputs

One JSON file per idea, `ideas/<n>-<name>.json`, with per-condition metrics on all, tune and
confirm questions, bootstrap CIs, raw p-values, within-idea and family Holm p-values,
acceptance flags, and the per-question arrays used by the tests (question indices and
numbers only). `ideas/summary.json` and `ideas/REPORT.md` close the family.
