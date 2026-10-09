# Twelve retrieval ideas on the personal knowledge-base bench

Andrey Maltsev, 2026-10-05/06. Development questions only; the sealed held-out set was not
opened. Results are findings, not decisions.

## Summary

Sixteen primary hypotheses from twelve ideas were pre-registered (`preregistration.md`,
committed before any experiment ran) and corrected together with Holm's method (m = 16;
hypotheses that were not run enter with p = 1). One idea was accepted: cross-encoder
reranking of the dense top 50 with the reranker's default instruction raised R@10 from 0.698
to 0.735 (+0.036, 95% CI +0.014 to +0.059, family Holm p = 0.007). Two ideas gave promising
signals that miss their pre-registered criteria: sentence-pruned packing raised pack
survival by 0.098 but failed its answer-quality guard, and link-graph personalised PageRank
gained +0.011 R@10 on the confirm half (CI excluding zero, below the +0.02 threshold). HyDE
showed a similar small gain (+0.009) at a cost of 14 s per query. The other ideas were null
or negative. Synopsis generation was over budget and was not run.

Effects are new minus comparator. "Confirm" means the confirm half (n = 408) after
selection on the tune half; "all" means all 817 questions with no selection.

| Idea | Hypothesis | Set | Comparator → new | Effect (95% CI) | Family Holm p | Status |
|---|---|---|---|---|---|---|
| 1 | H1a rerank top 50, default instruction | all | R@10 0.698 → 0.735 | +0.036 (+0.014, +0.059) | 0.007 | accepted |
| 1 | H1b rerank top 50, corpus instruction | all | R@10 0.698 → 0.710 | +0.012 (−0.011, +0.035) | 1.0 | not accepted |
| 2 | H2 sentence-pruned pack | confirm | survival 0.767 → 0.865 | +0.098 (+0.071, +0.127) | < 0.001 | not accepted (judge guard −5 points) |
| 3 | H3 query instruction (personal notes) | confirm | R@10 0.701 → 0.704 | +0.002 (−0.014, +0.019) | 1.0 | not accepted |
| 4 | H4a subtract shared mean | all | R@10 0.698 → 0.703 | +0.005 (−0.008, +0.017) | 1.0 | not accepted |
| 4 | H4b subtract separate means | all | R@10 0.698 → 0.681 | −0.017 (−0.034, −0.002) | 1.0 | not accepted |
| 4 | H4c project off shared direction | all | R@10 0.698 → 0.703 | +0.004 (−0.007, +0.016) | 1.0 | not accepted |
| 4 | H4d project off separate directions | all | R@10 0.698 → 0.684 | −0.015 (−0.031, +0.001) | 1.0 | not accepted |
| 5 | H5 Rocchio (a 0.6, k 5) | confirm | R@10 0.701 → 0.703 | +0.002 (−0.013, +0.017) | 1.0 | not accepted |
| 6 | H6 Hilbert partitions + SOAR (M 128, C 12) | confirm | R@10 vs exhaustive 0.699 | −0.043 (−0.066, −0.023) | 1.0 | not accepted |
| 7 | H7 Hilbert forest (L 64, k1 8) | confirm | R@10 vs exhaustive 0.699 | −0.059 (−0.090, −0.028) | 1.0 | not accepted |
| 8 | H8 chunk synopsis | all | — | — | 1.0 (p = 1) | not run (26 h projected > 10 h) |
| 9 | H9 link-graph PPR (alpha 0.3, beta 0.3) | confirm | R@10 0.701 → 0.712 | +0.011 (+0.001, +0.022) | 0.90 | not accepted (promising) |
| 10 | H10 HyDE (w 0.75) | confirm | R@10 0.701 → 0.711 | +0.009 (+0.001, +0.020) | 0.41 | not accepted |
| 11 | H11 adaptive-k | all | survival 0.755 → 0.563 | −0.192 (−0.220, −0.166) | 1.0 | not accepted |
| 12 | H12 MUVERA (≥ 5-chunk subset, n 314) | subset | R@10 0.716 → 0.458 | −0.258 (−0.305, −0.213) | 1.0 | not accepted |

## Common setup

- Corpus: frozen read-only snapshot of the live gbrain store, 1,223 notes with chunks and
  5,157 chunks (two more than the earlier manifest; deviation D3). The dense baseline on the
  snapshot reproduces the reference values exactly: R@10 0.698391, MRR 0.516824, nDCG@10
  0.537789, chunk-pack survival 0.755202 at 6,000 tokens.
- Dense rule: cosine to every chunk, best chunk wins the note, ties by slug. MRR is computed
  on the top-10 list, as in the baseline.
- Split: seed 20261005, 409 tune and 408 confirm questions (`split.json`). Parameters and
  variant choices were made on the tune half and tested on the confirm half; ideas without a
  choice were tested on all 817 questions.
- Statistics: percentile bootstrap CIs (10,000 resamples over questions); one-sided
  Wilcoxon signed-rank tests on non-zero paired differences; exact McNemar for survival.
  Acceptance needs the point estimate at the threshold, family Holm p < 0.05, and any guard.
- Deviation D1 (`deviations.md`, committed before ideas 6 and 7 ran): the pre-registered
  non-inferiority test (Wilcoxon on differences shifted by the margin) is invalid when most
  differences are zero, because the shift turns ties into many small positive values. For
  H6, H7 and H11 the larger of that p-value and a one-sided paired t-test on the shifted
  differences is used. This can only make acceptance harder, and none of the three meets its
  point-estimate rule anyway.

## Results by idea

### 1. Cross-encoder rerank of the dense top 50 — accepted (default instruction)

qwen3-reranker-0.6b over local MLX scored the winning chunk of each of the dense top 50
notes. With the default instruction, R@10 rose from 0.698 to 0.735 (+0.036, CI +0.014 to
+0.059), MRR from 0.517 to 0.536 (+0.019, CI −0.008 to +0.046) and nDCG@10 from 0.538 to
0.561. 91 questions improved and 59 worsened. Of the 111 questions whose relevant note sat
at dense rank 11 to 50, 56 moved into the top 10; 32 questions lost a top-10 hit. The
effect holds on both halves (tune +0.030, confirm +0.043). The corpus instruction written
for this study was worse: R@10 +0.012 (not significant) and MRR −0.037 (CI −0.063 to
−0.011). Latency was 4.4 s median and 6.2 s p95 per question for 50 documents on the local
GPU. An earlier gbrain evaluation that reranked the hybrid top 30 found no gain; the
difference is the candidate source, which here is the stronger dense ranking.

### 2. Sentence-pruned packing — not accepted (guard failed)

The free ceiling, the survival if pruned units of the dense top 30 all fitted, is hit@30 =
0.868, against 0.755 for the chunk pack. All three sentence caps (2, 3, 5) fitted all 30
notes in 6,000 tokens, so pruned survival reached the ceiling (cap 5 chosen by the tie rule):
on the confirm half 0.865 against 0.767 (+0.098, CI +0.071 to +0.127, McNemar 40 to 0,
p < 1e-12), using 3,914 instead of 5,482 tokens. This gain is partly built in, because
survival only asks whether the relevant slug is present. The pre-registered guard tests
whether the text still answers: on 100 confirm questions the local qwen3.8, blind to
condition, judged that the context contained the answer for 77% of pruned packs and 82% of
chunk packs (−5 points; the limit was −2; 4 pruned-only against 9 chunk-only verdicts). The
guard failed, so the survival gain is not accepted. The judge is a single local model with
a yes/no verdict and is lenient; its judgements are not validated against human labels.

### 3. Corpus-specific query instruction — not accepted

Re-embedding the 817 questions with the live format reproduced the saved vectors (cosine
0.999996). The personal-notes instruction won on the tune half (+0.010) but gave +0.002 on
the confirm half (CI −0.014 to +0.019) with MRR −0.011. The question-oriented instruction
was worse than the default on both halves.

### 4. Removing the mean direction — not accepted

Shared-mean subtraction (+0.005, CI −0.008 to +0.017) and projecting off the shared mean
direction (+0.004) were null. Using a separate query mean hurt (−0.017 and −0.015). The
chunk and question means are nearly parallel (cosine 0.87), so removing the common
direction changes rankings little; at this corpus size the anisotropy does not hurt cosine
ranking.

### 5. Rocchio pseudo-relevance feedback — not accepted

The tune half chose a = 0.6, k = 5 (+0.005 on tune). On confirm: +0.002 (CI −0.013 to
+0.017), MRR −0.004. With about one relevant note per question, the top-k chunks are mostly
non-relevant and feedback adds noise as often as signal.

### 6. Hilbert-ordered balanced partitions with centroid routing — not accepted

Exhaustive cosine over keyed chunks gives R@10 0.699. The best Hilbert setting under the
20% scan limit, chosen on tune, was SOAR with M = 128 and C = 12 (median 926 candidates on
confirm): R@10 −0.043 against exhaustive (CI −0.066 to −0.023), which fails the 0.03 margin.
At equal candidates, Hilbert partitions are far better than hk1 probes (about 0.45 against
0.30 at 320 candidates) and random subsets (0.35 at 1,015), and SOAR secondary assignment
adds 0.04 to 0.09 at equal candidates. Spherical k-means IVF, a comparator, dominates both: M = 128, C = 20
reaches 0.684 at 925 candidates (−0.015 against exhaustive, 18% scanned), which would meet
the H4 rule. That is a descriptive comparator result and has not been tested
confirmatorily. Sorting by a Hilbert key in an 8-dimensional projection keeps too little of
the 1,024-dimensional neighbourhood for the partition representatives to route well.

### 7. Hilbert forest — not accepted

Sixty-four zig-hilbert orders with distinct seeds on centred vectors. The tune half chose
L = 64, k1 = 8 (median 904 candidates): R@10 −0.059 on confirm (CI −0.090 to −0.028). The
best relevant chunk is in the candidate set for 90% of questions only at 53% or more of the
corpus scanned (L = 64, k1 = 32 or L = 32, k1 = 64), consistent with the LSH prediction of
about 71 independent 8-bit tables for 90% recall.

### 8. Per-chunk synopsis — not run within budget

On 20 random chunks the local qwen3.8 took 18.2 s per synopsis on average (median 18.8 s,
p95 30.4 s; about 1,210 prompt and 80 output tokens). The projection for 5,157 chunks is 26.1
hours, above the 10-hour budget, so the idea was not run (H8 enters Holm with p = 1). The
pre-registered limitation stands: the dev questions are LLM-written, which would bias any
LLM-synopsis gain upward.

### 9. Link-graph expansion — gate passed; PPR not accepted (promising)

Diagnostic: of 72 dense misses whose relevant note ranks beyond 50, 46 (64%) are one link
from a dense top-10 note and 69 (96%) within two links. The base rate for an arbitrary note
is 12% within one link and 69% within two, so the one-link enrichment (about five times) is
the informative part; the two-link figure mostly reflects hub pages such as topic indexes.
Personalised PageRank fused with dense scores (alpha = 0.3, beta = 0.3 chosen on tune;
+0.018 on tune) gave +0.011 R@10 on confirm (CI +0.001 to +0.022, raw p 0.069; 11 improved,
5 worsened) and MRR +0.009. Below the +0.02 threshold, so not accepted, but the only
consistent positive dense-side effect besides reranking.

### 10. HyDE / query2doc — not accepted

The local qwen3.8 (thinking off, no tools) wrote a three-to-five-sentence passage per
question, embedded as a document. The hypothetical passage alone (w = 0) was worse than the
question (R@10 0.661); the tune half chose w = 0.75 (+0.009 on tune). On confirm: +0.009
(CI +0.001 to +0.020, raw p 0.030, family Holm p 0.41; 8 improved, 4 worsened), MRR +0.001.
Added latency is 13.9 s median (p95 18.4 s) for generation plus 0.05 s for the embedding.
The questions are short keyword strings about private infrastructure, which a model without
access to the notes cannot answer, so the hypothetical passage adds little signal.

### 11. Adaptive-k cut-off — not accepted

Cutting at the largest score gap within the top 30 kept a median of 2 notes (47% of cuts
after the first note). Tokens fell by 70% (5,503 to 1,671, CI 68% to 71%), but survival
fell from 0.755 to 0.563 (−0.192, CI −0.220 to −0.166), far beyond the 0.02 margin. Dense
score gaps on this corpus do not mark the boundary of relevance.

### 12. MUVERA with hk1 level-1 buckets — not accepted

On the 314 questions whose relevant note has at least five chunks, best chunk gives R@10
0.716, mean of chunks 0.648 and the MUVERA encoding 0.458 (−0.258, CI −0.305 to −0.213).
With a single query vector, the encoding reduces to the note's chunks in the query's SimHash
bucket (or the nearest occupied bucket); a relevant chunk shares the query's level-1 bucket
for only 16% of these questions, so the encoding mostly scores the wrong chunks. MUVERA is
designed for multi-vector queries, which this bench does not have.

### Not applicable: late chunking

Qwen3-Embedding pools the last token; late chunking needs per-token mean pooling over a
long-context pass, which is not defined for this model.

## Implementation plans

### Accepted: dense top-50 rerank (idea 1)

Component: gbrain search configuration and reranker plugin; bench; agent harness for the
end-to-end check.

1. gbrain: feed the reranker from the dense chunk ranking (best chunk per note) instead of
   the hybrid list, with depth 50, the winning chunk text as the document, and the default
   instruction. Keep the instruction configurable but default it to the model's own; this
   study shows a hand-written corpus instruction hurts MRR.
2. Latency: 4.4 s median per query is acceptable for agent tool calls but not for
   interactive search. Before rollout, measure depth 30 and a 512-token document cap, and
   cache scores by (query, chunk id). Each of these needs its own pre-registered dev check
   against the depth-50 result.
3. Bench: add a `vector-chunk+rerank50` cell to `run_measure.py`, store per-question arrays,
   and run one pre-registered confirmation on the sealed held-out set (single look, same
   +0.03 criterion).
4. Agent harness (owned by the harness work): after gbrain exposes the cell, measure
   answer quality and wall time per task with and without reranking.
5. Risks: latency on CPU-only hosts; instruction sensitivity; documents above 1,024 tokens
   are truncated; dev questions written by an LLM may favour a cross-encoder.

### Promising: link-graph PageRank fusion (idea 9)

Component: gbrain retrieval post-processor (plugin) reading the `links` table; bench.

1. Bench: pre-register a follow-up that applies PPR fusion on top of the reranked list
   (idea 1) and down-weights hub notes (degree normalisation or excluding Topics/See Also
   index pages from propagation). The one-link enrichment suggests restricting propagation
   to one hop.
2. gbrain: compute the note graph once per sync and run PPR over 1,223 notes per query
   (milliseconds in numpy; sparse power iteration in TypeScript or SQL).
3. Risks: hub pages dominate two-hop expansion; the small effect (+0.011) may not survive a
   held-out test; links are author-written and uneven across note types.

### Promising: sentence-pruned packing (idea 2)

Component: gbrain evidence packer; agent harness (pack consumer); bench.

1. Bench: pre-register a hybrid pack, the full winning chunk for the top 10 and pruned
   sentences for ranks 11 to 30, which keeps the answer text of the top hits while adding
   coverage, and a sentence cap large enough to use the budget (pruned packs used only
   3,900 of 6,000 tokens).
2. Validate the judge first: human labels on 50 packs, then the guard with the validated
   judge, or require two judges to agree.
3. gbrain: implement sentence splitting at index time and store sentence vectors per chunk
   (about 89,000 sentences for the top-30 winning chunks of the dev set; all chunks would
   need a one-off embedding pass of roughly 15 to 25 minutes on MLX).
4. Risks: the survival metric rewards slug presence by construction; pruning can remove the
   sentence that carries the answer, which the guard detected.

### Descriptive: k-means IVF for a future ANN index (idea 6 comparator)

Component: zig-hilbert (a later `partitions` subcommand) or pgvector IVFFlat; bench.

At 5,157 chunks exhaustive cosine costs about 0.4 s for all 817 questions, so no ANN index
is needed now. If the corpus grows (see `scale-200k.json`), test pgvector IVFFlat (k-means
IVF) first; the data here favour k-means over Hilbert-ordered partitions at equal
candidates. A zig-hilbert `partitions` subcommand is only worth adding together with SOAR
secondary assignment, and only with a pre-registered comparison against k-means IVF.

### Not accepted, with a follow-up worth considering

- Synopsis (idea 8): to fit the budget, group chunks by note so the note prefix stays in
  the model's prompt cache, cap output at 60 tokens, or use a smaller local model; re-time
  and run overnight as a separate pre-registered study.
- HyDE (idea 10): +0.009 for 14 s per query is not worth adding to gbrain search; if
  revisited, generate from the dense top-3 chunks (query2doc with retrieved context)
  rather than from the question alone.
- Adaptive-k (idea 11): for the harness owner, the largest gap is a poor cut-off; a floor of
  at least 5 notes would be the minimum for a future variant.

## Reproduction

Scripts are in `ideas/`: `common.py` (snapshot cache, dense rule, metrics, statistics),
`idea01_rerank.py` to `idea12_muvera.py`, `summarize.py` (family Holm, `summary.json`).
Private text (note snapshot, sentence vectors, judge verdict files, synopsis and HyDE
generations) stays under `cache/` and `private/` on the local machine and is not committed.
