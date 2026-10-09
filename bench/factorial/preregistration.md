# Pre-registration: full factorial of the retrieval pipeline and a fractional agent-harness design

Written on 2026-10-05, before any measurement in this study was run. Factors, levels, skip
rules, metrics, analyses, correction methods and selection rules are fixed here. Results are
findings, not decisions. The sealed held-out file `qrels-test.json` is not opened. Any
departure is recorded with a time stamp in `factorial/deviations.md`; this file is not edited
after it is committed.

## 0. Common setup

- **Questions.** The 817 development questions of `qrels-dev.json` (SHA-256
  `87b6f6a2f0b87c8d55f8bf902e00099be366a363326295732e281a7b26a5f338`), saved query vectors
  `query-vectors.npy` (live instruction prefix).
- **Corpus.** The frozen snapshot of the twelve-idea study (`ideas/cache/corpus.pkl`, 1,223
  notes with chunks, 5,157 chunks; deviation D3 of that study), read read-only. Lexical scores
  are read from the live `content_chunks.search_vector` column with read-only SQL and TEMP
  tables, restricted to the snapshot's chunk ids. Nothing is written to the live database or
  configuration, and the corpus is not re-embedded.
- **Split.** The tune/confirm halves of `ideas/split.json` (seed 20261005; 409 tune, 408
  confirm questions).
- **Metrics.** R@10, MRR (on the top-10 list), nDCG@10 and hit@10 from `metrics.py`, exactly as
  in the twelve-idea study. Survival: at least one relevant note is in the 6,000-token pack.
  Tokens: `metrics.estimate_tokens` of the delivered pack. A pack walks units in rank order and
  keeps a unit if it fits in the remaining budget (no truncation).

## 1. Level 1: offline full factorial of the retrieval pipeline

### 1.1 Factors and levels

| Factor | Levels (reference first) |
|---|---|
| F1 first-stage search | dense (chunk vectors, cosine) / lexical-OR (chunk) / RRF hybrid of lexical-OR chunk and dense (k = 60, depth 50 each) |
| F2 note scoring for the dense part | best chunk / mean of chunks |
| F3 rerank of the top 50 | off / on (qwen3-reranker-0.6b, default instruction) |
| F4 link-graph personalised PageRank | off / on |
| F5 mean-direction removal (dense part) | off / on |
| F6 Rocchio PRF (dense part) | off / on |
| F7 candidate filter before scoring | none / hk1 level-1 probes with 16 ranges / Hilbert balanced partitions + SOAR |
| F8 evidence unit in the 6,000-token pack | chunk / window / section / page / sentence-pruned |

Fixed parameters, all taken from the twelve-idea study and not re-tuned here:
- F1 lexical-OR: `replace(websearch_to_tsquery('english', q)::text, ' & ', ' | ')::tsquery`
  against `content_chunks.search_vector`, chunk score `ts_rank_cd`, note score = maximum over
  its chunks (best chunk), ties by slug; the lexical list holds the top 50 notes with a match.
  RRF: `1/(60 + rank)` summed over the lexical top 50 and the dense top 50, ties by slug.
- F2 mean of chunks: the note vector is the normalised mean of all of the note's unit chunk
  vectors (in the space produced by F5); a note is eligible when at least one of its chunks
  passes F7.
- F3: the top 50 notes of the ranking produced by F1, F2, F4, F5, F6 and F7; each note is
  represented by its winning chunk (1.3); new top 10 = the 10 highest reranker scores among
  those 50, ties by previous rank; positions 11 to 50 are re-ordered by reranker score, notes
  beyond 50 keep their order.
- F4: personalised PageRank on the undirected note link graph (`ideas/cache/edges.json`),
  restart distribution proportional to the min-shifted first-stage scores of the top 20 notes,
  alpha = 0.3, 60 iterations, dangling mass restarts; fused score
  `(1 - beta) * minmax(first stage) + beta * minmax(PPR)` with beta = 0.3 (idea 9 chosen
  setting). Notes without a first-stage score take the minimum (0 after min-max); a note is in
  the ranking if it has a first-stage score or positive PPR mass.
- F5: subtract the shared mean of the unit chunk vectors from chunk and query vectors and
  re-normalise (idea 4 variant a, the best of the four variants).
- F6: `q' = normalise(0.6 q + 0.4 * mean of the top-5 chunk vectors)` of the initial chunk
  ranking over the F7 candidates, in the F5 space, then score again (idea 5 chosen setting).
- F7 hk1: zig-hilbert `key --dims 8 --bits 8 --seed 9e3779b97f4a7c15 --probe 1 --ranges 16`
  on the query vector, candidates = keyed chunks whose stored key falls in a returned range.
  Partitions: `hilbert+soar/M128/C12` of idea 6 (Hilbert order of keyed chunks cut into 128
  equal partitions, SOAR secondary assignment with lambda = 1, route to the 12 nearest
  partition means). Both filters use the original query vector. The filter restricts the
  chunks that any first-stage scorer may score: dense scores only candidate chunks, lexical
  scores only candidate chunks that match the query.
- F8: chunk = winning chunk text; window = winning chunk plus one neighbour on each side;
  section = `run_measure.section_span`; page = `compiled_truth`; these four pack the top 10
  notes. Sentence-pruned = idea 2 configuration: top 30 notes, winning chunk split into
  sentences with the idea 2 rule, the 5 sentences with the highest cosine to the unit query
  vector kept in original order, sentences embedded as documents through local MLX (no
  instruction), cached privately.

### 1.2 Pipeline order and skip rules

Order: F7 filter, then F1 scoring (with F5 and F6 applied to the dense part and F2 deciding
note scores of the dense part), then F4 fusion, then F3 rerank, then F8 packing.

Skip rule. F2, F5 and F6 act only on the dense part, so with F1 = lexical-OR they are inert;
those cells are not enumerated separately (lexical uses best chunk, F5 off, F6 off). Every
other combination is valid and enumerated:
- lexical: F3 x F4 x F7 x F8 = 2 x 2 x 3 x 5 = 60 cells;
- dense and hybrid: 2 x (2 x 2 x 2 x 2 x 2 x 3 x 5) = 960 cells;
- total 1,020 cells, of which 204 distinct rankings (F8 changes only the pack).

For the Shapley analysis an inert factor is evaluated as off (its value function equals the
cell with that factor at reference).

### 1.3 Winning chunk (evidence anchor) of a note

The chunk that gave the note its first-stage score: for dense parts the highest-cosine
candidate chunk in the F5/F6 space (also under F2 = mean, where it does not set the score);
for lexical the highest-`ts_rank_cd` candidate chunk; for hybrid the dense winner when the
note has a dense candidate chunk, else the lexical winner. A note that enters only through
F4 uses its first chunk (lowest chunk index).

### 1.4 Measured quantities per cell (all 817 questions)

R@10, MRR, nDCG@10, hit@10, survival, tokens delivered, candidates scored, rerank pairs, and
latency per query.

- Candidates scored: vectors compared in the first stage (chunk vectors; note vectors under
  F2 = mean; a second pass under F6 counts again; partition-mean comparisons of F7 counted
  separately) plus chunks scored lexically. Rerank pairs are reported separately.
- Latency is composed, not measured end to end: each component's per-query latency is
  measured once and summed for the components present in a cell. Components: query embedding
  (MLX, median of 30 single calls; needed whenever a dense part or a filter is present),
  dense scoring over the candidate count (numpy, single-query timing, scaled linearly by
  candidates scored), lexical SQL (median per-query time over a fixed sample of 100 questions),
  RRF fusion, PPR, mean removal, Rocchio (second pass), hk1 key and range scan, partition
  routing, reranker (median measured seconds per pair from the cached 50-document calls times
  the number of pairs), sentence pruning (cosine over precomputed sentence vectors; the
  on-the-fly sentence-embedding cost is reported separately). The composition is reported.

### 1.5 Reranker cost rule

Existing cache `ideas/cache/rerank-default.json` covers the dense top 50 under the reference
pipeline. Pairs are keyed by (question index, chunk id); only uncached pairs are scored, with
the same server, model, instruction and request format. Before scoring, the number of new
pairs is counted and the time is projected from the cached per-pair rate. Questions are scored
in a fixed random order (`default_rng(20261005).permutation(817)`), all needed pairs of one
question at a time. If not all questions are complete by 05:00, scoring stops and the
analysis set for **every** cell is the set of completed questions (a random subset; target at
least 50%), so all cells are compared on the same questions; full-set metrics of rerank-off
cells are reported as secondary. The same rule (question order, stop at 05:00) applies to the
sentence embeddings needed for sentence-pruned packs.

### 1.6 Analyses

1. **Main effects.** For each factor level versus its reference, the per-question contrast is
   the mean, over all valid combinations of the other factors, of (metric at level minus
   metric at reference). For F2, F5, F6 the averaging is over dense and hybrid cells only.
   Effect = mean over questions; 95% percentile bootstrap CI over questions (10,000 resamples,
   seed 20261005); two-sided Wilcoxon signed-rank test on non-zero per-question contrasts.
   Metrics: R@10, survival, tokens, latency (and MRR, nDCG@10 descriptively). F8 has no effect
   on ranking metrics and is excluded from the R@10 family. Latency is deterministic per cell
   (composed), so latency effects are reported without tests.
   Correction: Holm within each metric family over the main-effect contrasts (13 contrasts:
   F1 2, F2 1, F3 1, F4 1, F5 1, F6 1, F7 2, F8 4; 9 for R@10, which excludes F8).
2. **Two-factor interactions.** For each pair of factors and each pair of non-reference levels
   (a, b): per-question contrast `m(a,b) - m(a,ref) - m(ref,b) + m(ref,ref)` averaged over all
   valid combinations of the remaining factors; same CI and test. Correction: Benjamini-Hochberg
   FDR (q = 0.05) within each metric family (70 contrasts; 34 for R@10, which excludes F8), because the number of tests is large. Holm-adjusted values are reported as
   secondary.
3. **Shapley points.** For each "level profile" of the multi-level factors (F1 non-reference
   level in {lexical, hybrid}, F7 in {hk1, partitions}, F8 in {window, section, page, pruned};
   16 profiles), define a binary game with 8 players: player i "on" sets factor i to its
   profile level (binary factors to on), "off" to reference; inert factors are off. The value
   of a coalition is the mean metric of that cell. Exact Shapley values (all 2^8 coalitions,
   weights |S|!(n-|S|-1)!/n!) are computed per profile. A binary factor's points are the mean
   over the 16 profiles; a multi-level factor's points at level l are the mean over the
   profiles in which it takes l. The Banzhaf value (uniform average of marginal
   contributions) is reported alongside. Per-question Shapley values give bootstrap CIs.
   Metrics: R@10, survival, tokens, latency, MRR, nDCG@10.
4. **Leaderboards (top 20 per axis), selection on the tune half, values on both halves.**
   - Best recall: ranking configurations (204; F8 irrelevant) by tune R@10.
   - Best survival at the fixed 6,000-token budget: all cells by tune survival.
   - Fewest tokens: cells with tune survival >= best tune survival - 0.02, by tune mean tokens.
   - Fastest: ranking configurations with tune R@10 >= best tune R@10 - 0.02, by composed
     latency (packing excluded; sentence pruning adds to latency in pack axes only).
   Ties: by the next metric (MRR for recall axes, tokens for survival, survival for tokens,
   R@10 for latency), then by fewer active components.
5. **Confirmation.** For each axis the tune-chosen configuration is compared with the reference
   cell (dense, best chunk, all off, no filter, chunk pack) on the confirm half: R@10 and
   survival with paired tests (Wilcoxon for R@10, exact McNemar for survival), tokens and
   latency descriptively. Holm over these confirm-half tests (at most 4 primary tests:
   recall-axis R@10, survival-axis survival, tokens-axis survival non-inferiority with margin
   0.02 (paired t-test on difference plus margin, deviation D1 of the twelve-idea study),
   latency-axis R@10 non-inferiority with margin 0.02 (same test)).
6. **Pareto fronts.** Non-dominated cells on (R@10 max, tokens min, latency min) and on
   (survival max, tokens min, latency min), all 817 questions (or the analysis set of 1.5).

### 1.7 Reproduction checks (before analysis)

The factorial code must reproduce, on the snapshot, within 0.005: dense R@10 0.698;
lexical-OR chunk R@10 0.534; hybrid RRF 0.701; note-mean 0.683; dense rerank 0.735; centering
(H4a) 0.703; Rocchio chosen (all-817 value from idea 5); PPR chosen confirm-half 0.712; hk1
L1x16 0.3045; partitions chosen 0.645; chunk pack survival 0.755; pruned-pack survival 0.868.
A failed check is investigated and any fix is recorded as a deviation before the analysis.

## 2. Level 2: agent harness, fractional design

Run only after level 1 is complete. Uses the 10 pilot tasks (`agents/tasks.json`, private,
never committed) and the existing harness code (copied and extended under
`factorial/`, the original files unchanged). RTK is toggled only through the harness's
per-run mechanism (Claude Code `--settings` injection, Cursor per-condition HOME profile);
the user's global hook and settings files are never edited.

### 2.1 Factors

| Factor | Levels |
|---|---|
| Host | local Ollama qwen3.8 27B (`local_agent.py` loop) / Cursor CLI / Claude Code |
| Model tier (cloud) | large / small-fast / shared (one model available in both cloud hosts) |
| Retrieval tool | plain files (grep and read) / gbrain MCP (search and get_page) |
| RTK | off / on (cloud, files only; inert elsewhere) |
| Context | local: num_ctx 8K / 32K / 128K; cloud: gbrain evidence budget 2K / 6K / 24K tokens (gbrain only; inert for files) |
| Packing unit for gbrain results | chunk / page / best level-1 unit deliverable by gbrain (window or section, whichever has the larger level-1 main effect on survival on the tune half) |

Models, fixed by this rule after checking availability on 2026-10-05: Claude Code large
`claude-opus-5-5`, small `claude-haiku-4-5`, shared `claude-sonnet-5-5`; Cursor CLI large
`gpt-5.6-sol-high`, small `composer-2.5`, shared `claude-sonnet-5-5-medium`. If a model
fails to start, the next model of the same tier in the host list is used and recorded as a
deviation.

Cloud evidence budget and packing unit are enforced by a small stdio MCP relay in front of
`gbrain serve` that sets `token_budget` and `return_unit` on every search call (the model
cannot override them); everything else is passed through unchanged. Local runs pass the same
parameters directly; the local tool budget follows `local_agent.py` (a quarter of num_ctx).

### 2.2 Design

Three strata, each a pairwise-covering array over its non-inert factors (every pair of
levels of every two factors appears at least once):
- Cloud gbrain: host (2) x tier (3) x budget (3) x unit (3): Taguchi L9 orthogonal array
  (columns tier, budget, unit) crossed with host by assigning host = row parity, then
  completed so that each host x tier, host x budget and host x unit pair appears (rows added
  until covered; expected 9 to 12 rows).
- Cloud files: host (2) x tier (3) x RTK (2): full factorial, 12 rows.
- Local: files x num_ctx (3 rows) and gbrain x num_ctx (3) x unit (3) (9 rows), 12 rows.

Tasks. Each row runs on 5 tasks, assigned by rotation (row r gets tasks
`(r + j*2) mod 10`, j = 0..4) so that every task is used about equally. Runs are executed in
rounds: round j runs task j of every row (rows shuffled, seed 20261005), so stopping early
leaves a balanced design. Repeats: the shared-model rows of both cloud strata (host
comparison) are repeated twice more on their first two tasks, for spread. Cloud runs are capped
at about 150 in total; no new run starts after 07:30. Cloud runs go three at a time; local runs
one at a time.

### 2.3 Measures and analysis

Per run: page found (the reference slug appears in the SOURCES line), blind judge score 0/1/2
(`agents/judge.py` rubric and blinding, judge `claude-sonnet-5-5`), input and output tokens,
wall time, tool calls, errors. Analysis is exploratory: marginal means per factor level with
bootstrap CIs over runs; paired contrasts where runs match on task and all other factors.
Primary contrasts (Holm over these five, on page found and judge score separately): Cursor
CLI vs Claude Code on the shared model; gbrain vs files; RTK on vs off (tokens in, page found);
largest vs smallest context; best unit vs chunk. Exact sign tests on matched pairs.

## 3. Outputs

Under `factorial/`: `level1-cells.json` (per cell metrics on all, tune, confirm),
`level1-effects.json`, `level1-shapley.json`, `level1-leaderboard.json`, `level1-pareto.json`,
`level2-runs.json` (numeric fields only, no answers), `level2-summary.json`, `REPORT.md`
(findings per axis and an improvement-plan section), `deviations.md`. Scripts and numeric
results are mirrored to the paper repository under `bench/factorial/`. Private note text,
answers, judge transcripts and task files stay under `factorial/private/` and are never
committed.
