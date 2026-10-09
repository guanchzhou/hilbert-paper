# Clarifications and deviations: per-token efficiency study (M-series)

Written on 2026-10-06 before any result of the study was computed. The pre-registration
(`preregistration.md`, commit a3d8df1) leaves the details below open; each is fixed here, and the
run follows them. None changes a hypothesis, a grid, a split, a test or an acceptance threshold.

## Clarifications

**C1. Pack depth for methods that reorder.** For M3, M4 and M5 to M7 the pack walks the first 10
notes of the method's order with the baseline rule (each note's winning chunk kept whole if it
fits, skipped otherwise). The baseline also walks 10, so only the order differs between them. M1
and M2 select within their pool, whose size is part of their grid (M1) or 50 (M2).

**C2. M1 calibration data.** The isotonic map is fitted once, on the pool units of the tune half:
for each tune question, the top 50 notes of the baseline ranking, x = the note's best-chunk
cosine, y = 1 if the note is relevant. The same map serves every pool size and is applied to the
confirm half unchanged. Equal x values are merged first (mean of y, weighted by count). A new x
gets the value of the last fitted block whose start is at most x, and the first block's value below
the first start. Units are ordered by probability per token, ties in rank order.

**C3. M2 objective.** Similarities enter the coverage term as max(sim, 0), so coverage starts at 0
for the empty set, as the facility-location objective of Lin and Bilmes assumes non-negative
similarities. The greedy stops when no unit that fits has a positive gain. The best single unit is
compared by objective value. The pack is the selected set.

**C4. M4 order.** Greedy MAP inference runs until 10 units are chosen or the largest remaining
gain d_i^2 falls below 1e-12. If fewer than 10 are chosen, the rest of the pool follows in
baseline order.

**C5. Graph.** Nodes are the live notes with at least one embedded chunk. A link counts once per
ordered pair of notes; self links, and links to or from notes outside the node set, are dropped. The
undirected graph joins two notes if a link exists in either direction. The symmetric normalised
Laplacian follows Chung: L = I - D^(-1/2) W D^(-1/2) on notes with a positive degree, and a zero row
for an isolated note, so diffusion leaves its score unchanged.

**C6. M6 graph and seeds.** The k-nearest-neighbour graph keeps a note's k most cosine-similar
other notes among those with a positive cosine; the graph is made symmetric by taking the larger
weight of the two directions. The seeds y are the query cosines with the note vectors for the top
50 notes of the baseline ranking. F is rescaled per question onto [min s, max s] over all notes; a
constant F maps to min s.

**C7. M7 operator.** For a link from note i to note j, A_ij = 1. The advection operator is
L_adv = D_out - A^T, with D_out the out-degrees, so dx/dt = -L_adv x moves score along links and
conserves its total (Chapman and Mesbahi 2011). exp(-tM) is computed with `scipy.linalg.expm`.

**C8. M9 thresholds and loss.** Candidate thresholds are the cosines 1.000, 0.999, ..., 0.000 and
then minus infinity (no stop). At threshold λ, a question keeps the units of its 6,000-token
baseline pack whose note score is at least λ; because the walk is in decreasing score, these are a
prefix of the pack. Loss is 1 if none of them is relevant. On the tune half (n = 409), λ̂ is the
largest threshold with (n / (n + 1)) R̂(λ) + 1 / (n + 1) ≤ α (bound B = 1), where
α = the tune-half baseline miss rate at 6,000 tokens + 0.02. The primary test is the one-sided
Wilcoxon signed-rank test that baseline tokens minus M9 tokens is positive; the 20 percent
criterion is (mean baseline tokens - mean M9 tokens) / mean baseline tokens ≥ 0.20 on the confirm
half. Survival is compared with the 6,000-token baseline.

**C9. Selection ties.** Grids are walked in the order the pre-registration lists them, the first
parameter varying slowest; a later setting replaces the current choice only with a strictly higher
tune S̄.

**C10. Timing.** Wall time per question covers applying the chosen setting to the confirm half,
divided by its 408 questions. One-off work (fitting the isotonic map, the eigendecomposition, the
matrix exponentials, the neighbour graph and its factorisation) is reported separately.

**C11. Corpus check.** The run reads, at its start and at its end, the counts of live notes,
embedded chunks and links, and the latest `pages.updated_at` and `content_chunks.embedded_at`. If any
differs, the results are discarded and the run is repeated.

## Deviations

None. The tune-half baseline miss rate at 6,000 tokens is 0.2567, so M9's target is
α = 0.2767; the pre-registration names 0.245 as a placeholder to be replaced in this way.

## Run notes

- 2026-10-06, about 15:15 Helsinki time. Corpus: 1,228 live notes, of which 1,226 have embedded
  chunks; 5,169 chunks; 8,299 links, of which 5,591 directed pairs join two of those notes. The
  counts and timestamps were identical at the start and at the end of each run.
- The first attempt stopped before computing anything: Postgres orders slugs by its collation,
  not by code point, so the script now breaks ties with the slug order of `ideas/common.py`.
- The first complete run used float32 for M2's similarity matrix, and NumPy printed spurious
  floating-point warnings for that product (the values were correct to 1e-5 on every question).
  The committed script uses float64, as M3 and M4 already did; the rerun gave the identical
  family table.
- `crosscheck.py` recomputes the chosen setting of M1 to M7 on 20 confirm questions with
  zig-select 0.1.0 and zig-diffuse 0.1.0, which share no code with `run.py`. Every selection and
  order is equal; scores agree to within 3e-9.
