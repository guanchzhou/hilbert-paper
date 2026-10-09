# Pre-registration: relevant notes per token spent (M-series)

Written on 2026-10-06, before any of the experiments below were run. Every hypothesis, metric,
comparison, parameter grid, test and acceptance criterion is fixed here. Results are findings,
not decisions. The sealed held-out file `qrels-test.json` is not opened. The study is
exploratory with respect to the paper's confirmatory hypotheses H1 to H4, and confirmatory within
itself: each idea is tuned on one half of the development questions and tested once on the other.

## 1. Question

Which mathematical methods put more relevant notes into the evidence pack a model reads, at the
same number of tokens? The pack is the text gbrain hands to the model. Its default size is 6,000
tokens because gbrain's `DEFAULT_RETURN_BUDGET` is 6,000 (`src/core/search/evidence-delivery.ts`);
this study therefore also measures smaller budgets.

## 2. Common setup

**Data.** The 817 development questions in `qrels-dev.json` (SHA-256
`87b6f6a2f0b87c8d55f8bf902e00099be366a363326295732e281a7b26a5f338`); the saved query vectors
`query-vectors.npy` (817 x 1,024); and, read-only from gbrain Postgres at run time, the live chunk
vectors, chunk texts and the note-to-note links. The run records the corpus counts it saw.

**Baseline ("B").** Cosine between the unit query vector and every unit chunk vector; a note scores
the maximum over its chunks; notes ranked by that score, ties broken by slug. The baseline pack
walks the top 10 notes in rank order and keeps each note's winning chunk if it fits in the
remaining budget (units are not truncated; a unit that does not fit is skipped and later units are
still considered). This is the paper's chunk pack; at 6,000 tokens on all 817 questions it keeps a
relevant note for 0.755 of questions.

**Budgets.** b in {1,000, 2,000, 3,000, 4,000, 6,000} tokens, counted with
`metrics.estimate_tokens`.

**Candidate pool.** Methods that select or reorder units work on the winning chunks of the top 50
notes of the baseline ranking. Methods that change the ranking (M5 to M7) rank all notes, then pack
with the baseline rule.

**Metrics, per question.**
- Survival at budget b, S_b: 1 if at least one relevant note is in the pack, else 0.
- Primary outcome S̄: the mean of S_b over the five budgets (0 to 1).
- Secondary: tokens delivered at each budget; distinct relevant notes in the 6,000-token pack;
  R@10 of the method's ranking (M5 to M7 only); wall time per question.
- A ratio such as "relevant notes per token" is not used as an outcome: delivering one chunk would
  maximise it.

**Split.** The split of the twelve-idea study, `ideas/split.json`:
`perm = numpy.random.default_rng(20261005).permutation(817)`; tune = sorted first 409 indices,
confirm = sorted remaining 408. Every parameter in section 4 is chosen on the tune half by the
highest tune S̄ (ties: the smaller value in the grid order given). The chosen setting is tested once
on the confirm half. Both halves are reported.

**Statistics.** Effects are paired mean differences (method minus baseline) over confirm-half
questions, with 95% percentile bootstrap intervals (10,000 resamples, seed 20261005). The primary
test for M1 to M7 is the one-sided Wilcoxon signed-rank test on non-zero paired differences of S̄
(`zero_method="wilcox"`). Per-budget survival is tested with the exact McNemar test (descriptive,
Holm within the idea). Non-inferiority with margin m uses the shifted differences (method minus
baseline plus m) and reports the larger p of the one-sided Wilcoxon signed-rank test and the
one-sided paired t-test, because the shifted Wilcoxon test alone is unreliable when most paired
differences are zero (deviation D1 of the twelve-idea study).

**Multiplicity.** The eight primary p-values (M1 to M7 and M9) form one family, adjusted with
Holm's step-down procedure at α = 0.05.

**Acceptance.** An idea is accepted when its primary criterion in section 4 holds on the confirm
half and its Holm-adjusted primary p is below 0.05. Ideas are reported in the same table whether
accepted or not.

## 3. Sources

Each idea has a published source, checked on 2026-10-06.

| Idea | Source |
|---|---|
| M1 | Dantzig, G. B. (1957). Discrete-variable extremum problems. Operations Research 5(2), 266 (doi:10.1287/opre.5.2.266). Zadrozny, B., Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD 2002. |
| M2 | Lin, H., Bilmes, J. (2011). A class of submodular functions for document summarization. ACL-HLT 2011, 510–520. |
| M3 | Carbonell, J., Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. SIGIR 1998. |
| M4 | Kulesza, A., Taskar, B. (2012). Determinantal point processes for machine learning. Foundations and Trends in Machine Learning 5(2–3), 123–286. Chen, L., Zhang, G., Zhou, H. (2018). Fast greedy MAP inference for determinantal point process to improve recommendation diversity. NeurIPS 2018, 5627–5638. |
| M5 | Kondor, R. I., Lafferty, J. D. (2002). Diffusion kernels on graphs and other discrete input spaces. ICML 2002, 315–322. Chung, F. (2007). The heat kernel as the pagerank of a graph. PNAS 104(50), 19735–19740. |
| M6 | Zhou, D., Weston, J., Gretton, A., Bousquet, O., Schölkopf, B. (2004). Ranking on data manifolds. NIPS 16 (2003). |
| M7 | Chapman, A., Mesbahi, M. (2011). Advection on graphs. 50th IEEE CDC-ECC, 1461–1466. The paper concerns coordination of networked agents; applying advection to ranking is this study's adaptation, and no prior retrieval use was found. |
| M9 | Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. (2024). Conformal risk control. ICLR 2024 (arXiv:2208.02814). |

Idea M8 is not used, so that the numbering matches the plan.

## 4. Hypotheses

**M1, knapsack by value density.** Isotonic regression (pool-adjacent-violators) maps a note's
best-chunk cosine to the probability that the note is relevant, fitted on the tune half. Units in
the pool are added greedily in decreasing order of probability per token, each kept if it fits.
Grid: pool size {10, 20, 50}. H: S̄ improves by at least +0.02.

**M2, budgeted submodular coverage.** Objective F(S) = Σ_i r_i · max_{j∈S} sim(i, j) + λ · Σ_{j∈S} r_j
over the pool, where r_i is the query cosine of unit i and sim is the cosine between unit vectors.
Budgeted greedy: add the unit with the largest gain divided by cost^p (cost = tokens) that fits,
and compare with the best single unit that fits (the guarantee of the budgeted greedy). Grid:
λ in {0, 0.5, 1, 2}, p in {0.5, 1}. H: S̄ improves by at least +0.02.

**M3, maximal marginal relevance.** Select units in the order that maximises
λ · r_i − (1 − λ) · max_{j∈S} sim(i, j), then pack with the baseline rule. Grid: λ in
{0.5, 0.6, 0.7, 0.8, 0.9}. H: S̄ improves by at least +0.02.

**M4, determinantal point process.** Kernel L = diag(q) · K · diag(q), with K the Gram matrix of
unit chunk vectors and q_i = exp(α · r_i). Fast greedy MAP inference (Chen et al. 2018) gives an
order; pack in that order with the baseline rule. Grid: α in {2, 5, 10, 20}. H: S̄ improves by at
least +0.02.

**M5, heat diffusion on the link graph.** The note link graph is made undirected; L is its
symmetric normalised Laplacian. Note scores s (best-chunk cosine) become (1 − β) · s + β · exp(−tL) s,
computed by eigendecomposition. Grid: t in {0.5, 1, 2, 4}, β in {0.1, 0.3, 0.5}. H: S̄ improves by
at least +0.02.

**M6, manifold ranking.** A symmetric k-nearest-neighbour graph over note vectors (the mean of the
note's unit chunk vectors), weights equal to cosine, S = D^(−1/2) W D^(−1/2). Scores
F = (I − aS)^(−1) y with y_i = max(0, cos(q, note_i)) for the top 50 notes and 0 otherwise; the
final score is (1 − β) · s + β · F rescaled to the range of s. Grid: k in {10, 20}, a in
{0.5, 0.8, 0.95}, β in {0.3, 0.5}. H: S̄ improves by at least +0.02.

**M7, advection–diffusion on the directed link graph.** Links point from the note that contains
them to the note they name. With L_adv the advection operator of Chapman and Mesbahi (outflow along
directed edges) and L_sym the symmetric normalised Laplacian, scores become
(1 − β) · s + β · exp(−t · (γ · L_adv + (1 − γ) · L_sym)) s, computed with the matrix exponential.
Grid: t in {0.5, 1, 2}, γ in {0.25, 0.5, 0.75}, β in {0.1, 0.3}. H: S̄ improves by at least +0.02.

**M9, conformal risk control of the budget.** The baseline ranking is unchanged; per question the
pack stops after the first k units, where k is the smallest value whose calibrated risk of missing
every relevant note is at most the target. Loss for a question at threshold λ on the cosine of the
k-th unit: 1 if no relevant note is in the units kept, else 0 (monotone in λ). λ̂ is chosen on the
tune half by conformal risk control with target risk α = 0.245 + 0.02 (the baseline's 6,000-token
miss rate on the tune half plus 0.02; the tune-half value replaces 0.245 when computed). Primary
outcome: tokens delivered. H: on the confirm half, mean tokens fall by at least 20 percent against
the 6,000-token baseline (one-sided Wilcoxon signed-rank test, the primary p), and survival is
non-inferior with margin 0.02 (secondary test as in section 2). Accepted only if both hold. The
realised miss rate is reported against α.

## 5. Procedure and limits

- Implementation: Python with NumPy and SciPy on cached vectors; the live gbrain database is only
  read. CPU only; the study must not run at the same time as a GPU job or the 27B model.
- Each idea writes `bench/efficiency/m<k>.json` with the chosen parameters, both halves, the
  per-budget survival, the tests and the timings; `summary.json` gathers the family and the Holm
  adjustment.
- If the corpus changes during the run, the run is restarted from the beginning.
- Any departure from this document is recorded in `bench/efficiency/deviations.md` before results
  are read.
