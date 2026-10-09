# Pre-registration: six follow-up studies

Written on 2026-10-07, before any data in these studies was collected. Results are findings, not
decisions. Development questions only; `qrels-test.json` is not opened. Every judgement uses the
idea 2 judge (`ideas/idea02_prune.py`: local `qwen3.8:latest`, temperature 0, `think` off,
16,384-token window, JSON reply). No cloud model is used. One GPU-heavy job runs at a time, under a
read-only monitor that stops it at kernel memory-pressure level 4, more than 26 GB of swap in use,
or less than 15 GB of free disk (deviation D11 of the factorial study). Any departure is written to
`deviations.md` with its time before its results are read. Seeds: `numpy.random.default_rng(20261008)`
unless stated otherwise. All vectors come from the frozen snapshot (`ideas/cache/corpus.pkl`).

## S6. Hilbert keys as a map for compositional questions (first)

Idea (the author's): an agent splits a question such as "who in Cloud Core contributed to open
source" into facets; each facet has a region ("tree") of hk1 cells, and the answer is the
intersection of the regions. The paper so far tested hk1 only as a filter for one whole-question
vector.

Questions and gold.
- Link graph: live `links` rows of types `mentions`, empty, `related_to`, `see_also`, `related`,
  `topic`, `relates_to`, between `default`-source pages that are in the snapshot, self-links removed.
- A pair (A, B) of target pages qualifies when each has at most 60 distinct linking notes and they
  have 3 to 15 common linking notes. Sixty pairs are drawn without replacement; a page may appear in
  at most two drawn pairs (pairs that break this are skipped in draw order).
- Question: `Which notes are about both <title of A> and <title of B>?`. Gold: the notes that link to
  both A and B. A and B themselves are excluded from every result set and from the gold.
- The author checks 30 of the questions (the first 30 in draw order) on the labelling page: is the
  gold list correct and complete, yes or no. The share of yes is reported; S6 results are reported
  on all 60 and on the questions marked yes.

Arms (each returns a set of notes; A and B removed).
1. Whole-question dense: the question, embedded with the default query instruction, note scores by
   best chunk (`Bench.dense`); the top 10 and the top 30 notes.
2. Per-facet dense: each title embedded with the same instruction; the top 50 notes per facet; the
   intersection.
3. Per-facet hk1: each title vector keyed with `zig-hilbert key --probe 1 --ranges 16` (the paper's
   best hk1 setting); chunks whose stored hk1 key lies in a probe range (keyed chunks only), lifted to
   notes; the intersection of the two facets' note sets.
4. Links join: the notes linking to both pages. It reproduces the gold by construction and is
   reported only as an upper bound with its read cost, not as a competitor.

Measures per question: set recall |S∩G|/|G|; precision |S∩G|/|S| (0 for an empty set); size |S|;
notes examined (arm 1: 10 or 30; arm 2: the union of both facet lists; arm 3: the union of both
regions; arm 4: the union of the two pages' linking notes); tokens of the returned notes (gbrain's
estimate). For arm 3, also each facet region's coverage of the gold.

Tests: paired two-sided Wilcoxon on set recall, Holm over three: arm 3 vs arm 2 (primary), arm 2 vs
arm 1 at top 30, arm 3 vs arm 1 at top 30. Prediction, written before the run: arm 3 below arm 2,
because one sign key finds 0.304 of what exhaustive search finds and an intersection multiplies the
two facets' losses.

Not tested here: a Hilbert key over explicit attribute axes (team, project, time). Its attributes
would first have to be extracted, and automatic extraction in gbrain is off.

S6b, the agent. The first 20 questions; the level 2 local agent loop (`agents/local_agent.py`,
`qwen3.8:latest`, 8K window, evidence budget a quarter of the window) with three tool sets: (a) gbrain
`search` and `get_page`; (b) (a) plus `facet_intersect(a, b)` returning arm 2's set with titles and
slugs; (c) (a) plus `facet_intersect(a, b)` returning arm 3's set. The system prompt asks for the
slugs of the answering notes, one per line. Measures: set recall and precision of the slugs in the
answer against the gold, tool calls, input tokens, wall time. 60 runs; descriptive only, no test.
The agent searches the live brain, so notes newer than the snapshot count against precision.

## S1. Is the judge valid?

S1a, length. For each of the 150 questions of `factorial/judge-packs.json`, pack B (reference
chunk pack) is cut to the token count of that question's pack A (sentence pack): B's units are kept
in rank order while the running total stays within A's token count. The judge rates the cut pack
(B-cut) once. Tests against the cached judgements: exact McNemar, two-sided, B-cut vs A and B-cut vs
B, with the paired difference and a 95% bootstrap interval (10,000 resamples). Reading rule: length
is said to explain the gap if B-cut vs B is significant at 0.05 and B-cut is closer to A than to B;
content is said to explain it if B-cut vs B is not significant and B-cut vs A is. Other outcomes are
reported as they are. B-cut is judged alone, while A and B were judged in pairs; this is a known
difference.

S1b, human labels and prediction-powered inference. Fifty of the 150 questions are drawn; both
packs of each are labelled by the author: does the pack contain the information needed to answer
the question, yes or no. The page shows one question and one pack at a time in random order, never
the condition or the judge's answer, and saves every label at once. Estimands: the yes-rate of A, of
B, and the paired difference A minus B. The prediction-powered estimate is the judge's mean over all
150 questions plus the mean of (human minus judge) over the 50 labelled ones, with the normal
interval of Angelopoulos et al. (2023); the difference uses the paired values. Reported alongside:
the human-only estimate on the 50, the judge-only estimate on the 150, raw agreement and Cohen's
kappa.

## S2. How incomplete are the relevance labels?

One hundred development questions are drawn. For each, the union of the top 10 notes of the
reference configuration (dense, best chunk) and of the best-recall configuration of the factorial
leaderboard (with reranker), minus notes already labelled relevant, is judged with this prompt:

    You are judging relevance. Question:
    {q}

    Note:
    {text}

    Is this note relevant to the question, that is, does it contain information that answers the
    question or a substantial part of it? Reply with JSON only: {"answer": "yes"} or {"answer": "no"}

The note text is the note's chunks in order, cut at 6,000 tokens. If the number of calls counted
before judging exceeds 1,200, only the first 80 drawn questions are used. Outputs: the share of
unlabelled top-10 notes judged relevant (95% bootstrap interval over questions); R@10 of both
configurations with the original labels and with the extended labels (original plus judge-yes); the
reranker's gain under both, with a paired two-sided Wilcoxon. Thirty judged pairs (15 judged yes, 15
judged no, drawn) are labelled by the author on the same page; agreement is reported per stratum.

## S3. Cross-polytope LSH and whitening (zig-lsh)

Library `zig-lsh` (pure Zig 0.17, no dependencies) with a fast Hadamard rotation (three rounds of
seeded random signs, padded to a power of two), cross-polytope hashing (index and sign of the
largest-magnitude rotated coordinate, k hashes per table), hyperplane hashing (k sign bits of the
rotated vector), L tables, multi-probe. It is verified against an independent NumPy reference with
dense rotation matrices before any measurement.

Measurement on the 817 development questions and the snapshot chunks. Families: hyperplane with
k = 8 bits, cross-polytope with k = 1 (2,048 buckets at 1,024 dimensions). Preprocessing: raw,
centred (corpus chunk mean removed), whitened (centred, rotated onto the corpus covariance's
eigenvectors and scaled by 1/sqrt(eigenvalue + 1e-6); fitted on chunk vectors only, applied to
questions too). L in 1, 2, 4, 8, 16, 32, 64, 128; multi-probe off, and 1 and 4 extra probes per
table as secondary. Candidates are rescored by exact cosine on the original vectors. Measures: the
share of questions whose best relevant chunk collides in at least one table; the mean share of
chunks scanned; R@10 after rescoring.

Primary comparison: R@10 of cross-polytope against hyperplane, centred, each at the smallest L whose
mean scanned share reaches 10 percent, paired two-sided Wilcoxon. Secondary: the same at 20 percent;
the effect of whitening on each family at the same rule; and exhaustive-cosine R@10 on whitened
against raw vectors (Su et al. 2021), paired two-sided Wilcoxon. hk2 (`hk2-sweep.json`) is shown as
the reference curve.

## S4. Compression baselines against a hybrid pack

Same 150 questions and judge as S1. New packs, each within 6,000 tokens:
- H (hybrid): the best-survival ranking; the best chunk of each of the top 10 notes, then the pruned
  sentences (as in pack A) of the notes ranked 11 to 30.
- R (RECOMP extractive, `fangyuan/nq_extractive_compressor` at revision
  `46f98c1f57fa919c72d86ced16e3ece3403f9e9e`): pack A's candidate sentences (sentences of the top 30
  notes of the best-survival ranking) scored against the question with the compressor as its model
  card describes, kept best first up to pack A's token count for that question.
- L (LongLLMLingua, `llmlingua` 0.2.2 with `Qwen/Qwen2.5-1.5B` at revision
  `8faed761d45a263340a0528343f099c05c9a4323`): pack B's chunks as the contexts, the question as the
  question, question-aware compression with the library's LongLLMLingua defaults, target token count
  pack A's token count.

Packs are built first and the compressor models unloaded before the judge runs (450 calls, each pack
judged once). Tests: each of H, R and L against B, non-inferiority at the 0.02 margin (the larger of
the Wilcoxon and shifted paired-t p-values, as before), and exact McNemar against A. Survival and
tokens are reported for H and R; survival is not defined for L after token-level compression. RECOMP
was trained on English questions, and part of this corpus is Russian; this is a stated limitation.

## S5. How many questions does the reranker comparison need?

From the 817 development questions, the per-question R@10 differences of the best-recall
configuration against the reference, recomputed from the frozen snapshot. A paired bootstrap power
curve (5,000 resamples per size, two-sided Wilcoxon at 0.05) gives the number of questions for 80
percent power at two effect sizes: the held-out difference (+0.016) and the development difference
(+0.050), the differences being shifted to each effect size. No held-out data is used.

## Addendum (recorded 2026-10-07 19:14, before any S6c data): S6c and the author's acceptance rule

S6c repeats S6 with hk1 keys on sentences instead of chunks. The sentences are those of
`stage2.Sentences` (164,824 sentences of 5,068 chunks); the 455 sentences of the remaining 89
chunks are embedded the same way first. Each sentence gets an hk1 key (`zig-hilbert key`, the same
space and seed as the chunk keys).

Arms on the same 60 questions, A and B removed:
- sentence-dense: each facet title embedded as in S6; notes ranked by their best sentence;
  the top 50 notes per facet, intersected. Evidence of a returned note: its best sentence for each
  facet.
- sentence-hk1: each title's level-1 probe with 16 ranges over the sentence keys; a note is in
  a facet's region if any of its sentences is; the regions intersected. Evidence of a returned note:
  its sentences in either facet's region.

Measures as in S6, plus evidence tokens (the evidence sentences of the returned notes) and
retrieval time per question (from the facet vectors to the returned set, embedding excluded and
reported separately). Tests: paired two-sided Wilcoxon on set recall, sentence-hk1 vs
sentence-dense, and sentence-hk1 vs the chunk-level hk1 arm of S6, Holm over two. Prediction:
sentence-hk1 above the chunk-level hk1 arm (0.182) but below sentence-dense.

Acceptance rule, set by the author before this run: hk1 is accepted if, against its comparator, it
spends at least 25 percent fewer model tokens per correct answer and takes at most twice the time
per correct answer. Tokens per correct answer is the sum over questions of the tokens returned
divided by the sum of gold notes found; time per correct answer likewise with retrieval time. The
primary token measure is evidence tokens, whole-note tokens are secondary. The primary comparator of
sentence-hk1 is sentence-dense; the rule is also applied to the chunk-level arms of S6 and, after the fact and
descriptively, to the S6b agent runs (input tokens and wall time per gold note found).

## Addendum 2 (recorded 2026-10-08 13:49, before any S7 data): S7, decision models as judges

Two open decision models are compared with the 27B judge of S1, S2 and S4, at the author's request:
Cloudflare Clef-Flash (Hugging Face `Cloudflare/clef-flash`, revision
`fde727a287004204b7518dcc983fe64379776712`, Qwen3.5-9B backbone with a joint schema head) and
autotrust JEV-9B (`autotrust/JEV-9B`, revision `b63f651ce8ed64481d3f5e73ecdb05f740042f01`,
Qwen3.5-9B with a decision LoRA and a 24-slot head). Both answer a typed yes/no ("noul") question
with a probability in one forward pass. They run on this Mac's GPU (MPS) in bfloat16 through
`transformers`, one model at a time, under the D11 stop lines; the 27B judge is not loaded. Clef-Flash
runs first; JEV-9B is downloaded only if at least 15 GB of disk stays free.

Items, all already judged by the 27B judge:
- Sufficiency: the 150 questions of the answer-quality check with each of its six packs (A, B, B-cut,
  H, R, L), 900 items. State: `{"question": q, "context": pack}`; question: noul, instructions
  "Does the context contain the information needed to answer the question?".
- Relevance: the 1,059 pairs of S2. State: `{"question": q, "note": text}` with the same note text
  as S2; question: noul, instructions "Is this note relevant to the question, that is, does it contain
  information that answers the question or a substantial part of it?".
- Input length: Clef-Flash takes the whole state (its encoder accepts 16,384 tokens). JEV-9B is used as
  its card documents, with states cut to 1,024 tokens (first 60 percent and last 40 percent); this cuts
  most packs and many notes, and is a stated limitation of that model here.

Measures, per model and task: the probability of yes for every item; agreement with the 27B judge at a
0.5 threshold (raw agreement and Cohen's kappa, with bootstrap intervals over items); the area under
the ROC curve of the probability against the 27B labels; seconds per item. Replications of the key
quantities with each model's labels: the yes-rate of each pack type and the A minus B difference with
exact McNemar (S1, S4); the share of unlabelled notes judged relevant and the reranker's gain with
extended labels (S2). When the author's labels exist, the same agreement measures against them, for
each model and for the 27B judge. All of this is descriptive; nothing is accepted or rejected. Prior,
from Clef-Flash's model card: it scores 35.6 F1 on RAGTruth against 79.4 for the larger Clef, so it may
be weaker on context-grounding judgements than its other benchmarks suggest.

## Addendum 3 (recorded 2026-10-08 21:09, before any of these cells' data)

The author asked this session to run the 8 October 2026 investigation canvas after the S7 chain
exits. No cell starts while `s7_judge.py` or an S7 chain script is running. One heavy job at a
time, under the same D11 stop lines. Development questions only (`qrels-dev.json`, 817).
`qrels-test.json` stays sealed until the cells that pass are collected into one table. Baseline is
best-chunk cosine, R@10 0.698. Postgres stays. No new vector index and no new library. Results are
findings, not decisions.

Recall passes when the 95% interval for the difference against the named baseline sits above zero.
A token cell passes at 20% fewer tokens than the chunk pack and sufficiency within 0.02 of 0.793.
Fewer tokens that drop the page is a failure.

Wave A, no new model, reranker off except cell 5. Stop a line when its accept rule fails.
1. Rank each note by its best stored sentence; return the note. Recall at 10 against 0.698.
   arXiv 2312.06648.
2. Smooth the top hits over links, then keep the higher of the smoothed score and the original.
   Recall interval above zero, and the original top note is not lost. arXiv 2603.24925.
3. Forward-push PageRank from the top hits, on directed links. Match the existing PageRank recall
   in less time. arXiv 1908.10583.
4. Down-weight notes that sit close to every query, then search again. Recall interval above zero
   against the best ranker so far. arXiv 2508.02538.
5. Run the reranker only when the top two cosine scores are close. The threshold is chosen on the
   tuning half and applied once on the confirmation half, using reranker scores already stored.
   MRR stays inside the always-on spread; time nearer the no-reranker cost than the always-on cost.
   arXiv 2606.07923.
6. Count notes a later note replaces. No model. If the count is about zero, stop that line.
   arXiv 2605.06527.
7. Compare stored page-mean vectors with a fresh mean of the current chunk vectors. No new engine.
   If they already match, or if no separate stored mean exists, stop. arXiv 2203.16684.

Wave B uses the ranker that won wave A, on the same 150 packs as the sufficiency check.
8. Add chunks until a checker, separate from the 27B sufficiency judge, says the set can answer.
   At least 20% fewer tokens than the chunk pack, sufficiency within 0.02. arXiv 2506.05167.
9. Pack the winning sentence with the headings above it. Sufficiency within 0.02 of chunk packs,
   and fewer tokens. arXiv 2604.20849.
10. Drop sentences that are not evidence. Run only if cell 8 misses the token cut. Same bar as
    cell 8. arXiv 2501.16214.

Wave C, one heavy job at a time, only when its gate opens.
11. Late chunking. Skip: this encoder pools only the last token, already closed in the paper.
    arXiv 2409.04701.
12. Embed a short note written from the question and mix it 1:1 with the query. Skip: already
    measured (HyDE), and also skip if wave A already moved recall. arXiv 2212.10496.
13. Hide a fact a newer fact replaced. Only if cell 6 found real cases. Sufficiency up on those
    questions; recall at 10 on the 817 not down. arXiv 2605.06527.
14. Ignore a fact outside its start and end. Only if those dates exist. Same bar as cell 13.
    arXiv 2506.07270.
15. When two notes disagree and neither replaces the other, pack both sentences. Both sides
    present on the conflict set; recall at 10 on the 817 not down. arXiv 2608.13921.
16. On the 60 two-page questions, pick a set that covers each part. Set recall above 0.479,
    without the link join. arXiv 2507.06838.
17. Mark a query whose nearest notes do not form one cluster. Only after cell 4. The mark catches
    queries where the top 10 misses the page. arXiv 2406.07990.

Cells 1, 2, 3, 5, 8 and 9 are the six already specified in the paper, questions and methods only.
Cells 11 and 12, and a refresh of a note mean that is not stored separately, are recorded as skips
rather than re-run. Papers tagged later or research on that scan are not cells.
