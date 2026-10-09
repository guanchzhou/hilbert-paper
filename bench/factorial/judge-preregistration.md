# Pre-registration: do sentence-pruned packs still answer the question?

Written on 2026-10-07, before any judgement in this check was made. Results are findings, not
decisions.

## Question

The factorial's best-survival configuration (hybrid search, note-mean scores, link PageRank and
Rocchio feedback, with packs of the five best sentences of each of the top 30 notes) kept a
relevant note in the pack far more often than the reference chunk pack, on development and on
held-out questions. The twelve-idea study found that pruned packs were judged sufficient less
often (idea 2's guard). This check asks the same of the factorial's configuration.

## Design

- **Questions.** 150 questions drawn without replacement from the confirmation half of the 817
  development questions (`ideas/split.json`) with `numpy.random.default_rng(20261007)`. The
  held-out questions are not used.
- **Conditions.** (A) the best-survival configuration `hybrid|mean|off|on|off|on|none` with the
  `pruned` unit; (B) the reference, dense best-chunk scores with the chunk unit. Both packs are
  built exactly as `stage2.py` builds them, within 6,000 tokens, units joined with
  `"\n\n---\n\n"`, from the frozen snapshot.
- **Judge.** The idea 2 judge without change: local `qwen3.8:latest`, temperature 0, `think`
  off, 16,384-token window, JSON reply, the same prompt ("Does the context contain the
  information needed to answer the question?"). For each question the two packs are judged in an
  order drawn from the same generator. No cloud model is used.
- **Outcome.** The share of "yes" per condition, and the paired difference A minus B.
- **Test.** Non-inferiority with margin 0.02, as idea 2's guard: A is accepted as no worse when
  the one-sided p for (A - B + 0.02) > 0 is below 0.05, taking the larger p of the Wilcoxon
  signed-rank test and the paired t-test (deviation D1 of the ideas study). The exact two-sided
  McNemar test of A against B is reported alongside. Survival of each pack on the same questions
  is reported, so that a "yes" can be read against whether a relevant note was present.
- **Records.** Judge replies stay in `private/`; the result file holds the sample, the per-question
  judgements, the tests and timings.
