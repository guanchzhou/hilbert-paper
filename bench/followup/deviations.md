# Deviations from the follow-up pre-registration

## D1. S6b answer format (recorded 2026-10-07 16:37, before any S6b run)

The pre-registration says the S6b system prompt asks for the answering slugs one per line. The level 2
agent loop (`agents/local_agent.py`) already asks for a final line `SOURCES:` with comma-separated
slugs, and S6b keeps that prompt unchanged so that its runs stay comparable with level 2. Scoring
reads the slugs from that line. A and B are removed from the returned slugs before scoring, as in the
retrieval arms.

## D2. S6 summary statistics (recorded 2026-10-07 16:37)

`compositional.json` reports, besides the pre-registered means, the median token count, the median
set size and the share of empty sets per arm, because a few very long notes (up to 134,000 tokens)
dominate the mean token count. No test changed.

## D3. S5 effect-size method (recorded 2026-10-07 16:43, after seeing the pre-registered output)

The pre-registered method shifts the per-question differences to each effect size (subtract the
observed mean, add the target). It was run once and gave 97 percent power at 50 questions for the
+0.016 effect. That number is an artefact: most R@10 differences are exactly zero, the shift turns
every zero into the same small negative value, and the signed-rank test then reacts to hundreds of
tied small differences rather than to the comparison. Rescaling the differences would not help,
because the signed-rank test is invariant to a positive scale. The primary method is therefore
dilution: each resampled question keeps its observed difference with probability w and is zero
otherwise, with w = effect / observed mean, which models a gain that appears on fewer questions.
The development effect is the observed difference itself (+0.044 on all 817 questions; the
pre-registration's +0.050 was the confirmation-half figure, above the all-question mean, so dilution
cannot reach it). The shifted-method curve is still reported, marked as invalid.

## D4. S3 exploratory matched-cost comparison (recorded 2026-10-07 17:02, before the test below was computed)

The pre-registered primary comparison is not defined: without probes, cross-polytope with one hash
per table scans at most 9.4 percent of chunks at 128 tables, so it never reaches the 10 percent
point. That is reported as "not reached". An exploratory comparison is added at matched cost,
with this rule: the hyperplane reference is centred, 32 tables, no probes (13.0 percent scanned);
the cross-polytope configuration is the centred one with the highest R@10 among those scanning at
most that share (128 tables, no probes, 9.4 percent). The test is the paired two-sided Wilcoxon on
per-question R@10. It is labelled exploratory everywhere it is reported.

## D5. S4 compressor details (recorded 2026-10-07 17:06, before any S4 pack is built)

RECOMP's extractive compressor has no model card on Hugging Face; the repository holds a
sentence-transformers configuration (BERT, mean pooling, sentences cut at 100 tokens). R scores
each candidate sentence by the inner product of the mean-pooled question and sentence embeddings,
as in the RECOMP paper's Contriever-based compressor, and keeps sentences best first.
LongLLMLingua counts its target in the language model's tokens, while pack A's size is in gbrain's
estimate (characters / 4). For each question the target is pack A's estimate times that pack B's
ratio of Qwen tokens to estimated tokens, and the compressed pack's estimate is reported. The
LongLLMLingua settings are the library README's recommended ones (condition_in_question
after_condition, reorder_context sort, dynamic_context_compression_ratio 0.3, condition_compare
true, context_budget +100, rank_method longllmlingua), with target_token instead of rate.

## D6. S4 LongLLMLingua environment (recorded 2026-10-07 18:56, before any L pack was built)

`llmlingua` 0.2.2 iterates the model's attention cache as (key, value) pairs, which fails with the
current `transformers` release (its cache is an object). The throwaway environment for building the
L packs therefore pins `transformers` 4.44.2 and `sentence-transformers` below 4, on Python 3.12
(`tokenizers` 0.19.1, which that `transformers` needs, has no wheel for Python 3.13). The model,
revision and settings are unchanged. The H and R packs were built before this, with the unpinned
environment; they do not use `llmlingua`, and their RECOMP encoder is unchanged.

## D7. S4 L build stopped once by the memory stop line (recorded 2026-10-07 18:58)

The L build started five seconds after the 27B judge model was unloaded and reached kernel memory
pressure level 4 at 18:57, so the monitor stopped it after two packs, as the stop rules require.
A minute later pressure was back at level 1 with 23 GB free and no Ollama model loaded. The two packs
already written are kept (the build is deterministic and resumable); the chain now waits 30 seconds
after unloading before the next step, and the L build was restarted at 18:58.
The restarted build reached pressure level 4 again at 19:00 (swap in use 25.4 GB) after ten packs,
so the growth comes from LongLLMLingua on the GPU (MPS), not from the judge model. From 19:02 the L
build runs on the CPU, which keeps memory bounded; the packs already written are kept. The device
does not change the method, only rounding in the language model's scores.

## D8. S2 paused and stopped once (recorded 2026-10-08 12:50)

S2 was paused at the author's request at 10:21 (744 calls judged) and resumed at 10:41 from its
cache. At 12:38 the monitor stopped it at kernel memory-pressure level 4 (swap in use 22.0 GB),
with 1,031 of 1,059 calls judged. A minute later pressure was back at level 2, and the last 28 calls
ran from 12:39 to 12:49. Every judgement is cached per pair, so no call was repeated or lost.

## D9 (recorded 2026-10-08 14:30, before any S7 data): S7 runs with 8-bit weights

Loading Clef-Flash in bfloat16 (18 GB) reached memory-pressure level 4 three times (14:24 in the run
chain, 14:27 and 14:30 in one-item probes), the last one after the author agreed to stop gbrain's MLX
embedding and reranking servers for the run (they are restarted afterwards). As the author chose,
both decision models therefore run with int8 weight-only quantization (optimum-quanto through
transformers' QuantoConfig; activations stay bfloat16). Clef-Flash's output-embedding matrix, which
its schema head reads, and its unused vision tower stay in bfloat16. JEV-9B's LoRA adapter is applied
unmerged on the quantized backbone instead of being merged, which is the same function up to the
quantization. Quantization can shift the probabilities slightly, so S7 measures the models as run
here, not their bfloat16 releases.
Addition to D9 (14:34, before any S7 data): loading the int8 model straight onto the GPU also reached
level 4 (14:31), so both models are loaded and quantized on the CPU and then moved to the GPU; a probe
of that path stayed at level 2 or below, with 11.7 GB on the GPU for Clef-Flash.
Second addition to D9 (15:51): the Clef-Flash run reached level 4 at 15:49 after 300 of 1,959 items
(swap in use had grown from about 8.5 to 15.7 GB; time per input token was flat at 3.4–3.5 ms until
the last 50 items, 3.6 ms). The 300 cached judgements are kept. The run resumes in batches of 150 items,
each in a new process, so memory held by one process is released between batches; items, prompts and
the model are unchanged. Seconds per item are reported as medians.

## D10 (recorded 2026-10-08 17:21): S7 paused so the machine can move

At the author's request the Clef-Flash run was stopped at 17:21, at 606 of 1,959 items. The cache
private/s7-clef-flash.json is intact and the weights are kept. gbrain's MLX embedding and reranking
servers were restarted. Resume with private/s7-chain3.sh, which continues from the cache in batches
of 150 and then runs JEV-9B.
Addition to D10 (18:57): the author resumed the run. It continued from item 607, memory pressure at level 1.

## D11 (recorded 2026-10-09 00:35): JEV-9B did not judge

Clef-Flash finished all 1,959 items at 00:21. Its weights were deleted and JEV-9B's weights were
downloaded (disk 27 GB free). The judge then exited before any item. The offline snapshot check
required 21 files the download had been told to skip (`adapter_vllm`, `reports`, `vl`, `code`).
No JEV cache was written. The chain's "exit 0" is the date command's status; the traceback is the
record. The report written at 00:33 contains Clef-Flash only.

Before the retry, `s7_judge.py` passes those same ignore patterns when it opens the JEV snapshot.
The retry judges in batches of 150, one process each, under the same stop lines: a single long
process is what reached memory-pressure level 4 during Clef-Flash. The MLX servers are stopped for
the retry and restarted when it finishes.
Addition (00:48): the first batch finished all 150 items at memory-pressure level 1, then the
shell aborted assigning `status`, which zsh keeps read-only. The cache is kept. The retry
continues from item 151 with that variable renamed.

## D12 (recorded 2026-10-09 10:05): sufficiency judging in batches of six

Cells 8 and 9 ask the same 27B judge used for the chunk-pack yes-rate. One process judging every pack grew swap until `/System/Volumes/VM` had 14 GB free, which is the disk stop line, at memory-pressure level 2. No yes-rate had been computed. The cache of finished packs is kept. The judge now runs six new packs at a time and unloads between batches, under the same stop lines. The prompt, the temperature, and the packs are unchanged.

## D13 (recorded 2026-10-09 13:05): S9a draws 38 pairs, not 60

The frozen snapshot has 19 areas with at least 10 notes (Addendum 4 counted 18 on the live
database) and 154 qualifying pairs. With an area in at most three pairs and a note B in at most
two, the seeded draw stops at 38 pairs: three per area would allow at most 57, and several areas
have fewer qualifying notes B. The pre-registered result is the 38 pairs, reported with their
tests as specified. As an exploratory check, decided before running it, the same script raises
the area cap one step at a time until 60 pairs are drawn and reports the same arms and tests,
labelled exploratory (`s9a-explore.json`).
