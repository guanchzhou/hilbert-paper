# Factorial study of the retrieval pipeline and the local agent harness

Findings, not decisions. Pre-registration: `preregistration.md`; every departure, with its time, is in `deviations.md` (C1, C2, D1 to D11). Development questions only, except section 3.

## 0. What ran

- Level 1: all 1,020 cells (204 rankings times 5 evidence units) on all 817 development questions, after the reranker cache was completed (202,515 pairs).
- Level 2: 40 of the 198 planned runs, all with the local 27B model at 8K and 32K windows. The 138 cloud runs (Cursor CLI, Claude Code) were dropped by the author (D7), and the 20 runs at a 128K window exceeded this 32 GB machine's memory and were not run (D9).
- The level 2 judge ran on a work account by mistake (D10); judge scores are shown but marked.

## 1. Design

**Level 1** is an offline full factorial of the retrieval pipeline on the 817 development
questions. Eight factors are crossed: first-stage search F1 (dense chunk vectors, lexical OR
over chunk text, or reciprocal-rank fusion of both), note scoring F2 (best chunk or mean of
chunks), cross-encoder rerank of the top 50 F3, link-graph personalised PageRank F4,
mean-direction removal F5, Rocchio pseudo-relevance feedback F6, a candidate filter F7 (none,
hk1 level-1 probes with 16 ranges, Hilbert balanced partitions with SOAR), and the evidence
unit packed into 6,000 tokens F8 (chunk, window, section, page, sentence-pruned). F2, F5 and F6
act on the dense part only, so they are not crossed with the lexical search; that leaves 1,020
valid cells (204 distinct rankings times 5 units). Every parameter comes from the twelve-idea
study and none was re-tuned. Latency is composed from per-component timings measured once
(section 1.4 of the pre-registration), so it describes this machine and this corpus size, not
an end-to-end measurement.

Effects are paired over questions. A main effect is the per-question mean, over all matched
cells, of the metric at a level minus the metric at the reference level; an interaction is the
per-question double difference. Shapley points attribute each metric to the factors: for each
of the 16 profiles of non-reference levels of the multi-level factors, an exact Shapley value
is computed over 8 binary players (2^8 coalitions), then averaged. All choices of a "best"
configuration are made on the tune half and reported on the confirm half.

**Level 2** is a fractional design over the agent harness: host (local Ollama qwen3.8 27B,
Cursor CLI, Claude Code), model tier (large, small, shared), retrieval tool (files or gbrain
MCP), RTK (on or off), context (local num_ctx 8K/32K/128K; cloud gbrain evidence budget
2K/6K/24K) and gbrain packing unit (chunk, page, best level-1 unit). Three strata, each a
pairwise-covering array, on the 10 pilot tasks.

## 2. Level 1 results

### 2.1 Shapley points per factor (tier B, 817 questions)

Each value is the factor's average contribution to the metric over all combinations of the other factors, with a 95% bootstrap interval. Latency is composed from per-component timings.

| Factor = level | R@10 | survival | tokens | latency (ms) | MRR |
|---|---|---|---|---|---|
| F1=lexical | -0.0670 [-0.0829, -0.0507] | -0.1012 [-0.1173, -0.0850] | +6 | -49.7 | -0.0958 |
| F2=mean | -0.0050 [-0.0098, -0.0003] | -0.0065 [-0.0114, -0.0019] | -10 | -0.1 | -0.0045 |
| F3=on | +0.0500 [+0.0383, +0.0622] | +0.0568 [+0.0454, +0.0684] | +10 | +4364.0 | +0.0541 |
| F4=on | +0.0158 [+0.0127, +0.0190] | +0.0138 [+0.0108, +0.0168] | +32 | +103.1 | +0.0069 |
| F5=on | -0.0019 [-0.0054, +0.0015] | -0.0025 [-0.0056, +0.0007] | +9 | +0.0 | -0.0019 |
| F6=on | +0.0012 [-0.0020, +0.0043] | -0.0001 [-0.0031, +0.0031] | +13 | +0.1 | -0.0035 |
| F7=hk1 | -0.3673 [-0.3974, -0.3370] | -0.3501 [-0.3806, -0.3197] | -151 | -42.4 | -0.2280 |
| F8=window | +0.0000 [+0.0000, +0.0000] | -0.0803 [-0.0887, -0.0722] | +86 | +0.0 | +0.0000 |
| F8=section | +0.0000 [+0.0000, +0.0000] | -0.0208 [-0.0283, -0.0143] | +88 | +0.0 | +0.0000 |
| F8=page | +0.0000 [+0.0000, +0.0000] | -0.1298 [-0.1464, -0.1132] | -114 | +0.0 | +0.0000 |
| F8=pruned | +0.0000 [+0.0000, +0.0000] | +0.1105 [+0.0995, +0.1219] | -1701 | +0.7 | +0.0000 |
| F7=partitions | -0.0503 [-0.0666, -0.0349] | -0.0480 [-0.0642, -0.0326] | -1 | +1.7 | -0.0314 |
| F1=hybrid | +0.0096 [+0.0013, +0.0182] | +0.0014 [-0.0067, +0.0099] | +77 | +35.5 | -0.0097 |

### 2.2 Main effects (Holm-corrected)

R@10

| Contrast | effect | 95% CI | Holm p |
|---|---|---|---|
| F1=lexical | -0.0606 | [-0.0747, -0.0460] | 0 |
| F1=hybrid | +0.0076 | [+0.0010, +0.0144] | 0.012 |
| F2=mean | -0.0010 | [-0.0048, +0.0027] | 0.28 |
| F3=on | +0.0275 | [+0.0161, +0.0387] | 0 |
| F4=on | +0.0090 | [+0.0069, +0.0112] | 0 |
| F5=on | -0.0040 | [-0.0075, -0.0005] | 0.16 |
| F6=on | +0.0016 | [-0.0016, +0.0048] | 0.28 |
| F7=hk1 | -0.3881 | [-0.4190, -0.3568] | 0 |
| F7=partitions | -0.0546 | [-0.0711, -0.0393] | 0.28 |

survival

| Contrast | effect | 95% CI | Holm p |
|---|---|---|---|
| F1=lexical | -0.0827 | [-0.0957, -0.0696] | 0 |
| F1=hybrid | -0.0007 | [-0.0067, +0.0055] | 0.87 |
| F2=mean | -0.0001 | [-0.0035, +0.0031] | 0.00036 |
| F3=on | +0.0312 | [+0.0210, +0.0411] | 0 |
| F4=on | +0.0084 | [+0.0063, +0.0106] | 0 |
| F5=on | -0.0030 | [-0.0058, -0.0001] | 0.27 |
| F6=on | -0.0003 | [-0.0031, +0.0025] | 0.87 |
| F7=hk1 | -0.3680 | [-0.3988, -0.3371] | 0 |
| F7=partitions | -0.0533 | [-0.0697, -0.0378] | 0.17 |
| F8=window | -0.0787 | [-0.0874, -0.0707] | 0 |
| F8=section | -0.0228 | [-0.0302, -0.0162] | 0 |
| F8=page | -0.1283 | [-0.1446, -0.1121] | 0 |
| F8=pruned | +0.0979 | [+0.0872, +0.1090] | 0 |

tokens

| Contrast | effect | 95% CI | Holm p |
|---|---|---|---|
| F1=lexical | -56.6287 | [-102.5171, -15.0380] | 0.17 |
| F1=hybrid | +42.2166 | [+34.3580, +50.3287] | 0 |
| F2=mean | -5.8249 | [-8.5867, -3.1244] | 1.2e-05 |
| F3=on | -0.0649 | [-10.0489, +9.0664] | 0.0028 |
| F4=on | +13.0853 | [+10.5243, +15.7902] | 0 |
| F5=on | +12.8001 | [+8.8724, +16.7782] | 0 |
| F6=on | +6.6084 | [+3.4411, +9.7336] | 0 |
| F7=hk1 | -123.0081 | [-139.9631, -106.6612] | 0 |
| F7=partitions | +5.6037 | [-0.6568, +11.9598] | 0.0022 |
| F8=window | +111.5858 | [+93.0573, +131.1058] | 0 |
| F8=section | +108.0531 | [+92.6105, +124.8118] | 0 |
| F8=page | -94.0697 | [-117.1804, -70.4014] | 0 |
| F8=pruned | -1738.8797 | [-1770.1172, -1707.6485] | 0 |

latency

| Contrast | effect | 95% CI | Holm p |
|---|---|---|---|
| F1=lexical | -78.4482 | [-98.8452, -59.9110] | n/a |
| F1=hybrid | +35.4988 | [+35.4988, +35.4988] | n/a |
| F2=mean | -0.0485 | [-0.0487, -0.0484] | n/a |
| F3=on | +4381.8169 | [+4376.6335, +4386.7399] | n/a |
| F4=on | +59.6340 | [+55.2522, +64.3742] | n/a |
| F5=on | +0.0028 | [+0.0028, +0.0028] | n/a |
| F6=on | +0.1224 | [+0.1222, +0.1226] | n/a |
| F7=hk1 | -27.3818 | [-34.5968, -20.8028] | n/a |
| F7=partitions | +0.2611 | [-0.2197, +0.6748] | n/a |
| F8=window | +0.0000 | [+0.0000, +0.0000] | n/a |
| F8=section | +0.0000 | [+0.0000, +0.0000] | n/a |
| F8=page | +0.0000 | [+0.0000, +0.0000] | n/a |
| F8=pruned | +0.7493 | [+0.7493, +0.7493] | n/a |

### 2.3 Largest two-factor interactions

R@10

| Interaction | size | 95% CI | BH q |
|---|---|---|---|
| F1=lexical x F7=hk1 | +0.1059 | [+0.0833, +0.1285] | 0 |
| F1=lexical x F3=on | +0.0940 | [+0.0746, +0.1139] | 0 |
| F1=lexical x F7=partitions | +0.0334 | [+0.0177, +0.0488] | 0.00015 |
| F3=on x F7=hk1 | -0.0236 | [-0.0416, -0.0052] | 0.0019 |
| F3=on x F7=partitions | -0.0188 | [-0.0292, -0.0083] | 3e-06 |
| F1=lexical x F4=on | +0.0179 | [+0.0111, +0.0249] | 1e-06 |
| F2=mean x F7=hk1 | +0.0174 | [+0.0107, +0.0245] | 0.00069 |
| F1=hybrid x F3=on | +0.0115 | [-0.0005, +0.0235] | 0.32 |

survival

| Interaction | size | 95% CI | BH q |
|---|---|---|---|
| F1=lexical x F3=on | +0.1266 | [+0.1079, +0.1452] | 0 |
| F1=lexical x F7=hk1 | +0.1265 | [+0.1057, +0.1474] | 0 |
| F7=hk1 x F8=pruned | -0.0660 | [-0.0812, -0.0509] | 0 |
| F1=lexical x F8=pruned | +0.0571 | [+0.0439, +0.0704] | 0 |
| F7=hk1 x F8=page | +0.0486 | [+0.0323, +0.0649] | 0 |
| F7=hk1 x F8=window | +0.0467 | [+0.0341, +0.0592] | 0 |
| F1=lexical x F7=partitions | +0.0439 | [+0.0309, +0.0564] | 0 |
| F3=on x F7=hk1 | -0.0301 | [-0.0465, -0.0139] | 0 |

tokens

| Interaction | size | 95% CI | BH q |
|---|---|---|---|
| F1=lexical x F8=page | -266.3352 | [-301.7380, -232.1777] | 0 |
| F1=lexical x F8=window | -244.3259 | [-272.2738, -217.3517] | 0 |
| F7=hk1 x F8=pruned | -201.9179 | [-240.2455, -162.9711] | 0 |
| F7=hk1 x F8=window | +169.0861 | [+134.4522, +205.9877] | 0 |
| F1=lexical x F8=section | -163.5867 | [-187.7172, -140.9685] | 0 |
| F7=hk1 x F8=section | +160.6029 | [+127.7162, +195.4332] | 0 |
| F1=hybrid x F8=page | -156.7371 | [-173.5938, -139.9002] | 0 |
| F3=on x F8=page | +146.4612 | [+123.2978, +169.4081] | 0 |

### 2.4 Leaderboards, chosen on the tune half and reported on both halves

best_recall (eligible 204)

| # | configuration | R@10 tune / confirm | survival tune / confirm | tokens | latency ms |
|---|---|---|---|---|---|
| 1 | hybrid + mean-of-chunks + rerank + PPR | 0.734 / 0.752 | 0.782 / 0.801 | 5612 | 4505 |
| 2 | hybrid + rerank + PPR + centering | 0.734 / 0.752 | 0.773 / 0.799 | 5604 | 4505 |
| 3 | hybrid + rerank + centering | 0.734 / 0.751 | 0.773 / 0.799 | 5601 | 4467 |
| 4 | hybrid + mean-of-chunks + rerank + PPR + Rocchio | 0.733 / 0.763 | 0.775 / 0.801 | 5598 | 4506 |
| 5 | hybrid + rerank + PPR | 0.732 / 0.760 | 0.785 / 0.809 | 5609 | 4505 |

best_survival (eligible 1020)

| # | configuration | R@10 tune / confirm | survival tune / confirm | tokens | latency ms |
|---|---|---|---|---|---|
| 1 | hybrid + mean-of-chunks + PPR + Rocchio + unit=pruned | 0.712 / 0.693 | 0.912 / 0.880 | 3976 | 100 |
| 2 | hybrid + mean-of-chunks + rerank + PPR + unit=pruned | 0.734 / 0.752 | 0.907 / 0.897 | 3952 | 4506 |
| 3 | hybrid + rerank + PPR + unit=pruned | 0.732 / 0.760 | 0.907 / 0.904 | 3966 | 4506 |
| 4 | dense + mean-of-chunks + PPR + Rocchio + unit=pruned | 0.711 / 0.698 | 0.905 / 0.887 | 3984 | 64 |
| 5 | hybrid + mean-of-chunks + rerank + PPR + Rocchio + unit=pruned | 0.733 / 0.763 | 0.905 / 0.904 | 3989 | 4506 |

fewest_tokens (eligible 23)

| # | configuration | R@10 tune / confirm | survival tune / confirm | tokens | latency ms |
|---|---|---|---|---|---|
| 1 | hybrid + mean-of-chunks + PPR + centering + Rocchio + unit=pruned | 0.712 / 0.707 | 0.897 / 0.885 | 3924 | 100 |
| 2 | dense + rerank + PPR + unit=pruned | 0.728 / 0.750 | 0.895 / 0.912 | 3908 | 4471 |
| 3 | hybrid + mean-of-chunks + PPR + unit=pruned | 0.706 / 0.694 | 0.895 / 0.880 | 3912 | 99 |
| 4 | hybrid + mean-of-chunks + rerank + PPR + centering + unit=pruned | 0.728 / 0.747 | 0.895 / 0.900 | 3921 | 4506 |
| 5 | dense + PPR + centering + Rocchio + unit=pruned | 0.704 / 0.716 | 0.892 / 0.885 | 3943 | 64 |

fastest (eligible 30)

| # | configuration | R@10 tune / confirm | survival tune / confirm | tokens | latency ms |
|---|---|---|---|---|---|
| 1 | dense + PPR | 0.714 / 0.712 | 0.748 / 0.775 | 5547 | 63 |
| 2 | dense + PPR + centering | 0.720 / 0.722 | 0.758 / 0.767 | 5569 | 63 |
| 3 | hybrid + PPR + centering + Rocchio | 0.717 / 0.713 | 0.736 / 0.745 | 5681 | 99 |
| 4 | dense + mean-of-chunks + rerank | 0.717 / 0.725 | 0.768 / 0.772 | 5537 | 4431 |
| 5 | dense + rerank | 0.725 / 0.744 | 0.773 / 0.792 | 5548 | 4431 |

confirmation

- recall axis: R@10 vs reference: hybrid + mean-of-chunks + rerank + PPR + unit=chunk: diff +0.0505 [+0.0219, +0.0804], p 0.00053, Holm 0.00053
- survival axis: survival vs reference: hybrid + mean-of-chunks + PPR + Rocchio + unit=pruned: diff +0.1127 [+0.0809, +0.1446], p 0, Holm 0
- tokens axis: survival non-inferiority (margin 0.02) vs reference: hybrid + mean-of-chunks + PPR + centering + Rocchio + unit=pruned: diff +0.1176 [+0.0882, +0.1495], p 0, Holm 0
- latency axis: R@10 non-inferiority (margin 0.02) vs reference: dense + PPR + unit=chunk: diff +0.0108 [+0.0006, +0.0224], p 0, Holm 0

recall_tokens_latency: 64 non-dominated cells

survival_tokens_latency: 58 non-dominated cells

## 3. Held-out confirmation

Run once on the 817 sealed held-out questions (SHA-256 `96510536278c37e3...`), protocol D8. Paired against the reference configuration on the same questions.

| run | configuration | unit | R@10 | MRR | survival | tokens |
|---|---|---|---|---|---|---|
| best_recall | `hybrid / mean / on / on / off / off / none` | chunk | 0.740 | 0.521 | 0.781 | 5,574 |
| best_survival | `hybrid / mean / off / on / off / on / none` | pruned | 0.722 | 0.488 | 0.897 | 3,926 |
| reference | `dense / best / off / off / off / off / none` | chunk | 0.725 | 0.522 | 0.752 | 5,443 |

- best_recall: R@10 +0.016 [-0.009, +0.041], Holm p 0.49; survival +0.029 [+0.004, +0.056], Holm p 0.12; tokens +131.
- best_survival: R@10 -0.003 [-0.027, +0.023], Holm p 0.88; survival +0.146 [+0.121, +0.171], Holm p < 0.001; tokens -1,517.

### 3.1 Do sentence packs answer the question?

Pre-registered in `judge-preregistration.md`: 150 confirmation-half questions, the local judge `qwen3.8:latest` asked whether each pack contains the information needed to answer, both packs judged per question in a random order.

| pack | judged sufficient | survival | sufficient when a relevant note survives | tokens |
|---|---|---|---|---|
| best-survival, pruned sentences | 0.713 | 0.893 | 0.761 | 3,972 |
| reference, chunks | 0.793 | 0.753 | 0.867 | 5,415 |

Difference -0.080 [-0.147, -0.013]; non-inferiority at the 0.02 margin not accepted (p 0.96, the larger of the Wilcoxon and shifted t p-values; the Wilcoxon value alone is degenerate here because 124 of 150 differences are tied). Exact McNemar, two-sided: 7 questions favour the pruned pack, 19 the chunk pack, p 0.029. Median judge time 41 s per pack (D11 records the memory stop lines).

## 4. Level 2: the local agent

40 runs over 10 pilot tasks. Samples are small; differences below are descriptive unless a test is given. Judge scores come from D10.

**Notes as files or through gbrain**

| level | runs | errors | page found | judge (0-2) | input tokens | wall time (s) | tool calls |
|---|---|---|---|---|---|---|---|
| files | 10 | 4 | 0.30 | 0.60 | 21,125 | 219 | 5.2 |
| gbrain | 30 | 8 | 0.40 | 1.00 | 29,339 | 276 | 4.1 |

**Context window**

| level | runs | errors | page found | judge (0-2) | input tokens | wall time (s) | tool calls |
|---|---|---|---|---|---|---|---|
| 32768 | 20 | 7 | 0.35 | 0.90 | 37,188 | 379 | 3.5 |
| 8192 | 20 | 5 | 0.40 | 0.90 | 17,383 | 144 | 5.2 |

**gbrain packing unit**

| level | runs | errors | page found | judge (0-2) | input tokens | wall time (s) | tool calls |
|---|---|---|---|---|---|---|---|
| chunk | 10 | 3 | 0.40 | 1.00 | 27,815 | 285 | 4.5 |
| page | 10 | 3 | 0.40 | 1.00 | 31,744 | 294 | 4.0 |
| section | 10 | 2 | 0.40 | 1.00 | 28,458 | 248 | 3.8 |

Matched pairs, gbrain against files: page found +0.10 over 10 pairs (sign test, Holm p 1); judge +0.40 (Holm p 0.12). The pre-registered contrast of the largest and smallest window has no pairs, because no 128K run exists.

## 5. What beats what

- **Recall.** The reranker is the largest positive factor (+0.050 R@10), then link PageRank (+0.016) and hybrid search (+0.010). The hk1 filter costs -0.367 and keyword search alone -0.067. Best configuration: hybrid + mean-of-chunks + rerank + PPR, confirm R@10 0.752; on held-out questions its gain over the reference shrank to +0.016 (Holm p 0.49).
- **Speed.** The reranker adds 4.4 s per question; every other factor adds at most 103 ms.
- **Tokens and survival.** Sentence-pruned packs keep a relevant note more often (+0.110 survival) with -1,701 tokens; whole-page packs lose -0.130. Best: hybrid + mean-of-chunks + PPR + Rocchio + unit=pruned, confirm survival 0.880. The pre-registered judge (section 3.1) found its pruned packs sufficient for 0.713 of questions against 0.793 for chunk packs, so the survival gain does not carry over to answers.
- **Local model, context window.** 32K against 8K: page found 0.35 vs 0.40, input tokens 37,188 vs 17,383, wall time 379 vs 144 s. The larger window costs about 2.1 times the tokens and 2.6 times the time with no measured gain. 128K does not fit in memory.
- **Cursor, Claude Code, RTK, evidence budget.** Not measured in this round (cloud stratum dropped).

## 6. Implementation plans (work to verify, not decisions)

1. Reranker on the hybrid top 50 with link PageRank: its recall gain did not reach significance on the held-out questions (+0.016, Holm p 0.49), so re-measure it on more questions, with depth 30 and score caching against its 4.4 s cost, before enabling it for agents.
2. Link PageRank: a cheap recall gain; implement as a gbrain post-processor over the links table and re-measure stacked on the reranker.
3. Sentence-pruned packs: replicated on held-out questions (+0.146 survival, 124 vs 5 discordant questions); but judged sufficient less often than chunk packs (-0.080, non-inferiority not accepted), so the measurements do not support them as the default unit; a pack that keeps whole chunks for the top notes and adds pruned sentences only below them would need its own pre-registered judged test.
4. Keep chunk packs as the default unit; do not use the hk1 filter for candidate selection.
5. Local agent: prefer an 8K window; the 32K window doubled cost without a measured gain.

## 7. Follow-up checks (bench/followup, pre-registered)

- Length: chunk packs cut to the sentence packs' length are judged sufficient 0.793, the same as full chunk packs (0.793); against sentence packs 19 vs 7 (p 0.029). Reading: content explains the gap.
- Compression: hybrid 0.807 (5,972 tokens), LongLLMLingua 0.793 (3,680), RECOMP extractive 0.627 (3,972), chunk packs 0.793; none passes non-inferiority at 0.02.
- Reranker power: about 1,079 questions for 80 percent power at the held-out gain; labels judged incomplete for 42 percent of unlabelled top-10 notes, and with the judge's labels the reranker's gain is +0.119 (original labels +0.026).
- Hilbert facet map: rejected by the author's cost rule (chunk keys 7.5x the tokens and 3.2x the time per correct answer of dense facets).
- Author's labels: agreement with the judge 0.83 (kappa 0.50) on 100 packs. Prediction-powered A minus B is -0.060 (95% CI -0.227 to +0.107); the author's own difference on the 50 is +0.000. Relevance agreement 0.73 where the judge said yes and 0.53 where it said no.
- Gold check: 25 of 30 lists marked correct and complete. On those, hk1 facet recall 0.176 against dense facets 0.426.

