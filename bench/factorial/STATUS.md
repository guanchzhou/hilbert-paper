# Factorial study status

Updated 2026-10-06 17:00. The first night was interrupted by a memory panic at 04:58
(deviations.md, D3). Night 2 preflight passed (D4); a loader memory fault was fixed (D5).

| Part | State |
|---|---|
| Pre-registration | committed (cbba31c) |
| Runner and analysis code, level 2 design | committed (2086c35) |
| Component latencies | measured (level1-latency.json) |
| Sentence vectors for pruned packs | complete, 75,555 sentences (local cache) |
| Reranker pairs for rerank-on cells | complete, 202,515 pairs (local cache) |
| Level 1 cell evaluation | all 1,020 cells on all 817 questions (stage2.py) |
| Level 1 tier A analysis | done: 510 rerank-off cells, 817 questions (level1-*-tierA.json) |
| Level 1 tier B analysis | done: 1,020 cells, 817 questions (level1-*.json) |
| Level 2 agent runs | local stratum running since 00:41 (D7); cloud stratum dropped |

Resume (second night): `./resume_night2.sh` runs the reranker fill (one worker), evaluates all
cells (stage2.py), then the tier A and tier B analyses, under mem_watchdog.sh. Tier A (510
rerank-off cells, all 817 questions) needs no further reranker scores and can run first if
time is short. Level 2 runs afterwards, with the 27B model loaded and the watchdog running.

At about 10 reranker pairs per second the remaining 190,170 pairs need about 5.3 hours.
