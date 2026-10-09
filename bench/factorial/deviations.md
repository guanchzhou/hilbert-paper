# Deviations and clarifications

The pre-registration (`preregistration.md`, commit cbba31c) is not edited. Entries are in time
order, Europe/Helsinki time.

## C1. Interaction count (2026-10-05 22:42, before any analysis on measured data)

The pre-registration states 70 two-factor interaction contrasts per metric (34 for R@10). Three
of them are undefined under the pre-registered skip rule, because lexical cells exist only with
F2 = best, F5 = off and F6 = off: F1=lexical x F2=mean, F1=lexical x F5=on and F1=lexical x
F6=on. The families therefore hold 67 contrasts (31 for R@10). This follows from the skip rule
and changes no test.

## C2. Sidedness of the confirmation tests (2026-10-05 22:42, before any analysis on measured data)

Section 1.6 item 5 does not state the sidedness of the two superiority tests. Both are run
two-sided (Wilcoxon for R@10 on the recall axis, exact McNemar for survival on the survival
axis), the more conservative choice. The two non-inferiority tests are one-sided by definition.

## D1. Level 2 task assignment (2026-10-05 22:44, before any level 2 run)

The pre-registered rotation (row r gets tasks `(r + 2j) mod 10`) keeps the parity of r, and in
the cloud-gbrain stratum host is assigned by row parity, so every Cursor CLI row would get only
even-numbered tasks and every Claude Code row only odd-numbered ones: host would be confounded
with task. Replacement, fixed before any run: each row gets a block of 5 consecutive tasks
`(s + j) mod 10`, j = 0..4, run in rounds j as pre-registered. Cloud-gbrain rows: s = the row's
position in the stratum (0..9). Cloud-files rows: s = 3 x tier index (large 0, small 1,
shared 2), so the four rows of a tier (two hosts x RTK on/off) share the same tasks and the
host and RTK contrasts are matched on task. Local rows: s = 5 for every row (tasks t05 to t09),
so all local rows share tasks and every local contrast is matched on task. Counts are unchanged
(198 planned runs, 138 cloud, 60 local).

## D2. Late start gate: reranker deadline and a two-tier analysis (2026-10-06 00:56, before any GPU measurement)

The start gate (the twelve-idea study finished and committed) was expected around 01:00. At
00:55 that study was still running idea 10 (HyDE generation on the local qwen3.8, 41 of 817
questions done at about 14 s each), so the gate is now expected around 04:00 to 04:30. Under the
pre-registered 05:00 reranker deadline only about 15% of questions could be completed, and rule
1.5 would then shrink every cell, including the 510 rerank-off cells that need no new reranker
scores, to that small subset. Two changes, fixed now, before any reranker or sentence-embedding
call of this study:

1. The deadline for new reranker pairs and sentence embeddings moves from 05:00 to 06:00. Level 1
   has priority over level 2 in the task brief; level 2 keeps its 07:30 stop and runs in whatever
   time remains after level 1.
2. The analysis is reported in two tiers.
   - Tier B (pre-registered, rule 1.5 unchanged): all 1,020 cells on the questions complete in
     every cell, a random subset fixed by the pre-registered question order.
   - Tier A (added): the 510 rerank-off cells (F3 = off; 7 factors, 102 rankings) on all 817
     questions, with the same main-effect, interaction, Shapley (7 players, 2^7 coalitions),
     leaderboard, confirmation and Pareto procedures, and the same correction rules. Tier A needs
     no new reranker scores; if sentence embeddings are incomplete at 06:00, its sentence-pruned
     cells use the questions whose pruned packs are complete and this is reported.
   Findings about F3 come from tier B only. Findings about the other factors are read from
   tier A (larger n) and checked against tier B.

## D3. Interruption by a system panic (recorded 2026-10-06 08:45)

The machine rebooted at 04:58 during the GPU fill. The kernel panic report gives a userspace
watchdog timeout (WindowServer missed check-ins for 123 s) with the memory compressor at 100% of
its segment limit, 86 swapfiles and low swap space. At that time the sentence embedder, the
reranker fill (two workers) and the loaded 27B chat model were running together.

State at the interruption: the start gate was met at about 04:00; component latencies were
measured (level1-latency.json); the sentence fill finished (75,555 sentences in 1,945 s); the
reranker fill reached 50 of 817 questions (12,345 of 202,515 needed pairs, cached and resumable);
no level 1 cell was evaluated and no level 2 run was made. Nothing measured is lost, and no
result was computed, so no analysis choice was made after seeing data.

Change for the resumed run, fixed now: heavy steps run one at a time, the reranker fill uses one
worker, the 27B model is unloaded during level 1, and a memory watchdog pauses heavy processes
below 20% free memory and resumes them above 35% (mem_watchdog.sh, resume_night2.sh). The
reranker deadline for the resumed night is set when that night starts and is recorded here
before the fill resumes.

## D4. Database server upgraded between nights (recorded 2026-10-06 12:10)

At the author's request the gbrain cluster was upgraded from PostgreSQL 16.15 with pgvector 0.8.6
to PostgreSQL 18.6 with pgvector 0.8.7 (pg_upgrade --clone, then ALTER EXTENSION UPDATE and a
full ANALYZE). Before the upgrade the cluster was backed up as a logical dump, test-restored into
a scratch 18.6 cluster with identical row counts on all 116 tables, and as an archive of the
stopped data directory; the 16.15 data directory was kept unchanged. After the upgrade all 116
tables had the same row counts, all 637 indexes were valid, and nearest-neighbour queries used
the same HNSW index.

All measurements made before this point (night 1 and every earlier module) ran on 16.15 with
pgvector 0.8.6. Runs from night 2 on run on 18.6 with pgvector 0.8.7. Recall and ranking do not
depend on the server version for exact computations; HNSW results and latencies may differ, so
any latency or HNSW-recall comparison across the two periods is reported as crossing this change.

Also before night 2, the gbrain CLI moved from 0.60.64.0 (all measurements so far) to 0.60.85.0,
and its migrations were re-applied (schema version 212). Runs from night 2 on use 0.60.85.0.

Night 2 preflight (B0, 2026-10-06 16:37). A fresh logical dump (`pg_dump -Fc`, 110.6 MB, plus
globals) is in `~/.gbrain/backups/night2-preflight-2026-10-06T1637/`. It was restored into a
scratch PostgreSQL 18.6 cluster on a private socket: no restore errors, and all 119 tables (the
schema has gained three since the upgrade) have the same row counts as the live database, 28,054
rows in total; no table changed while the dump ran. The scratch cluster was then deleted. At
that time: PostgreSQL 18.6, pgvector 0.8.7, gbrain 0.60.85.0; 1,228 live pages and 5,174 chunks,
all embedded. The benchmark MCP client lists 138 tools, and search with `return_unit` and
`token_budget` and get_page both work.

## D5. Memory fault in stage2.py fixed before its first full run (recorded 2026-10-06 17:05)

The first two night-2 attempts at `stage2.py` were stopped before any cell was evaluated: the
process grew to 82 GB (81 GB compressed or swapped) within about two minutes, and the system ran
out of swap. Cause: the loader for the second batch of sentence vectors indexed `z["V"]` inside
its loop, and NumPy's `NpzFile` decompresses the whole 295 MB array on every access, so each of
the 5,068 stored slices kept its own full copy alive. The array is now read once before the loop;
the loader peaks at 1.7 GB. No value computed by the pipeline changes. Separately, gbrain's MLX
embedding and reranking servers had grown to about 24 GB each after seven hours and were
restarted at the author's request at 16:47; both answered health checks afterwards, and level 1
does not call them.

## D6. Reranker fill deadline and memory guard (recorded 2026-10-06 17:10, before the fill resumed)

The fill resumes at about 17:10 with one worker and the deadline 23:30 (rule 1.5: no question
starts after it; questions not complete in every cell are left out of tier B). After the server
restart in D5, `rerank_fill.py --check` re-scored question 0's 50 pairs and matched the cached
scores exactly (largest difference 0.0). Besides `mem_watchdog.sh`, a guard checks the memory of
the reranker server every 30 seconds and stops the fill if it exceeds 12 GB; the scores saved up
to then are kept, and a stop is recorded here.

Guard stop at 17:01:22, 30 seconds after the start, before any new score was saved: the reranker
server went from 5.6 GB to 24.0 GB. MLX keeps freed buffers in its cache up to its memory limit
(about 24 GB on this 32 GB machine), and gbrain's two MLX servers set no cache limit; this is
also the likely origin of the night-1 memory panic (D3). At the author's request both servers now
call `mx.set_cache_limit` with 2 GB (environment variables `MLX_RERANK_CACHE_MB` and
`MLX_EMBED_CACHE_MB`; the original files are kept as `server.py.bak-2026-10-06`) and were
restarted. Afterwards the embedder answered, and `rerank_fill.py --check` again matched the cached
scores exactly (largest difference 0.0), with the reranker server at 3.7 GB. The fill then
resumed under the same guard. The cache limit changes memory use only, not the model's scores.

Deadline moved (recorded 2026-10-06 22:14, before any tier B result). The fill had run from
20:19 to 22:12 at 12.6 pairs per second and saved 97,865 of 202,515 pairs; the author asked to
"run all", so the fill was stopped and restarted with the deadline 07:00, enough to finish. Tier B
then runs automatically after the fill, as in `resume_night2.sh`.

## D7. Level 2 tonight: local runs only, with a hard memory stop (recorded 2026-10-06 22:25, before any level 2 run)

The author chose to run only the 60 local runs of the level 2 design tonight; the 138 cloud runs
(Cursor CLI and Claude Code) are not run, so the level 2 report will cover the local stratum
only. The local runs start after tier B, no new run after 07:30. The pilot ran only the 8K
window; the 32K and 128K windows are untested on this 32 GB machine. A guard therefore checks
every 15 seconds while a local run is active and, if free memory falls below 10 percent or swap
in use exceeds 20 GB, stops the runner and unloads the 27B model; runs not started by then are
reported as not run. (Corrected at 22:30, before any level 2 run: the first version tested free
swap, which macOS keeps small because it grows swap on demand.)

## D8. Held-out confirmation protocol (recorded 2026-10-06 22:39, before qrels-test.json is opened)

The pre-registration does not define the final held-out step. At the author's request it is: the
best-recall and the best-survival configurations of the tier B leaderboard (`best_recall` and
`best_survival`, first row each), each run once on the sealed held-out questions, together with
the reference configuration (dense best-chunk scores, chunk pack) on the same questions for a
paired comparison. Tests are two-sided: Wilcoxon signed-rank on recall at 10 and exact McNemar on
survival, Holm-corrected over the four tests. `heldout.py` implements it and runs after level 2,
so that only one GPU job runs at a time.

Before the test run, `heldout.py --questions dev` must reproduce `stage2.py` exactly (recall at 10
and survival on every development question) for the three configurations, through the same
code; the test run refuses to start otherwise. A dry run with the tier A winners passed. Two
details were fixed by that check: questions are embedded in batches of 32, as the development
vectors were (other batch sizes change the vectors by up to 5e-4 per component through
padding), and keyword scores are recomputed against the frozen snapshot's chunk texts with
gbrain's chunk search vector, `setweight(to_tsvector('english', chunk_text), 'B')`, in TEMP tables.
On the development questions this matches the night-1 keyword scores on 1,244,728 of 1,245,274
pairs; the other 546 belong to one question (index 218), where every score is 0.4 higher, one
more matched term, which points to a stemming difference between PostgreSQL 16, on which the
stored vectors and the night-1 queries were built, and 18, on which both are now built.

## D9. Level 2 restarted without the pausing watchdog (recorded 2026-10-07 00:55, before any level 2 result)

Level 2 started at 00:41 with `mem_watchdog.sh` still running from tier B. With the 27B model
loaded (17 GB at a 32K window) free memory sits at 11 to 16 percent, below the watchdog's
20 percent pause line; the watchdog paused the first run, the idle model was unloaded after
five minutes, the run resumed, the model reloaded, and the run was paused again. A paused run's
wall time would include the pause, so that run was stopped before it wrote a record and will be
run again. Level 2 now runs without the pausing watchdog, protected only by the hard stop of D7
(below 10 percent free or more than 20 GB swap in use), and under `caffeinate` so that the
machine cannot sleep during a run. `level2_run.py` gained an option `--max-ctx` that skips local
runs with a larger window; it is used only if the hard stop ends the 128K runs (author's
instruction: skip the remaining 128K runs and continue with the others).

Hard stop at 00:55:57 on the first 128K run (r24-t05-a): with the model at 20 GB, free memory fell
to 7 percent and swap in use to 13 GB. The run wrote no record. As instructed by the author, the
remaining 128K runs are not run (20 of the 60 local runs); level 2 continued at 01:22 with the 8K
and 32K windows (`--max-ctx 32768`), with the hard stop restarted. The level 2 report covers
those two windows; whether a 128K window helps the local model cannot be answered on this
32 GB machine.

Second hard stop at 01:52:01 during a 32K run, after 17 completed runs: free memory touched
9 percent while swap in use was steady at 8.6 GB (in the 128K stop it rose from 8 to 13 GB in
30 seconds). Free memory had gone below 10 percent only these two times. The interrupted run
wrote no record and is run again. To let the 8K and 32K runs finish while keeping the
protection that matters (swap growth, the condition of the night-1 panic), the free-memory
line was lowered from 10 to 5 percent at 01:55; the 20 GB swap line is unchanged. Level 2
continued with `--max-ctx 32768`.

## D10. Level 2 judge ran on a work account (recorded 2026-10-07 04:20)

Level 2 finished at 04:16 with 40 local runs (8K and 32K windows). `level2_aggregate.py` then
called its pre-registered blind judge, Claude Sonnet through the Claude Code CLI, for the 25
answers not already in the pilot's judge cache. That CLI is logged in to the author's employer
organisation. The author had dropped the cloud runs in part to keep this study off work
accounts, and this use was not intended; it was noticed after the calls were made. No further
judge calls are made. The judge scores are reported with this note, and the level 2 results that
do not depend on the judge (page found, tokens, wall time, tool calls) are reported separately.

Held-out run (D8) at 04:20 to 05:16: the dev check passed for the tier B winners, then the test run
scored 40,850 reranker pairs and embedded the sentences of 5 new chunks. The reference
configuration reached R@10 0.7246 on the held-out questions, the same value as the earlier
held-out cell (`bench/heldout.json`). In `heldout-level1.json` the survival tests were labelled
"one-sided" by the shared `mcnemar` helper although they were computed two-sided as specified
(for example 75 against 51 discordant pairs gives p = 0.040 two-sided, 0.020 one-sided); the
label was corrected after the run, and no value changed.

## D11. Answer-quality judge continued past 20 GB of swap (recorded 2026-10-07 10:10, at 84 of 150 questions)

The judge of `judge-preregistration.md` runs the local 27B model with a 16K window. A read-only
monitor was to stop it at more than 20 GB of swap in use, the line of
D7. At 10:04 swap in use reached 21.2 GB, after Ollama restarted its model runner at 09:57; the
kernel memory pressure level was 2 (warning, not critical), 13 percent of memory was free and the
swap volume had 44 GB of disk free. The judge was not stopped, because its results are cached
per question and a restart would reload the same model to the same size. It continues under a
monitor that stops it at pressure level 4 (critical), more than 26 GB of swap in use or less than
15 GB of free disk. No judge result was read before this decision; judge timing is not an outcome.

Found after the run: gbrain's autopilot embedded the nine Zig repositories registered as new
sources that morning (1,881 chunks) between 08:49 and 10:02, so the embedding server ran alongside
the judge in that period, against the plan to embed them only after the judge. The last of those
embeddings finished at 10:02, just before the swap peak. The judge finished all 150 questions at
11:31 with a highest swap in use of 20.8 GB under the new monitor.
