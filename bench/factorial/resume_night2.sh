#!/bin/zsh
# Resume the factorial study on the second night, one heavy process at a time, under the memory watchdog.
cd "$(dirname "$0")"
log() { echo "$(date '+%H:%M:%S') $*"; }
free_pct() { memory_pressure -Q | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}'; }
wait_free() {
  while (( $(free_pct) < ${1:-40} )); do
    log "waiting for memory: $(free_pct)% free"
    sleep 60
  done
}

./mem_watchdog.sh 20 35 > private/mem_watchdog.log 2>&1 &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

# Level 1 needs the reranker and the embedder, not the 27B chat model; unload it.
ollama stop qwen3.8:latest 2>/dev/null

wait_free 40
log "rerank fill"
python3 rerank_fill.py --deadline "${RERANK_DEADLINE:-05:30}" --workers 1

wait_free 40
log "evaluate all cells"
python3 stage2.py

wait_free 40
log "level 1 analysis, tier A (rerank-off cells, all 817 questions)"
python3 analyze.py --tier-a

wait_free 40
log "level 1 analysis, tier B (all cells, questions complete in every cell)"
python3 analyze.py

log "level 1 done; level 2 (agents) is started separately: it needs the 27B model loaded"
