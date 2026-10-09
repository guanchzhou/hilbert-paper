#!/bin/zsh
# Pause heavy bench processes when system memory runs low, resume them when it recovers.
# The 2026-10-06 run ended in a watchdog panic with the memory compressor at its segment limit
# and swap space exhausted, so heavy steps must never push the machine that far again.
#
# Usage: ./mem_watchdog.sh [pause_below_pct] [resume_above_pct]
PAUSE=${1:-20}
RESUME=${2:-35}
PATTERN='rerank_fill.py|sentence_fill.py|stage[12].py|analyze.py|level2_run.py|agent_local.py'
log() { echo "$(date '+%H:%M:%S') $*"; }
free_pct() { memory_pressure -Q | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}'; }
paused=0
while true; do
  f=$(free_pct)
  pids=$(pgrep -f "$PATTERN" | tr '\n' ' ')
  if (( ! paused )) && (( f < PAUSE )) && [[ -n $pids ]]; then
    log "free ${f}% < ${PAUSE}%: pausing $pids"
    kill -STOP ${=pids}
    paused=1
  elif (( paused )) && (( f > RESUME )); then
    log "free ${f}% > ${RESUME}%: resuming $pids"
    [[ -n $pids ]] && kill -CONT ${=pids}
    paused=0
  fi
  sleep 15
done
