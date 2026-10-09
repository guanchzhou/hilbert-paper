#!/bin/zsh
# Wait for the twelve-idea study to finish (REPORT.md present, its results committed, no idea process
# for 5 consecutive minutes), then start the GPU parts of level 1.
cd "$(dirname "$0")"
IDEAS=~/.gbrain/eval/bench/ideas
REPO=~/Development/hilbert-paper
log() { echo "$(date '+%H:%M:%S') $*"; }
busy() { pgrep -f 'idea[0-9]+_|summarize.py' >/dev/null; }
while true; do
  a=0; c=0
  [[ -f $IDEAS/REPORT.md ]] && a=1
  [[ -z "$(git -C $REPO status --short bench/ideas)" ]] && c=1
  if (( a && c )) && ! busy; then
    quiet=1
    for i in 1 2 3 4 5; do
      sleep 60
      if busy; then quiet=0; break; fi
    done
    if (( quiet )); then
      log "GATE MET"
      break
    fi
  fi
  log "waiting: report=$a committed=$c busy=$(busy && echo 1 || echo 0)"
  sleep 300
done
python3 latency.py --embed
python3 sentence_fill.py > private/sentence_fill.log 2>&1 &
python3 rerank_fill.py --check
python3 rerank_fill.py --deadline 06:00 --workers 2
wait
log "GPU fill finished"
