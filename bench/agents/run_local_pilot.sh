#!/bin/sh
# Local pilot: 5 tasks x 3 conditions x 2 context windows on Ollama.
cd "$(dirname "$0")"
for ctx in 8192 32768; do
  for t in t00 t01 t02 t03 t04; do
    for c in files gbrain-chunk gbrain-page; do
      python3 local_agent.py --task "$t" --condition "$c" --num-ctx "$ctx" --out runs/local_pilot.jsonl
    done
  done
done
# re-run cells whose last attempt recorded an error, once, with the same code as the rest
python3 - <<'PY'
import json, subprocess
rows = [json.loads(l) for l in open("runs/local_pilot.jsonl")]
last = {}
for r in rows:
    last[(r["task"], r["condition"], r["num_ctx"])] = r
for (t, c, n), r in last.items():
    if r["error"]:
        subprocess.run(["python3", "local_agent.py", "--task", t, "--condition", c, "--num-ctx", str(n), "--out", "runs/local_pilot.jsonl"])
PY
