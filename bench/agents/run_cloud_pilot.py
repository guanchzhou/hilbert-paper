#!/usr/bin/env python3
"""Cloud pilot: every (host, model, condition, task) cell once, shuffled, three at a time, resumable."""

import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "cloud_pilot.jsonl"
TASKS = ["t00", "t01", "t02", "t03", "t04"]
CONDITIONS = ["files-rtk-off", "files-rtk-on", "gbrain"]
MODELS = {
    "claude": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"],
    "cursor": ["claude-opus-5-5-medium", "claude-sonnet-5-5-medium", "gpt-5.6-sol-high", "composer-2.5"],
}


def done() -> set:
    if not OUT.exists():
        return set()
    return {(r["host"], r["model"], r["condition"], r["task"]) for r in map(json.loads, OUT.read_text().splitlines())
            if not r.get("error")}


def run(cell: tuple) -> None:
    host, model, condition, task = cell
    subprocess.run(["python3", str(HERE / "cloud_agent.py"), "--host", host, "--model", model,
                    "--condition", condition, "--task", task, "--out", str(OUT)])


cells = [(h, m, c, t) for h, ms in MODELS.items() for m in ms for c in CONDITIONS for t in TASKS]
random.Random(2026).shuffle(cells)
todo = [c for c in cells if c not in done()]
print(f"{len(todo)} of {len(cells)} cells to run", flush=True)
with ThreadPoolExecutor(3) as pool:
    list(pool.map(run, todo))
