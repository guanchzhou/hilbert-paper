#!/usr/bin/env python3
"""Execute the level 2 design in rounds: cloud runs three at a time, local runs one at a time,
both streams in parallel. No new run starts after the deadline; cloud runs are capped. Resumable."""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "private" / "level2-runs.jsonl"
LOCAL_TIMEOUT = 900
lock = threading.Lock()


def deadline_at(hhmm: str) -> datetime:
    hh, mm = map(int, hhmm.split(":"))
    now = datetime.now()
    d = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return d if d > now else d + timedelta(days=1)


def done_ids() -> set:
    if not OUT.exists():
        return set()
    return {json.loads(line)["run_id"] for line in OUT.read_text().splitlines() if line.strip()}


def cloud_cmd(row: dict, run: dict) -> list:
    host = "cursor" if row["host"] == "cursor" else "claude"
    cmd = [sys.executable, str(HERE / "agent_cloud.py"), "--task", run["task"], "--host", host,
           "--model", row["model"], "--condition", row["condition"], "--run-id", run["run_id"], "--out", str(OUT)]
    if row["condition"] == "gbrain":
        cmd += ["--budget", str(row["budget"]), "--unit", row["unit"]]
    return cmd


def local_cmd(row: dict, run: dict) -> list:
    cmd = [sys.executable, str(HERE / "agent_local.py"), "--task", run["task"], "--tool", row["tool"],
           "--num-ctx", str(row["num_ctx"]), "--run-id", run["run_id"], "--out", str(OUT)]
    if row["tool"] == "gbrain":
        cmd += ["--unit", row["unit"]]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="07:30")
    ap.add_argument("--cloud-cap", type=int, default=150)
    ap.add_argument("--only", choices=["cloud", "local"], default=None)
    ap.add_argument("--max-ctx", type=int, default=None, help="skip local runs with a larger window")
    a = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    design = json.loads((HERE / "level2-design.json").read_text())
    rows = design["rows"]
    stop = deadline_at(a.deadline)
    done = done_ids()
    cloud = [r for r in design["runs"] if rows[r["row"]]["stratum"] != "local" and r["run_id"] not in done]
    local = [r for r in design["runs"] if rows[r["row"]]["stratum"] == "local" and r["run_id"] not in done]
    if a.max_ctx is not None:
        local = [r for r in local if rows[r["row"]]["num_ctx"] <= a.max_ctx]
    cloud.sort(key=lambda r: r["round"])
    local.sort(key=lambda r: r["round"])
    started = {"cloud": len([1 for i in done if rows[int(i[1:3])]["stratum"] != "local"])}
    print(f"{len(cloud)} cloud and {len(local)} local runs to go; deadline {stop}", flush=True)

    def run_cloud(run: dict) -> None:
        with lock:
            if datetime.now() >= stop or started["cloud"] >= a.cloud_cap:
                return
            started["cloud"] += 1
        subprocess.run(cloud_cmd(rows[run["row"]], run))

    def run_local_stream() -> None:
        for run in local:
            if datetime.now() >= stop:
                return
            row = rows[run["row"]]
            t0 = time.perf_counter()
            try:
                subprocess.run(local_cmd(row, run), timeout=LOCAL_TIMEOUT)
            except subprocess.TimeoutExpired:
                rec = {"run_id": run["run_id"], "host": "local", "model": row["model"], "tool": row["tool"],
                       "num_ctx": row["num_ctx"], "unit": row.get("unit"), "task": run["task"], "sources": [],
                       "page_found": False, "answer": "", "wall_s": time.perf_counter() - t0, "input_tokens": 0,
                       "output_tokens": 0, "tool_calls": 0, "steps": 0, "error": "timeout"}
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps({"run_id": run["run_id"], "error": "timeout"}), flush=True)

    threads = []
    if a.only != "cloud":
        t = threading.Thread(target=run_local_stream)
        t.start()
        threads.append(t)
    if a.only != "local":
        with ThreadPoolExecutor(3) as pool:
            list(pool.map(run_cloud, cloud))
    for t in threads:
        t.join()
    print("finished at", datetime.now().strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
