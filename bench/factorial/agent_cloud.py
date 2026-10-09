#!/usr/bin/env python3
"""One cloud agent run for the level 2 design. Extends agents/cloud_agent.py (unchanged) with a pinned
gbrain evidence budget and packing unit (through mcp_relay.py) and per-cell isolation, so concurrent
runs never share a profile or MCP config file."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agents"))
import cloud_agent as ca  # noqa: E402

PRIVATE = HERE / "private"
RAW = PRIVATE / "raw"
TASKS = HERE.parent / "agents" / "tasks.json"


def relay_servers(budget: int, unit: str) -> dict:
    return {"mcpServers": {"gbrain": {"command": sys.executable, "args": [str(HERE / "mcp_relay.py")],
                                      "env": {"HOME": str(ca.REAL_HOME), "RELAY_BUDGET": str(budget),
                                              "RELAY_UNIT": unit}}}}


def cursor_cmd(model: str, condition: str, prompt: str, budget: int, unit: str, tag: str):
    name = f"cursor-{condition}" + (f"-{budget}-{unit}" if condition == "gbrain" else "")
    home = ca.PROFILES / name
    (home / ".cursor").mkdir(parents=True, exist_ok=True)
    (home / ".config" / "cursor").mkdir(parents=True, exist_ok=True)
    (home / "Library").mkdir(exist_ok=True)
    link = home / "Library" / "Keychains"
    if not link.is_symlink():
        link.symlink_to(ca.REAL_HOME / "Library" / "Keychains")
    shutil.copy(ca.REAL_HOME / ".cursor" / "cli-config.json", home / ".cursor" / "cli-config.json")
    shutil.copy(ca.REAL_HOME / ".config" / "cursor" / "cli-config.json", home / ".config" / "cursor" / "cli-config.json")
    hooks, mcp = home / ".cursor" / "hooks.json", home / ".cursor" / "mcp.json"
    hooks.unlink(missing_ok=True)
    mcp.unlink(missing_ok=True)
    if condition == "files-rtk-on":
        hooks.write_text(json.dumps(ca.RTK_HOOK_CURSOR))
    if condition == "gbrain":
        mcp.write_text(json.dumps(relay_servers(budget, unit)))
    cwd = ca.EMPTY if condition == "gbrain" else ca.KB
    cmd = ["cursor-agent", "-p", f"{ca.SYSTEM}\n\n{prompt}", "--output-format", "stream-json", "--model", model,
           "--force", "--trust", "--workspace", str(cwd)]
    if condition == "gbrain":
        cmd.append("--approve-mcps")
    if ca.SANDBOX:
        cmd += ["--sandbox", ca.SANDBOX]
    return cmd, {"HOME": str(home)}, cwd


def claude_cmd(model: str, condition: str, prompt: str, budget: int, unit: str, tag: str):
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", model,
           "--setting-sources", "project,local", "--no-session-persistence", "--disable-slash-commands",
           "--append-system-prompt", ca.SYSTEM, "--strict-mcp-config", "--dangerously-skip-permissions"]
    if "haiku" not in model:
        cmd += ["--effort", "medium"]
    if condition == "gbrain":
        cfg = PRIVATE / "mcp" / f"{tag}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(relay_servers(budget, unit)))
        cmd += ["--tools", "", "--mcp-config", str(cfg),
                "--allowedTools", "mcp__gbrain__search", "mcp__gbrain__get_page"]
        cwd = ca.EMPTY
    else:
        cmd += ["--tools", "Bash,Read,Grep,Glob"]
        if condition == "files-rtk-on":
            cmd += ["--settings", json.dumps(ca.RTK_HOOK_CLAUDE)]
        cwd = ca.KB
    return cmd, {}, cwd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--host", required=True, choices=["claude", "cursor"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["files-rtk-off", "files-rtk-on", "gbrain"])
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--unit", default="")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = next(t for t in json.loads(TASKS.read_text()) if t["id"] == a.task)
    ca.EMPTY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    where = ca.WHERE["gbrain" if a.condition == "gbrain" else "files"]
    prompt = f"{where} {ca.FENCE}\n\nQuestion: {task['query']}"
    tag = a.run_id.replace("/", "-")
    build = claude_cmd if a.host == "claude" else cursor_cmd
    cmd, env_extra, cwd = build(a.model, a.condition, prompt, a.budget, a.unit, tag)
    env = {**os.environ, **env_extra}
    t0 = time.perf_counter()
    error, proc = None, None
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=ca.TIMEOUT)
        stdout = proc.stdout
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        error = "timeout"
    wall = time.perf_counter() - t0
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    (RAW / f"{tag}.jsonl").write_text(stdout)
    stats = ca.parse_claude(events) if a.host == "claude" else ca.parse_cursor(events)
    if error or not events:
        stats["error"] = error or ((proc.stderr[:300] if proc else "") if not events else stats["error"])
    answer = stats.pop("answer")
    stats["escapes"] = ca.escapes(stats.pop("tool_args"), cwd)
    stats["escaped"] = bool(stats["escapes"])
    m = re.findall(r"SOURCES:\s*(.+)", answer, flags=re.I)
    sources = [s.strip().strip("`*").removesuffix(".md").removeprefix("./") for s in m[-1].split(",")] if m else []
    rec = {"run_id": a.run_id, "host": a.host, "model": a.model, "condition": a.condition,
           "tool": "gbrain" if a.condition == "gbrain" else "files",
           "rtk": "on" if a.condition == "files-rtk-on" else "off", "budget": a.budget or None,
           "unit": a.unit or None, "task": a.task, "relevant": task["relevant"], "sources": sources,
           "page_found": task["relevant"] in sources, "answer": answer, "wall_s": wall, **stats}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("run_id", "page_found", "wall_s", "input_tokens", "output_tokens",
                                               "tool_calls", "error")}), flush=True)


if __name__ == "__main__":
    main()
