#!/usr/bin/env python3
"""Cloud agent run: one task, one host, one model, one condition, one JSON line of measurements.

Hosts
  claude  Claude Code (claude -p, stream-json)
  cursor  Cursor CLI (cursor-agent -p, stream-json)

Conditions
  files-rtk-off  shell and file tools over the read-only corpus copy, no hooks at all
  files-rtk-on   same, plus only the RTK rewrite hook
  gbrain         gbrain MCP search and get_page from an empty workspace, no hooks

Isolation never touches the user's global settings. Claude Code drops user settings with
--setting-sources project,local and injects the RTK hook with --settings. Cursor runs under a
per-condition HOME (profiles/cursor-*) holding a copy of cli-config.json, the keychain symlink
the login needs, and either nothing, the RTK hook, or the gbrain MCP server.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from mcp_client import GBRAIN

HERE = Path(__file__).resolve().parent
SCRATCH = Path.home() / ".cache" / "kbbench"
KB = SCRATCH / "kb"
EMPTY = SCRATCH / "empty"
PROFILES = SCRATCH / "profiles"
RAW = HERE / "runs" / "raw"
TIMEOUT = 420
SANDBOX = os.environ.get("KBBENCH_CURSOR_SANDBOX", "")
REAL_HOME = Path.home()

SYSTEM = (
    "You answer questions from a personal knowledge base of markdown notes. Use the tools to find the notes "
    "that answer the question. Be brief. End your final answer with one line: SOURCES: followed by the slugs "
    "of the notes you used, comma-separated."
)
WHERE = {
    "files": "The notes are the markdown files under the current directory; a note's slug is its path without .md. "
             "Do not modify any file.",
    "gbrain": "The notes are only reachable through the gbrain MCP tools search and get_page; search results carry slugs. "
              "Use no other tool.",
}
FENCE = "Stay inside the current workspace: do not read, search or list anything outside it."
RTK_HOOK_CLAUDE = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}]}}
RTK_HOOK_CURSOR = {"version": 1, "hooks": {"preToolUse": [{"command": "rtk hook cursor", "matcher": "Shell"}]}}
MCP_SERVERS = {"mcpServers": {"gbrain": {"command": GBRAIN[0], "args": GBRAIN[1:], "env": {"HOME": str(Path.home())}}}}


def tokens_est(text: str) -> int:
    return (len(text) + 3) // 4


def cursor_profile(condition: str) -> Path:
    home = PROFILES / f"cursor-{condition}"
    (home / ".cursor").mkdir(parents=True, exist_ok=True)
    (home / ".config" / "cursor").mkdir(parents=True, exist_ok=True)
    (home / "Library").mkdir(exist_ok=True)
    link = home / "Library" / "Keychains"
    if not link.is_symlink():
        link.symlink_to(REAL_HOME / "Library" / "Keychains")
    shutil.copy(REAL_HOME / ".cursor" / "cli-config.json", home / ".cursor" / "cli-config.json")
    shutil.copy(REAL_HOME / ".config" / "cursor" / "cli-config.json", home / ".config" / "cursor" / "cli-config.json")
    hooks = home / ".cursor" / "hooks.json"
    mcp = home / ".cursor" / "mcp.json"
    hooks.unlink(missing_ok=True)
    mcp.unlink(missing_ok=True)
    if condition == "files-rtk-on":
        hooks.write_text(json.dumps(RTK_HOOK_CURSOR))
    if condition == "gbrain":
        mcp.write_text(json.dumps(MCP_SERVERS))
    return home


def claude_cmd(model: str, condition: str, prompt: str) -> tuple[list, dict, Path]:
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", model,
           "--setting-sources", "project,local", "--no-session-persistence", "--disable-slash-commands",
           "--append-system-prompt", SYSTEM, "--strict-mcp-config", "--dangerously-skip-permissions"]
    if "haiku" not in model:
        cmd += ["--effort", "medium"]
    if condition == "gbrain":
        cfg = HERE / "runs" / "gbrain-mcp.json"
        cfg.write_text(json.dumps(MCP_SERVERS))
        cmd += ["--tools", "", "--mcp-config", str(cfg),
                "--allowedTools", "mcp__gbrain__search", "mcp__gbrain__get_page"]
        cwd = EMPTY
    else:
        cmd += ["--tools", "Bash,Read,Grep,Glob"]
        if condition == "files-rtk-on":
            cmd += ["--settings", json.dumps(RTK_HOOK_CLAUDE)]
        cwd = KB
    return cmd, {}, cwd


def cursor_cmd(model: str, condition: str, prompt: str) -> tuple[list, dict, Path]:
    home = cursor_profile(condition)
    cwd = EMPTY if condition == "gbrain" else KB
    cmd = ["cursor-agent", "-p", f"{SYSTEM}\n\n{prompt}", "--output-format", "stream-json", "--model", model,
           "--force", "--trust", "--workspace", str(cwd)]
    if condition == "gbrain":
        cmd.append("--approve-mcps")
    if SANDBOX:
        cmd += ["--sandbox", SANDBOX]
    return cmd, {"HOME": str(home)}, cwd


def parse_claude(events: list) -> dict:
    out = {"tool_calls": 0, "tool_names": [], "tool_args": [], "tool_result_tokens": 0, "answer": "", "error": None}
    for d in events:
        t = d.get("type")
        if t == "assistant":
            for c in d["message"].get("content", []):
                if c.get("type") == "tool_use":
                    out["tool_calls"] += 1
                    out["tool_names"].append(c["name"])
                    out["tool_args"].append(json.dumps(c.get("input", {})))
        elif t == "user":
            for c in d["message"].get("content", []) if isinstance(d["message"].get("content"), list) else []:
                if c.get("type") == "tool_result":
                    body = c.get("content")
                    text = body if isinstance(body, str) else json.dumps(body)
                    out["tool_result_tokens"] += tokens_est(text)
        elif t == "result":
            u = d.get("usage", {})
            out["answer"] = d.get("result") or ""
            out["input_tokens"] = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
            out["uncached_input_tokens"] = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
            out["output_tokens"] = u.get("output_tokens", 0)
            out["cost_usd"] = d.get("total_cost_usd")
            out["turns"] = d.get("num_turns")
            if d.get("is_error"):
                out["error"] = str(d.get("result"))[:300]
    return out


def parse_cursor(events: list) -> dict:
    out = {"tool_calls": 0, "tool_names": [], "tool_args": [], "tool_result_tokens": 0, "answer": "", "error": None}
    last_text = ""
    for d in events:
        t = d.get("type")
        if t == "tool_call" and d.get("subtype") == "completed":
            call = d.get("tool_call", {})
            name = next(iter(call), "unknown")
            out["tool_calls"] += 1
            out["tool_names"].append(name.removesuffix("ToolCall"))
            out["tool_args"].append(json.dumps(call.get(name, {}).get("args", {})))
            out["tool_result_tokens"] += tokens_est(json.dumps(call.get(name, {}).get("result", "")))
        elif t == "assistant":
            last_text = "".join(c.get("text", "") for c in d["message"].get("content", []))
        elif t == "result":
            u = d.get("usage", {})
            out["answer"] = d.get("result") or last_text
            out["input_tokens"] = u.get("inputTokens", 0) + u.get("cacheReadTokens", 0) + u.get("cacheWriteTokens", 0)
            out["uncached_input_tokens"] = u.get("inputTokens", 0) + u.get("cacheWriteTokens", 0)
            out["output_tokens"] = u.get("outputTokens", 0)
            out["cost_usd"] = None
            out["turns"] = None
            if d.get("is_error"):
                out["error"] = str(d.get("result"))[:300]
    return out


def escapes(tool_args: list, cwd: Path) -> list:
    """Tool arguments that reach outside the workspace: absolute paths elsewhere, home or parent references."""
    bad = []
    for args in tool_args:
        for path in re.findall(r"(?:/Users|/private|/tmp|/var|/etc|~)[^\s\"',;|]*|\.\./", args):
            if not path.startswith(str(cwd)) and not path.startswith(str(PROFILES)):
                bad.append(path[:120])
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--host", required=True, choices=["claude", "cursor"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["files-rtk-off", "files-rtk-on", "gbrain"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = next(t for t in json.loads((HERE / "tasks.json").read_text()) if t["id"] == a.task)
    EMPTY.mkdir(exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    where = WHERE["gbrain" if a.condition == "gbrain" else "files"]
    prompt = f"{where} {FENCE}\n\nQuestion: {task['query']}"
    build = claude_cmd if a.host == "claude" else cursor_cmd
    cmd, env_extra, cwd = build(a.model, a.condition, prompt)
    env = {**os.environ, **env_extra}
    t0 = time.perf_counter()
    error = None
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
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
    tag = f"{a.host}_{a.model}_{a.condition}_{a.task}".replace("/", "-")
    (RAW / f"{tag}.jsonl").write_text(stdout)
    stats = parse_claude(events) if a.host == "claude" else parse_cursor(events)
    if error or not events:
        stats["error"] = error or (proc.stderr[:300] if not events else stats["error"])
    answer = stats.pop("answer")
    stats["escapes"] = escapes(stats.pop("tool_args"), cwd)
    stats["escaped"] = bool(stats["escapes"])
    m = re.findall(r"SOURCES:\s*(.+)", answer, flags=re.I)
    sources = [s.strip().strip("`*").removesuffix(".md").removeprefix("./") for s in m[-1].split(",")] if m else []
    rec = {"host": a.host, "model": a.model, "condition": a.condition,
           "rtk": "on" if a.condition == "files-rtk-on" else "off", "task": a.task, "relevant": task["relevant"],
           "sources": sources, "page_found": task["relevant"] in sources, "answer": answer, "wall_s": wall, **stats}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("host", "model", "condition", "task", "page_found", "wall_s",
                                               "input_tokens", "output_tokens", "tool_calls", "error")}))


if __name__ == "__main__":
    main()
