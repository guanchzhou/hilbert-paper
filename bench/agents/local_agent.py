#!/usr/bin/env python3
"""Local agent loop on Ollama: one task, one condition, one JSON line of measurements.

Conditions
  files        grep and read over the read-only corpus copy
  gbrain-chunk gbrain MCP search returning chunks, plus get_page
  gbrain-page  gbrain MCP search returning whole pages, plus get_page
"""

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from mcp_client import GBRAIN, MCP

HERE = Path(__file__).resolve().parent
KB = HERE / "kb"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MAX_STEPS = 8

SYSTEM = (
    "You answer questions from a personal knowledge base of markdown notes. Use the tools to find the notes "
    "that answer the question. Be brief. End your final answer with one line: SOURCES: followed by the slugs "
    "of the notes you used, comma-separated."
)

FILE_TOOLS = [
    {"type": "function", "function": {"name": "grep", "description": "Search note text with a regular expression. Returns matching lines as slug:line:text, at most 40 lines.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read one note by slug.",
     "parameters": {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}}},
]
GBRAIN_TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search the knowledge base. Returns ranked evidence with slugs.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_page", "description": "Read one note by slug.",
     "parameters": {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}}},
]


def tokens_est(text: str) -> int:
    return (len(text) + 2) // 3


def clip(text: str, max_tokens: int) -> str:
    limit = max_tokens * 3
    return text if len(text) <= limit else text[:limit] + f"\n[clipped at {max_tokens} tokens]"


def run_tool(name: str, args: dict, condition: str, mcp: MCP | None, tool_cap: int) -> str:
    if name == "grep":
        proc = subprocess.run(["rg", "-i", "-n", "--max-count", "3", "-e", args.get("pattern", ""), str(KB)],
                              capture_output=True, text=True)
        lines = []
        for line in proc.stdout.splitlines()[:40]:
            rel = line[len(str(KB)) + 1:]
            rel = re.sub(r"\.md:", ":", rel, count=1)
            lines.append(rel[:300])
        return clip("\n".join(lines) or "no matches", tool_cap)
    if name == "read":
        path = KB / f"{args.get('slug', '').removesuffix('.md')}.md"
        return clip(path.read_text() if path.is_file() else "no such note", tool_cap)
    if name == "search":
        unit = "chunk" if condition == "gbrain-chunk" else "page"
        out = mcp.tool("search", {"query": args.get("query", ""), "limit": 10, "return_unit": unit,
                                  "snippet_chars": 0, "token_budget": tool_cap})
        return clip(out, tool_cap)
    if name == "get_page":
        out = mcp.tool("get_page", {"slug": args.get("slug", ""), "fuzzy": True})
        return clip(out, tool_cap)
    return "unknown tool"


def chat(model: str, messages: list, tools: list, num_ctx: int, think) -> dict:
    body = {"model": model, "messages": messages, "tools": tools, "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": 0}, "think": think}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--condition", required=True, choices=["files", "gbrain-chunk", "gbrain-page"])
    ap.add_argument("--model", default="qwen3.8:latest")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--think", default="false")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = next(t for t in json.loads((HERE / "tasks.json").read_text()) if t["id"] == a.task)
    think = {"false": False, "true": True}.get(a.think, a.think)
    tool_cap = max(500, a.num_ctx // 4)
    tools = FILE_TOOLS if a.condition == "files" else GBRAIN_TOOLS
    mcp = MCP(GBRAIN) if a.condition.startswith("gbrain") else None
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["query"]}]
    stats = {"prompt_tokens": 0, "output_tokens": 0, "tool_calls": 0, "tool_result_tokens": 0,
             "max_prompt_tokens": 0, "steps": 0, "context_overflow": False, "compactions": 0, "overflow_retries": 0,
             "error": None}
    answer = ""
    t0 = time.perf_counter()
    try:
        for step in range(MAX_STEPS):
            for _ in range(4):
                try:
                    r = chat(a.model, messages, tools, a.num_ctx, think)
                    break
                except urllib.error.HTTPError as exc:
                    old = next((m for m in messages if m["role"] == "tool" and not m["content"].startswith("[dropped")), None)
                    if exc.code != 500 or old is None:
                        raise
                    old["content"] = "[dropped to fit the context window]"
                    stats["compactions"] += 1
                    stats["overflow_retries"] += 1
            else:
                raise RuntimeError("context overflow after compaction")
            stats["steps"] += 1
            pt = r.get("prompt_eval_count", 0)
            stats["prompt_tokens"] += pt
            stats["output_tokens"] += r.get("eval_count", 0)
            stats["max_prompt_tokens"] = max(stats["max_prompt_tokens"], pt)
            if pt >= a.num_ctx - 64:
                stats["context_overflow"] = True
            msg = r["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = msg.get("content", "")
                break
            per_call = max(300, int(0.4 * a.num_ctx / len(calls)))
            for c in calls:
                fn = c["function"]
                result = run_tool(fn["name"], fn.get("arguments") or {}, a.condition, mcp, min(tool_cap, per_call))
                stats["tool_calls"] += 1
                stats["tool_result_tokens"] += tokens_est(result)
                messages.append({"role": "tool", "content": result, "tool_name": fn["name"]})
            while sum(tokens_est(m.get("content") or "") for m in messages) > 0.75 * a.num_ctx:
                old = next((m for m in messages if m["role"] == "tool" and not m["content"].startswith("[dropped")), None)
                if old is None:
                    break
                old["content"] = "[dropped to fit the context window]"
                stats["compactions"] += 1
        else:
            answer = messages[-1].get("content", "") if messages[-1].get("role") == "assistant" else ""
            stats["error"] = "max_steps"
    except Exception as exc:  # recorded, not raised: a failed run is a result
        stats["error"] = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if mcp:
            mcp.close()
    wall = time.perf_counter() - t0
    m = re.search(r"SOURCES:\s*(.+)", answer, flags=re.I)
    sources = [s.strip().strip("`").removesuffix(".md") for s in m.group(1).split(",")] if m else []
    rec = {"host": "local-ollama", "model": a.model, "condition": a.condition, "num_ctx": a.num_ctx,
           "think": a.think, "rtk": "n/a", "task": a.task, "relevant": task["relevant"],
           "sources": sources, "page_found": task["relevant"] in sources, "answer": answer,
           "wall_s": wall, **stats}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("task", "condition", "num_ctx", "page_found", "wall_s", "prompt_tokens",
                                           "output_tokens", "tool_calls", "steps", "error")}))


if __name__ == "__main__":
    main()
