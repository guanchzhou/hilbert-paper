#!/usr/bin/env python3
"""One local agent run (Ollama) for the level 2 design. Same loop as agents/local_agent.py (unchanged),
with the gbrain packing unit as a parameter; the gbrain evidence budget stays a quarter of num_ctx."""

import argparse
import json
import re
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agents"))
import local_agent as la  # noqa: E402
from mcp_client import GBRAIN, MCP  # noqa: E402

TASKS = HERE.parent / "agents" / "tasks.json"


def run_tool(name: str, args: dict, tool: str, unit: str, mcp, cap: int) -> str:
    if name in ("grep", "read"):
        return la.run_tool(name, args, "files", None, cap)
    if name == "search":
        out = mcp.tool("search", {"query": args.get("query", ""), "limit": 10, "return_unit": unit,
                                  "snippet_chars": 0, "token_budget": cap})
        return la.clip(out, cap)
    if name == "get_page":
        return la.clip(mcp.tool("get_page", {"slug": args.get("slug", ""), "fuzzy": True}), cap)
    return "unknown tool"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tool", required=True, choices=["files", "gbrain"])
    ap.add_argument("--unit", default="chunk")
    ap.add_argument("--model", default="qwen3.8:latest")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = next(t for t in json.loads(TASKS.read_text()) if t["id"] == a.task)
    tool_cap = max(500, a.num_ctx // 4)
    tools = la.FILE_TOOLS if a.tool == "files" else la.GBRAIN_TOOLS
    mcp = MCP(GBRAIN) if a.tool == "gbrain" else None
    messages = [{"role": "system", "content": la.SYSTEM}, {"role": "user", "content": task["query"]}]
    stats = {"prompt_tokens": 0, "output_tokens": 0, "tool_calls": 0, "tool_result_tokens": 0,
             "max_prompt_tokens": 0, "steps": 0, "context_overflow": False, "compactions": 0, "error": None}
    answer = ""
    t0 = time.perf_counter()
    try:
        for _ in range(la.MAX_STEPS):
            for _ in range(4):
                try:
                    r = la.chat(a.model, messages, tools, a.num_ctx, False)
                    break
                except urllib.error.HTTPError as exc:
                    old = next((m for m in messages if m["role"] == "tool" and not m["content"].startswith("[dropped")), None)
                    if exc.code != 500 or old is None:
                        raise
                    old["content"] = "[dropped to fit the context window]"
                    stats["compactions"] += 1
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
                result = run_tool(fn["name"], fn.get("arguments") or {}, a.tool, a.unit, mcp, min(tool_cap, per_call))
                stats["tool_calls"] += 1
                stats["tool_result_tokens"] += la.tokens_est(result)
                messages.append({"role": "tool", "content": result, "tool_name": fn["name"]})
            while sum(la.tokens_est(m.get("content") or "") for m in messages) > 0.75 * a.num_ctx:
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
    rec = {"run_id": a.run_id, "host": "local", "model": a.model, "tool": a.tool, "num_ctx": a.num_ctx,
           "unit": a.unit if a.tool == "gbrain" else None, "budget": tool_cap if a.tool == "gbrain" else None,
           "rtk": "n/a", "task": a.task, "relevant": task["relevant"], "sources": sources,
           "page_found": task["relevant"] in sources, "answer": answer, "wall_s": wall,
           "input_tokens": stats["prompt_tokens"], **stats}
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("run_id", "page_found", "wall_s", "prompt_tokens", "output_tokens",
                                           "tool_calls", "steps", "error")}), flush=True)


if __name__ == "__main__":
    main()
