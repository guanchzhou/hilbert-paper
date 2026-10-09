#!/usr/bin/env python3
"""S6b: the level 2 local agent on 20 compositional questions with three tool sets.

(a) gbrain search and get_page; (b) plus facet_intersect(a, b) returning the per-facet dense
intersection; (c) plus facet_intersect(a, b) returning the per-facet hk1 intersection. The agent loop
is agents/local_agent.py, unchanged (deviation D1). Runs are appended to private/s6-agent.jsonl and
skipped when already present; the summary goes to compositional-agent.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent / "agents"))
sys.path.insert(0, str(HERE.parent))
import local_agent as la  # noqa: E402
from common import CACHE, Bench, embed, unit  # noqa: E402
from idea03_instruction import INSTRUCTIONS, prefix  # noqa: E402
from investigate import hilbert, key_int  # noqa: E402
from mcp_client import GBRAIN, MCP  # noqa: E402

MODEL, NUM_CTX, N_QUESTIONS, FACET_K = "qwen3.8:latest", 8192, 20, 50
LOG = HERE / "private" / "s6-agent.jsonl"
FACET_TOOL = {"type": "function", "function": {
    "name": "facet_intersect",
    "description": "Find notes about both of two topics. Give each topic as a short phrase. Returns the slugs and titles of the notes in the intersection.",
    "parameters": {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}}}


class Facets:
    def __init__(self, b: Bench) -> None:
        self.b = b
        h = np.load(CACHE / "hk1-default.npz")
        self.ckey, self.keyed = h["ckey"], h["keyed"]

    def run(self, kind: str, a: str, bb: str) -> list[str]:
        V = unit(embed([prefix(INSTRUCTIONS["default"]) + t for t in (a, bb)]))
        b = self.b
        if kind == "dense":
            sets = []
            for s in b.note_scores(b.chunk_sim(V))[0]:
                sets.append({b.pages[p] for p in b.order(s[None, :])[0][:FACET_K]})
        else:
            sets = []
            for r in sorted(hilbert(V, ["--probe", "1", "--ranges", "16"]), key=lambda r: int(r["id"])):
                mask = np.zeros(len(self.ckey), dtype=bool)
                for lo, hi in r["ranges"]:
                    mask |= (self.ckey >= np.uint64(key_int(lo))) & (self.ckey <= np.uint64(key_int(hi)))
                mask &= self.keyed
                sets.append({b.pages[p] for p in np.unique(b.page_of_row[mask])})
        return sorted(sets[0] & sets[1])


def one_run(q: dict, toolset: str, facets: Facets, titles: dict, mcp: MCP) -> dict:
    tools = list(la.GBRAIN_TOOLS) + ([FACET_TOOL] if toolset != "search" else [])
    tool_cap = max(500, NUM_CTX // 4)
    messages = [{"role": "system", "content": la.SYSTEM}, {"role": "user", "content": q["question"]}]
    stats = {"prompt_tokens": 0, "output_tokens": 0, "tool_calls": 0, "facet_calls": 0, "steps": 0, "error": None}
    answer, t0 = "", time.perf_counter()
    try:
        for _ in range(la.MAX_STEPS):
            r = la.chat(MODEL, messages, tools, NUM_CTX, False)
            stats["steps"] += 1
            stats["prompt_tokens"] += r.get("prompt_eval_count", 0)
            stats["output_tokens"] += r.get("eval_count", 0)
            msg = r["message"]
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = msg.get("content", "")
                break
            per_call = max(300, int(0.4 * NUM_CTX / len(calls)))
            for c in calls:
                fn, args = c["function"]["name"], c["function"].get("arguments") or {}
                cap = min(tool_cap, per_call)
                if fn == "facet_intersect":
                    stats["facet_calls"] += 1
                    found = facets.run(toolset, str(args.get("a", "")), str(args.get("b", "")))
                    result = la.clip("\n".join(f"{s} | {titles.get(s, s)}" for s in found) or "no notes in the intersection", cap)
                elif fn == "search":
                    result = la.clip(mcp.tool("search", {"query": args.get("query", ""), "limit": 10, "return_unit": "chunk",
                                                         "snippet_chars": 0, "token_budget": cap}), cap)
                elif fn == "get_page":
                    result = la.clip(mcp.tool("get_page", {"slug": args.get("slug", ""), "fuzzy": True}), cap)
                else:
                    result = "unknown tool"
                stats["tool_calls"] += 1
                messages.append({"role": "tool", "content": result, "tool_name": fn})
            while sum(la.tokens_est(m.get("content") or "") for m in messages) > 0.75 * NUM_CTX:
                old = next((m for m in messages if m["role"] == "tool" and not m["content"].startswith("[dropped")), None)
                if old is None:
                    break
                old["content"] = "[dropped to fit the context window]"
        else:
            stats["error"] = "max_steps"
            answer = messages[-1].get("content", "") if messages[-1].get("role") == "assistant" else ""
    except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"[:300]
    m = re.search(r"SOURCES:\s*(.+)", answer, flags=re.I)
    sources = {s.strip().strip("`").removesuffix(".md") for s in m.group(1).split(",")} if m else set()
    sources -= {q["a"], q["b"], ""}
    gold = set(q["gold"])
    hit = len(sources & gold)
    return {"run_id": f"q{q['id']}-{toolset}", "question": q["id"], "toolset": toolset, "sources": sorted(sources),
            "recall": hit / len(gold), "precision": hit / len(sources) if sources else 0.0,
            "wall_s": time.perf_counter() - t0, "answer": answer[-600:], **stats}


def main() -> None:
    qs = json.loads((HERE / "compositional-questions.json").read_text())["questions"][:N_QUESTIONS]
    b = Bench()
    facets = Facets(b)
    titles = {q["a"]: q["title_a"] for q in qs} | {q["b"]: q["title_b"] for q in qs}
    import subprocess, os  # noqa: E401
    url = json.loads(Path(os.path.expanduser("~/.gbrain/config.json")).read_text())["database_url"]
    out = subprocess.run(["/opt/homebrew/opt/postgresql@18/bin/psql", url, "-At", "-F", "\x1f", "-c",
                          "select slug, title from pages where source_id = 'default' and deleted_at is null"],
                         capture_output=True, text=True, check=True).stdout
    titles |= dict(line.split("\x1f", 1) for line in out.splitlines() if "\x1f" in line)
    LOG.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["run_id"] for l in LOG.read_text().splitlines()} if LOG.exists() else set()
    rng = np.random.default_rng(20261008)
    plan = [(q, t) for q in qs for t in rng.permutation(["search", "dense", "hk1"]).tolist()]
    mcp = MCP(GBRAIN)
    try:
        for q, t in plan:
            if f"q{q['id']}-{t}" in done:
                continue
            rec = one_run(q, t, facets, titles, mcp)
            with open(LOG, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec["run_id"], round(rec["recall"], 2), round(rec["precision"], 2), rec["tool_calls"], rec["facet_calls"],
                  round(rec["wall_s"]), rec["error"], flush=True)
    finally:
        mcp.close()
    runs = [json.loads(l) for l in LOG.read_text().splitlines()]
    summary = {}
    for t in ("search", "dense", "hk1"):
        rs = [r for r in runs if r["toolset"] == t]
        summary[t] = {k: float(np.mean([r[k] for r in rs])) for k in ("recall", "precision", "tool_calls", "facet_calls", "prompt_tokens", "wall_s")}
        summary[t]["runs"], summary[t]["errors"] = len(rs), sum(1 for r in rs if r["error"])
    (HERE / "compositional-agent.json").write_text(json.dumps(
        {"preregistration": "preregistration.md#s6", "deviations": ["D1"], "model": MODEL, "num_ctx": NUM_CTX,
         "n_questions": N_QUESTIONS, "summary": summary,
         "runs": [{k: r[k] for k in r if k != "answer"} for r in runs]}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
