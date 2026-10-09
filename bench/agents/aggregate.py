#!/usr/bin/env python3
"""Aggregate the agent pilot: one row per condition cell, then paired tests per comparison axis.

A comparison awards a point to the better side of a metric only when the Holm-adjusted p-value
(over every test in the file) is below 0.05; otherwise the direction is reported as a lean.
Writes runs/agent_pilot_summary.json (aggregates only, no answers).
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
JUDGES = ["claude-sonnet-5-5", "cursor-gpt-5.6-sol-medium"]
LARGE = {"claude-opus-5-5", "claude-opus-5-5-medium", "gpt-5.6-sol-high"}
SMALL = {"claude-haiku-4-5", "composer-2.5"}
FAMILY = {"claude-opus-5-5": "opus-5.5", "claude-opus-5-5-medium": "opus-5.5",
          "claude-sonnet-5-5": "sonnet-5.5", "claude-sonnet-5-5-medium": "sonnet-5.5"}


def judge_key(rec):
    import hashlib
    return hashlib.sha256(f"{rec['task']}\0{rec['answer']}".encode()).hexdigest()[:16]


def load() -> list:
    scores = {}
    for j in JUDGES:
        p = RUNS / f"judge_{j}.jsonl"
        if p.exists():
            for r in map(json.loads, p.read_text().splitlines()):
                if r["score"] is not None:
                    scores.setdefault(r["key"], {})[j] = r["score"]
    rows = []
    for name in ("cloud_pilot.jsonl", "local_pilot.jsonl"):
        p = RUNS / name
        if not p.exists():
            continue
        last = {}
        for r in map(json.loads, p.read_text().splitlines()):
            cell = (r["host"], r["model"], r["condition"], r.get("num_ctx"), r["task"])
            if cell not in last or last[cell]["error"] or not r["error"]:
                last[cell] = r
        for r in last.values():
            base = r["relevant"].rsplit("/", 1)[-1]
            r["page_found_loose"] = r["page_found"] or any(s.rsplit("/", 1)[-1] == base for s in r["sources"])
            js = scores.get(judge_key(r), {})
            r["judges"] = js
            r["judge"] = float(np.mean(list(js.values()))) if js else None
            if r["host"] == "local-ollama":
                r["host"] = "local"
                r["input_tokens"] = r["prompt_tokens"]
            r["window"] = r["num_ctx"] if r["host"] == "local" else "default"
            rows.append(r)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def table(rows) -> list:
    cells = defaultdict(list)
    for r in rows:
        cells[(r["host"], r["model"], r["condition"], r["rtk"], str(r["window"]))].append(r)
    out = []
    for (host, model, cond, rtk, window), rs in sorted(cells.items()):
        js = [r["judge"] for r in rs if r["judge"] is not None]
        out.append({"host": host, "model": model, "condition": cond, "rtk": rtk, "window": window, "n": len(rs),
                    "errors": sum(bool(r["error"]) for r in rs), "escaped": sum(bool(r.get("escaped")) for r in rs),
                    "page_found": sum(r["page_found_loose"] for r in rs) / len(rs),
                    "judge": float(np.mean(js)) if js else None,
                    "tokens_in": med([r["input_tokens"] for r in rs]),
                    "tokens_out": med([r["output_tokens"] for r in rs]),
                    "tool_result_tokens": med([r["tool_result_tokens"] for r in rs]),
                    "wall_s": med([r["wall_s"] for r in rs]),
                    "tool_calls": med([r["tool_calls"] for r in rs]),
                    "shell_calls": med([sum(n in ("Bash", "shell") for n in r.get("tool_names", [])) for r in rs]),
                    "cost_usd": med([r.get("cost_usd") for r in rs])})
    return out


METRICS = [  # name, field, higher is better
    ("page found", "page_found_loose", True),
    ("judge score", "judge", True),
    ("input tokens", "input_tokens", False),
    ("tool result tokens", "tool_result_tokens", False),
    ("wall time", "wall_s", False),
    ("tool calls", "tool_calls", False),
]


def paired(rows, axis, side_of, match_of):
    """Pair runs that agree on match_of() and differ on side_of(); test each metric."""
    by = defaultdict(dict)
    for r in rows:
        side = side_of(r)
        if side is not None:
            by[match_of(r)][side] = r
    sides = sorted({s for d in by.values() for s in d})
    results = []
    for a, b in combinations(sides, 2):
        pairs = [(d[a], d[b]) for d in by.values() if a in d and b in d]
        if len(pairs) < 3:
            continue
        for name, field, higher in METRICS:
            xs = [(p[field], q[field]) for p, q in pairs if p[field] is not None and q[field] is not None]
            if len(xs) < 3:
                continue
            x = np.array([float(u) for u, _ in xs])
            y = np.array([float(v) for _, v in xs])
            if field == "page_found_loose":
                b01 = int(((x == 1) & (y == 0)).sum())
                b10 = int(((x == 0) & (y == 1)).sum())
                p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
                stat = {"a_only": b01, "b_only": b10}
            else:
                d = x - y
                p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
                stat = {}
            ma, mb = float(np.mean(x)) if field in ("page_found_loose", "judge") else float(np.median(x)), \
                float(np.mean(y)) if field in ("page_found_loose", "judge") else float(np.median(y))
            better = None if ma == mb else (a if (ma > mb) == higher else b)
            results.append({"axis": axis, "a": a, "b": b, "metric": name, "n": len(xs), "a_value": ma, "b_value": mb,
                            "better": better, "p": float(p), **stat})
    return results


def pooled(rows):
    """One synthetic row per (local|cloud, files|gbrain, task): the mean of every run in that group."""
    groups = defaultdict(list)
    for r in rows:
        if r["condition"] not in ("files", "files-rtk-off", "gbrain", "gbrain-chunk"):
            continue
        kind = "local" if r["host"] == "local" else "cloud"
        groups[(kind, "files" if r["condition"].startswith("files") else "gbrain", r["task"])].append(r)
    out = []
    for (kind, cond, task), rs in groups.items():
        row = {"host": kind, "condition": cond, "task": task}
        for _, field, _ in METRICS:
            vals = [float(r[field]) for r in rs if r[field] is not None]
            row[field] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def holm(results):
    order = sorted(range(len(results)), key=lambda i: results[i]["p"])
    m = len(results)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * results[i]["p"]))
        results[i]["p_holm"] = running
        results[i]["point"] = results[i]["better"] if running < 0.05 else None
    return results


def main():
    rows = load()
    cloud = [r for r in rows if r["host"] in ("claude", "cursor")]
    local = [r for r in rows if r["host"] == "local"]
    tests = []
    tests += paired([r for r in cloud if r["condition"].startswith("files")], "RTK on vs off (cloud, files)",
                    lambda r: r["rtk"], lambda r: (r["host"], r["model"], r["task"]))
    tests += paired([r for r in cloud if r["condition"] in ("files-rtk-off", "gbrain")], "gbrain MCP vs plain files (cloud)",
                    lambda r: r["condition"], lambda r: (r["host"], r["model"], r["task"]))
    tests += paired([r for r in cloud if r["model"] in FAMILY], "Cursor vs Claude Code (same model)",
                    lambda r: r["host"], lambda r: (FAMILY[r["model"]], r["condition"], r["task"]))
    tests += paired(cloud, "large vs small model (cloud)",
                    lambda r: "large" if r["model"] in LARGE else "small" if r["model"] in SMALL else None,
                    lambda r: (r["host"], r["condition"], r["task"], r["model"] in LARGE))
    tests += paired(local, "local: files vs gbrain chunk vs gbrain page",
                    lambda r: r["condition"], lambda r: (r["num_ctx"], r["task"]))
    tests += paired(local, "local: small (8k) vs large (32k) window",
                    lambda r: str(r["num_ctx"]), lambda r: (r["condition"], r["task"]))
    tests += paired(pooled(rows), "local model vs cloud models (mean per task and tool family)",
                    lambda r: r["host"], lambda r: (r["condition"], r["task"]))
    tests = holm(tests)
    summary = {"rows": len(rows), "table": table(rows), "tests": tests}
    (RUNS / "agent_pilot_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(rows)} runs")
    for t in summary["table"]:
        print(f"{t['host']:6} {t['model'][:24]:24} {t['condition']:13} rtk={t['rtk']:3} win={t['window']:5} n={t['n']} "
              f"err={t['errors']} esc={t['escaped']} found={t['page_found']:.2f} judge={t['judge'] if t['judge'] is None else round(t['judge'], 2)} "
              f"in={t['tokens_in']} out={t['tokens_out']} wall={t['wall_s'] and round(t['wall_s'])} calls={t['tool_calls']}")
    for t in tests:
        print(f"[{t['axis']}] {t['metric']}: {t['a']}={t['a_value']:.3g} {t['b']}={t['b_value']:.3g} n={t['n']} "
              f"p={t['p']:.3g} holm={t['p_holm']:.3g} point={t['point']} lean={t['better']}")


if __name__ == "__main__":
    main()
