#!/usr/bin/env python3
"""Level 2: blind judge, numeric per-run table (no answers), factor marginals, matched contrasts.

Answers stay in private/; the committed files carry numbers, factor levels and task ids only.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
AGENTS = HERE.parent / "agents"
RUNS = HERE / "private" / "level2-runs.jsonl"
REPO = Path.home() / "Development/hilbert-paper/bench/factorial"
JUDGE = "claude-sonnet-5-5"
SEED = 20261005
METRICS = ("page_found", "judge", "input_tokens", "output_tokens", "tool_result_tokens", "wall_s", "tool_calls")


def key(task: str, answer: str) -> str:
    return hashlib.sha256(f"{task}\0{answer}".encode()).hexdigest()[:16]


def boot(x: np.ndarray) -> list:
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def holm(ps: dict) -> dict:
    items = sorted((k, v) for k, v in ps.items() if v is not None)
    items.sort(key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * p))
        out[k] = run
    return out


def sign_test(d: list) -> dict:
    d = np.array([x for x in d if x is not None], dtype=float)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else None
    return {"pairs": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None, "ci95": boot(d),
            "positive": pos, "negative": neg, "p_sign_two_sided": p}


def main() -> None:
    judge = "--no-judge" not in sys.argv
    design = json.loads((HERE / "level2-design.json").read_text())
    rows = design["rows"]
    recs = {}
    for line in RUNS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["run_id"]] = r
    if judge:
        tmp = HERE / "private" / "level2-runs-dedup.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in recs.values()))
        subprocess.run([sys.executable, str(AGENTS / "judge.py"), "--runs", str(tmp), "--judge", JUDGE], check=False)
    cache_path = AGENTS / "runs" / f"judge_{JUDGE}.jsonl"
    scores = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            j = json.loads(line)
            if j.get("score") is not None:
                scores[j["key"]] = j["score"]
    table = []
    for rid, r in sorted(recs.items()):
        row = rows[int(rid[1:3])]
        err = r.get("error")
        t = {"run_id": rid, "row": row["row"], "stratum": row["stratum"], "host": row["host"], "tier": row["tier"],
             "model": row["model"], "tool": "gbrain" if row.get("condition", row.get("tool")) == "gbrain" else "files",
             "rtk": ("on" if row.get("condition") == "files-rtk-on" else "off") if row["stratum"] == "cloud-files" else None,
             "budget": row.get("budget"), "num_ctx": row.get("num_ctx"), "unit": row.get("unit"),
             "task": r["task"], "rep": "abc".index(rid[-1]),
             "page_found": float(bool(r.get("page_found"))),
             "judge": scores.get(key(r["task"], r.get("answer", ""))),
             "input_tokens": r.get("input_tokens") or r.get("prompt_tokens") or 0,
             "output_tokens": r.get("output_tokens") or 0, "tool_result_tokens": r.get("tool_result_tokens") or 0,
             "wall_s": r.get("wall_s"), "tool_calls": r.get("tool_calls") or 0,
             "error": None if not err else ("timeout" if "timeout" in str(err) else
                                            "max_steps" if "max_steps" in str(err) else
                                            "context" if "500" in str(err) or "overflow" in str(err) else "other"),
             "escaped": bool(r.get("escaped")), "context_overflow": bool(r.get("context_overflow")),
             "compactions": r.get("compactions")}
        table.append(t)

    def marg(sel, by):
        out = {}
        groups = defaultdict(list)
        for t in table:
            if sel(t):
                groups[str(t[by])].append(t)
        for g, ts in sorted(groups.items()):
            rec = {"n": len(ts), "errors": sum(1 for x in ts if x["error"])}
            for m in METRICS:
                v = np.array([x[m] for x in ts if x[m] is not None], dtype=float)
                rec[m] = {"mean": float(v.mean()) if len(v) else None, "ci95": boot(v), "n": int(len(v))}
            out[g] = rec
        return out
    cloud = lambda t: t["stratum"] != "local"  # noqa: E731
    marginals = {
        "host (all strata)": marg(lambda t: True, "host"),
        "host, shared model, cloud": marg(lambda t: cloud(t) and t["tier"] == "shared", "host"),
        "model (cloud)": marg(cloud, "model"),
        "tier (cloud)": marg(cloud, "tier"),
        "tool (cloud)": marg(cloud, "tool"),
        "tool (local)": marg(lambda t: t["stratum"] == "local", "tool"),
        "rtk (cloud files)": marg(lambda t: t["stratum"] == "cloud-files", "rtk"),
        "evidence budget (cloud gbrain)": marg(lambda t: t["stratum"] == "cloud-gbrain", "budget"),
        "num_ctx (local)": marg(lambda t: t["stratum"] == "local", "num_ctx"),
        "num_ctx (local gbrain)": marg(lambda t: t["stratum"] == "local" and t["tool"] == "gbrain", "num_ctx"),
        "unit (cloud gbrain)": marg(lambda t: t["stratum"] == "cloud-gbrain", "unit"),
        "unit (local gbrain)": marg(lambda t: t["stratum"] == "local" and t["tool"] == "gbrain", "unit"),
    }

    def pairs(sel_a, sel_b, match, metric):
        A, B = defaultdict(list), defaultdict(list)
        for t in table:
            if t[metric] is None:
                continue
            if sel_a(t):
                A[match(t)].append(t[metric])
            elif sel_b(t):
                B[match(t)].append(t[metric])
        return [float(np.mean(A[k]) - np.mean(B[k])) for k in A if k in B]
    best = design["best_unit"]
    contrasts = {
        "Cursor CLI vs Claude Code, shared model": (
            lambda t: cloud(t) and t["tier"] == "shared" and t["host"] == "cursor",
            lambda t: cloud(t) and t["tier"] == "shared" and t["host"] == "claude",
            lambda t: (t["stratum"], t["rtk"], t["budget"], t["unit"], t["task"], t["rep"])),
        "gbrain vs files": (
            lambda t: t["tool"] == "gbrain", lambda t: t["tool"] == "files" and t["rtk"] in (None, "off"),
            lambda t: (t["host"], t["model"], t["num_ctx"], t["task"])),
        "RTK on vs off": (
            lambda t: t["rtk"] == "on", lambda t: t["rtk"] == "off",
            lambda t: (t["host"], t["model"], t["task"], t["rep"])),
        "largest vs smallest context": (
            lambda t: t["num_ctx"] == 131072 or t["budget"] == 24000,
            lambda t: t["num_ctx"] == 8192 or t["budget"] == 2000,
            lambda t: (t["host"], t["model"], t["tool"], t["unit"], t["task"])),
        f"best unit ({best}) vs chunk": (
            lambda t: t["unit"] == best, lambda t: t["unit"] == "chunk",
            lambda t: (t["host"], t["model"], t["num_ctx"], t["budget"], t["task"])),
    }
    tests = {}
    for metric in ("page_found", "judge", "input_tokens", "wall_s"):
        fam = {}
        for name, (a, b, m) in contrasts.items():
            fam[name] = sign_test(pairs(a, b, m, metric))
        hm = holm({k: v["p_sign_two_sided"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = hm.get(k)
        tests[metric] = fam
    secondary = {
        "Cursor CLI vs Claude Code, all tiers matched (different models)": {
            m: sign_test(pairs(lambda t: cloud(t) and t["host"] == "cursor", lambda t: cloud(t) and t["host"] == "claude",
                               lambda t: (t["stratum"], t["tier"], t["rtk"], t["task"], t["rep"]), m))
            for m in ("page_found", "judge", "input_tokens", "wall_s")},
        "large vs small model, cloud": {
            m: sign_test(pairs(lambda t: cloud(t) and t["tier"] == "large", lambda t: cloud(t) and t["tier"] == "small",
                               lambda t: (t["stratum"], t["host"], t["rtk"], t["task"]), m))
            for m in ("page_found", "judge", "input_tokens", "wall_s")},
    }
    planned = design["runs"]
    cov = {"planned": len(planned), "completed": len(table),
           "completed_cloud": sum(1 for t in table if t["stratum"] != "local"),
           "completed_local": sum(1 for t in table if t["stratum"] == "local"),
           "judged": sum(1 for t in table if t["judge"] is not None),
           "rows_with_any_run": len({t["row"] for t in table}), "rows": len(rows),
           "rounds_completed_local": sorted({r["round"] for r in planned if r["run_id"] in recs
                                             and rows[r["row"]]["stratum"] == "local"})}
    out_runs = {"note": "numeric fields, factor levels and task ids only; answers are private", "runs": table}
    summary = {"coverage": cov, "judge": JUDGE, "marginals": marginals, "primary_contrasts": tests,
               "secondary_contrasts": secondary,
               "test": "exact two-sided sign test on matched-pair differences (pairs matched on task and the listed "
                       "factors, replicate means within a pair cell); Holm over the five primary contrasts per metric"}
    for name, obj in (("level2-runs", out_runs), ("level2-summary", summary)):
        p = HERE / f"{name}.json"
        p.write_text(json.dumps(obj, indent=1) + "\n")
        REPO.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, REPO / p.name)
    print(json.dumps(cov))


if __name__ == "__main__":
    main()
