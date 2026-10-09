#!/usr/bin/env python3
"""Blind judge: score each answer 0/1/2 against the reference note, without host, model or condition.

The judge sees the question, the reference note text and the answer with its SOURCES line removed.
Answers are shuffled and keyed by a hash, so the judge never sees which run produced which answer.
Scores are cached in runs/judge_<judge>.jsonl and reused on re-runs.
"""

import argparse
import hashlib
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_CHARS = 12000

RUBRIC = (
    "You grade an answer to a question about a personal knowledge base. The reference note is the ground truth.\n"
    "Score 2: the answer is correct and covers the main point the reference note gives for this question.\n"
    "Score 1: partly correct or incomplete, with no claim that contradicts the reference.\n"
    "Score 0: wrong, contradicts the reference, says it could not find the answer, or is empty.\n"
    'Reply with JSON only: {"score": 0|1|2, "reason": "<one sentence>"}'
)


def key(rec: dict) -> str:
    return hashlib.sha256(f"{rec['task']}\0{rec['answer']}".encode()).hexdigest()[:16]


def strip_sources(answer: str) -> str:
    return re.sub(r"(?im)^.*SOURCES:.*$", "", answer).strip()


def ask(judge: str, prompt: str) -> str:
    if judge.startswith("cursor:"):
        cmd = ["cursor-agent", "-p", prompt, "--output-format", "json", "--model", judge.split(":", 1)[1],
               "--mode", "ask", "--trust", "--workspace", str(HERE / "judge-empty")]
        (HERE / "judge-empty").mkdir(exist_ok=True)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    else:
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", judge, "--tools", "",
               "--setting-sources", "project,local", "--strict-mcp-config", "--no-session-persistence"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/tmp").stdout
    return json.loads(out).get("result", "")


def score_one(judge: str, task: dict, answer: str) -> dict:
    prompt = (f"{RUBRIC}\n\nQuestion: {task['query']}\n\nReference note:\n{task['reference'][:REF_CHARS]}\n\n"
              f"Answer to grade:\n{strip_sources(answer) or '(empty)'}")
    for _ in range(3):
        try:
            text = ask(judge, prompt)
            m = re.search(r"\{.*\}", text, flags=re.S)
            got = json.loads(m.group(0))
            if got.get("score") in (0, 1, 2):
                return got
        except Exception as exc:  # retried, then recorded
            text = f"{type(exc).__name__}: {exc}"
    return {"score": None, "reason": text[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--judge", default="claude-sonnet-5-5")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in json.loads((HERE / "tasks.json").read_text())}
    cache_path = HERE / "runs" / f"judge_{a.judge.replace(':', '-')}.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {r["key"]: r for r in map(json.loads, cache_path.read_text().splitlines()) if r["score"] is not None}
    todo = {}
    for path in a.runs:
        for rec in map(json.loads, Path(path).read_text().splitlines()):
            k = key(rec)
            if k not in cache:
                todo[k] = (rec["task"], rec["answer"])
    items = list(todo.items())
    random.Random(7).shuffle(items)
    print(f"{len(items)} answers to judge, {len(cache)} cached", flush=True)

    def work(item):
        k, (tid, answer) = item
        got = score_one(a.judge, tasks[tid], answer)
        return {"key": k, "task": tid, **got}

    with ThreadPoolExecutor(4) as pool, open(cache_path, "a") as fh:
        for row in pool.map(work, items):
            fh.write(json.dumps(row) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
