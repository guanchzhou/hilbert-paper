#!/usr/bin/env python3
"""S10 gate for arm D, as pre-registered in Addendum 5.

For 100 seeded tuning-half questions, the local generator writes three questions for the question's
best relevant chunk. D runs if the median centred cosine of the real question with the normalised
mean of the generated question vectors exceeds its median centred cosine with the chunk by at least
0.05, and the measured speed projects the whole snapshot to at most six hours. Chunk vectors are
centred by the snapshot's chunk mean; generated vectors and the real question, for that comparison,
by the mean of the 100 generated means. Raw cosines are reported too. The generator runs with the
MLX servers stopped; embedding runs after they are restarted.

    s10_gate.py generate   # MLX servers stopped, Ollama loaded
    s10_gate.py score      # MLX servers running
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE))
from common import Bench, embed, post, unit  # noqa: E402
from idea03_instruction import INSTRUCTIONS, prefix  # noqa: E402
from s6_questions import psql  # noqa: E402

SEED = 20261009
N_GATE = 100
MODEL = "qwen3:4b"
OLLAMA = "http://127.0.0.1:11434/api/chat"
MAX_CHARS = 6000
GAIN = 0.05
MAX_HOURS = 6.0
PRIVATE = HERE.parent / "private" / "s10-gate-generated.json"
PROMPT = ("Write three different questions that the passage below answers. Write them in the language "
          "of the passage, one per line, without numbering.\n\nNote title: {title}\n\nPassage:\n{text}")


def gate_rows(b: Bench) -> list[tuple[int, int]]:
    sim = b.chunk_sim(b.Qn)
    rows = {}
    for i, s in enumerate(b.slugs):
        rows.setdefault(s, []).append(i)
    rng = np.random.default_rng(SEED)
    out = []
    for qi in rng.permutation(b.tune):
        cand = [r for s in b.rels[qi] for r in rows.get(s, [])]
        if cand:
            out.append((int(qi), max(cand, key=lambda r: sim[qi, r])))
        if len(out) == N_GATE:
            break
    return out


def parse(text: str) -> list[str]:
    lines = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", l).strip() for l in text.splitlines()]
    return [l for l in lines if len(l) > 3][:3]


def generate_one(title: str, text: str) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    r = post(OLLAMA, {"model": MODEL, "stream": False, "think": False, "keep_alive": "10m",
                      "messages": [{"role": "user", "content": PROMPT.format(title=title, text=text[:MAX_CHARS])}],
                      "options": {"temperature": 0, "seed": SEED, "num_predict": 160}})
    return parse(r["message"]["content"]), time.perf_counter() - t0


def generate() -> None:
    b = Bench()
    titles = dict(psql("select slug, title from pages where source_id = 'default' and deleted_at is null"))
    pairs = gate_rows(b)
    generate_one("warm-up", "warm-up")
    out = []
    for qi, ci in pairs:
        qs, sec = generate_one(titles.get(b.slugs[ci]) or b.slugs[ci], b.texts[ci])
        out.append({"qi": qi, "chunk": ci, "questions": qs, "seconds": sec})
        print(len(out), round(sec, 2), len(qs), flush=True)
    post(OLLAMA, {"model": MODEL, "keep_alive": 0, "messages": []})
    PRIVATE.write_text(json.dumps({"model": MODEL, "rows": out}, ensure_ascii=False, indent=1) + "\n")


def score() -> None:
    b = Bench()
    g = json.loads(PRIVATE.read_text())["rows"]
    texts = [q for r in g for q in r["questions"]]
    V = unit(embed([prefix(INSTRUCTIONS["default"]) + t for t in texts]))
    means, k = [], 0
    for r in g:
        n = len(r["questions"])
        means.append(V[k:k + n].mean(axis=0))
        k += n
    G = unit(np.array(means))
    qi = np.array([r["qi"] for r in g])
    ci = np.array([r["chunk"] for r in g])
    Q, C = b.Qn[qi], b.Cn[ci]
    mu_c, mu_g = b.Cn.mean(axis=0), G.mean(axis=0)
    cos = lambda X, Y: np.einsum("ij,ij->i", unit(X), unit(Y))
    chunk_c, gen_c = cos(Q - mu_c, C - mu_c), cos(Q - mu_g, G - mu_g)
    chunk_r, gen_r = cos(Q, C), cos(Q, G)
    sec = np.array([r["seconds"] for r in g])
    hours = float(sec.mean() * len(b.slugs) / 3600)
    gain = float(np.median(gen_c) - np.median(chunk_c))
    out = {"preregistration": "preregistration.md#addendum-5", "model": json.loads(PRIVATE.read_text())["model"],
           "n": len(g), "questions_per_chunk": {"mean": float(np.mean([len(r["questions"]) for r in g])),
                                                 "min": min(len(r["questions"]) for r in g)},
           "centred_cosine_median": {"question_chunk": float(np.median(chunk_c)), "question_generated": float(np.median(gen_c))},
           "raw_cosine_median": {"question_chunk": float(np.median(chunk_r)), "question_generated": float(np.median(gen_r))},
           "gain_centred": gain, "gain_required": GAIN,
           "seconds_per_chunk": {"mean": float(sec.mean()), "median": float(np.median(sec)), "p95": float(np.percentile(sec, 95))},
           "projected_hours": hours, "max_hours": MAX_HOURS,
           "share_closer_centred": float((gen_c > chunk_c).mean()),
           "passes": bool(gain >= GAIN and hours <= MAX_HOURS),
           "private": "note names, note text and question texts are kept out of the repository"}
    (HERE / "s10-gate.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"generate": generate, "score": score}[sys.argv[1]]()
