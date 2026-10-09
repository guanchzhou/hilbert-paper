#!/usr/bin/env python3
"""Idea 3: corpus-specific query instruction for Qwen3-Embedding. Re-embeds only the 817 questions."""

from pathlib import Path

import numpy as np

from common import CACHE, Bench, embed, halves, unit, wilcoxon, write_result

INSTRUCTIONS = {
    "default": "Given a web search query, retrieve relevant passages that answer the query",
    "personal-notes": ("Given a question about the author's personal notes on software, infrastructure, research "
                       "and genealogy, retrieve the note passage that answers it"),
    "question-oriented": "Given a question, retrieve the passage of a note that answers the question",
}


def prefix(task: str) -> str:
    return f"Instruct: {task}\nQuery:"


def main() -> None:
    b = Bench()
    check = embed([prefix(INSTRUCTIONS["default"]) + b.queries[0]])[0]
    drift = float(np.max(np.abs(check - b.Q[0])))
    cos_check = float(unit(check) @ b.Qn[0])
    print("default check max abs diff", drift, "cos", cos_check, flush=True)
    vecs = {"default": b.Q}
    for name in ("personal-notes", "question-oriented"):
        path = CACHE / f"query-vectors-{name}.npy"
        if not path.exists():
            np.save(path, embed([prefix(INSTRUCTIONS[name]) + q for q in b.queries]))
        vecs[name] = np.load(path)
    res = {}
    for name, V in vecs.items():
        s, _ = b.dense(unit(V))
        res[name] = b.per_question(b.lists(s))
        print(name, {k: round(v, 4) for k, v in halves(b, res[name])["tune"].items()}, flush=True)
    chosen = max(("personal-notes", "question-oriented"), key=lambda n: res[n]["R@10"][b.tune].mean())
    c = b.confirm
    t = wilcoxon(res[chosen]["R@10"][c], res["default"]["R@10"][c])
    t["MRR_diff"] = float((res[chosen]["MRR"][c] - res["default"]["MRR"][c]).mean())
    t["nDCG_diff"] = float((res[chosen]["nDCG@10"][c] - res["default"]["nDCG@10"][c]).mean())
    t["threshold"] = 0.02
    write_result("3-instruction", {
        "idea": 3, "name": "corpus-specific query instruction",
        "format": "Instruct: <task>\\nQuery: + question, no separator (live format)",
        "instructions": INSTRUCTIONS,
        "default_reembed_check": {"max_abs_diff_vs_saved": drift, "cosine_vs_saved": cos_check},
        "selection": "better of personal-notes and question-oriented by tune-half R@10",
        "chosen": chosen,
        "conditions": {n: halves(b, r) for n, r in res.items()},
        "tune_half_test": wilcoxon(res[chosen]["R@10"][b.tune], res["default"]["R@10"][b.tune]),
        "primary": {"H3 query instruction": t},
        "per_question": {"confirm_idx": c, **{f"{n}_R@10": r["R@10"] for n, r in res.items()}},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"])


if __name__ == "__main__":
    main()
