#!/usr/bin/env python3
"""Idea 2: pack more notes into 6,000 tokens by pruning each winning chunk to its most query-relevant sentences.

Sentence vectors and judge transcripts stay in cache/ and private/ (never committed).
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from common import CACHE, OLLAMA_URL, PRIVATE, SEED, Bench, embed, halves, mcnemar, post, unit, write_result
from metrics import estimate_tokens

DEPTH = 30
CAPS = (2, 3, 5)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
JUDGE_MODEL = "qwen3.8:latest"
JUDGE_PROMPT = ("You are checking whether retrieved context is sufficient. Question:\n{q}\n\nContext:\n{ctx}\n\n"
                "Does the context contain the information needed to answer the question? "
                'Reply with JSON only: {{"answer": "yes"}} or {{"answer": "no"}}')


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(text) if s and s.strip()]


def pack_text(units: list[tuple[str, str]]) -> tuple[list[str], str, int]:
    used, kept, parts = 0, [], []
    for slug, text in units:
        cost = estimate_tokens(text)
        if used + cost <= 6000:
            used += cost
            kept.append(slug)
            parts.append(text)
    return kept, "\n\n---\n\n".join(parts), used


def judge(q: str, ctx: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = post(OLLAMA_URL, {"model": JUDGE_MODEL, "stream": False, "think": False, "format": "json",
                            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 20},
                            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(q=q, ctx=ctx)}]},
               timeout=1200)
    text = out["message"]["content"]
    try:
        ans = json.loads(text).get("answer", "").strip().lower()
    except Exception:  # noqa: BLE001
        ans = "yes" if "yes" in text.lower() else "no"
    return ans, time.perf_counter() - t0


def main(run_judge: bool) -> None:
    b = Bench()
    s, win = b.dense()
    order = b.order(s)[:, :DEPTH]
    base_surv, base_tok = b.chunk_pack(s, win)
    fr = b.first_rank(s)
    ceiling = ((fr > 0) & (fr <= DEPTH)).astype(float)

    rows = sorted({int(win[qi, p]) for qi in range(b.n) for p in order[qi]})
    sent_of = {r: sentences(b.texts[r]) for r in rows}
    vec_path = CACHE / "sentence-vectors.npz"
    flat = [x for r in rows for x in sent_of[r]]
    if vec_path.exists():
        V = np.load(vec_path)["V"]
    else:
        t0 = time.perf_counter()
        V = embed(flat, batch=64)
        print("embedded", len(flat), "sentences in", round(time.perf_counter() - t0, 1), "s", flush=True)
        np.savez(vec_path, V=V)
    assert len(V) == len(flat)
    Vn = unit(V)
    offset, pos = {}, 0
    for r in rows:
        offset[r] = (pos, pos + len(sent_of[r]))
        pos += len(sent_of[r])

    def pruned_units(qi: int, cap: int) -> list[tuple[str, str]]:
        units = []
        for p in order[qi]:
            r = int(win[qi, p])
            a, e = offset[r]
            sims = Vn[a:e] @ b.Qn[qi]
            keep = sorted(np.argsort(-sims)[:cap])
            units.append((b.pages[p], "\n".join(sent_of[r][j] for j in keep)))
        return units

    res = {}
    for cap in CAPS:
        surv, tok, notes = np.zeros(b.n), np.zeros(b.n), np.zeros(b.n)
        for qi in range(b.n):
            kept, _, tok[qi] = pack_text(pruned_units(qi, cap))
            surv[qi] = 1.0 if any(k in b.rels[qi] for k in kept) else 0.0
            notes[qi] = len(kept)
        res[cap] = {"survival": surv, "tokens": tok, "notes_packed": notes}
        print(cap, {h: round(v["survival"], 4) for h, v in halves(b, res[cap]).items()},
              round(tok.mean()), round(notes.mean(), 1), flush=True)
    chosen = max(CAPS, key=lambda c: (res[c]["survival"][b.tune].mean(), c))
    c = b.confirm
    t = mcnemar(res[chosen]["survival"][c], base_surv[c])
    t["threshold"] = 0.03
    t["tokens_pruned_mean_confirm"] = float(res[chosen]["tokens"][c].mean())
    t["tokens_chunk_pack_mean_confirm"] = float(base_tok[c].mean())

    guard = {"run": False, "reason": "H2 threshold not met; guard not needed (pre-registered)"}
    passes = t["diff"] >= 0.03 and t["p"] < 0.05
    if passes and run_judge:
        rng = np.random.default_rng(SEED)
        sample = sorted(rng.choice(c, size=100, replace=False).tolist())
        PRIVATE.mkdir(exist_ok=True)
        jpath = PRIVATE / "judge-idea2.json"
        done = json.loads(jpath.read_text()) if jpath.exists() else {}
        for qi in sample:
            if str(qi) in done:
                continue
            _, pruned_ctx, _ = pack_text(pruned_units(qi, chosen))
            _, chunk_ctx, _ = pack_text([(b.pages[p], b.texts[int(win[qi, p])]) for p in order[qi, :10]])
            first = "pruned" if rng.random() < 0.5 else "chunk"
            out = {}
            for cond in ([first] + [x for x in ("pruned", "chunk") if x != first]):
                ctx = pruned_ctx if cond == "pruned" else chunk_ctx
                out[cond], out[cond + "_seconds"] = judge(b.queries[qi], ctx)
            done[str(qi)] = out
            jpath.write_text(json.dumps(done))
            print("judge", qi, out, flush=True)
        yes_p = np.array([done[str(q)]["pruned"] == "yes" for q in sample], dtype=float)
        yes_c = np.array([done[str(q)]["chunk"] == "yes" for q in sample], dtype=float)
        guard = {"run": True, "n": 100, "sample_idx": sample, "yes_pruned": float(yes_p.mean()),
                 "yes_chunk": float(yes_c.mean()), "diff": float((yes_p - yes_c).mean()),
                 "pass": bool(yes_p.mean() >= yes_c.mean() - 0.02),
                 "per_question_yes_pruned": yes_p, "per_question_yes_chunk": yes_c}
    elif passes:
        guard = {"run": False, "reason": "pending judge run"}
    write_result("2-prune", {
        "idea": 2, "name": "sentence-pruned packing of the dense top 30",
        "split_rule": SPLIT.pattern, "depth": DEPTH, "sentences_embedded": len(flat),
        "free_ceiling": {"definition": "hit@30 of the dense ranking: survival if every pruned top-30 unit fitted",
                         **halves(b, {"survival": ceiling})},
        "chunk_pack": halves(b, {"survival": base_surv, "tokens": base_tok}),
        "caps": {str(cap): halves(b, r) for cap, r in res.items()},
        "selection": "sentence cap with the best tune-half survival; ties to the larger cap",
        "chosen_cap": chosen,
        "tune_half_test": mcnemar(res[chosen]["survival"][b.tune], base_surv[b.tune]),
        "primary": {"H2 pruned pack survival": t},
        "judge_guard": guard,
        "per_question": {"confirm_idx": c, "chunk_survival": base_surv, "pruned_survival": res[chosen]["survival"],
                         "pruned_tokens": res[chosen]["tokens"], "pruned_notes": res[chosen]["notes_packed"]},
    }, Path(__file__))
    print("chosen", chosen, t["diff"], t["ci95"], t["p"], guard.get("pass"))


if __name__ == "__main__":
    import sys
    main(run_judge="--judge" in sys.argv)
