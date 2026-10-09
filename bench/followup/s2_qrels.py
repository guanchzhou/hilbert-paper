#!/usr/bin/env python3
"""S2: how incomplete are the relevance labels? As pre-registered in preregistration.md.

For 100 development questions, the unlabelled notes in the union of the top 10 of the reference
configuration and of the best-recall configuration are judged for relevance by the local 27B judge.
Judgements are cached in private/s2-judge.json (resumable). Writes qrels-extended.json (slugs and
judgements only) and private/s2-pairs-for-labels.json (30 pairs for the labelling page).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / "factorial"
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(FACTORIAL))
from common import OLLAMA_URL, post, rounded, wilcoxon  # noqa: E402
from idea02_prune import JUDGE_MODEL  # noqa: E402
from metrics import estimate_tokens, recall_at_k  # noqa: E402
from pipeline import CACHE, DEPTH, Cfg, Data, rerank_order  # noqa: E402
from stage1 import cached_pairs  # noqa: E402

REF = Cfg("dense", "best", "off", "off", "off", "off", "none")
BEST = Cfg("hybrid", "mean", "on", "on", "off", "off", "none")
N, CAP_CALLS, CAP_N, NOTE_TOKENS = 100, 1200, 80, 6000
PROMPT = ("You are judging relevance. Question:\n{q}\n\nNote:\n{text}\n\nIs this note relevant to the question, that is, "
          "does it contain information that answers the question or a substantial part of it? Reply with JSON only: "
          '{{"answer": "yes"}} or {{"answer": "no"}}')
JCACHE = HERE / "private" / "s2-judge.json"


def judge(q: str, text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = post(OLLAMA_URL, {"model": JUDGE_MODEL, "stream": False, "think": False, "format": "json",
                            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 20},
                            "messages": [{"role": "user", "content": PROMPT.format(q=q, text=text)}]}, timeout=1200)
    raw = out["message"]["content"]
    try:
        ans = json.loads(raw).get("answer", "").strip().lower()
    except Exception:  # noqa: BLE001
        ans = "yes" if "yes" in raw.lower() else "no"
    return ans, time.perf_counter() - t0


def main() -> None:
    t0 = time.time()
    d = Data()
    b = d.b
    z = np.load(CACHE / "pre-rankings.npz")
    pre = {k: (z["top"][i].astype(np.int64), z["rows"][i].astype(np.int64)) for i, k in enumerate(z["keys"])}
    scores = cached_pairs(d)
    path = CACHE / "rerank-pairs.json"
    if path.exists():
        scores.update({k: v["score"] for k, v in json.loads(path.read_text()).items()})

    def top10(cfg: Cfg, qi: int) -> list[str] | None:
        t, r = pre[cfg.pre().key]
        t, r = t[qi], r[qi]
        if cfg.f3 == "on":
            n = int((t >= 0).sum())
            sc = [scores.get(f"{qi}:{b.ids[int(x)]}") for x in r[:n]]
            if any(v is None for v in sc):
                return None
            o = rerank_order(np.arange(DEPTH), np.array(sc + [0.0] * (DEPTH - n)))
            o = np.concatenate([o[:n], np.arange(n, DEPTH)])
            t = t[o]
        return [b.pages[int(x)] for x in t[:10] if x >= 0]

    rng = np.random.default_rng(20261008)
    drawn = [int(x) for x in rng.choice(b.n, size=N, replace=False)]
    lists = {qi: (top10(REF, qi), top10(BEST, qi)) for qi in drawn}
    pairs = {qi: sorted((set(lists[qi][0]) | set(lists[qi][1] or [])) - set(b.rels[qi])) for qi in drawn}
    calls = sum(len(v) for v in pairs.values())
    sample = drawn if calls <= CAP_CALLS else drawn[:CAP_N]
    calls_used = sum(len(pairs[qi]) for qi in sample)
    print("calls", calls, "-> using", len(sample), "questions,", calls_used, "calls", flush=True)

    def note_text(slug: str) -> str:
        i = b.page_index[slug]
        out, used = [], 0
        for row in range(b.starts[i], b.ends[i]):
            t = b.texts[row]
            if used + estimate_tokens(t) > NOTE_TOKENS:
                out.append(t[: max(0, (NOTE_TOKENS - used) * 4)])
                break
            out.append(t)
            used += estimate_tokens(t)
        return "\n\n".join(out)

    done = json.loads(JCACHE.read_text()) if JCACHE.exists() else {}
    JCACHE.parent.mkdir(exist_ok=True)
    for qi in sample:
        for s in pairs[qi]:
            key = f"{qi}|{s}"
            if key in done:
                continue
            ans, secs = judge(b.queries[qi], note_text(s))
            done[key] = {"answer": ans, "seconds": secs}
            JCACHE.write_text(json.dumps(done))
            print("judge", len(done), "of", calls_used, ans, round(secs, 1), flush=True)

    yes_by_q = {qi: [s for s in pairs[qi] if done[f"{qi}|{s}"]["answer"] == "yes"] for qi in sample}
    n_unlab = np.array([len(pairs[qi]) for qi in sample], dtype=float)
    n_yes = np.array([len(yes_by_q[qi]) for qi in sample], dtype=float)
    boot = np.random.default_rng(20261008)
    ratios = []
    for _ in range(10000):
        idx = boot.integers(0, len(sample), len(sample))
        ratios.append(n_yes[idx].sum() / max(1.0, n_unlab[idx].sum()))
    res = {}
    for name, cfg_i in (("reference", 0), ("best_recall", 1)):
        orig, ext = [], []
        for qi in sample:
            top = lists[qi][cfg_i] or []
            orig.append(recall_at_k(top, set(b.rels[qi]), 10))
            ext.append(recall_at_k(top, set(b.rels[qi]) | set(yes_by_q[qi]), 10))
        res[name] = {"original": np.array(orig), "extended": np.array(ext)}
    gain = {}
    for lab in ("original", "extended"):
        t = wilcoxon(res["best_recall"][lab], res["reference"][lab], alternative="two-sided")
        t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
        gain[lab] = t
    out = {"preregistration": "preregistration.md#s2", "judge_model": JUDGE_MODEL, "prompt": PROMPT,
           "questions_drawn": N, "questions_used": len(sample), "calls_counted": calls, "calls_used": calls_used,
           "cap_rule": f"more than {CAP_CALLS} calls -> first {CAP_N} drawn questions",
           "unlabelled_judged_relevant": {"share": float(n_yes.sum() / max(1.0, n_unlab.sum())),
                                          "ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                                          "pairs": int(n_unlab.sum()), "yes": int(n_yes.sum()),
                                          "questions_with_any_yes": int((n_yes > 0).sum())},
           "R@10": {k: {lab: float(v[lab].mean()) for lab in v} for k, v in res.items()},
           "reranker_gain": gain, "judge_seconds_median": float(np.median([v["seconds"] for v in done.values()])),
           "seconds": time.time() - t0,
           "per_question": [{"qi": qi, "unlabelled": pairs[qi], "judged_relevant": yes_by_q[qi]} for qi in sample]}
    (HERE / "qrels-extended.json").write_text(json.dumps(rounded(out), indent=1) + "\n")

    yes_pairs = [(qi, s) for qi in sample for s in yes_by_q[qi]]
    no_pairs = [(qi, s) for qi in sample for s in pairs[qi] if s not in yes_by_q[qi]]
    pick = np.random.default_rng(20261009)
    chosen = [yes_pairs[i] for i in pick.permutation(len(yes_pairs))[:15]] + [no_pairs[i] for i in pick.permutation(len(no_pairs))[:15]]
    items = [{"kind": "relevance", "id": f"rel-{qi}-{s}", "qi": qi, "slug": s, "judge": done[f"{qi}|{s}"]["answer"],
              "question": b.queries[qi], "text": f"{s}\n\n{note_text(s)}"} for qi, s in chosen]
    order = pick.permutation(len(items))
    (HERE / "private" / "s2-pairs-for-labels.json").write_text(json.dumps([items[i] for i in order], ensure_ascii=False))
    print(json.dumps(rounded({k: out[k] for k in ("questions_used", "calls_used", "unlabelled_judged_relevant", "R@10")})))


if __name__ == "__main__":
    main()
