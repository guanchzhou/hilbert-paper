"""Fixed metrics for the knowledge-search bench.

nDCG discounts with log2(rank+1) (rank is 1-based). Tokens are
ceil((len-cjk)/4)+cjk. CJK is Han (U+4E00–U+9FFF), Hiragana, Katakana,
and Hangul syllables — the same ranges gbrain's estimateTokens uses.
"""

import math
import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return math.ceil((len(text) - cjk) / 4) + cjk


def recall_at_k(hits: list[str], relevant: set[str], k: int = 10) -> float:
    if k <= 0 or not hits or not relevant:
        return 0.0
    top = _dedupe(hits)[:k]
    return sum(1 for h in top if h in relevant) / len(relevant)


def precision_at_k(hits: list[str], relevant: set[str], k: int = 10) -> float:
    if k <= 0 or not hits or not relevant:
        return 0.0
    top = _dedupe(hits)[:k]
    return sum(1 for h in top if h in relevant) / k


def mrr(hits: list[str], relevant: set[str]) -> float:
    if not hits or not relevant:
        return 0.0
    for i, h in enumerate(_dedupe(hits)):
        if h in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(hits: list[str], relevant: set[str], k: int = 10) -> float:
    if k <= 0 or not hits or not relevant:
        return 0.0
    top = _dedupe(hits)[:k]
    dcg = 0.0
    for i, h in enumerate(top):
        if h in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_n = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    if idcg == 0:
        return 0.0
    return min(1.0, max(0.0, dcg / idcg))


def summarize(lists: list[list[str]], qrels: list[dict], k: int = 10) -> dict:
    n = len(qrels)
    p = r = m = g = 0.0
    for hits, q in zip(lists, qrels):
        rel = set(q["relevant"])
        p += precision_at_k(hits, rel, k)
        r += recall_at_k(hits, rel, k)
        m += mrr(hits, rel)
        g += ndcg_at_k(hits, rel, k)
    return {
        "n": n,
        "P@10": p / n,
        "R@10": r / n,
        "MRR": m / n,
        "nDCG@10": g / n,
    }


def _dedupe(hits: list[str]) -> list[str]:
    seen = set()
    out = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out
