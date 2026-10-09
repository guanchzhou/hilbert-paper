#!/usr/bin/env python3
"""S6c: hk1 keys on sentences for the compositional questions, and the author's acceptance rule,
as pre-registered in the addendum of preregistration.md.

Arms: sentence-dense (notes by best sentence per facet, top 50 per facet, intersected) and
sentence-hk1 (a note is in a facet's region if any of its sentences is). The chunk-level arms of S6
are rerun here with timing, and their set recall must equal compositional.json. Writes
compositional-sentences.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "factorial"))
sys.path.insert(0, str(HERE.parent / "ideas"))
sys.path.insert(0, str(HERE.parent))
from pipeline import Data, sentences  # noqa: E402
import stage2  # noqa: E402
from common import CACHE, embed, holm, unit, wilcoxon  # noqa: E402
from idea03_instruction import INSTRUCTIONS, prefix  # noqa: E402
from investigate import hilbert, key_int  # noqa: E402
from metrics import estimate_tokens  # noqa: E402

FACET_K, LEVEL, RANGES, BATCH = 50, 1, 16, 4000
PRIVATE = HERE / "private"


def two_sided(x: np.ndarray, y: np.ndarray) -> dict:
    t = wilcoxon(x, y, alternative="two-sided")
    t["test"] = "two-sided Wilcoxon signed-rank on non-zero paired differences"
    return t


def main() -> None:
    d = Data()
    b = d.b
    sent = stage2.Sentences(d)
    miss = [r for r in range(len(b.texts)) if r not in sent.vec]
    mpath = PRIVATE / "s6c-missing.npz"
    if not mpath.exists():
        flat = [s for r in miss for s in sentences(b.texts[r])]
        np.savez_compressed(mpath, V=embed(flat, batch=64) if flat else np.zeros((0, 1024), np.float32))
    Vm = np.load(mpath)["V"]
    pos = 0
    for r in miss:
        n = len(sentences(b.texts[r]))
        v = Vm[pos:pos + n]
        sent.vec[r] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        sent.text[r] = sentences(b.texts[r])
        pos += n
    assert pos == len(Vm)

    rows = sorted(sent.vec)
    S = np.concatenate([sent.vec[r] for r in rows]).astype(np.float32)
    s_note = np.concatenate([np.full(len(sent.vec[r]), b.page_of_row[r]) for r in rows])
    s_text = [t for r in rows for t in sent.text[r]]
    s_tok = np.array([estimate_tokens(t) for t in s_text])
    order = np.argsort(s_note, kind="stable")
    S, s_note, s_tok = S[order], s_note[order], s_tok[order]
    s_text = [s_text[i] for i in order]
    starts = np.searchsorted(s_note, np.arange(len(b.pages)))
    print("sentences", len(S), "missing embedded", len(Vm), flush=True)

    kpath = PRIVATE / "s6c-keys.npy"
    if kpath.exists():
        skey = np.load(kpath)
    else:
        skey = np.zeros(len(S), dtype=np.uint64)
        for s0 in range(0, len(S), BATCH):
            for r in hilbert(S[s0:s0 + BATCH], []):
                skey[s0 + int(r["id"])] = np.uint64(key_int(r["marker"]))
            print("keyed", min(s0 + BATCH, len(S)), flush=True)
        np.save(kpath, skey)

    spec = json.loads((HERE / "compositional-questions.json").read_text())
    qs = spec["questions"]
    VA = unit(embed([prefix(INSTRUCTIONS["default"]) + q["title_a"] for q in qs]))
    VB = unit(embed([prefix(INSTRUCTIONS["default"]) + q["title_b"] for q in qs]))
    h = np.load(CACHE / "hk1-default.npz")
    ckey, keyed = h["ckey"], h["keyed"]

    t0 = time.perf_counter()
    probes_a = sorted(hilbert(VA, ["--probe", str(LEVEL), "--ranges", str(RANGES)]), key=lambda r: int(r["id"]))
    probes_b = sorted(hilbert(VB, ["--probe", str(LEVEL), "--ranges", str(RANGES)]), key=lambda r: int(r["id"]))
    probe_s = (time.perf_counter() - t0) / len(qs)

    def in_ranges(keys: np.ndarray, probe: dict) -> np.ndarray:
        mask = np.zeros(len(keys), dtype=bool)
        for lo, hi in probe["ranges"]:
            mask |= (keys >= np.uint64(key_int(lo))) & (keys <= np.uint64(key_int(hi)))
        return mask

    def note_tokens(notes: set[int]) -> int:
        return int(sum(b.chunk_tokens[b.starts[p]:b.ends[p]].sum() for p in notes))

    def top_notes(score: np.ndarray, drop: set[int]) -> list[int]:
        out = []
        for p in b.order(score[None, :])[0]:
            if int(p) not in drop:
                out.append(int(p))
            if len(out) == FACET_K:
                break
        return out

    rows_out, arms = [], ["chunk-dense", "chunk-hk1", "sentence-dense", "sentence-hk1"]
    for i, q in enumerate(qs):
        drop = {b.page_index[q["a"]], b.page_index[q["b"]]}
        gold = {b.page_index[s] for s in q["gold"]}
        res = {}

        t = time.perf_counter()
        sa, sb = b.note_scores(b.chunk_sim(np.stack([VA[i], VB[i]])))[0]
        found = set(top_notes(sa, drop)) & set(top_notes(sb, drop))
        res["chunk-dense"] = (found, note_tokens(found), time.perf_counter() - t)

        t = time.perf_counter()
        ra = {int(p) for p in np.unique(b.page_of_row[in_ranges(ckey, probes_a[i]) & keyed])} - drop
        rb = {int(p) for p in np.unique(b.page_of_row[in_ranges(ckey, probes_b[i]) & keyed])} - drop
        found = ra & rb
        res["chunk-hk1"] = (found, note_tokens(found), time.perf_counter() - t + 2 * probe_s)

        t = time.perf_counter()
        sim = np.stack([VA[i], VB[i]]) @ S.T
        best = np.maximum.reduceat(sim, starts, axis=1)
        ta, tb = top_notes(best[0], drop), top_notes(best[1], drop)
        found = set(ta) & set(tb)
        ev = 0
        for p in found:
            lo, hi = starts[p], starts[p + 1] if p + 1 < len(starts) else len(S)
            pick = {lo + int(np.argmax(sim[0, lo:hi])), lo + int(np.argmax(sim[1, lo:hi]))}
            ev += int(s_tok[list(pick)].sum())
        res["sentence-dense"] = (found, ev, time.perf_counter() - t)

        t = time.perf_counter()
        ma, mb = in_ranges(skey, probes_a[i]), in_ranges(skey, probes_b[i])
        na = {int(p) for p in np.unique(s_note[ma])} - drop
        nb = {int(p) for p in np.unique(s_note[mb])} - drop
        found = na & nb
        sel = (ma | mb) & np.isin(s_note, list(found))
        res["sentence-hk1"] = (found, int(s_tok[sel].sum()), time.perf_counter() - t + 2 * probe_s)

        row = {"id": q["id"], "gold": len(gold)}
        for a in arms:
            found, tok, secs = res[a]
            hit = len(found & gold)
            row[a] = {"recall": hit / len(gold), "precision": hit / len(found) if found else 0.0, "size": len(found),
                      "found": hit, "evidence_tokens": tok if a.startswith("sentence") else note_tokens(found),
                      "note_tokens": note_tokens(found), "seconds": secs}
        rows_out.append(row)

    s6 = json.loads((HERE / "compositional.json").read_text())["per_question"]
    for r, old in zip(rows_out, s6):
        assert abs(r["chunk-dense"]["recall"] - old["dense_facets"]["recall"]) < 1e-12, r["id"]
        assert abs(r["chunk-hk1"]["recall"] - old["hk1_facets"]["recall"]) < 1e-12, r["id"]

    def per_correct(a: str, field: str) -> float | None:
        found = sum(r[a]["found"] for r in rows_out)
        return None if found == 0 else sum(r[a][field] for r in rows_out) / found

    summary = {a: {"recall": float(np.mean([r[a]["recall"] for r in rows_out])),
                   "precision": float(np.mean([r[a]["precision"] for r in rows_out])),
                   "size_median": float(np.median([r[a]["size"] for r in rows_out])),
                   "gold_found": int(sum(r[a]["found"] for r in rows_out)),
                   "evidence_tokens_per_correct": per_correct(a, "evidence_tokens"),
                   "note_tokens_per_correct": per_correct(a, "note_tokens"),
                   "seconds_per_correct": per_correct(a, "seconds"),
                   "seconds_per_question": float(np.mean([r[a]["seconds"] for r in rows_out]))} for a in arms}

    def rule(new: str, comp: str, tok: str) -> dict:
        tn, tc = summary[new][tok], summary[comp][tok]
        sn, sc = summary[new]["seconds_per_correct"], summary[comp]["seconds_per_correct"]
        if None in (tn, tc, sn, sc):
            return {"new": new, "comparator": comp, "accepted": False, "note": "no gold found by one arm"}
        return {"new": new, "comparator": comp, "token_measure": tok, "token_ratio": tn / tc, "time_ratio": sn / sc,
                "accepted": bool(tn / tc <= 0.75 and sn / sc <= 2.0)}

    agent = json.loads((HERE / "compositional-agent.json").read_text())["runs"]
    gold_n = {q["id"]: len(q["gold"]) for q in qs}

    def agent_cost(ts: str) -> dict:
        rs = [r for r in agent if r["toolset"] == ts]
        found = sum(r["recall"] * gold_n[r["question"]] for r in rs)
        return {"tokens_per_correct": None if found == 0 else sum(r["prompt_tokens"] for r in rs) / found,
                "seconds_per_correct": None if found == 0 else sum(r["wall_s"] for r in rs) / found, "gold_found": found}

    ag = {t: agent_cost(t) for t in ("search", "dense", "hk1")}
    R = {a: np.array([r[a]["recall"] for r in rows_out]) for a in arms}
    tests = {"sentence-hk1 vs sentence-dense": two_sided(R["sentence-hk1"], R["sentence-dense"]),
             "sentence-hk1 vs chunk-hk1": two_sided(R["sentence-hk1"], R["chunk-hk1"])}
    for k, p in holm({k: t["p"] for k, t in tests.items()}).items():
        tests[k]["p_holm"] = p
    out = {"preregistration": "preregistration.md#addendum", "n": len(rows_out), "sentences": int(len(S)),
           "sentences_embedded_here": int(len(Vm)), "facet_k": FACET_K, "hk1_probe": {"level": LEVEL, "ranges": RANGES},
           "probe_seconds_per_facet": probe_s, "summary": summary, "tests": tests,
           "acceptance_rule": "hk1 accepted if token_ratio <= 0.75 and time_ratio <= 2.0 (per correct answer)",
           "acceptance": {"sentence-hk1 vs sentence-dense (primary)": rule("sentence-hk1", "sentence-dense", "evidence_tokens_per_correct"),
                          "sentence-hk1 vs sentence-dense, whole notes": rule("sentence-hk1", "sentence-dense", "note_tokens_per_correct"),
                          "chunk-hk1 vs chunk-dense": rule("chunk-hk1", "chunk-dense", "note_tokens_per_correct"),
                          "S6b agent, hk1 tool vs dense tool (after the fact)": {
                              "token_ratio": None if not ag["dense"]["tokens_per_correct"] or ag["hk1"]["tokens_per_correct"] is None else ag["hk1"]["tokens_per_correct"] / ag["dense"]["tokens_per_correct"],
                              "time_ratio": None if not ag["dense"]["seconds_per_correct"] or ag["hk1"]["seconds_per_correct"] is None else ag["hk1"]["seconds_per_correct"] / ag["dense"]["seconds_per_correct"],
                              "agent": ag}},
           "per_question": rows_out}
    a = out["acceptance"]["S6b agent, hk1 tool vs dense tool (after the fact)"]
    a["accepted"] = bool(a["token_ratio"] is not None and a["token_ratio"] <= 0.75 and a["time_ratio"] <= 2.0)
    (HERE / "compositional-sentences.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in summary.items():
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items()})
    for k, v in out["acceptance"].items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "agent"})
    for k, t in tests.items():
        print(k, round(t["diff"], 4), "p", t["p"], "holm", t["p_holm"])


if __name__ == "__main__":
    main()
