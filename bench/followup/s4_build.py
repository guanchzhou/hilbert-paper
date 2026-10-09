#!/usr/bin/env python3
"""S4: build the hybrid (H), RECOMP extractive (R) and LongLLMLingua (L) packs for the 150 questions
of the answer-quality check, as pre-registered (deviation D5 for compressor details). Texts go to
private/s4-packs.json; nothing is judged here.

Run: uv run --no-project --with numpy --with scipy --with torch --with sentence-transformers \
         --with transformers --with accelerate --with llmlingua==0.2.2 python s4_build.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from packs import Packs, pack
from metrics import estimate_tokens  # noqa: E402  (on the path that packs sets up)

HERE = Path(__file__).resolve().parent
OUT = HERE / "private" / "s4-packs.json"
RECOMP = ("fangyuan/nq_extractive_compressor", "46f98c1f57fa919c72d86ced16e3ece3403f9e9e")
LINGUA = ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323")
DEVICE = os.environ.get("S4_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
STAGES = os.environ.get("S4_STAGES", "HRL")


def main() -> None:
    t0 = time.time()
    p = Packs()
    p.check()
    b = p.b
    done = json.loads(OUT.read_text()) if OUT.exists() else {}

    for qi in p.sample:
        key = str(qi)
        if key in done and "H" in done[key]:
            continue
        a_text, a_surv, a_tok = p.a(qi)
        h_text, h_surv, h_tok = pack(p.units_h(qi), b.rels[qi])
        done.setdefault(key, {})["A_tokens"] = a_tok
        done[key]["H"] = {"text": h_text, "survival": h_surv, "tokens": h_tok}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False))
    print("H built", round(time.time() - t0, 1), flush=True)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(RECOMP[0], revision=RECOMP[1], device=DEVICE)
    for qi in p.sample:
        key = str(qi)
        if "R" in done[key]:
            continue
        t, r = p.pre_a["top"][qi], p.pre_a["rows"][qi]
        ok = t >= 0
        cands = []
        for n, w in zip(t[:30][ok[:30]], r[:30][ok[:30]]):
            for s in p.sent.text[int(w)]:
                cands.append((b.pages[int(n)], s))
        qv = enc.encode([b.queries[qi]], convert_to_numpy=True, normalize_embeddings=False)[0]
        sv = enc.encode([s for _, s in cands], convert_to_numpy=True, normalize_embeddings=False, batch_size=64)
        order = np.argsort(-(sv @ qv), kind="stable")
        text, surv, tok = pack([cands[i] for i in order], b.rels[qi], done[key]["A_tokens"])
        done[key]["R"] = {"text": text.replace("\n\n---\n\n", "\n"), "survival": surv, "tokens": tok, "candidates": len(cands)}
    OUT.write_text(json.dumps(done, ensure_ascii=False))
    del enc
    print("R built", round(time.time() - t0, 1), flush=True)
    if "L" not in STAGES:
        return

    from llmlingua import PromptCompressor

    lingua = PromptCompressor(model_name=LINGUA[0], device_map=DEVICE, model_config={"revision": LINGUA[1]})
    for qi in p.sample:
        key = str(qi)
        if "L" in done[key]:
            continue
        contexts = [text for _, text in p.units_b(qi)]
        b_text, _, b_tok = p.b_pack(qi)
        kept = []
        used = 0
        for c in contexts:
            if used + estimate_tokens(c) <= 6000:
                used += estimate_tokens(c)
                kept.append(c)
        lm_tokens = len(lingua.tokenizer.encode("\n\n".join(kept)))
        target = max(1, round(done[key]["A_tokens"] * lm_tokens / max(1, b_tok)))
        res = lingua.compress_prompt(kept, question=b.queries[qi], target_token=target,
                                     condition_in_question="after_condition", reorder_context="sort",
                                     dynamic_context_compression_ratio=0.3, condition_compare=True,
                                     context_budget="+100", rank_method="longllmlingua")
        text = res["compressed_prompt"]
        if text.rstrip().endswith(b.queries[qi].strip()):
            text = text.rstrip()[: -len(b.queries[qi].strip())].rstrip()
        done[key]["L"] = {"text": text, "tokens": estimate_tokens(text), "target_lm_tokens": target,
                          "lm_tokens_before": lm_tokens, "lm_tokens_after": res.get("compressed_tokens")}
        OUT.write_text(json.dumps(done, ensure_ascii=False))
        print("L", qi, done[key]["L"]["tokens"], "vs A", done[key]["A_tokens"], flush=True)
    print("all built", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
