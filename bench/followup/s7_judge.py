#!/usr/bin/env python3
"""S7: one decision model judges the items of private/s7-items.json (as pre-registered in
preregistration.md, addendum 2), with int8 weights (deviation D9). Usage: s7_judge.py
clef-flash|jev-9b [limit]. The probability of yes and the seconds of each item are cached in
private/s7-<model>.json (resumable).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODELS = {"clef-flash": ("Cloudflare/clef-flash", "fde727a287004204b7518dcc983fe64379776712"),
          "jev-9b": ("autotrust/JEV-9B", "b63f651ce8ed64481d3f5e73ecdb05f740042f01")}
INSTRUCTIONS = {
    "sufficiency": "Does the context contain the information needed to answer the question?",
    "relevance": ("Is this note relevant to the question, that is, does it contain information that answers "
                  "the question or a substantial part of it?"),
}
JEV_MAX_TOKENS = 1024
DEVICE = "mps"


def int8():
    from transformers import QuantoConfig
    return QuantoConfig(weights="int8", modules_to_not_convert=["lm_head", "visual"])


def state(item: dict) -> dict:
    return {"question": item["question"], ("context" if item["task"] == "sufficiency" else "note"): item["text"]}


def clef(path: Path):
    sys.path.insert(0, str(path))
    from joint_schema_model import load_release_model, systemone
    model, processor = load_release_model(path, device="cpu", quantization_config=int8())
    model.to(DEVICE)

    def p_yes(item: dict) -> dict:
        out = systemone(model, processor, {"model": "clef-flash", "state": state(item),
                                           "questions": {"answer": {"type": "noul", "instructions": INSTRUCTIONS[item["task"]]}}})
        return {"p_yes": float(out["answers"]["answer"]["noul"]), "input_tokens": int(out["usage"]["input_tokens"])}
    return p_yes


def jev(path: Path):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cpu"},
                                                quantization_config=int8())
    PeftModel.from_pretrained(base, path / "adapter").eval()
    model = base.to(DEVICE)
    head = load_file(path / "head.safetensors")
    cfg = json.loads((path / "judge_config.json").read_text())
    temp = json.loads((path / "calibration.json").read_text())["per_kind"]["noul"]
    w, b = head["proj.weight"].to(DEVICE).float(), head["proj.bias"].to(DEVICE).float()
    start, _ = cfg["slots"]["ranges"]["noul"]

    def text(st: str, question: str) -> str:
        return f"[kind] noul\n[state] {st}\n[question] {question}\n[options]\nfalse\ntrue\n[decision]:"

    @torch.inference_mode()
    def p_yes(item: dict) -> dict:
        st, q = json.dumps(state(item), ensure_ascii=False), INSTRUCTIONS[item["task"]]
        budget = JEV_MAX_TOKENS - len(tok(text("", q), add_special_tokens=False).input_ids)
        sid = tok(st, add_special_tokens=False).input_ids
        cut = len(sid) > budget
        if cut:
            head_n = int(budget * 0.6)
            st = tok.decode(sid[:head_n]) + tok.decode(sid[len(sid) - (budget - head_n):])
        ids = tok(text(st, q), return_tensors="pt", add_special_tokens=False).to(DEVICE)
        h = model.model(**ids).last_hidden_state[0, -1].float()
        z = (w @ h + b) / temp
        return {"p_yes": float(torch.softmax(z[start:start + 2], 0)[1]), "input_tokens": int(ids.input_ids.shape[1]),
                "truncated": cut}
    return p_yes


def main() -> None:
    name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    from huggingface_hub import snapshot_download
    repo, rev = MODELS[name]
    ignore = ["adapter_vllm/*", "reports/*", "vl/*", "code/*"] if name == "jev-9b" else None
    path = Path(snapshot_download(repo, revision=rev, local_files_only=True, ignore_patterns=ignore))
    items = json.loads((HERE / "private" / "s7-items.json").read_text())["items"][:limit]
    cache = HERE / "private" / f"s7-{name}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    t0 = time.perf_counter()
    p_yes = (clef if name == "clef-flash" else jev)(path)
    print("loaded", round(time.perf_counter() - t0, 1), "s", flush=True)
    for n, item in enumerate(items, 1):
        key = f"{item['task']}|{item['id']}"
        if key in done:
            continue
        t = time.perf_counter()
        r = p_yes(item)
        torch.mps.empty_cache()
        done[key] = {**r, "seconds": time.perf_counter() - t}
        cache.write_text(json.dumps(done))
        print(name, n, "of", len(items), item["task"], round(r["p_yes"], 3), "27b", item["judge27b"],
              round(done[key]["seconds"], 2), "s", r["input_tokens"], "tok", flush=True)


if __name__ == "__main__":
    main()
