#!/usr/bin/env python3
"""Local labelling page for S1b (100 packs), S6 (30 gold checks) and S2 (30 relevance pairs).

Run once with --build to write private/label-items.json (texts stay private), then serve:
    python label_server.py --build      # S1b and S6 items; --add-s2 appends the S2 pairs later
    python label_server.py              # http://127.0.0.1:8765
The page shows one item at a time and never the condition or the judge's answer. Each label is
written to private/human-labels.json at once, so labelling can stop and resume.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "private" / "label-items.json"
LABELS = HERE / "private" / "human-labels.json"
SEED = 20261008

PROMPTS = {
    "pack": "Does this context contain the information needed to answer the question?",
    "gold": "Is this list correct and complete: every listed note is about both topics, and no note about both is missing?",
    "relevance": "Is this note relevant to the question, that is, does it contain information that answers it or a substantial part of it?",
}


def build() -> None:
    from packs import Packs

    p = Packs()
    p.check()
    rng = np.random.default_rng(SEED)
    qs = sorted(int(x) for x in rng.choice(np.array(p.sample), size=50, replace=False))
    packs = []
    for qi in qs:
        for cond, fn in (("A", p.a), ("B", p.b_pack)):
            packs.append({"kind": "pack", "qi": qi, "cond": cond, "question": p.b.queries[qi], "text": fn(qi)[0]})
    order = rng.permutation(len(packs))
    items = [dict(packs[i], id=f"pack-{n}") for n, i in enumerate(order)]

    spec = json.loads((HERE / "compositional-questions.json").read_text())
    b = p.b
    for q in spec["questions"][:30]:
        lines = []
        for s in q["gold"]:
            i = b.page_index[s]
            excerpt = b.texts[b.starts[i]][:400].replace("\n", " ")
            lines.append(f"{s}\n    {excerpt}")
        items.append({"kind": "gold", "id": f"gold-{q['id']}", "s6": q["id"], "question": q["question"],
                      "text": f"Topic A: {q['title_a']} ({q['a']})\nTopic B: {q['title_b']} ({q['b']})\n\nGold list ({len(q['gold'])} notes):\n\n" + "\n\n".join(lines)})
    ITEMS.parent.mkdir(exist_ok=True)
    ITEMS.write_text(json.dumps({"seed": SEED, "pack_questions": qs, "items": items}, ensure_ascii=False))
    print(len(items), "items written; pack questions", len(qs))


def add_s2() -> None:
    data = json.loads(ITEMS.read_text())
    s2 = json.loads((HERE / "private" / "s2-pairs-for-labels.json").read_text())
    have = {it["id"] for it in data["items"]}
    for it in s2:
        if it["id"] not in have:
            data["items"].append(it)
    ITEMS.write_text(json.dumps(data, ensure_ascii=False))
    print(len(data["items"]), "items after adding S2 pairs")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Labels</title>
<style>body{font:15px/1.45 -apple-system,sans-serif;max-width:900px;margin:24px auto;padding:0 16px}
pre{white-space:pre-wrap;background:#f6f7f9;padding:12px;border-radius:6px;max-height:60vh;overflow:auto;font:13px/1.4 ui-monospace,monospace}
button{font-size:16px;padding:8px 22px;margin-right:10px}.q{font-weight:600;font-size:17px}.p{color:#555}</style></head>
<body><div id="prog"></div><p class="p" id="prompt"></p><p class="q" id="question"></p><pre id="text"></pre>
<button onclick="send('yes')">Yes (y)</button><button onclick="send('no')">No (n)</button>
<script>
let cur=null;
async function next(){const r=await fetch('/api/next');const j=await r.json();
 document.getElementById('prog').textContent=j.done+' of '+j.total+' labelled';
 if(!j.item){document.getElementById('question').textContent='All done. Thank you.';document.getElementById('text').textContent='';document.getElementById('prompt').textContent='';cur=null;return}
 cur=j.item;document.getElementById('prompt').textContent=j.prompt;document.getElementById('question').textContent=cur.question;document.getElementById('text').textContent=cur.text;window.scrollTo(0,0)}
async function send(v){if(!cur)return;await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cur.id,label:v})});next()}
document.addEventListener('keydown',e=>{if(e.key==='y')send('yes');if(e.key==='n')send('no')});next();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/next":
            items = json.loads(ITEMS.read_text())["items"]
            labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
            todo = [it for it in items if it["id"] not in labels]
            item = {k: todo[0][k] for k in ("id", "question", "text")} if todo else None
            self._json({"done": len(labels), "total": len(items), "item": item,
                        "prompt": PROMPTS[todo[0]["kind"]] if todo else ""})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/label":
            self.send_error(404)
            return
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if msg.get("label") not in ("yes", "no"):
            self.send_error(400)
            return
        labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        labels[msg["id"]] = msg["label"]
        LABELS.write_text(json.dumps(labels))
        self._json({"ok": True})

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--add-s2", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.add_s2:
        add_s2()
    else:
        print("http://127.0.0.1:8765")
        HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()


if __name__ == "__main__":
    main()
