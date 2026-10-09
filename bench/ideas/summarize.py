#!/usr/bin/env python3
"""Family-wide Holm correction over the 16 pre-registered hypotheses, acceptance flags, summary.json."""

import json
import shutil
from pathlib import Path

from common import IDEAS, REPO, holm, rounded

FAMILY = ["H1a", "H1b", "H2", "H3", "H4a", "H4b", "H4c", "H4d", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12"]
FILES = {1: "1-rerank", 2: "2-prune", 3: "3-instruction", 4: "4-centering", 5: "5-rocchio", 6: "6-partitions",
         7: "7-forest", 8: "8-synopsis", 9: "9-ppr", 10: "10-hyde", 11: "11-adaptive-k", 12: "12-muvera"}
THRESH = {"H1a": 0.03, "H1b": 0.03, "H2": 0.03, "H3": 0.02, "H4a": 0.02, "H4b": 0.02, "H4c": 0.02, "H4d": 0.02,
          "H5": 0.02, "H6": -0.03, "H7": -0.03, "H8": 0.03, "H9": 0.02, "H10": 0.03, "H11": -0.02, "H12": 0.03}


def idea_of(h: str) -> int:
    return int("".join(ch for ch in h[1:] if ch.isdigit()))


def main() -> None:
    found, docs = {}, {}
    for n, name in FILES.items():
        path = IDEAS / f"{name}.json"
        if path.exists():
            docs[n] = json.loads(path.read_text())
            for key, t in docs[n].get("primary", {}).items():
                hid = key.split()[0]
                if hid in FAMILY:
                    found[hid] = (n, key, t)
    ps = {h: (found[h][2].get("p", 1.0) if h in found and found[h][2].get("diff") is not None else 1.0)
          for h in FAMILY}
    adj = holm(ps)
    rows = []
    for h in FAMILY:
        row = {"id": h, "idea": idea_of(h), "p": ps[h], "p_holm_family": adj[h], "threshold": THRESH[h]}
        if h not in found or found[h][2].get("diff") is None:
            row.update({"status": "not run", "accepted": False})
            rows.append(row)
            continue
        n, key, t = found[h]
        row.update({"hypothesis": key, "diff": t["diff"], "ci95": t["ci95"], "n": t.get("n")})
        ok = t["diff"] >= THRESH[h] and adj[h] < 0.05
        why = []
        if t["diff"] < THRESH[h]:
            why.append(f"effect {t['diff']:+.4f} below threshold {THRESH[h]:+.2f}")
        if adj[h] >= 0.05:
            why.append(f"family Holm p {adj[h]:.3g} >= 0.05")
        if h in ("H5", "H9") and not t.get("guard_pass", False):
            ok = False
            why.append(f"MRR guard failed ({t.get('MRR_diff'):+.4f})")
        if h in ("H6", "H7") and not t.get("candidate_rule_pass", False):
            ok = False
            why.append("candidate rule failed")
        if h == "H11" and t.get("token_reduction", 0) < 0.30:
            ok = False
            why.append(f"token reduction {t.get('token_reduction'):.2f} < 0.30")
        if h == "H2":
            g = docs[n].get("judge_guard", {})
            if not g.get("pass", False):
                ok = False
                why.append("judge guard not passed" if g.get("run") else "judge guard not run")
        row.update({"status": "accepted" if ok else "not accepted", "accepted": ok, "reasons": why})
        rows.append(row)
        t["p_holm_family"] = adj[h]
        t["accepted"] = ok
    for n, d in docs.items():
        (IDEAS / f"{FILES[n]}.json").write_text(json.dumps(rounded(d), indent=1, ensure_ascii=False) + "\n")
        shutil.copy(IDEAS / f"{FILES[n]}.json", REPO / f"{FILES[n]}.json")
    out = {"family": FAMILY, "m": len(FAMILY), "correction": "Holm step-down over all 16; not-run hypotheses enter with p = 1",
           "hypotheses": rows,
           "ideas": {str(n): {"file": FILES[n], "accepted": any(r["accepted"] for r in rows if r["idea"] == n),
                              "hypotheses": [r["id"] for r in rows if r["idea"] == n]} for n in FILES}}
    (IDEAS / "summary.json").write_text(json.dumps(rounded(out), indent=1) + "\n")
    shutil.copy(IDEAS / "summary.json", REPO / "summary.json")
    shutil.copy(Path(__file__), REPO / "summarize.py")
    for r in rows:
        print(r["id"], r["status"], r.get("diff"), r["p"], r["p_holm_family"], r.get("reasons"))


if __name__ == "__main__":
    main()
