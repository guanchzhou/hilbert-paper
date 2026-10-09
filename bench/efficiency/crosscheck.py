#!/usr/bin/env python3
"""Cross-check of run.py against two independent implementations, zig-select and zig-diffuse.

For the setting each idea chose, the first 20 confirm-half questions are recomputed with the Zig
command-line tools and compared with the Python code of run.py: selections and orders must be
equal, diffused scores must agree to 1e-9. Writes crosscheck.json and copies it, with this script,
to the repository.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

import run

SELECT = Path.home() / "Development/zig-select/zig-out/bin/zig-select"
DIFFUSE = Path.home() / "Development/zig-diffuse/zig-out/bin/zig-diffuse"
QUESTIONS = 20


def call(binary: Path, cmd: str, payload: dict) -> dict:
    proc = subprocess.run([str(binary), cmd], input=json.dumps(payload), capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def main() -> None:
    d = run.Data()
    chosen = {k: json.loads((run.HERE / f"{k.lower()}.json").read_text())["params"] for k in
              ("M1", "M2", "M3", "M4", "M5", "M6", "M7")}
    qs = [int(q) for q in d.confirm[:QUESTIONS]]
    report = {}

    # M1: isotonic map and value-density selection.
    xs, ys = [], []
    for qi in d.tune:
        notes = d.base_order[qi, :run.POOL]
        xs.append(d.S[qi, notes])
        ys.append(d.relmask[qi, notes].astype(float))
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys)
    model = run.fit_isotonic(x, y)
    pool = chosen["M1"]["pool"]
    probe = np.unique(np.concatenate([d.S[qi, d.base_order[qi, :pool]] for qi in qs]).astype(float))
    iso = call(SELECT, "isotonic", {"x": x.tolist(), "y": y.tolist(), "predict": probe.tolist()})
    iso_diff = float(np.abs(np.array(iso["predicted"]) - run.predict_isotonic(model, probe)).max())
    same = 0
    for qi in qs:
        notes = d.base_order[qi, :pool]
        prob = run.predict_isotonic(model, d.S[qi, notes].astype(float))
        cost = [d.unit_of(qi, int(p))[1] for p in notes]
        order = np.argsort(-(prob / np.array(cost)), kind="stable")
        costs = [cost[i] for i in order]
        for b in run.BUDGETS:
            py = sorted(int(order[i]) for i in run.walk(costs, b))
            z = call(SELECT, "knapsack", {"values": prob.tolist(), "costs": cost, "budget": b, "best_single": False})
            same += py == sorted(z["kept"])
    report["M1"] = {"isotonic_max_abs_diff": f"{iso_diff:.2e}", "selections_equal": same, "of": len(qs) * len(run.BUDGETS)}

    # M2, M3, M4: pool selections and orders.
    for k in ("M2", "M3", "M4"):
        same = total = 0
        for qi in qs:
            notes = d.base_order[qi, :run.POOL]
            U = d.Cn[d.win[qi, notes]].astype(float)
            sim = U @ U.T
            rel = d.S[qi, notes].astype(float)
            cost = np.array([d.unit_of(qi, int(p))[1] for p in notes])
            if k == "M2":
                lam, p = chosen[k]["lam"], chosen[k]["p"]
                for b in run.BUDGETS:
                    py = sorted(run.m2_select(rel, sim, cost, b, lam, p))
                    z = call(SELECT, "submodular", {"relevance": rel.tolist(), "similarity": sim.tolist(),
                                                    "costs": cost.tolist(), "budget": b, "lambda": lam, "p": p})
                    same += py == sorted(z["kept"])
                    total += 1
            elif k == "M3":
                lam = chosen[k]["lam"]
                py = run.mmr_order(rel, sim, lam, run.DEPTH)
                z = call(SELECT, "mmr", {"relevance": rel.tolist(), "similarity": sim.tolist(), "lambda": lam})
                same += py == z["order"][:run.DEPTH]
                total += 1
            else:
                alpha = chosen[k]["alpha"]
                q = np.exp(alpha * (rel - rel.max()))
                L = q[:, None] * sim * q[None, :]
                py = run.dpp_order(rel, sim, alpha, run.DEPTH)[:run.DEPTH]
                z = call(SELECT, "dpp", {"kernel": L.tolist(), "max_items": run.DEPTH, "eps": 1e-12})
                same += py == z["order"][:run.DEPTH]
                total += 1
        report[k] = {"equal": same, "of": total}

    # M5, M6, M7: diffused scores.
    W = run.undirected(d)
    L_sym, _ = run.sym_laplacian(W)
    lam_e, V = np.linalg.eigh(L_sym)
    und = [[i, int(j)] for i in range(d.N) for j in np.flatnonzero(W[i]) if j > i]
    directed = [[i, j] for i, j in d.directed]
    from scipy import linalg
    A = np.zeros((d.N, d.N))
    for i, j in d.directed:
        A[i, j] = 1.0
    L_adv = np.diag(A.sum(axis=1)) - A.T

    t, beta = chosen["M5"]["t"], chosen["M5"]["beta"]
    diffs, tops = [], 0
    for qi in qs[:5]:
        s = d.S[qi].astype(float)
        py = (1 - beta) * s + beta * (V @ (np.exp(-t * lam_e) * (V.T @ s)))
        z = np.array(call(DIFFUSE, "heat", {"n": d.N, "edges": und, "scores": s.tolist(), "t": t, "beta": beta})["scores"])
        diffs.append(float(np.abs(py - z).max()))
        tops += list(d.order(py[None])[0, :10]) == list(d.order(z[None])[0, :10])
    report["M5"] = {"max_abs_diff": f"{max(diffs):.2e}", "top10_equal": tops, "of": len(diffs)}

    t, gamma, beta = chosen["M7"]["t"], chosen["M7"]["gamma"], chosen["M7"]["beta"]
    E = linalg.expm(-t * (gamma * L_adv + (1 - gamma) * L_sym))
    diffs, tops = [], 0
    for qi in qs[:5]:
        s = d.S[qi].astype(float)
        py = (1 - beta) * s + beta * (E @ s)
        z = np.array(call(DIFFUSE, "advect", {"n": d.N, "edges": directed, "scores": s.tolist(), "t": t,
                                              "gamma": gamma, "beta": beta})["scores"])
        diffs.append(float(np.abs(py - z).max()))
        tops += list(d.order(py[None])[0, :10]) == list(d.order(z[None])[0, :10])
    report["M7"] = {"max_abs_diff": f"{max(diffs):.2e}", "top10_equal": tops, "of": len(diffs)}

    k, a = chosen["M6"]["k"], chosen["M6"]["a"]
    Nn = np.zeros((d.N, d.Cn.shape[1]))
    for p, (s0, s1) in enumerate(zip(d.starts, d.ends)):
        Nn[p] = d.Cn[s0:s1].mean(axis=0)
    Nn = run.unit(Nn).astype(float)
    cos_nn = Nn @ Nn.T
    np.fill_diagonal(cos_nn, -np.inf)
    nbr = np.argsort(-cos_nn, axis=1, kind="stable")[:, :k]
    Wk = np.zeros((d.N, d.N))
    for i in range(d.N):
        for j in nbr[i]:
            if cos_nn[i, j] > 0:
                Wk[i, j] = max(Wk[i, j], cos_nn[i, j])
                Wk[j, i] = max(Wk[j, i], cos_nn[i, j])
    _, Sk = run.sym_laplacian(Wk)
    knn = call(DIFFUSE, "knn", {"vectors": Nn.tolist(), "k": k})["edges"]
    py_edges = sorted((i, int(j)) for i in range(d.N) for j in nbr[i] if cos_nn[i, j] > 0)
    z_edges = sorted((int(e[0]), int(e[1])) for e in knn)
    diffs = []
    qn = d.Qn.astype(float) @ Nn.T
    for qi in qs[:5]:
        y = np.zeros(d.N)
        top = d.base_order[qi, :run.POOL]
        y[top] = np.maximum(0, qn[qi, top])
        py = np.linalg.solve(np.eye(d.N) - a * Sk, y)
        z = np.array(call(DIFFUSE, "manifold", {"n": d.N, "edges": knn, "a": a, "y": y.tolist()})["f"])
        diffs.append(float(np.abs(py - z).max()))
    report["M6"] = {"knn_edges_equal": py_edges == z_edges, "edges": len(z_edges), "max_abs_diff_F": f"{max(diffs):.2e}",
                    "of": len(diffs)}

    out = {"questions": qs, "settings": chosen, "tools": {"zig-select": str(SELECT), "zig-diffuse": str(DIFFUSE)},
           "results": report}
    path = run.HERE / "crosscheck.json"
    path.write_text(json.dumps(run.rounded(out), indent=1) + "\n")
    shutil.copy(path, run.REPO / path.name)
    shutil.copy(Path(__file__), run.REPO / "crosscheck.py")
    print(json.dumps(run.rounded(report), indent=1))


if __name__ == "__main__":
    main()
