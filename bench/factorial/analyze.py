#!/usr/bin/env python3
"""Level 1 analysis (pre-registration 1.6): main effects, two-factor interactions, Shapley points,
leaderboards with tune-half selection and confirm-half tests, Pareto fronts."""

import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

from pipeline import CACHE, F1, F7, F8, HERE, LEVELS, NAMES, Bench, rankings
from common import SEED, boot_ci, holm, mcnemar, rounded, wilcoxon

REPO = Path.home() / "Development/hilbert-paper/bench/factorial"
REF = ("dense", "best", "off", "off", "off", "off", "none", "chunk")
RANK_METRICS = ("R@10", "MRR", "nDCG@10", "hit@10")
EFFECT_METRICS = ("R@10", "survival", "tokens", "latency", "MRR", "nDCG@10")
TESTED = ("R@10", "survival", "tokens")


def bh(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        running = min(running, p * m / (i + 1))
        out[k] = min(1.0, running)
    return out


def wilcoxon2(d: np.ndarray) -> float:
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        return 1.0
    return float(stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue)


class Cells:
    def __init__(self, test: bool = False, tier: str = "B") -> None:
        self.tier = tier
        self.b = Bench()
        z = dict(np.load(CACHE / "level1-perq.npz"))
        if test:
            keys = list(z["keys"])
            for i, k in enumerate(keys):
                if "|on|" in k and k.split("|")[2] == "on":
                    parts = k.split("|")
                    parts[2] = "off"
                    j = keys.index("|".join(parts))
                    for m in ("R_10", "MRR", "nDCG_10", "hit_10"):
                        z[m][i] = np.where(np.isnan(z[m][i]), z[m][j], z[m][i])
                    for m in ("surv", "tok"):
                        z[m][i] = np.where(np.isnan(z[m][i]), z[m][j], z[m][i])
            for m in ("surv", "tok"):
                z[m] = np.where(np.isnan(z[m]), 0.0, z[m])
        lat = json.loads((HERE / "level1-latency.json").read_text())
        self.lat = lat
        cfgs = rankings()
        assert list(z["keys"]) == [c.key for c in cfgs]
        surv, tok = z["surv"], z["tok"]
        keep = np.array([tier == "B" or c.f3 == "off" for c in cfgs])
        z["R_10"] = np.where(keep[:, None], z["R_10"], 0.0)
        surv = np.where(keep[:, None, None], surv, 0.0)
        tok = np.where(keep[:, None, None], tok, 0.0)
        finite = np.isfinite(surv).all(axis=(0, 1)) & np.isfinite(tok).all(axis=(0, 1)) & np.isfinite(z["R_10"]).all(0)
        self.Q = np.nonzero(finite)[0]
        self.n_all = self.b.n
        q = self.Q
        tune = np.isin(q, self.b.tune)
        self.halves = {"all": np.ones(len(q), bool), "tune": tune, "confirm": ~tune}
        self.data, self.index = [], {}
        embed = lat.get("embed_query_ms", {}).get("median", float("nan"))
        for ci, c in enumerate(cfgs):
            if not keep[ci]:
                continue
            dense_part = c.f1 in ("dense", "hybrid")
            base = 0.0
            if dense_part or c.f7 != "none":
                base += embed
            if c.f1 in ("lexical", "hybrid"):
                base += lat["lexical_sql_ms"]
            if dense_part and c.f5 == "on":
                base += lat["centre_ms"]
            if dense_part and c.f6 == "on":
                base += lat["rocchio_extra_ms"]
            if c.f1 == "hybrid":
                base += lat["rrf_ms"]
            if c.f4 == "on":
                base += lat["ppr_ms"]
            if c.f7 == "hk1":
                base += lat["hk1_ms"]
            if c.f7 == "partitions":
                base += lat["partitions_route_ms"] + 128 * lat["dense_ms_per_vector"]
            rank_lat = (base + lat["dense_ms_per_vector"] * z["cand_dense"][ci][q]
                        + lat["rerank_ms_per_pair"] * z["pairs"][ci][q])
            for ui, u in enumerate(F8):
                key = (c.f1, c.f2, c.f3, c.f4, c.f5, c.f6, c.f7, u)
                m = {"R@10": z["R_10"][ci][q], "MRR": z["MRR"][ci][q], "nDCG@10": z["nDCG_10"][ci][q],
                     "hit@10": z["hit_10"][ci][q], "survival": surv[ci, ui][q], "tokens": tok[ci, ui][q],
                     "candidates": z["cand"][ci][q], "rerank_pairs": z["pairs"][ci][q],
                     "rank_latency": rank_lat,
                     "latency": rank_lat + (lat["prune_ms_precomputed_sentences"] if u == "pruned" else 0.0)}
                self.index[key] = len(self.data)
                self.data.append((key, m))
        # secondary: full-set metrics of cells finite on all 817 questions
        self.full = {}
        for ci, c in enumerate(cfgs):
            if not keep[ci]:
                continue
            for ui, u in enumerate(F8):
                if np.isfinite(surv[ci, ui]).all() and np.isfinite(z["R_10"][ci]).all():
                    key = (c.f1, c.f2, c.f3, c.f4, c.f5, c.f6, c.f7, u)
                    self.full[key] = {"R@10": float(z["R_10"][ci].mean()), "survival": float(surv[ci, ui].mean()),
                                      "tokens": float(tok[ci, ui].mean())}

    def m(self, key, metric):
        return self.data[self.index[key]][1][metric]

    def summary(self, key) -> dict:
        m = self.data[self.index[key]][1]
        out = {}
        for h, sel in self.halves.items():
            out[h] = {k: float(v[sel].mean()) for k, v in m.items()}
        out["latency_median_ms"] = float(np.median(m["latency"]))
        out["candidates_median"] = float(np.median(m["candidates"]))
        return out


def dict_key(key) -> dict:
    return dict(zip(NAMES, key))


def label(key) -> str:
    parts = []
    names = {"F1": None, "F2": "mean-of-chunks", "F3": "rerank", "F4": "PPR", "F5": "centering", "F6": "Rocchio"}
    parts.append(key[0])
    for i, n in enumerate(("F2", "F3", "F4", "F5", "F6"), start=1):
        if key[i] not in ("off", "best"):
            parts.append(names[n])
    if key[6] != "none":
        parts.append("filter=" + key[6])
    parts.append("unit=" + key[7])
    return " + ".join(parts)


def contrast(C: Cells, metric: str, fi: int, lvl: str, gi: int | None = None, lvl2: str | None = None):
    diffs = []
    for key, _ in C.data:
        if key[fi] != LEVELS[fi][0]:
            continue
        if gi is not None and key[gi] != LEVELS[gi][0]:
            continue
        a = list(key)
        a[fi] = lvl
        a = tuple(a)
        if a not in C.index:
            continue
        if gi is None:
            diffs.append(C.m(a, metric) - C.m(key, metric))
            continue
        bb = list(key)
        bb[gi] = lvl2
        bb = tuple(bb)
        ab = list(a)
        ab[gi] = lvl2
        ab = tuple(ab)
        if bb not in C.index or ab not in C.index:
            continue
        diffs.append(C.m(ab, metric) - C.m(a, metric) - C.m(bb, metric) + C.m(key, metric))
    if not diffs:
        return None, 0
    return np.mean(diffs, axis=0), len(diffs)


def effects(C: Cells) -> dict:
    out = {"main": {}, "interactions": {}}
    for metric in EFFECT_METRICS:
        main, inter = {}, {}
        for fi, levels in enumerate(LEVELS):
            if fi == 7 and metric in RANK_METRICS:
                continue
            for lvl in levels[1:]:
                d, n = contrast(C, metric, fi, lvl)
                if d is None:
                    continue
                lo, hi = boot_ci(d)
                main[f"{NAMES[fi]}={lvl}"] = {"effect": float(d.mean()), "ci95": [lo, hi], "cells_paired": n,
                                              "p": wilcoxon2(d) if metric in TESTED else None,
                                              "improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum())}
        for fi, gi in itertools.combinations(range(8), 2):
            if 7 in (fi, gi) and metric in RANK_METRICS:
                continue
            for la in LEVELS[fi][1:]:
                for lb in LEVELS[gi][1:]:
                    d, n = contrast(C, metric, fi, la, gi, lb)
                    if d is None:
                        continue
                    lo, hi = boot_ci(d)
                    inter[f"{NAMES[fi]}={la} x {NAMES[gi]}={lb}"] = {
                        "interaction": float(d.mean()), "ci95": [lo, hi], "cells_paired": n,
                        "p": wilcoxon2(d) if metric in TESTED else None}
        if metric in TESTED:
            hm = holm({k: v["p"] for k, v in main.items()})
            for k in main:
                main[k]["p_holm"] = hm[k]
            bq = bh({k: v["p"] for k, v in inter.items()})
            hq = holm({k: v["p"] for k, v in inter.items()})
            for k in inter:
                inter[k]["q_bh"] = bq[k]
                inter[k]["p_holm"] = hq[k]
        out["main"][metric] = main
        out["interactions"][metric] = inter
        print(metric, len(main), "main,", len(inter), "interactions", flush=True)
    return out


def shapley(C: Cells) -> dict:
    players = [i for i in range(8) if not (C.tier == "A" and i == 2)]
    n = len(players)
    w = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    metrics = ("R@10", "survival", "tokens", "latency", "MRR", "nDCG@10")
    per_level = {}  # (factor, level) -> list of per-question arrays per metric (shapley, banzhaf)
    for p1, p7, p8 in itertools.product(F1[1:], F7[1:], F8[1:]):
        prof = [p1, "mean", "on", "on", "on", "on", p7, p8]

        def cell(S):
            key = list(REF)
            for i in S:
                key[i] = prof[i]
            if key[0] == "lexical":
                key[1], key[4], key[5] = "best", "off", "off"
            return tuple(key)
        vals = {}
        for r in range(n + 1):
            for S in itertools.combinations(players, r):
                vals[frozenset(S)] = cell(S)
        for i in players:
            others = [j for j in players if j != i]
            acc = {m: 0.0 for m in metrics}
            ban = {m: 0.0 for m in metrics}
            cnt = 0
            for r in range(n):
                for S in itertools.combinations(others, r):
                    S0 = frozenset(S)
                    S1 = S0 | {i}
                    for m in metrics:
                        dm = C.m(vals[S1], m) - C.m(vals[S0], m)
                        acc[m] = acc[m] + w[r] * dm
                        ban[m] = ban[m] + dm
                    cnt += 1
            lvl = prof[i]
            per_level.setdefault((NAMES[i], lvl), []).append(
                ({m: acc[m] for m in metrics}, {m: ban[m] / cnt for m in metrics}))
    out = {}
    for (f, lvl), runs in per_level.items():
        rec = {"profiles": len(runs)}
        for m in metrics:
            sh = np.mean([r[0][m] for r in runs], axis=0)
            bz = np.mean([r[1][m] for r in runs], axis=0)
            rec[m] = {"shapley": float(sh.mean()), "ci95": list(boot_ci(sh)), "banzhaf": float(bz.mean())}
        out[f"{f}={lvl}"] = rec
    return out


def leaderboards(C: Cells) -> dict:
    tune, conf = C.halves["tune"], C.halves["confirm"]
    ref = REF

    def mean(key, m, sel):
        return float(C.m(key, m)[sel].mean())

    def active(key):
        return sum(1 for k, r in zip(key, REF) if k != r)
    ranking_keys = [k for k, _ in C.data if k[7] == "chunk"]
    all_keys = [k for k, _ in C.data]

    def row(key, axis_metric=None):
        s = C.summary(key)
        cfg = dict_key(key)
        if axis_metric == "rank":
            cfg.pop("F8")
        return {"config": cfg,
                "label": label(key) if axis_metric != "rank" else label(key).rsplit(" + unit=", 1)[0],
                "tune": {m: s["tune"][m] for m in ("R@10", "MRR", "nDCG@10", "survival", "tokens", "latency", "rank_latency")},
                "confirm": {m: s["confirm"][m] for m in ("R@10", "MRR", "nDCG@10", "survival", "tokens", "latency", "rank_latency")},
                "all": {m: s["all"][m] for m in ("R@10", "MRR", "nDCG@10", "hit@10", "survival", "tokens", "latency",
                                                 "rank_latency", "candidates", "rerank_pairs")},
                "latency_median_ms": s["latency_median_ms"]}
    boards = {}
    rec = sorted(ranking_keys, key=lambda k: (-mean(k, "R@10", tune), -mean(k, "MRR", tune), active(k)))
    boards["best_recall"] = rec
    sv = sorted(all_keys, key=lambda k: (-mean(k, "survival", tune), mean(k, "tokens", tune), active(k)))
    boards["best_survival"] = sv
    best_s = mean(sv[0], "survival", tune)
    tk = sorted([k for k in all_keys if mean(k, "survival", tune) >= best_s - 0.02],
                key=lambda k: (mean(k, "tokens", tune), -mean(k, "survival", tune), active(k)))
    boards["fewest_tokens"] = tk
    best_r = mean(rec[0], "R@10", tune)
    fs = sorted([k for k in ranking_keys if mean(k, "R@10", tune) >= best_r - 0.02],
                key=lambda k: (mean(k, "rank_latency", tune), -mean(k, "R@10", tune), active(k)))
    boards["fastest"] = fs
    out = {"selection": "tune half; values reported on tune, confirm and all", "n_questions": int(len(C.Q)),
           "thresholds": {"best_tune_survival": best_s, "best_tune_R@10": best_r}}
    for name, keys in boards.items():
        kind = "rank" if name in ("best_recall", "fastest") else "cell"
        out[name] = {"eligible": len(keys), "top20": [row(k, kind) for k in keys[:20]]}
    # confirmation on the confirm half vs the reference cell
    tests = {}
    r0 = ref
    k = boards["best_recall"][0]
    tests["recall axis: R@10 vs reference"] = {"config": label(k), **wilcox_two(C.m(k, "R@10")[conf], C.m(r0, "R@10")[conf])}
    k = boards["best_survival"][0]
    t = mcnemar(C.m(k, "survival")[conf], C.m(r0, "survival")[conf], alternative="two-sided")
    t["test"] = "exact McNemar (binomial on discordant pairs), two-sided"
    tests["survival axis: survival vs reference"] = {"config": label(k), **t}
    k = boards["fewest_tokens"][0]
    t = wilcoxon(C.m(k, "survival")[conf], C.m(r0, "survival")[conf], margin=0.02)
    t["tokens_new"] = float(C.m(k, "tokens")[conf].mean())
    t["tokens_reference"] = float(C.m(r0, "tokens")[conf].mean())
    tests["tokens axis: survival non-inferiority (margin 0.02) vs reference"] = {"config": label(k), **t}
    k = boards["fastest"][0]
    t = wilcoxon(C.m(k, "R@10")[conf], C.m(r0, "R@10")[conf], margin=0.02)
    t["latency_new_ms"] = float(C.m(k, "rank_latency")[conf].mean())
    t["latency_reference_ms"] = float(C.m(r0, "rank_latency")[conf].mean())
    tests["latency axis: R@10 non-inferiority (margin 0.02) vs reference"] = {"config": label(k), **t}
    hm = holm({k: v["p"] for k, v in tests.items()})
    for kk in tests:
        tests[kk]["p_holm"] = hm[kk]
    out["confirmation"] = tests
    out["reference"] = row(r0, "cell")
    return out


def wilcox_two(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    lo, hi = boot_ci(d)
    return {"mean_new": float(a.mean()), "mean_reference": float(b.mean()), "diff": float(d.mean()),
            "ci95": [lo, hi], "p": wilcoxon2(d), "test": "two-sided Wilcoxon signed-rank on non-zero differences",
            "n": int(len(d))}


def pareto(C: Cells, objs: tuple) -> list:
    pts = []
    for key, m in C.data:
        pts.append((key, [float(m[o].mean()) * (-1 if o in ("R@10", "survival") else 1) for o in objs]))
    front = []
    for key, v in pts:
        dominated = any(all(x <= y for x, y in zip(u, v)) and any(x < y for x, y in zip(u, v)) for _, u in pts)
        if not dominated:
            front.append({"label": label(key), "config": dict_key(key),
                          **{o: float(C.m(key, o).mean()) for o in objs}})
    return sorted(front, key=lambda r: r[objs[1]])


SUFFIX = ""


def main() -> None:
    import sys
    global HERE, REPO, SUFFIX
    test = "--test" in sys.argv
    tier = "A" if "--tier-a" in sys.argv else "B"
    SUFFIX = "-tierA" if tier == "A" else ""
    if test:
        HERE = REPO = Path("/tmp/factorial-test")
        HERE.mkdir(exist_ok=True)
        lat = json.loads((Path(__file__).resolve().parent / "level1-latency.json").read_text())
        lat.setdefault("embed_query_ms", {"median": 50.0})
        (HERE / "level1-latency.json").write_text(json.dumps(lat))
    C = Cells(test, tier)
    print("analysis set", len(C.Q), "questions", flush=True)
    cells = {"tier": C.tier, "n_questions_analysis_set": int(len(C.Q)), "analysis_set_idx": C.Q.tolist(),
             "n_cells": len(C.data), "cells": []}
    for key, _ in C.data:
        cells["cells"].append({"config": dict_key(key), "label": label(key), **C.summary(key),
                               "all817_secondary": C.full.get(key)})
    cells["latency_components_ms"] = C.lat
    write("level1-cells", cells)
    eff = effects(C)
    eff.update({"n_questions": int(len(C.Q)), "contrast": "per-question mean over matched cells of (level - reference)",
                "test": "two-sided Wilcoxon signed-rank on non-zero per-question contrasts; latency not tested (composed)",
                "correction": "Holm within metric over main effects; Benjamini-Hochberg within metric over interactions"})
    write("level1-effects", eff)
    sh = shapley(C)
    write("level1-shapley", {"tier": C.tier, "n_questions": int(len(C.Q)), "method": "exact Shapley over binary players (8; 7 in tier A without F3) per level "
                             "profile (16 profiles); multi-level factors averaged over profiles with that level; "
                             "inert factors evaluated as off; Banzhaf alongside", "points": sh})
    write("level1-leaderboard", leaderboards(C))
    write("level1-pareto", {"n_questions": int(len(C.Q)),
                            "recall_tokens_latency": pareto(C, ("R@10", "tokens", "latency")),
                            "survival_tokens_latency": pareto(C, ("survival", "tokens", "latency"))})
    print("done", flush=True)


def write(name: str, obj: dict) -> None:
    path = HERE / f"{name}{SUFFIX}.json"
    path.write_text(json.dumps(rounded(obj), indent=1, ensure_ascii=False) + "\n")
    REPO.mkdir(parents=True, exist_ok=True)
    if REPO != HERE:
        shutil.copy(path, REPO / path.name)


if __name__ == "__main__":
    main()
