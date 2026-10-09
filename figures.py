#!/usr/bin/env python3
"""Build every figure of the paper from bench/*.json into figures/*.svg."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "bench"
OUT = ROOT / "figures"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "svg.fonttype": "path",
    "svg.hashsalt": "hilbert-paper",
    "figure.dpi": 100,
})

INK = "#1a1a1a"
GREY = "#8c8c8c"
LIGHT = "#d9d9d9"
ACCENT = "#1f4e79"
WARN = "#a23b2a"
W = 5.5

LABELS = {
    "H1": "Dense chunk vs lexical note (AND)",
    "H2": "Dense chunk vs dense note mean",
    "H3": "Chunk pack vs page pack (survival)",
    "H4": "hk1 level 1, 16 ranges vs dense chunk",
    "E1": "Lexical note, OR vs AND",
    "E2": "RRF hybrid (note lexical) vs dense chunk",
    "E3": "Chunk pack vs section pack (survival)",
    "E4": "Lexical chunk OR vs dense chunk",
    "E5": "RRF hybrid (chunk lexical) vs dense chunk",
}


def load(name: str):
    return json.loads((BENCH / name).read_text())


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.3f}", dy=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=0.7, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, linespacing=1.15)

    def arrow(x1, y1, x2, y2, style="-|>", color=INK, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                                     linewidth=0.7, color=color, linestyle=ls))

    box(1, 44, 14, 10, "Note\n1,225 pages")
    box(21, 44, 15, 10, "Chunks\n5,155")
    box(42, 44, 18, 10, "Qwen3\nembedding,\n1,024 dims")
    box(66, 44, 15, 10, "pgvector\nHNSW")
    box(86, 44, 13, 10, "Cosine\ntop chunks")
    arrow(16, 49, 20.5, 49)
    arrow(36.5, 49, 41.5, 49)
    arrow(60.5, 49, 65.5, 49)
    arrow(81.5, 49, 85.5, 49)

    box(66, 26, 15, 10, "Best chunk\nwins the page")
    box(86, 26, 13, 10, "Pack unit\n6,000 tokens", fill="#eef3f8", edge=ACCENT)
    arrow(92.5, 43.5, 74, 36.5)
    arrow(81.5, 31, 85.5, 31)

    box(42, 8, 18, 12, "zig-hilbert:\nprojection,\nquantizer,\n8-d curve", fill="#f7f7f7")
    box(66, 8, 15, 12, "hk1 key\nchunk_hilbert\n5,079 rows", fill="#f7f7f7")
    box(86, 8, 13, 12, "Prefix range\nthen cosine\nrescore", fill="#f7f7f7")
    arrow(51, 43.5, 51, 20.5, color=GREY, ls="--")
    arrow(60.5, 14, 65.5, 14, color=GREY)
    arrow(81.5, 14, 85.5, 14, color=GREY)
    arrow(92.5, 20.5, 92.5, 25.5, color=GREY, ls="--")

    box(1, 26, 14, 10, "Keyword\ntsvector")
    box(21, 26, 15, 10, "ts_rank_cd\npage or chunk")
    arrow(8, 43.5, 8, 36.5)
    arrow(16, 31, 20.5, 31)
    ax.text(51, 3, "dashed: the filter under test, not used in production", ha="center", fontsize=6.8, color=GREY)
    save(fig, "fig01-pipeline")


CELLS = [("keyword-page", "Keyword,\nwhole note"), ("keyword-chunk", "Keyword,\nchunks"),
         ("vector-page", "Vector,\npage mean"), ("vector-chunk", "Vector,\nchunks")]


def fig_retrieval():
    grid = load("retrieval-grid.json")["cells"]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    width = 0.26
    xs = range(len(CELLS))
    for k, (metric, color) in enumerate([("R", ACCENT), ("MRR", GREY), ("nDCG", LIGHT)]):
        vals = [grid[c][metric] for c, _ in CELLS]
        bars = ax.bar([x + (k - 1) * width for x in xs], vals, width, color=color, edgecolor=INK, linewidth=0.4,
                      label={"R": "Recall at 10", "MRR": "MRR", "nDCG": "nDCG at 10"}[metric])
        label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(list(xs), [n for _, n in CELLS])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score, 817 development queries")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, "fig02-retrieval")


def fig_hit_at_k():
    pq = load("per-query.json")["hit_at_k"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    styles = {"keyword-page": (GREY, "--"), "keyword-chunk": (GREY, ":"),
              "vector-page": (INK, "--"), "vector-chunk": (ACCENT, "-")}
    for name, label in CELLS:
        c, ls = styles[name]
        ax.plot(range(1, 11), pq[name], color=c, linestyle=ls, marker="o", markersize=2.5, linewidth=1,
                label=label.replace("\n", " "))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("k, the number of pages returned")
    ax.set_ylabel("Queries with a relevant page\nin the top k (fraction)")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, loc="center right")
    save(fig, "fig03-hit-at-k")


def fig_overlap():
    pq = load("per-query.json")
    pairs = [
        ("overlap_keyword_page_vs_vector_chunk", "keyword-page", "vector-chunk", "Keyword note\nvs vector chunk"),
        ("overlap_vector_page_vs_vector_chunk", "vector-page", "vector-chunk", "Vector page mean\nvs vector chunk"),
        ("overlap_keyword_page_vs_keyword_chunk", "keyword-page", "keyword-chunk", "Keyword note\nvs keyword chunk"),
    ]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for row, (key, a, b, label) in enumerate(pairs):
        d = pq[key]
        parts = [(d["both"], ACCENT, "both"), (d[f"only_{a}"], GREY, "first only"),
                 (d[f"only_{b}"], "#9db7d1", "second only"), (d["neither"], "white", "neither")]
        left = 0
        for val, color, name in parts:
            ax.barh(row, val, left=left, color=color, edgecolor=INK, linewidth=0.4,
                    label=name if row == 0 else None)
            if val >= 18:
                ax.text(left + val / 2, row, str(val), ha="center", va="center", fontsize=7,
                        color="white" if color == ACCENT else INK)
            left += val
    ax.set_yticks(range(len(pairs)), [p[3] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 817)
    ax.set_xlabel("Queries, out of 817, with a relevant page in the top 10")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.45, 1.0))
    save(fig, "fig04-overlap")


def fig_tokens_recall():
    grid = load("retrieval-grid.json")["cells"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    offsets = {"keyword-page": (6, -3), "keyword-chunk": (6, -3), "vector-page": (-8, 6), "vector-chunk": (6, -10)}
    for name, label in CELLS:
        c = grid[name]
        color = ACCENT if name == "vector-chunk" else INK
        ax.scatter(c["tokens"], c["R"], s=22, color=color, zorder=3)
        dx, dy = offsets[name]
        ax.annotate(label.replace("\n", " "), (c["tokens"], c["R"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(3000, 300000)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Tokens in the full text of the ten pages returned (mean per query, log scale)")
    ax.set_ylabel("Recall at 10")
    ax.axvline(6000, color=WARN, linewidth=0.7, linestyle="--")
    ax.text(6300, 0.74, "6,000-token budget", color=WARN, fontsize=7)
    save(fig, "fig05-tokens-recall")


def fig_units():
    units = load("evidence-units.json")["units"]
    order = ["chunk", "window", "section", "page"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw={"wspace": 0.45})
    surv = [units[u]["relevant_in_pack"] for u in order]
    colors = [ACCENT, GREY, GREY, WARN]
    bars = ax1.bar(order, surv, color=colors, edgecolor=INK, linewidth=0.4)
    label_bars(ax1, bars, "{:.3f}")
    ax1.set_ylim(0, 0.9)
    ax1.set_ylabel("Queries whose relevant page\nis inside the pack (fraction)")
    ax1.set_title("Relevant page survives", loc="left")
    drop = [units[u]["hits_dropped"] for u in order]
    tok = [units[u]["tokens_delivered"] for u in order]
    bars = ax2.bar(order, drop, color=colors, edgecolor=INK, linewidth=0.4)
    for b, t in zip(bars, tok):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{t:,.0f} tok", ha="center", fontsize=6.5)
    ax2.set_ylim(0, 6)
    ax2.set_ylabel("Ranked hits that did not fit\n(mean per query, of 10)")
    ax2.set_title("Hits dropped by the budget", loc="left")
    save(fig, "fig06-units")


def fig_hk1_sweep():
    sweep = load("hilbert-sweep.json")
    rnd = load("random-filter.json")["cells"]
    cells = sweep["cells"]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for level, color, marker in [(1, ACCENT, "o"), (2, WARN, "s")]:
        pts = sorted((max(c["mean_candidates"], 0.5), c["R@10"], c["ranges"]) for c in cells if c["level"] == level)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, markersize=3.5, linewidth=1,
                label=f"hk1, probe level {level}")
        for x, y, r in pts:
            ax.annotate(f"{r}", (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color=color)
    ax.plot([c["candidates"] for c in rnd], [c["R@10_mean"] for c in rnd], color=GREY, linestyle="--",
            marker="x", markersize=3.5, linewidth=0.9, label="random subset, same size")
    lvl0 = next(c for c in cells if c["level"] == 0)
    ax.scatter([lvl0["mean_candidates"]], [lvl0["R@10"]], marker="D", s=18, color=INK, zorder=3,
               label="hk1 level 0 (whole key space)")
    cos = sweep["cosine"]
    ax.scatter([cos["candidates"]], [cos["R@10"]], marker="*", s=70, color=ACCENT, zorder=2,
               label="cosine, no filter")
    ax.annotate("level 0 and cosine coincide:\n0.699 and 0.698 at about 5,100", (lvl0["mean_candidates"], lvl0["R@10"]),
                textcoords="offset points", xytext=(-12, 2), ha="right", va="center", fontsize=6.8)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 9000)
    ax.set_ylim(-0.02, 0.8)
    ax.set_xlabel("Candidate chunks per query (mean, log scale); numbers on points are range counts")
    ax.set_ylabel("Recall at 10 after cosine rescore")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "fig07-hk1-sweep")


def fig_neighbour():
    ch = load("neighbour-chunk.json")
    pg = load("neighbour-page.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.4})
    for d, color, label in [(ch, ACCENT, "chunk keys"), (pg, GREY, "page-mean keys")]:
        rows = [r for r in d["by_shared_bits"] if r["n"] >= 50 and r["bits"] <= 20]
        ax1.plot([r["bits"] for r in rows], [r["mean_cosine"] for r in rows], color=color, marker="o",
                 markersize=2.5, linewidth=1, label=label)
    ax1.set_xlabel("Shared prefix bits of two keys")
    ax1.set_ylabel("Mean cosine of the two vectors")
    ax1.set_title("Bit counts with at least 50 pairs", loc="left")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_ylim(0.2, 0.7)
    hist = ch["nearest_neighbor"]["shared_bits_histogram"]
    bins = list(range(0, 17))
    counts = [hist.get(str(b), 0) for b in bins]
    tail = sum(v for k, v in hist.items() if 17 <= int(k) < 64)
    ax2.bar([str(b) for b in bins] + ["17–63", "64"], counts + [tail, hist.get("64", 0)], color=ACCENT,
            edgecolor=INK, linewidth=0.3)
    ax2.set_xticks(range(0, 19, 3), [str(b) for b in range(0, 17, 3)] + ["17–63"])
    ax2.set_xlabel("Shared bits with the true nearest chunk")
    ax2.set_ylabel("Chunks")
    ax2.set_title("Nearest neighbour, 5,079 chunks", loc="left")
    save(fig, "fig08-neighbour")


def fig_latency():
    live = load("pgvector-5143.json")
    sc = load("scale-200k.json")
    rows = [
        ("pgvector top-10, live chunks", live["median_ms"], live["p95_ms"], ACCENT),
        ("pgvector top-10, 200,000 vectors", sc["in_session"]["median_ms"], sc["in_session"]["p95_ms"], ACCENT),
        ("pgvector top-10, 200,000, new client each", sc["range_scan"]["pgvector_median_ms"],
         sc["range_scan"]["pgvector_p95_ms"], GREY),
        ("one hk1 prefix range, 200,000, new client", sc["range_scan"]["hilbert_one_range_ms"], None, WARN),
    ]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for i, (name, med, p95, color) in enumerate(rows):
        ax.barh(i, med, color=color, edgecolor=INK, linewidth=0.4)
        txt = f"{med:.2f} ms" + (f", p95 {p95:.2f}" if p95 else "")
        ax.text(med * 1.15, i, txt, va="center", fontsize=7)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5000)
    ax.set_xlabel("Median latency, milliseconds (log scale)")
    save(fig, "fig09-latency")


def fig_harness():
    rtk = load("rtk-bytes.json")
    hp = load("harness-pages.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.55, "width_ratios": [1.4, 1]})
    names = {"ls_la_hilbert_paper": "ls -la", "git_status_hilbert_paper": "git status",
             "git_log_hilbert_paper": "git log -5", "rg_typst_hilbert_paper": "rg -l", "find_md_depth2_zig_hilbert": "find"}
    cmds = rtk["commands"]
    ys = range(len(cmds))
    ax1.barh([y - 0.18 for y in ys], [c["bytes_before"] for c in cmds], 0.36, color=LIGHT, edgecolor=INK,
             linewidth=0.4, label="raw")
    ax1.barh([y + 0.18 for y in ys], [c["bytes_after"] for c in cmds], 0.36, color=ACCENT, edgecolor=INK,
             linewidth=0.4, label="through RTK")
    ax1.set_yticks(list(ys), [names.get(c["name"], c["name"]) for c in cmds])
    ax1.invert_yaxis()
    ax1.set_xlabel("Output bytes")
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(f"RTK: {rtk['bytes_after']} of {rtk['bytes_before']} bytes", loc="left")
    n = hp["n"]
    vals = [round(hp["vault_hit_rate"] * n), round(hp["vault_token_hit_rate"] * n), round(hp["gbrain_hit_rate"] * n)]
    bars = ax2.bar(["ripgrep,\nexact", "ripgrep,\ntokens", "gbrain\nsearch"], vals, color=[LIGHT, GREY, ACCENT],
                   edgecolor=INK, linewidth=0.4)
    label_bars(ax2, bars, "{:.0f}", dy=0.4)
    ax2.set_ylim(0, n)
    ax2.set_ylabel(f"Questions, of {n}, where the\nrelevant page was found")
    ax2.set_title("Thirty page questions", loc="left")
    save(fig, "fig10-harness")


def fig_heldout():
    grid = load("retrieval-grid.json")["cells"]["vector-chunk"]
    ho = load("heldout.json")
    ev = load("earlier-evals.json")
    hyb = ev["hybrid_heldout"]
    metrics = [("R@10", "R", "Recall at 10"), ("MRR", "MRR", "MRR"), ("nDCG@10", "nDCG", "nDCG at 10")]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    width = 0.26
    xs = range(len(metrics))
    series = [
        ("vector chunks, development", [grid[g] for _, g, _ in metrics], ACCENT),
        ("vector chunks, held-out", [ho[h] for h, _, _ in metrics], "#9db7d1"),
        ("gbrain hybrid, held-out", [hyb[h] for h, _, _ in metrics], GREY),
    ]
    for k, (name, vals, color) in enumerate(series):
        bars = ax.bar([x + (k - 1) * width for x in xs], vals, width, color=color, edgecolor=INK, linewidth=0.4,
                      label=name)
        label_bars(ax, bars, "{:.2f}")
    ax.set_xticks(list(xs), [m[2] for m in metrics])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Score, 817 queries per set")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    save(fig, "fig11-heldout")


def fig_appendix():
    ev = load("earlier-evals.json")["dev"]
    order = [("baseline", "Hybrid,\nbaseline"), ("title", "Hybrid,\ntitle prefix"), ("rerank", "Hybrid,\nplus reranker")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"wspace": 0.4, "width_ratios": [1.6, 1]})
    width = 0.26
    xs = range(len(order))
    for k, (m, color) in enumerate([("R@10", ACCENT), ("MRR", GREY), ("nDCG@10", LIGHT)]):
        bars = ax1.bar([x + (k - 1) * width for x in xs], [ev[o][m] for o, _ in order], width, color=color,
                       edgecolor=INK, linewidth=0.4, label=m)
        label_bars(ax1, bars, "{:.2f}")
    ax1.set_xticks(list(xs), [o[1] for o in order])
    ax1.set_ylim(0, 0.85)
    ax1.legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
    ax1.set_ylabel("Score, 817 development queries")
    lat = load("earlier-evals.json")["latency_seconds"]
    bars = ax2.bar(["reranker\noff", "reranker\non"], [lat["off"], lat["on"]], color=[ACCENT, WARN], edgecolor=INK,
                   linewidth=0.4)
    label_bars(ax2, bars, "{:.2f} s", dy=0.05)
    ax2.set_ylabel("Seconds per query")
    ax2.set_ylim(0, 4.5)
    save(fig, "figA1-earlier")


def fig_dataset():
    d = load("investigation.json")["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw={"wspace": 0.45})
    qw = d["question_words"]["hist"]
    axes[0].bar(range(len(qw)), qw, color=ACCENT, width=0.85)
    axes[0].set_xlabel("Words per question")
    axes[0].set_ylabel("Questions")
    rp = d["relevant_per_question"]["hist"]
    axes[1].bar(range(len(rp)), rp, color=GREY, width=0.7)
    axes[1].set_xlabel("Relevant pages per question")
    axes[1].set_xticks(range(len(rp)))
    ct = d["chunk_tokens"]["values_sample"]
    axes[2].hist(ct, bins=30, color=ACCENT, range=(0, 1600))
    axes[2].set_xlabel("Tokens per chunk")
    axes[2].set_ylabel("Chunks (1 in 5)")
    save(fig, "fig00-dataset")


def fig_keyword():
    c = load("investigation.json")["cells"]
    order = [("keyword-and-page", "AND,\nnote"), ("keyword-and-chunk", "AND,\nchunk"), ("keyword-or-page", "OR,\nnote"),
             ("keyword-or-chunk", "OR,\nchunk"), ("hybrid-rrf", "RRF,\nOR note"), ("hybrid-rrf-chunk", "RRF,\nOR chunk"),
             ("vector-chunk", "Vector,\nchunk")]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    xs = range(len(order))
    w = 0.38
    b1 = ax.bar([x - w / 2 for x in xs], [c[k]["recall"] for k, _ in order], w, color=ACCENT, edgecolor=INK,
                linewidth=0.4, label="Recall at 10")
    b2 = ax.bar([x + w / 2 for x in xs], [c[k]["mrr"] for k, _ in order], w, color=LIGHT, edgecolor=INK,
                linewidth=0.4, label="MRR")
    label_bars(ax, b1, "{:.2f}")
    label_bars(ax, b2, "{:.2f}")
    ax.set_xticks(list(xs), [n for _, n in order], fontsize=7)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score, 817 development questions")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "fig12-keyword")


def fig_effects():
    inv = load("investigation.json")
    rows = []
    for group in ("confirmatory", "exploratory"):
        for k, v in inv[group].items():
            rows.append((k.split(" ", 1)[0], LABELS[k.split(" ", 1)[0]], v["diff"], v["ci_lo"], v["ci_hi"], v["p_holm"], group))
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for i, (tag, name, d, lo, hi, p, g) in enumerate(rows):
        color = ACCENT if g == "confirmatory" else GREY
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.4)
        ax.scatter([d], [i], color=color, s=16, zorder=3, marker="o" if p < 0.05 else "D",
                   facecolors=color if p < 0.05 else "white")
        ax.text(0.62, i, f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}", va="center", fontsize=6.8)
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_yticks(range(len(rows)), [f"{r[0]}  {r[1]}" for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.55, 0.75)
    ax.set_xlabel("Paired difference with 95% bootstrap interval (recall at 10, or survival share)")
    ax.text(0.62, -1.0, "Holm-adjusted", fontsize=6.8, color=GREY)
    save(fig, "fig13-effects")


def fig_lsh():
    cen = load("centering.json")["variants"]
    qa = load("lsh-qa.json")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.35}, sharey=True)
    for ax, key, title, qkey in [(axes[0], "raw", "Raw vectors", "raw"),
                                 (axes[1], "centred-corpus-mean", "Mean-centred vectors", "centred")]:
        cal = [b for b in cen[key]["calibration"] if b["n"] >= 100]
        x = [b["lo"] + 0.05 for b in cal]
        ax.plot(x, [b["cell_pred"] for b in cal], color=GREY, linestyle="--", linewidth=1,
                label=r"model $(1-\theta/\pi)^8$")
        ax.plot(x, [b["cell_obs"] for b in cal], color=ACCENT, marker="o", markersize=2.5, linewidth=1,
                label="observed, chunk pairs")
        q = qa[qkey]
        ax.scatter([q["cos_quartiles"][1]], [q["same_cell_observed"]], marker="*", s=60, color=WARN, zorder=4,
                   label="observed, question and answer")
        ax.scatter([q["cos_quartiles"][1]], [q["same_cell_predicted"]], marker="x", s=30, color=WARN, zorder=4,
                   label="model, question and answer")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Cosine of the pair")
    axes[0].set_ylabel("Share of pairs in the same\nlevel-1 cell (log scale)")
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    save(fig, "fig14-lsh-model")


def fig_occupancy():
    cen = load("centering.json")["variants"]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for key, color, label in [("raw", WARN, "raw vectors"), ("centred-corpus-mean", ACCENT, "mean-centred")]:
        occ = cen[key]["occupancy_sorted"]
        ax.plot(range(1, 257), occ, color=color, linewidth=1, label=label)
    ax.axhline(sum(cen["raw"]["occupancy_sorted"]) / 256, color=GREY, linestyle="--", linewidth=0.8,
               label="uniform")
    ax.set_xlabel("Level-1 cell, ordered by size")
    ax.set_ylabel("Chunks in the cell")
    ax.legend(frameon=False)
    save(fig, "fig15-occupancy")


def fig_centring_filter():
    cen = load("centering.json")["variants"]
    rnd = load("random-filter.json")["cells"]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for key, color, label in [("raw", ACCENT, "hk1, raw vectors"), ("centred-corpus-mean", WARN, "hk1, mean-centred")]:
        pts = sorted((v["mean_candidates"], v["recall"]) for v in cen[key]["filter"].values() if v["level"] == 1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", markersize=3, linewidth=1,
                label=f"{label}, level 1")
    ax.plot([c["candidates"] for c in rnd], [c["R@10_mean"] for c in rnd], color=GREY, linestyle="--", marker="x",
            markersize=3, linewidth=0.9, label="random subset")
    ax.set_xscale("log")
    ax.set_xlabel("Candidate chunks per question (mean, log scale)")
    ax.set_ylabel("Recall at 10 after rescore")
    ax.set_ylim(0, 0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig16-centring-filter")


def fig_failures():
    f = load("investigation.json")["failures"]
    parts = [("relevant_not_in_corpus", "relevant note\nnot in corpus"), ("duplicate_in_top10", "duplicate note\nin top 10"),
             ("rank_11_20", "rank\n11 to 20"), ("rank_21_50", "rank\n21 to 50"), ("rank_51_100", "rank\n51 to 100"),
             ("rank_over_100", "rank\nover 100")]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    bars = ax.bar([p[1] for p in parts], [f[p[0]] for p in parts], color=[GREY, GREY, ACCENT, ACCENT, WARN, WARN],
                  edgecolor=INK, linewidth=0.4)
    label_bars(ax, bars, "{:.0f}", dy=0.6)
    ax.set_ylabel("Questions")
    ax.set_title(f"{f['misses']} questions with no relevant note in the chunk-cosine top 10", loc="left")
    ax.tick_params(axis="x", labelsize=7)
    save(fig, "fig17-failures")


def fig_engines():
    v = load("arango-vector.json")["results"]
    hop = load("arango-hop.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.45, "width_ratios": [1.7, 1]})
    for engine, color, marker, label in [("arangodb", WARN, "s", "ArangoDB vector index (nProbe)"),
                                         ("pgvector", ACCENT, "o", "pgvector HNSW (ef_search)")]:
        pts = [r for r in v if r["engine"] == engine and "full scan" not in r["index"]]
        ax1.plot([r["median_ms"] for r in pts], [r["chunk_recall_vs_exact_at_10"] for r in pts], color=color,
                 marker=marker, markersize=3.5, linewidth=1, label=label)
        for r in pts:
            tag = r.get("nProbe", r.get("ef_search"))
            if engine == "pgvector" and tag not in (50, 400):
                continue
            off = (4, -9) if engine == "arangodb" else ((-4, -10) if tag == 50 else (4, 4))
            ax1.annotate(str(tag), (r["median_ms"], r["chunk_recall_vs_exact_at_10"]), textcoords="offset points",
                         xytext=off, fontsize=6.3, color=color, ha="right" if off[0] < 0 else "left")
        ex = next(r for r in v if r["engine"] == engine and "full scan" in r["index"])
        ax1.scatter([ex["median_ms"]], [1.0], marker=marker, s=26, facecolors="white", edgecolors=color, zorder=3)
        ax1.annotate("full scan", (ex["median_ms"], 1.0), textcoords="offset points",
                     xytext=(0, -11) if engine == "arangodb" else (0, 6), ha="center", fontsize=6.3, color=color)
    ax1.set_xscale("log")
    ax1.set_xlim(1, 20)
    ax1.set_ylim(0.5, 1.04)
    ax1.set_xlabel("Median latency per query, ms (log scale)")
    ax1.set_ylabel("Exact top-10 chunks found")
    ax1.legend(frameon=False, loc="lower right", fontsize=6.8)
    bars = ax2.bar(["AQL", "SQL"], [hop["arangodb_aql"]["median_ms"], hop["postgres_sql"]["median_ms"]],
                   color=[WARN, ACCENT], edgecolor=INK, linewidth=0.4)
    label_bars(ax2, bars, "{:.2f} ms", dy=0.03)
    ax2.set_ylabel("Median latency, one hop, ms")
    ax2.set_title(f"One hop, {hop['starts']} start notes", loc="left")
    ax2.set_ylim(0, 1.8)
    save(fig, "fig18-engines")


IDEA_LABELS = {
    "H1a": "1  Rerank top 50, default instruction",
    "H1b": "1  Rerank top 50, corpus instruction",
    "H2": "2  Sentence-pruned packing (survival)",
    "H3": "3  Corpus query instruction",
    "H4a": "4  Subtract shared mean",
    "H4b": "4  Subtract separate means",
    "H4c": "4  Project off shared direction",
    "H4d": "4  Project off separate directions",
    "H5": "5  Rocchio feedback",
    "H6": "6  Hilbert partitions + SOAR (vs exhaustive)",
    "H7": "7  Hilbert forest (vs exhaustive)",
    "H9": "9  Link-graph PageRank",
    "H10": "10 HyDE",
    "H11": "11 Adaptive cut-off (survival)",
    "H12": "12 MUVERA (notes with 5+ chunks)",
}


def fig_ideas():
    s = load("ideas/summary.json")
    rows = [h for h in s["hypotheses"] if h.get("diff") is not None]
    fig, ax = plt.subplots(figsize=(W, 3.9))
    lo_all = min(h["ci95"][0] for h in rows)
    hi_all = max(max(h["ci95"][1], h["threshold"]) for h in rows)
    value_x = hi_all + 0.025
    for i, h in enumerate(rows):
        lo, hi = h["ci95"]
        color = ACCENT if h["accepted"] else GREY
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.4)
        ax.scatter([h["diff"]], [i], color=color, s=16, zorder=3)
        ax.scatter([h["threshold"]], [i], marker="|", s=90, color=WARN, zorder=2)
        guard_failed = any("guard" in r for r in h.get("reasons", []))
        label = f"{h['diff']:+.3f}" + ("  accepted" if h["accepted"] else "  failed judge check" if guard_failed else "")
        ax.text(value_x, i, label, va="center", ha="left", fontsize=6.3, color=color,
                fontweight="bold" if h["accepted"] else "normal")
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_xlim(lo_all - 0.02, value_x + 0.17)
    ax.set_xticks([t / 10 for t in range(-3, 3) if lo_all - 0.02 <= t / 10 <= hi_all])
    ax.set_yticks(range(len(rows)), [IDEA_LABELS.get(h["id"], h["id"]) for h in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference with 95% bootstrap interval (recall at 10 unless marked)")
    ax.scatter([], [], color=ACCENT, s=16, label="measured difference, accepted")
    ax.scatter([], [], color=GREY, s=16, label="measured difference, not accepted")
    ax.scatter([], [], marker="|", s=90, color=WARN, label="threshold fixed before the run")
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3,
              handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig19-ideas")


FIGURES = [fig_ideas, fig_engines, fig_dataset, fig_pipeline, fig_retrieval, fig_hit_at_k, fig_overlap, fig_tokens_recall, fig_units,
           fig_hk1_sweep, fig_neighbour, fig_latency, fig_harness, fig_heldout, fig_appendix, fig_keyword, fig_effects,
           fig_lsh, fig_occupancy, fig_centring_filter, fig_failures]


def main() -> int:
    for f in FIGURES:
        f()
    made = sorted(p.name for p in OUT.glob("*.svg"))
    print("\n".join(made))
    return 0 if len(made) >= len(FIGURES) else 1


if __name__ == "__main__":
    sys.exit(main())
