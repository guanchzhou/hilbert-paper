#!/usr/bin/env python3
"""Render the level 1 and level 2 result tables as markdown (for REPORT.md)."""

import json
import sys
from pathlib import Path

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def f(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}" if isinstance(x, float) and d else f"{x:.{d}f}"


def shapley() -> None:
    s = json.loads((HERE / "level1-shapley.json").read_text())["points"]
    print("| Factor = level | R@10 | survival | tokens | latency (ms) | MRR |")
    print("|---|---|---|---|---|---|")
    for k, v in s.items():
        print(f"| {k} | {v['R@10']['shapley']:+.4f} [{v['R@10']['ci95'][0]:+.4f}, {v['R@10']['ci95'][1]:+.4f}] "
              f"| {v['survival']['shapley']:+.4f} [{v['survival']['ci95'][0]:+.4f}, {v['survival']['ci95'][1]:+.4f}] "
              f"| {v['tokens']['shapley']:+.0f} | {v['latency']['shapley']:+.1f} | {v['MRR']['shapley']:+.4f} |")


def main_effects() -> None:
    e = json.loads((HERE / "level1-effects.json").read_text())["main"]
    for metric in ("R@10", "survival", "tokens", "latency"):
        print(f"\n{metric}\n")
        print("| Contrast | effect | 95% CI | Holm p |")
        print("|---|---|---|---|")
        for k, v in e[metric].items():
            p = v.get("p_holm")
            print(f"| {k} | {v['effect']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                  f"{'n/a' if p is None else f'{p:.2g}'} |")


def top_interactions(n: int = 8) -> None:
    e = json.loads((HERE / "level1-effects.json").read_text())["interactions"]
    for metric in ("R@10", "survival", "tokens"):
        items = sorted(e[metric].items(), key=lambda kv: -abs(kv[1]["interaction"]))[:n]
        print(f"\n{metric}\n")
        print("| Interaction | size | 95% CI | BH q |")
        print("|---|---|---|---|")
        for k, v in items:
            print(f"| {k} | {v['interaction']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['q_bh']:.2g} |")


def boards(n: int = 5) -> None:
    lb = json.loads((HERE / "level1-leaderboard.json").read_text())
    for ax in ("best_recall", "best_survival", "fewest_tokens", "fastest"):
        print(f"\n{ax} (eligible {lb[ax]['eligible']})\n")
        print("| # | configuration | R@10 tune / confirm | survival tune / confirm | tokens | latency ms |")
        print("|---|---|---|---|---|---|")
        for i, r in enumerate(lb[ax]["top20"][:n], 1):
            lat = r["all"]["rank_latency"] if ax in ("best_recall", "fastest") else r["all"]["latency"]
            print(f"| {i} | {r['label']} | {r['tune']['R@10']:.3f} / {r['confirm']['R@10']:.3f} | "
                  f"{r['tune']['survival']:.3f} / {r['confirm']['survival']:.3f} | {r['all']['tokens']:.0f} | {lat:.0f} |")
    print("\nconfirmation\n")
    for k, v in lb["confirmation"].items():
        print(f"- {k}: {v['config']}: diff {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
              f"p {v['p']:.2g}, Holm {v['p_holm']:.2g}")


def pareto() -> None:
    p = json.loads((HERE / "level1-pareto.json").read_text())
    for k in ("recall_tokens_latency", "survival_tokens_latency"):
        print(f"\n{k}: {len(p[k])} non-dominated cells")


if __name__ == "__main__":
    shapley()
    main_effects()
    top_interactions()
    boards()
    pareto()
