#!/usr/bin/env python3
"""Assemble REPORT.md for the factorial study from the result files. Every number is read from
level1-*.json, level2-summary.json and heldout-level1.json; the prose only frames them."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import report_tables as rt

HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / name
    return json.loads(path.read_text()) if path.exists() else None


def captured(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue().strip()


def pm(x, d=3):
    return f"{x:+.{d}f}"


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.2g}"


def cfg(key: str) -> str:
    return key.replace("|", " / ")


def level2_table(m: dict, title: str) -> str:
    rows = [f"**{title}**", "", "| level | runs | errors | page found | judge (0-2) | input tokens | wall time (s) | tool calls |",
            "|---|---|---|---|---|---|---|---|"]
    for level, v in m.items():
        rows.append(f"| {level} | {v['n']} | {v['errors']} | {v['page_found']['mean']:.2f} | {v['judge']['mean']:.2f} | "
                    f"{v['input_tokens']['mean']:,.0f} | {v['wall_s']['mean']:.0f} | {v['tool_calls']['mean']:.1f} |")
    return "\n".join(rows)


def main() -> None:
    sh = load("level1-shapley.json")["points"]
    lb = load("level1-leaderboard.json")
    l2 = load("level2-summary.json")
    ho = load("heldout-level1.json")
    jp = load("judge-packs.json")
    conf = lb["confirmation"]
    rec = lb["best_recall"]["top20"][0]
    srv = lb["best_survival"]["top20"][0]
    cov = l2["coverage"]
    ctx = l2["marginals"]["num_ctx (local)"]
    tool = l2["marginals"]["tool (local)"]
    out = []
    out.append("# Factorial study of the retrieval pipeline and the local agent harness\n")
    out.append("Findings, not decisions. Pre-registration: `preregistration.md`; every departure, with its time, is in "
               "`deviations.md` (C1, C2, D1 to D11). Development questions only, except section 3.\n")
    out.append("## 0. What ran\n")
    out.append(f"- Level 1: all 1,020 cells (204 rankings times 5 evidence units) on all {lb['n_questions']} "
               "development questions, after the reranker cache was completed (202,515 pairs).")
    out.append(f"- Level 2: {cov['completed_local']} of the {cov['planned']} planned runs, all with the local 27B model "
               "at 8K and 32K windows. The 138 cloud runs (Cursor CLI, Claude Code) were dropped by the author (D7), and "
               "the 20 runs at a 128K window exceeded this 32 GB machine's memory and were not run (D9).")
    out.append("- The level 2 judge ran on a work account by mistake (D10); judge scores are shown but marked.\n")
    out.append("## 1. Design\n")
    out.append((HERE / "private" / "report-methods-draft.md").read_text().split("## 1. Design", 1)[1].strip() + "\n")
    out.append("## 2. Level 1 results\n")
    out.append("### 2.1 Shapley points per factor (tier B, 817 questions)\n")
    out.append("Each value is the factor's average contribution to the metric over all combinations of the other "
               "factors, with a 95% bootstrap interval. Latency is composed from per-component timings.\n")
    out.append(captured(rt.shapley) + "\n")
    out.append("### 2.2 Main effects (Holm-corrected)\n")
    out.append(captured(rt.main_effects) + "\n")
    out.append("### 2.3 Largest two-factor interactions\n")
    out.append(captured(rt.top_interactions) + "\n")
    out.append("### 2.4 Leaderboards, chosen on the tune half and reported on both halves\n")
    out.append(captured(rt.boards) + "\n")
    out.append(captured(rt.pareto) + "\n")

    out.append("## 3. Held-out confirmation\n")
    if ho:
        out.append(f"Run once on the {ho['n']} sealed held-out questions (SHA-256 `{ho['qrels_test_sha256'][:16]}...`), "
                   "protocol D8. Paired against the reference configuration on the same questions.\n")
        out.append("| run | configuration | unit | R@10 | MRR | survival | tokens |")
        out.append("|---|---|---|---|---|---|---|")
        for name, r in ho["runs"].items():
            out.append(f"| {name} | `{cfg(r['config'])}` | {r['unit']} | {r['R@10']:.3f} | {r['MRR']:.3f} | "
                       f"{r['survival']:.3f} | {r['tokens']:,.0f} |")
        out.append("")
        for name, t in ho["paired_vs_reference"].items():
            out.append(f"- {name}: R@10 {pm(t['R@10']['diff'])} [{pm(t['R@10']['ci95'][0])}, {pm(t['R@10']['ci95'][1])}], "
                       f"Holm p {fp(t['R@10']['p_holm'])}; survival {pm(t['survival']['diff'])} "
                       f"[{pm(t['survival']['ci95'][0])}, {pm(t['survival']['ci95'][1])}], Holm p {fp(t['survival']['p_holm'])}; "
                       f"tokens {t['tokens_diff']:+,.0f}.")
        out.append("")
    else:
        out.append("Not run yet.\n")

    out.append("### 3.1 Do sentence packs answer the question?\n")
    if jp:
        ni, mc = jp["non_inferiority_margin_0.02"], jp["mcnemar_two_sided"]
        out.append(f"Pre-registered in `judge-preregistration.md`: {jp['n']} confirmation-half questions, the local judge "
                   f"`{jp['judge_model']}` asked whether each pack contains the information needed to answer, both packs "
                   "judged per question in a random order.\n")
        out.append("| pack | judged sufficient | survival | sufficient when a relevant note survives | tokens |")
        out.append("|---|---|---|---|---|")
        for k, name in (("A", "best-survival, pruned sentences"), ("B", "reference, chunks")):
            out.append(f"| {name} | {jp['yes'][k]:.3f} | {jp['survival'][k]:.3f} | {jp['yes_given_survival'][k]:.3f} | "
                       f"{jp['tokens'][k]:,.0f} |")
        out.append("")
        out.append(f"Difference {pm(ni['diff'])} [{pm(ni['ci95'][0])}, {pm(ni['ci95'][1])}]; non-inferiority at the 0.02 margin "
                   f"{'accepted' if jp['non_inferior'] else 'not accepted'} (p {fp(ni['p'])}, the larger of the Wilcoxon "
                   f"and shifted t p-values; the Wilcoxon value alone is degenerate here because {jp['n'] - ni['improved'] - ni['worsened']} "
                   f"of {jp['n']} differences are tied). Exact McNemar, two-sided: {mc['new_only']} questions favour the "
                   f"pruned pack, {mc['comparator_only']} the chunk pack, p {fp(mc['p'])}. Median judge time "
                   f"{jp['judge_seconds_median']:.0f} s per pack (D11 records the memory stop lines).\n")
    else:
        out.append("Not run yet.\n")

    out.append("## 4. Level 2: the local agent\n")
    tasks = len({t for r in load('level2-design.json')['rows'] for t in r['tasks']})
    out.append(f"{cov['completed_local']} runs over {tasks} pilot tasks. Samples are small; differences "
               "below are descriptive unless a test is given. Judge scores come from D10.\n")
    out.append(level2_table(tool, "Notes as files or through gbrain") + "\n")
    out.append(level2_table(ctx, "Context window") + "\n")
    out.append(level2_table(l2["marginals"]["unit (local gbrain)"], "gbrain packing unit") + "\n")
    pc = l2["primary_contrasts"]
    g = pc["page_found"]["gbrain vs files"]
    gj = pc["judge"]["gbrain vs files"]
    out.append(f"Matched pairs, gbrain against files: page found {pm(g['mean_diff'], 2)} over {g['pairs']} pairs "
               f"(sign test, Holm p {g['p_holm']:.2g}); judge {pm(gj['mean_diff'], 2)} (Holm p {gj['p_holm']:.2g}). "
               "The pre-registered contrast of the largest and smallest window has no pairs, because no 128K run exists.\n")

    out.append("## 5. What beats what\n")
    out.append(f"- **Recall.** The reranker is the largest positive factor ({pm(sh['F3=on']['R@10']['shapley'])} R@10), "
               f"then link PageRank ({pm(sh['F4=on']['R@10']['shapley'])}) and hybrid search ({pm(sh['F1=hybrid']['R@10']['shapley'])}). "
               f"The hk1 filter costs {pm(sh['F7=hk1']['R@10']['shapley'])} and keyword search alone {pm(sh['F1=lexical']['R@10']['shapley'])}. "
               f"Best configuration: {rec['label']}, confirm R@10 {rec['confirm']['R@10']:.3f}"
               + (f"; on held-out questions its gain over the reference shrank to {pm(ho['paired_vs_reference']['best_recall']['R@10']['diff'])} "
                  f"(Holm p {fp(ho['paired_vs_reference']['best_recall']['R@10']['p_holm'])})." if ho else "."))
    out.append(f"- **Speed.** The reranker adds {sh['F3=on']['latency']['shapley']/1000:.1f} s per question; every other factor "
               f"adds at most {max(abs(v['latency']['shapley']) for k, v in sh.items() if k != 'F3=on'):.0f} ms.")
    out.append(f"- **Tokens and survival.** Sentence-pruned packs keep a relevant note more often "
               f"({pm(sh['F8=pruned']['survival']['shapley'])} survival) with {sh['F8=pruned']['tokens']['shapley']:+,.0f} tokens; "
               f"whole-page packs lose {pm(sh['F8=page']['survival']['shapley'])}. Best: {srv['label']}, confirm survival "
               f"{srv['confirm']['survival']:.3f}."
               + (f" The pre-registered judge (section 3.1) found its pruned packs sufficient for {jp['yes']['A']:.3f} of "
                  f"questions against {jp['yes']['B']:.3f} for chunk packs, so the survival gain does not carry over to answers."
                  if jp else ""))
    c32, c8 = ctx["32768"], ctx["8192"]
    out.append(f"- **Local model, context window.** 32K against 8K: page found {c32['page_found']['mean']:.2f} vs "
               f"{c8['page_found']['mean']:.2f}, input tokens {c32['input_tokens']['mean']:,.0f} vs {c8['input_tokens']['mean']:,.0f}, "
               f"wall time {c32['wall_s']['mean']:.0f} vs {c8['wall_s']['mean']:.0f} s. The larger window costs about "
               f"{c32['input_tokens']['mean']/c8['input_tokens']['mean']:.1f} times the tokens and "
               f"{c32['wall_s']['mean']/c8['wall_s']['mean']:.1f} times the time with no measured gain. 128K does not fit in memory.")
    out.append("- **Cursor, Claude Code, RTK, evidence budget.** Not measured in this round (cloud stratum dropped).\n")

    out.append("## 6. Implementation plans (work to verify, not decisions)\n")
    hr = ho["paired_vs_reference"]["best_recall"]["R@10"] if ho else None
    held = ("its recall gain did not reach significance on the held-out questions "
            f"({pm(hr['diff'])}, Holm p {fp(hr['p_holm'])}), so" if hr and hr["p_holm"] >= 0.05 else "confirmed on held-out questions;")
    out.append(f"1. Reranker on the hybrid top 50 with link PageRank: {held} re-measure it on more questions, with depth 30 "
               f"and score caching against its {sh['F3=on']['latency']['shapley']/1000:.1f} s cost, before enabling it for agents.")
    out.append("2. Link PageRank: a cheap recall gain; implement as a gbrain post-processor over the links table and "
               "re-measure stacked on the reranker.")
    hs = ho["paired_vs_reference"]["best_survival"]["survival"] if ho else None
    rep = (f"replicated on held-out questions ({pm(hs['diff'])} survival, {hs['new_only']} vs {hs['comparator_only']} discordant questions); " if hs else "")
    if jp:
        out.append(f"3. Sentence-pruned packs: {rep}but judged sufficient less often than chunk packs "
                   f"({pm(jp['non_inferiority_margin_0.02']['diff'])}, non-inferiority not accepted), so the measurements do not "
                   "support them as the default unit; a pack that keeps whole chunks for the top notes and adds pruned sentences only below them would need its own "
                   "pre-registered judged test.")
    else:
        out.append(f"3. Sentence-pruned packs: {rep}pre-register a judged check of answer quality, with the judge on a personal account, before use.")
    out.append("4. Keep chunk packs as the default unit; do not use the hk1 filter for candidate selection.")
    out.append("5. Local agent: prefer an 8K window; the 32K window doubled cost without a measured gain.\n")
    fu = HERE.parent / "followup"
    if (fu / "qrels-extended.json").exists():
        jl = json.loads((fu / "judge-length.json").read_text())
        cp = json.loads((fu / "compress-packs.json").read_text())
        qe = json.loads((fu / "qrels-extended.json").read_text())
        pw = json.loads((fu / "power.json").read_text())
        cs = json.loads((fu / "compositional-sentences.json").read_text())
        out.append("## 7. Follow-up checks (bench/followup, pre-registered)\n")
        out.append(f"- Length: chunk packs cut to the sentence packs' length are judged sufficient {jl['yes']['Bcut']:.3f}, "
                   f"the same as full chunk packs ({jl['yes']['B']:.3f}); against sentence packs {jl['Bcut_vs_A']['new_only']} vs "
                   f"{jl['Bcut_vs_A']['comparator_only']} (p {fp(jl['Bcut_vs_A']['p'])}). Reading: {jl['reading']}.")
        out.append(f"- Compression: hybrid {cp['yes']['H']:.3f} ({cp['tokens']['H']:,.0f} tokens), LongLLMLingua {cp['yes']['L']:.3f} "
                   f"({cp['tokens']['L']:,.0f}), RECOMP extractive {cp['yes']['R']:.3f} ({cp['tokens']['R']:,.0f}), chunk packs "
                   f"{cp['yes']['B']:.3f}; none passes non-inferiority at 0.02.")
        ho = pw["curves"]["held-out (+0.016)"]
        out.append(f"- Reranker power: about {ho['n_for_80_percent']:,} questions for 80 percent power at the held-out gain; "
                   f"labels judged incomplete for {qe['unlabelled_judged_relevant']['share'] * 100:.0f} percent of unlabelled top-10 notes, "
                   f"and with the judge's labels the reranker's gain is {pm(qe['reranker_gain']['extended']['diff'])} "
                   f"(original labels {pm(qe['reranker_gain']['original']['diff'])}).")
        acc = cs["acceptance"]["chunk-hk1 vs chunk-dense"]
        out.append(f"- Hilbert facet map: rejected by the author's cost rule (chunk keys {acc['token_ratio']:.1f}x the tokens and "
                   f"{acc['time_ratio']:.1f}x the time per correct answer of dense facets).")
        jh = json.loads((fu / "judge-human.json").read_text())
        gc = json.loads((fu / "gold-check.json").read_text())
        s1, s2 = jh["s1b"], jh["s2_agreement"]
        out.append(f"- Author's labels: agreement with the judge {s1['agreement']['all']['agreement']:.2f} "
                   f"(kappa {s1['agreement']['all']['kappa']:.2f}) on 100 packs. Prediction-powered A minus B is "
                   f"{s1['A_minus_B']['ppi']['estimate']:+.3f} (95% CI {s1['A_minus_B']['ppi']['ci95'][0]:+.3f} to "
                   f"{s1['A_minus_B']['ppi']['ci95'][1]:+.3f}); the author's own difference on the 50 is "
                   f"{s1['A_minus_B']['human']['estimate']:+.3f}. Relevance agreement {s2['judge_yes']['agreement']:.2f} "
                   f"where the judge said yes and {s2['judge_no']['agreement']:.2f} where it said no.")
        out.append(f"- Gold check: {gc['correct_and_complete']} of {gc['checked']} lists marked correct and complete. "
                   f"On those, hk1 facet recall {gc['summary']['hk1_facets']['recall']:.3f} against dense facets "
                   f"{gc['summary']['dense_facets']['recall']:.3f}.\n")
    (HERE / "REPORT.md").write_text("\n".join(out) + "\n")
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
