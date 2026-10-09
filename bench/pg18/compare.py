"""Pair every database-dependent result measured on PostgreSQL 16.15 with its repeat on 18.6.

Reads the original files one level up and the repeats in this directory; writes comparison.json.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    old = {name: load(BENCH / name) for name in ("pgvector-5143.json", "retrieval-grid.json", "hilbert-sweep.json",
                                                   "scale-200k.json", "arango-vector.json", "arango-hop.json",
                                                   "arango-load.json", "manifest.json")}
    new = {name: load(HERE / name) for name in ("versions.json", "pgvector-live.json", "keyword.json", "hk1-scans.json",
                                                  "scale-in-session.json", "scale-range-scan.json", "arango-vector.json",
                                                  "arango-hop.json", "arango-load.json")}
    rows = []

    def row(group, measure, a, b, unit, comparable=True, note=""):
        rows.append({"group": group, "measure": measure, "pg16": a, "pg18": b, "unit": unit,
                     "comparable": comparable, "note": note})

    o, n = old["pgvector-5143.json"], new["pgvector-live.json"]
    row("pgvector, live chunks", "top-10 median", o["median_ms"], n["median_ms"], "ms")
    row("pgvector, live chunks", "top-10 p95", o["p95_ms"], n["p95_ms"], "ms")

    g, k = old["retrieval-grid.json"]["cells"], new["keyword.json"]
    for cell in ("keyword-page", "keyword-chunk"):
        row("keyword search", f"{cell} recall at 10", g[cell]["R"], k[cell]["R"], "")
        row("keyword search", f"{cell} time, 817 questions", g[cell]["seconds"], k[cell]["seconds"], "s")

    sweep = {(c["level"], c["ranges"]): c for c in old["hilbert-sweep.json"]["cells"]}
    for p in new["hk1-scans.json"]["probes"]:
        c = sweep[(p["level"], p["ranges"])]
        row("hk1 range scans", f"level {p['level']}, {p['ranges']} ranges, 817 questions", c["scan_seconds"],
            p["scan_seconds"], "s", note=f"median candidates {c['median_candidates']} then {p['median_candidates']}")

    s_old, s_new = old["scale-200k.json"]["in_session"], new["scale-in-session.json"]
    row("200,000 vectors, one session", "top-10 median", s_old["median_ms"], s_new["median_ms"], "ms")
    row("200,000 vectors, one session", "top-10 p95", s_old["p95_ms"], s_new["p95_ms"], "ms")
    row("200,000 vectors, one session", "HNSW index size", s_old["index_bytes"], s_new["index_bytes"], "bytes")
    row("200,000 vectors, one session", "HNSW build", s_old["index_build_s"], s_new["index_build_s"], "s",
        note="the original build settings were not recorded")
    row("200,000 vectors, one session", "recall at 10 against exact", s_old["exact_recall_at_10"],
        s_new["exact_recall_at_10"], "", comparable=False,
        note="the original run's code was not kept; the noisy copies were rebuilt from its description, "
             "so the data set differs")
    r_old, r_new = old["scale-200k.json"]["range_scan"], new["scale-range-scan.json"]
    row("200,000 vectors, new psql per query", "top-10 median", r_old["pgvector_median_ms"], r_new["pgvector_median_ms"], "ms")
    row("200,000 vectors, new psql per query", "one hk1 prefix range",
        r_old["hilbert_one_range_ms"], r_new["hilbert_one_range_ms"], "ms",
        note=f"rows {r_old['hilbert_one_range_rows']} then {r_new['hilbert_one_range_rows']}; random keys differ per run")

    def by_setting(doc):
        return {r["setting"]: r for r in doc["results"]}
    va, vb = by_setting(old["arango-vector.json"]), by_setting(new["arango-vector.json"])
    for setting in vb:
        if setting not in va:
            continue
        engine = "ArangoDB 3.12.10 then 3.12.12" if setting.startswith("arango") else "pgvector"
        row(f"engine comparison, {engine}", f"{setting}, median", va[setting]["median_ms"], vb[setting]["median_ms"], "ms")
        row(f"engine comparison, {engine}", f"{setting}, exact top-10 found",
            va[setting]["chunk_recall_vs_exact_at_10"], vb[setting]["chunk_recall_vs_exact_at_10"], "")
    ha, hb = old["arango-hop.json"], new["arango-hop.json"]
    row("one-hop expansion", "AQL median", ha["arangodb_aql"]["median_ms"], hb["arangodb_aql"]["median_ms"], "ms")
    row("one-hop expansion", "SQL median", ha["postgres_sql"]["median_ms"], hb["postgres_sql"]["median_ms"], "ms")
    row("one-hop expansion", "identical neighbour sets", ha["identical_sets"], hb["identical_sets"], "starts")

    doc = {
        "pg16": {"server_version": "16.15", "pgvector": "0.8.6", "arangodb": old["arango-load.json"]["server"]["version"],
                 "embedded_chunks": old["manifest.json"]["chunk_count"]},
        "pg18": {**new["versions.json"], "arangodb": new["arango-load.json"]["server"]["version"]},
        "design": "replication on the upgraded server, compared with stored results; not a side-by-side test, "
                  "because PostgreSQL 16 was removed. The corpus grew between the runs and machine load differed.",
        "rows": rows,
    }
    (HERE / "comparison.json").write_text(json.dumps(doc, indent=2) + "\n")
    for r in rows:
        flag = "" if r["comparable"] else "  (not comparable)"
        print(f"{r['group']:<45} {r['measure']:<52} {r['pg16']!s:>14} -> {r['pg18']!s:<14} {r['unit']}{flag}")


if __name__ == "__main__":
    main()
