#!/usr/bin/env python3
"""S9b: a 3-D Hilbert key over (area, type, month) as an index for attribute boxes.

Four tables hold the same rows, each with one access path and physically ordered by it: a B-tree
on the Hilbert index of the ordinals, a B-tree on the Z-order key, a composite B-tree on (area,
type, month), and three single-column B-trees (random heap order). Boxes fix an area, a type or
both and span 1 to 24 months. Curve keys cover a box with exact ranges merged smallest gap first
down to 64. Per box and path: shared buffers touched, median execution time of 20 runs, ranges,
rows fetched and returned. Runs at the 1,186 live notes and at 200,000 rows drawn from their
tuples, in a scratch database that is dropped afterwards. Writes s9b.json.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import psycopg
from psycopg.conninfo import make_conninfo

HERE = Path(__file__).resolve().parent
SEED = 20261009
BITS, DIMS = 8, 3
BOXES_PER_SHAPE = 60
MAX_RANGES = 64
RUNS = 20
SIZES = ("live", 200_000)
PAYLOAD = "x" * 200
SCRATCH = "s9b_scratch"
PATHS = ("hilbert", "zorder", "composite", "bitmap")
SHAPES = ("area_type_months", "area_months", "type_months")


def hilbert_index(X: np.ndarray, bits: int) -> np.ndarray:
    """Skilling's AxesToTranspose, then the transpose read out most significant bit first."""
    X = X.astype(np.int64).copy()
    n = X.shape[1]
    q = 1 << (bits - 1)
    while q > 1:
        p = q - 1
        for i in range(n):
            hi = (X[:, i] & q) != 0
            X[hi, 0] ^= p
            t = (X[:, 0] ^ X[:, i]) & p
            t[hi] = 0
            X[:, 0] ^= t
            X[:, i] ^= t
        q >>= 1
    for i in range(1, n):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(len(X), dtype=np.int64)
    q = 1 << (bits - 1)
    while q > 1:
        t[(X[:, n - 1] & q) != 0] ^= q - 1
        q >>= 1
    X ^= t[:, None]
    return interleave(X, bits)


def interleave(X: np.ndarray, bits: int) -> np.ndarray:
    h = np.zeros(len(X), dtype=np.int64)
    for j in range(bits - 1, -1, -1):
        for i in range(X.shape[1]):
            h = (h << 1) | ((X[:, i] >> j) & 1)
    return h


def check_hilbert() -> dict:
    b = 4
    g = np.array(np.meshgrid(*[np.arange(1 << b)] * DIMS, indexing="ij")).reshape(DIMS, -1).T
    h = hilbert_index(g, b)
    order = np.argsort(h)
    bijection = bool(np.array_equal(np.sort(h), np.arange(len(g))))
    steps = np.abs(np.diff(g[order], axis=0)).sum(axis=1)
    unit = bool((steps == 1).all())
    zsteps = np.abs(np.diff(g[np.argsort(interleave(g, b))], axis=0)).sum(axis=1)
    assert bijection and unit, "Hilbert transform failed its check"
    return {"grid": f"{1 << b}^{DIMS}", "bijection": bijection, "unit_steps": unit,
            "zorder_unit_steps_share": float((zsteps == 1).mean())}


def area(slug: str) -> str:
    parts = slug.removeprefix("obsidian/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 3 else parts[0]


def live_rows(url: str) -> tuple[np.ndarray, dict]:
    with psycopg.connect(url, autocommit=True, prepare_threshold=None) as c:
        rows = c.execute("select slug, type, effective_date from pages "
                         "where source_id = 'default' and deleted_at is null").fetchall()
    areas = sorted({area(s) for s, _, _ in rows})
    types = sorted({t for _, t, _ in rows})
    m0 = min(d.year * 12 + d.month - 1 for *_, d in rows)
    a_ix, t_ix = {a: i for i, a in enumerate(areas)}, {t: i for i, t in enumerate(types)}
    X = np.array([(a_ix[area(s)], t_ix[t], d.year * 12 + d.month - 1 - m0) for s, t, d in rows], dtype=np.int64)
    dom = {"areas": len(areas), "types": len(types), "months": int(X[:, 2].max()) + 1}
    assert max(dom.values()) <= 1 << BITS
    return X, dom


def draw_boxes(X: np.ndarray, dom: dict, rng: np.random.Generator) -> list[dict]:
    boxes = []
    for shape in SHAPES:
        for _ in range(BOXES_PER_SHAPE):
            a, t, m = X[rng.integers(len(X))]
            length = int(rng.integers(1, 25))
            m_hi = min(int(m) + length - 1, dom["months"] - 1)
            lo = [int(a), int(t), int(m)]
            hi = [int(a), int(t), m_hi]
            if shape == "area_months":
                lo[1], hi[1] = 0, dom["types"] - 1
            if shape == "type_months":
                lo[0], hi[0] = 0, dom["areas"] - 1
            boxes.append({"shape": shape, "lo": lo, "hi": hi, "months": m_hi - int(m) + 1})
    return boxes


def curve_ranges(box: dict, key) -> tuple[list[tuple[int, int]], int]:
    axes = [np.arange(l, h + 1) for l, h in zip(box["lo"], box["hi"])]
    cells = np.array(np.meshgrid(*axes, indexing="ij")).reshape(DIMS, -1).T
    k = np.sort(key(cells))
    breaks = np.flatnonzero(np.diff(k) > 1)
    starts = np.concatenate([[k[0]], k[breaks + 1]])
    ends = np.concatenate([k[breaks], [k[-1]]])
    exact = len(starts)
    if exact > MAX_RANGES:
        gaps = starts[1:] - ends[:-1]
        close = np.argsort(gaps, kind="stable")[: exact - MAX_RANGES]
        keep = np.ones(exact - 1, dtype=bool)
        keep[close] = False
        starts = np.concatenate([[starts[0]], starts[1:][keep]])
        ends = np.concatenate([ends[:-1][keep], [ends[-1]]])
    return [(int(s), int(e)) for s, e in zip(starts, ends)], exact


def attr_where(box: dict, dom: dict) -> str:
    (a0, t0, m0), (a1, t1, m1) = box["lo"], box["hi"]
    parts = []
    if a0 == a1:
        parts.append(f"area = {a0}")
    if t0 == t1:
        parts.append(f"type = {t0}")
    parts.append(f"month between {m0} and {m1}")
    return " and ".join(parts)


def build(c: psycopg.Connection, X: np.ndarray) -> None:
    h = hilbert_index(X, BITS)
    z = interleave(X, BITS)
    c.execute("drop table if exists base, t_hilbert, t_zorder, t_composite, t_bitmap")
    c.execute("create unlogged table base (id int primary key, area int2, type int2, month int2, "
              "h int4, z int4, payload text)")
    with c.cursor().copy("copy base (id, area, type, month, h, z, payload) from stdin") as cp:
        for i, ((a, t, m), hh, zz) in enumerate(zip(X.tolist(), h.tolist(), z.tolist())):
            cp.write_row((i, a, t, m, hh, zz, PAYLOAD))
    for name, cols in (("t_hilbert", "h"), ("t_zorder", "z"), ("t_composite", "area, type, month")):
        c.execute(f"create table {name} as select * from base")
        c.execute(f"create index {name}_ix on {name} ({cols})")
        c.execute(f"cluster {name} using {name}_ix")
    c.execute("create table t_bitmap as select * from base order by md5(id::text)")
    for col in ("area", "type", "month"):
        c.execute(f"create index t_bitmap_{col} on t_bitmap ({col})")
    c.execute("drop table base")
    c.execute("vacuum analyze")


def scan_counts(plan: dict) -> tuple[int, int]:
    """Rows the scan nodes fetched (kept plus removed by filter or recheck), and rows returned."""
    fetched = 0

    def walk(n: dict) -> None:
        nonlocal fetched
        if n.get("Relation Name"):
            loops = n.get("Actual Loops", 1)
            fetched += int(round((n.get("Actual Rows", 0) + n.get("Rows Removed by Filter", 0)
                                  + n.get("Rows Removed by Index Recheck", 0)) * loops))
        for ch in n.get("Plans", []):
            walk(ch)

    walk(plan)
    return fetched, int(plan["Actual Rows"])


def node_types(plan: dict) -> list[str]:
    out = [plan["Node Type"]]
    for ch in plan.get("Plans", []):
        out += node_types(ch)
    return out


def measure(c: psycopg.Connection, sql: str) -> dict:
    c.execute(sql).fetchall()
    first = None
    times, planning = [], []
    for r in range(RUNS):
        e = c.execute("explain (analyze, buffers, format json) " + sql).fetchone()[0][0]
        if first is None:
            first = e
        times.append(e["Execution Time"])
        planning.append(e["Planning Time"])
    p = first["Plan"]
    fetched, returned = scan_counts(p)
    return {"buffers": int(p.get("Shared Hit Blocks", 0) + p.get("Shared Read Blocks", 0)),
            "ms": statistics.median(times), "planning_ms": statistics.median(planning),
            "fetched": fetched, "returned": returned, "nodes": sorted(set(node_types(p)))}


def run_size(c: psycopg.Connection, X: np.ndarray, dom: dict, boxes: list[dict]) -> list[dict]:
    build(c, X)
    c.execute("set enable_seqscan = off")
    c.execute("set max_parallel_workers_per_gather = 0")
    keys = {"hilbert": lambda cells: hilbert_index(cells, BITS), "zorder": lambda cells: interleave(cells, BITS)}
    out = []
    for box in boxes:
        where = attr_where(box, dom)
        row = {"shape": box["shape"], "months": box["months"]}
        ids = {}
        for path in PATHS:
            rec = {}
            if path in keys:
                ranges, exact = curve_ranges(box, keys[path])
                col = "h" if path == "hilbert" else "z"
                cover = " or ".join(f"{col} between {lo} and {hi}" for lo, hi in ranges)
                sql = f"select id, payload from t_{path} where ({cover}) and {where}"
                rec.update(ranges=len(ranges), exact_ranges=exact)
            else:
                sql = f"select id, payload from t_{path} where {where}"
            rec.update(measure(c, sql))
            ids[path] = sorted(r[0] for r in c.execute(sql).fetchall())
            row[path] = rec
        assert all(ids[p] == ids["composite"] for p in PATHS), f"paths disagree on {box}"
        out.append(row)
    return out


def summarise(rows: list[dict]) -> dict:
    s = {}
    for shape in SHAPES:
        rs = [r for r in rows if r["shape"] == shape]
        s[shape] = {"boxes": len(rs), "rows_returned_median": float(np.median([r["composite"]["returned"] for r in rs]))}
        for path in PATHS:
            v = [r[path] for r in rs]
            d = {"buffers_median": float(np.median([x["buffers"] for x in v])),
                 "buffers_mean": float(np.mean([x["buffers"] for x in v])),
                 "ms_median": float(np.median([x["ms"] for x in v])),
                 "planning_ms_median": float(np.median([x["planning_ms"] for x in v])),
                 "fetched_over_returned": float(sum(x["fetched"] for x in v) / max(1, sum(x["returned"] for x in v))),
                 "plans": sorted({" + ".join(x["nodes"]) for x in v})}
            if path in ("hilbert", "zorder"):
                d["ranges_median"] = float(np.median([x["ranges"] for x in v]))
                d["exact_ranges_median"] = float(np.median([x["exact_ranges"] for x in v]))
                d["capped_share"] = float(np.mean([x["exact_ranges"] > MAX_RANGES for x in v]))
            s[shape][path] = d
        others = [p for p in PATHS if p != "hilbert"]
        best = min(others, key=lambda p: s[shape][p]["buffers_median"])
        hb, ob = s[shape]["hilbert"]["buffers_median"], s[shape][best]["buffers_median"]
        s[shape]["acceptance"] = {
            "best_other": best, "buffer_ratio": hb / ob if ob else None,
            "time_ratio": s[shape]["hilbert"]["ms_median"] / s[shape][best]["ms_median"],
            "hilbert_wins": bool(ob and hb <= 0.75 * ob and s[shape]["hilbert"]["ms_median"] <= 2 * s[shape][best]["ms_median"]),
            "hilbert_vs_zorder_buffer_ratio": hb / s[shape]["zorder"]["buffers_median"]}
    return s


def main() -> None:
    url = json.loads(Path(os.path.expanduser("~/.gbrain/config.json")).read_text())["database_url"]
    check = check_hilbert()
    X_live, dom = live_rows(url)
    rng = np.random.default_rng(SEED)
    boxes = draw_boxes(X_live, dom, rng)
    X_big = X_live[rng.integers(len(X_live), size=SIZES[1])]
    with psycopg.connect(url, autocommit=True, prepare_threshold=None) as admin:
        version = admin.execute("show server_version").fetchone()[0]
        admin.execute(f"drop database if exists {SCRATCH}")
        admin.execute(f"create database {SCRATCH}")
    results = {}
    try:
        with psycopg.connect(make_conninfo(url, dbname=SCRATCH), autocommit=True, prepare_threshold=None) as c:
            for size, X in (("live", X_live), (str(SIZES[1]), X_big)):
                rows = run_size(c, X, dom, boxes)
                results[size] = {"rows": len(X), "summary": summarise(rows), "per_box": rows}
                print(size, json.dumps({sh: results[size]["summary"][sh]["acceptance"] for sh in SHAPES}))
    finally:
        with psycopg.connect(url, autocommit=True, prepare_threshold=None) as admin:
            admin.execute(f"drop database if exists {SCRATCH}")
    out = {"preregistration": "preregistration.md#addendum-4", "seed": SEED, "server_version": version,
           "hilbert_check": check, "domain": dom, "bits_per_axis": BITS, "max_ranges": MAX_RANGES, "runs": RUNS,
           "boxes_per_shape": BOXES_PER_SHAPE, "settings": {"enable_seqscan": "off", "max_parallel_workers_per_gather": 0},
           "ordinals": "areas and types in name order, months counted from the earliest note's month",
           "results": results}
    (HERE / "s9b.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
