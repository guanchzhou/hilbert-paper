#!/usr/bin/env python3
"""Evidence units under different token budgets.

The paper measures the four evidence units (chunk, window, section, whole note) at one budget,
6,000 tokens, which is gbrain's DEFAULT_RETURN_BUDGET. This repeats the same packing at
1,000, 2,000, 4,000, 6,000, 12,000 and 24,000 tokens with the functions of run_measure.py:
the dense chunk ranking of all 817 development questions, the top 10 notes walked in rank order,
each unit kept whole if it fits in the remaining budget. Read-only; writes budget-sweep.json.
"""

import json
from pathlib import Path

import numpy as np

import run_measure as rm

BENCH = Path(__file__).resolve().parent
BUDGETS = [1000, 2000, 4000, 6000, 12000, 24000]


def main() -> None:
    env, password = rm.pg_env()
    qrels = json.loads(rm.QRELS.read_text())
    vectors = np.load(BENCH / "query-vectors.npy").astype(np.float32)
    qnorm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    truth, _ids, slugs, sources, texts, indexes, matrix = rm.load_corpus(env, password)
    _pl, _ps, _cl, _cs, winners, by_page, pos_of = rm.vector_rankings(qnorm, matrix, slugs, sources, texts, indexes)
    rows = []
    for budget in BUDGETS:
        rm.BUDGET = budget
        packed = rm.pack_units(qrels, winners, by_page, pos_of, truth)
        for unit, rec in packed.items():
            rows.append({"budget": budget, "unit": unit, "relevant_in_pack": rec["relevant_in_pack"],
                         "tokens_delivered": rec["tokens_delivered"], "hits_dropped": rec["hits_dropped"]})
            print(budget, unit, rec["relevant_in_pack"], rec["tokens_delivered"], flush=True)
    doc = {"budgets": BUDGETS, "questions": len(qrels), "chunks": int(matrix.shape[0]),
           "default_budget_source": "gbrain src/core/search/evidence-delivery.ts DEFAULT_RETURN_BUDGET = 6000",
           "pack": "top 10 notes of the dense chunk ranking, units kept whole if they fit, skipped otherwise",
           "rows": rows}
    (BENCH / "budget-sweep.json").write_text(json.dumps(doc, indent=2) + "\n")


if __name__ == "__main__":
    main()
