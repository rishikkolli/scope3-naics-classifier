"""R-series retrieval ablations on gov_test, in priority order (design.md section 2).

  R1 bm25           BM25 over all chunk types
  R2 dense          e5 with query/passage prefixes
  R3 (chunk study)  R1/R2 recall with definitions only vs. + index phrases
  R4 hybrid         reciprocal rank fusion of R1 + R2

Also runs E1: cosine-similarity distributions for gold vs. top non-gold code.
Writes eval/runs/R<i>_<variant>.json with config, per-item ranked codes and recall@k.

Usage:  python scripts/run_retrieval_eval.py [--variants bm25 dense hybrid] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_gov                              # noqa: E402
from src.naics_corpus import build_chunks                  # noqa: E402
from src.retrieval import (BM25Retriever, DenseRetriever,  # noqa: E402
                           HybridRetriever, recall_at_k)

KS = [1, 5, 10, 20, 50]
RUNS = ROOT / "eval" / "runs"


def query_text(row) -> str:
    return f"{row.description} (supplier: {row.recipient})".strip()


def evaluate(name: str, ranked: list[list[str]], gold: list[str], items: list[str],
             extra: dict | None = None) -> dict:
    rec = {f"recall@{k}": round(recall_at_k(gold, ranked, k), 4) for k in KS}
    print(f"{name:28s} " + "  ".join(f"R@{k}={rec[f'recall@{k}']:.3f}" for k in KS))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"run": name, "n": len(gold), "metrics": rec, **(extra or {}),
           "items": [{"item_id": i, "gold": g, "top20": r[:20]}
                     for i, g, r in zip(items, gold, ranked)]}
    (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1))
    return rec


def e1_similarity_study(dense: DenseRetriever, queries: list[str], gold: list[str]) -> dict:
    """Cosine of best gold chunk vs. best non-gold chunk, per query (feedback item 2)."""
    q = dense.encode_queries(queries)
    sims = q @ dense.emb.T                                   # n_queries x n_chunks
    codes = np.array([c.naics6 for c in dense.chunks])
    gold_best, other_best = [], []
    for i, g in enumerate(gold):
        m = codes == g
        if m.any():
            gold_best.append(float(sims[i, m].max()))
            other_best.append(float(sims[i, ~m].max()))
    gb, ob = np.array(gold_best), np.array(other_best)
    stats = {
        "n": len(gb),
        "gold_best_cosine": {"mean": round(gb.mean(), 4), "p10": round(np.percentile(gb, 10), 4),
                             "p90": round(np.percentile(gb, 90), 4)},
        "top_nongold_cosine": {"mean": round(ob.mean(), 4), "p10": round(np.percentile(ob, 10), 4),
                               "p90": round(np.percentile(ob, 90), 4)},
        "margin_mean": round(float((gb - ob).mean()), 4),
        "gold_above_nongold_share": round(float((gb > ob).mean()), 4),
    }
    (RUNS / "E1_similarity.json").write_text(json.dumps(stats, indent=1))
    print("E1 similarity:", json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--limit", type=int, default=None, help="cap test items (smoke test)")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    test = load_gov("gov_test")
    if args.limit:
        test = test.head(args.limit)
    gold = test["naics6"].tolist()
    items = test["item_id"].tolist()
    queries = [query_text(r) for r in test.itertuples()]
    print(f"gov_test: {len(test)} items")

    chunks_full = build_chunks(ROOT / "data" / "raw")
    chunks_def = [c for c in chunks_full if c.kind in ("title", "definition")]
    print(f"chunks: {len(chunks_full)} (defs only: {len(chunks_def)})")

    results: dict[str, dict] = {}
    bm25 = dense = None

    if "bm25" in args.variants:
        t0 = time.time()
        # R3 chunk study, lexical side: definitions only vs + index phrases
        r = BM25Retriever(chunks_def)
        ranked = [[c.naics6 for c in r.search(q, args.k)] for q in queries]
        results["R1a"] = evaluate("R1a_bm25_defs_only", ranked, gold, items)
        bm25 = BM25Retriever(chunks_full)
        ranked = [[c.naics6 for c in bm25.search(q, args.k)] for q in queries]
        results["R1"] = evaluate("R1_bm25_all_chunks", ranked, gold, items)
        print(f"  bm25 took {time.time() - t0:.0f}s")

    if "dense" in args.variants or "hybrid" in args.variants:
        t0 = time.time()
        dense = DenseRetriever(chunks_full, cache_dir=ROOT / "data" / "index")
        q_embs = dense.encode_queries(queries)
        ranked = [[c.naics6 for c in dense.search_encoded(e, args.k)] for e in q_embs]
        results["R2"] = evaluate("R2_dense_e5", ranked, gold, items)
        print(f"  dense took {time.time() - t0:.0f}s")
        e1_similarity_study(dense, queries, gold)

    if "hybrid" in args.variants and bm25 is not None and dense is not None:
        hybrid = HybridRetriever(bm25, dense)
        ranked = [[c.naics6 for c in hybrid.search(q, args.k, q_emb=e)]
                  for q, e in zip(queries, q_embs)]
        results["R4"] = evaluate("R4_hybrid_rrf", ranked, gold, items)

    print("\nDone. Full per-item outputs in eval/runs/.")


if __name__ == "__main__":
    main()
