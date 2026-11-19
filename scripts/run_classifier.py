"""S-series LLM runs, one change per run (design.md Stages).

  S0c zero_shot    description -> NAICS-6 directly
  S1  rag          + choose among top-k from the frozen retriever
  S2  rag_fewshot  + nearest labeled neighbors from gov_pool
  S3  rag_fewshot --with-context   + amount & recipient in the prompt

Needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env (see src/classify.py). Writes eval/runs/<stage>.json with per-item predictions,
samples (for self-consistency), verbal confidence, cost and latency.

Usage:  python scripts/run_classifier.py --stage S1 [--limit 50] [--k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import classify                                   # noqa: E402
from src.data import load_gov                              # noqa: E402
from src.naics_corpus import build_chunks                  # noqa: E402
from src.retrieval import (BM25Retriever, DenseRetriever, HybridRetriever,  # noqa: E402
                           LogregRetriever)

RUNS = ROOT / "eval" / "runs"


def build_retriever(kind: str = "hybrid"):
    chunks = build_chunks(ROOT / "data" / "raw")
    if kind == "bm25":
        return BM25Retriever(chunks)
    if kind == "logreg":
        # same query text the LLM loop builds: "<description> (supplier: <recipient>)"
        pool = load_gov("gov_pool")
        texts = (pool["description"] + " (supplier: " + pool["recipient"] + ")").tolist()
        return LogregRetriever(chunks, texts, pool["naics6"].tolist())
    return HybridRetriever(BM25Retriever(chunks),
                           DenseRetriever(chunks, cache_dir=ROOT / "data" / "index"))


def fewshot_index(pool):
    """Nearest labeled neighbors by TF-IDF cosine (cheap, no extra embedding run)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    texts = (pool["description"] + " " + pool["recipient"]).tolist()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X)
    def neighbors(query: str):
        _, idx = nn.kneighbors(vec.transform([query]))
        return [(pool["description"].iloc[i][:120], pool["naics6"].iloc[i]) for i in idx[0]]
    return neighbors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["S0c", "S1", "S2", "S3"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--retriever", choices=["hybrid", "bm25", "logreg"], default="hybrid",
                    help="candidate generator for S1-S3: hybrid (BM25+e5), bm25 (no "
                         "embedding model), logreg (supervised top-k from gov_pool)")
    ap.add_argument("--run-id", default=None, help="output name (default: the stage)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        sys.exit("No API key: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env "
                 "(copy .env.example to .env)")

    test = load_gov("gov_test")
    if args.limit:
        test = test.head(args.limit)

    retr = build_retriever(args.retriever) if args.stage != "S0c" else None
    shots = fewshot_index(load_gov("gov_pool")) if args.stage in ("S2", "S3") else None

    preds, t0 = [], time.time()
    for row in test.itertuples():
        query = f"{row.description} (supplier: {row.recipient})"
        ctx = {"merchant": row.recipient} | ({"amount": row.amount} if args.stage == "S3" else {})
        if args.stage == "S0c":
            p = classify.zero_shot(row.description, item_id=row.item_id,
                                   n_samples=args.n_samples, **ctx)
        else:
            cands = retr.search(query, args.k)
            if args.stage == "S1":
                p = classify.rag(row.description, cands, item_id=row.item_id,
                                 n_samples=args.n_samples, **ctx)
            else:
                p = classify.rag_fewshot(row.description, cands, shots(query),
                                         item_id=row.item_id, n_samples=args.n_samples, **ctx)
        preds.append(p)
        if len(preds) % 25 == 0:
            print(f"  {len(preds)}/{len(test)}  cost so far ${sum(x.cost_usd for x in preds):.2f}")

    RUNS.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or args.stage
    out = {"run": run_id, "stage": args.stage, "retriever": args.retriever, "n": len(preds), "k": args.k, "provider": classify.PROVIDER, "model": classify.MODEL,
           "n_samples": args.n_samples, "wall_seconds": round(time.time() - t0, 1),
           "cost_usd": round(sum(p.cost_usd for p in preds), 4),
           "items": [vars(p) for p in preds]}
    (RUNS / f"{run_id}.json").write_text(json.dumps(out, indent=1))
    print(f"saved eval/runs/{run_id}.json  (${out['cost_usd']:.2f}, "
          f"{out['wall_seconds']:.0f}s). Score it with scripts/score_run.py {run_id}")


if __name__ == "__main__":
    main()
