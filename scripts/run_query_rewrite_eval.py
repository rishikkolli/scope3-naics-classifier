"""R5: does LLM query rewriting help? (design.md section 2, priority R5)

`src/classify.py:llm_rewrite` has been implemented since the first LLM commit but no
script ever called it, so R5 was the one technique in the plan with no result either way.
Milestone 3 tested a free deterministic stand-in (strip boilerplate and part numbers,
`scripts/run_baseline_ablations.py` variant A6) which changed six-digit accuracy by
nothing; that said little about an LLM rewrite, which can *resolve* a part number into a
product noun rather than delete it.

This script rewrites every gov_test description once with the LLM, caches the rewrites,
and then measures everything downstream locally and for free:

  1. supervised accuracy on rewritten queries (the pool keeps its original text -- R5 is
     query rewriting, so only the query changes)
  2. few-shot neighbour recall: does the gold code appear among the k nearest pool rows
  3. candidate recall@k from the logreg generator -- the ceiling S1-S3 inherit

Re-running is free once `eval/runs/R5_rewrites.json` exists; delete it to pay again.

Usage:  python scripts/run_query_rewrite_eval.py [--limit N] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_gov                                      # noqa: E402
from src.emissions import FactorLookup, to_2022_usd                # noqa: E402
from src.evaluate import emissions_error, hierarchical_accuracy    # noqa: E402

RUNS = ROOT / "eval" / "runs"
CACHE = RUNS / "R5_rewrites.json"
FY_YEAR = 2024


# --------------------------------------------------------------- 1. the rewrites
def collect_rewrites(test, force: bool) -> tuple[list[str], dict]:
    """One LLM call per row, cached. Returns (rewrites, usage)."""
    from src import classify

    cached = json.loads(CACHE.read_text()) if CACHE.exists() and not force else {}
    rewrites, usage = [], {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                           "cached": 0, "failures": 0}

    # instrument the shared completion helper so this run reports real token cost,
    # without changing llm_rewrite's signature for its other caller (retrieval.rewrite_query)
    original = classify._complete

    def counting(*a, **kw):
        texts, tin, tout = original(*a, **kw)
        usage["calls"] += 1
        usage["tokens_in"] += tin
        usage["tokens_out"] += tout
        return texts, tin, tout

    classify._complete = counting
    t0 = time.time()
    try:
        for i, row in enumerate(test.itertuples()):
            key = f"{i}"
            if key in cached:
                rewrites.append(cached[key])
                usage["cached"] += 1
                continue
            try:
                out = classify.llm_rewrite(row.description, row.recipient)
            except Exception as e:                      # keep the run going; log the gap
                print(f"  row {i} failed ({type(e).__name__}); keeping original text")
                usage["failures"] += 1
                out = row.description
            rewrites.append(out)
            cached[key] = out
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(test)}  ${usage['cost_usd']:.4f} so far", flush=True)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cached, indent=1))
            usage["cost_usd"] = (usage["tokens_in"] * classify._PRICE_IN
                                 + usage["tokens_out"] * classify._PRICE_OUT) / 1e6
    finally:
        classify._complete = original
    usage["wall_seconds"] = round(time.time() - t0, 1)
    usage["model"] = classify.MODEL
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cached, indent=1))
    return rewrites, usage


# --------------------------------------------------------------- 2. scoring
def word_tfidf():
    return TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))


def score(gold, pred, test, lookup) -> dict:
    m = {f"acc@{d}": round(hierarchical_accuracy(gold, pred, d), 4) for d in (2, 4, 6)}
    groups = {c: lookup.group(c) for c in set(gold) | set(pred)}
    pairs = [(g, p) for g, p in zip(gold, pred) if groups.get(g) is not None]
    m["factor_equiv_acc"] = round(float(np.mean([groups[g] == groups.get(p) for g, p in pairs])), 4)
    amt = np.array([to_2022_usd(a, FY_YEAR) for a in test["amount"]])
    fg = np.array([lookup.factor(g) if lookup.factor(g) is not None else np.nan for g in gold])
    fp = np.array([lookup.factor(p) if lookup.factor(p) is not None else 0.0 for p in pred])
    ok = ~np.isnan(fg)
    m["emissions_rel_error"] = round(
        emissions_error(amt[ok] * fg[ok], amt[ok] * fp[ok])["relative"], 4)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ignore the cache and pay again")
    args = ap.parse_args()

    pool, test = load_gov("gov_pool"), load_gov("gov_test")
    if args.limit:
        test = test.head(args.limit)
    lookup = FactorLookup(ROOT / "data" / "raw")
    gold = test["naics6"].tolist()

    print(f"rewriting {len(test)} descriptions ...")
    rewrites, usage = collect_rewrites(test, args.force)
    print(f"done: {usage['calls']} calls, {usage['cached']} from cache, "
          f"{usage['failures']} failures, ${usage['cost_usd']:.4f}, {usage['wall_seconds']}s")

    orig_d = test["description"].astype(str).tolist()
    recip = test["recipient"].astype(str).tolist()
    pool_text = (pool["description"].astype(str) + " " + pool["recipient"].astype(str)).tolist()
    pool_codes = pool["naics6"].to_numpy()

    # ---------------- supervised accuracy, original vs rewritten query
    clf = make_pipeline(word_tfidf(), LogisticRegression(max_iter=2000, C=4.0))
    clf.fit(pool_text, pool["naics6"])
    variants = {
        "original description + supplier (= S0b)": [f"{d} {r}" for d, r in zip(orig_d, recip)],
        "rewritten description + supplier": [f"{d} {r}" for d, r in zip(rewrites, recip)],
        "rewritten description only": list(rewrites),
        "original description only": list(orig_d),
    }
    results = {}
    for name, texts in variants.items():
        results[name] = score(gold, list(clf.predict(texts)), test, lookup)
        print(f"  {name:42s} acc@6={results[name]['acc@6']:.4f} acc@2={results[name]['acc@2']:.4f}")

    # ---------------- is the drop significant? paired bootstrap, same items
    rng = np.random.default_rng(0)
    base = np.array(list(clf.predict(variants["original description + supplier (= S0b)"]))) == np.array(gold)
    rw = np.array(list(clf.predict(variants["rewritten description + supplier"]))) == np.array(gold)
    d = rw.astype(float) - base.astype(float)
    samp = d[rng.integers(0, len(d), size=(5000, len(d)))].mean(axis=1)
    boot = {"delta_acc6": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(samp, 2.5)), 4),
                     round(float(np.percentile(samp, 97.5)), 4)],
            "p_two_sided": round(float(2 * min((samp <= 0).mean(), (samp >= 0).mean())), 4),
            "wins": int((rw & ~base).sum()), "losses": int((~rw & base).sum())}
    print(f"  rewritten minus original, acc@6: {boot['delta_acc6']:+.4f} {boot['ci95']} "
          f"p={boot['p_two_sided']} ({boot['wins']}W/{boot['losses']}L)")

    # ---------------- few-shot neighbour recall and candidate recall@k
    vec = word_tfidf()
    X = vec.fit_transform(pool_text)
    queries = {
        "original": [f"{d} (supplier: {r})" for d, r in zip(orig_d, recip)],
        "rewritten": [f"{d} (supplier: {r})" for d, r in zip(rewrites, recip)],
    }
    fewshot = {}
    for name, q in queries.items():
        nn = np.argsort(-linear_kernel(vec.transform(q), X), axis=1)
        fewshot[name] = {k: round(float(np.mean(
            [gold[i] in set(pool_codes[nn[i, :k]]) for i in range(len(gold))])), 4)
            for k in (1, 3, 5, 10, 20)}
        print(f"  few-shot recall ({name}): {fewshot[name]}")

    cand = {}
    cand_texts = {"original": [f"{d} {r}" for d, r in zip(orig_d, recip)],
                  "rewritten": [f"{d} {r}" for d, r in zip(rewrites, recip)]}
    for name, texts in cand_texts.items():
        P = clf[-1].predict_proba(clf[0].transform(texts))
        ranked = clf[-1].classes_[np.argsort(-P, axis=1)]
        cand[name] = {k: round(float(np.mean(
            [gold[i] in set(ranked[i, :k]) for i in range(len(gold))])), 4)
            for k in (1, 3, 5, 10, 20, 50)}
        print(f"  candidate recall ({name}): {cand[name]}")

    # ---------------- what the rewrites did to the text
    shorter = sum(1 for a, b in zip(orig_d, rewrites) if len(b) < len(a))
    text_stats = {
        "mean_chars_original": round(float(np.mean([len(d) for d in orig_d])), 1),
        "mean_chars_rewritten": round(float(np.mean([len(d) for d in rewrites])), 1),
        "mean_tokens_original": round(float(np.mean([len(d.split()) for d in orig_d])), 2),
        "mean_tokens_rewritten": round(float(np.mean([len(d.split()) for d in rewrites])), 2),
        "share_shorter": round(shorter / len(orig_d), 4),
        "share_unchanged": round(sum(1 for a, b in zip(orig_d, rewrites) if a.strip() == b.strip())
                                 / len(orig_d), 4),
    }
    print(f"  text: {text_stats}")

    out = {"n": len(test), "usage": usage, "supervised": results, "bootstrap_vs_original": boot,
           "fewshot_neighbour_recall": fewshot, "candidate_recall_at_k": cand,
           "text_stats": text_stats,
           "examples": [{"original": o, "rewritten": w, "gold": g}
                        for o, w, g in list(zip(orig_d, rewrites, gold))[:15]]}
    (RUNS / "R5_query_rewrite.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RUNS / 'R5_query_rewrite.json'}")


if __name__ == "__main__":
    main()
