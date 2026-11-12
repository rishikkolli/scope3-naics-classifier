"""S0b: train the supervised baseline (B1) and score it with full metrics.

TF-IDF + logistic regression on gov_pool (description + recipient -> NAICS-6),
evaluated on gov_test: accuracy at 2/4/6 digits, factor-equivalent accuracy, and
emissions error. This is the bar the generative pipeline must beat (design.md Stages).

Also scores S0a (B0): retrieval-only top-1 from the R-series run files, if present.

Usage:  python scripts/train_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_gov                      # noqa: E402
from src.emissions import FactorLookup, to_2022_usd  # noqa: E402
from src.evaluate import emissions_error, hierarchical_accuracy  # noqa: E402

RUNS = ROOT / "eval" / "runs"
FY_YEAR = 2024   # award amounts are nominal FY2024 dollars


def score(run_id: str, test, pred: list[str], lookup: FactorLookup, extra: dict | None = None) -> dict:
    gold = test["naics6"].tolist()
    m = {f"acc@{d}": round(hierarchical_accuracy(gold, pred, d), 4) for d in (2, 4, 6)}
    groups = {c: lookup.group(c) for c in set(gold) | set(pred)}
    pairs = [(g, p) for g, p in zip(gold, pred) if groups.get(g) is not None]
    m["factor_equiv_acc"] = round(float(np.mean(
        [groups[g] == groups.get(p) for g, p in pairs])), 4)

    amt22 = np.array([to_2022_usd(a, FY_YEAR) for a in test["amount"]])
    fg = np.array([lookup.factor(g) or np.nan for g in gold])
    fp = np.array([lookup.factor(p) or np.nan for p in pred])
    ok = ~np.isnan(fg)
    m["gold_factor_coverage"] = round(float(ok.mean()), 4)
    fp_filled = np.where(np.isnan(fp), 0.0, fp)   # unmappable prediction = 0 kg estimated
    em = emissions_error(amt22[ok] * fg[ok], amt22[ok] * fp_filled[ok])
    m["emissions_rel_error"] = round(em["relative"], 4)
    m["emissions_net_bias_rel"] = round(em["net_bias_kg"] / max((amt22[ok] * fg[ok]).sum(), 1e-9), 4)
    per_item = np.abs(fp_filled[ok] - fg[ok]) / fg[ok]
    m["median_item_factor_rel_error"] = round(float(np.median(per_item)), 4)

    print(f"{run_id:22s} " + "  ".join(f"{k}={v}" for k, v in m.items()))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"run": run_id, "n": len(gold), "metrics": m, **(extra or {}),
           "items": [{"item_id": i, "gold": g, "pred": p}
                     for i, g, p in zip(test["item_id"], gold, pred)]}
    (RUNS / f"{run_id}.json").write_text(json.dumps(out, indent=1))
    return m


def s0b_tfidf_logreg(pool, test, lookup: FactorLookup) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def text(df):
        return (df["description"] + " " + df["recipient"]).tolist()

    t0 = time.time()
    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1),
    )
    clf.fit(text(pool), pool["naics6"])
    train_s = time.time() - t0
    print(f"B1 trained on {len(pool)} rows, {len(set(pool['naics6']))} classes, {train_s:.0f}s")
    pred = list(clf.predict(text(test)))
    score("S0b_tfidf_logreg", test, pred, lookup,
          {"train_rows": len(pool), "train_seconds": round(train_s, 1)})


def s0a_retrieval_top1(test, lookup: FactorLookup) -> None:
    """B0 from the best completed R-series run file."""
    # frozen retriever = R2 dense: best recall@1 (0.150 vs hybrid 0.114) and best @50
    for run in ["R2_dense_e5", "R4_hybrid_rrf", "R1_bm25_all_chunks"]:
        f = RUNS / f"{run}.json"
        if f.exists():
            data = json.loads(f.read_text())
            top1 = {it["item_id"]: (it["top20"][0] if it["top20"] else "")
                    for it in data["items"]}
            pred = [top1.get(i, "") for i in test["item_id"]]
            score("S0a_retrieval_top1", test, pred, lookup, {"retriever_run": run})
            return
    print("S0a skipped: no R-series run file yet (run scripts/run_retrieval_eval.py first)")


def main() -> None:
    pool, test = load_gov("gov_pool"), load_gov("gov_test")
    lookup = FactorLookup(ROOT / "data" / "raw")
    s0a_retrieval_top1(test, lookup)
    s0b_tfidf_logreg(pool, test, lookup)


if __name__ == "__main__":
    main()
