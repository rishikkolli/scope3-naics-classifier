"""S0b_tfidf_logreg_proba: the S0b baseline (B1) with its probabilities logged.

Same model and data as S0b in train_baseline.py (TF-IDF + logistic regression on gov_pool,
description + recipient -> NAICS-6). It also saves what the S4 confidence code reads:
the top-20 classes as `candidates` with `candidate_scores` (their probabilities) and the top
probability as `confidence`. This lets C2/C4, abstention, E4 and E5 run end to end before the
LLM stages exist. S0b stays the official B1 number.

Usage:  python scripts/train_baseline_proba.py
        python scripts/run_confidence_eval.py S0b_tfidf_logreg_proba
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

RUNS = ROOT / "eval" / "runs"
RUN_ID = "S0b_tfidf_logreg_proba"


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def text(df):
        return (df["description"] + " " + df["recipient"]).tolist()

    pool, test = load_gov("gov_pool"), load_gov("gov_test")
    t0 = time.time()
    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    clf.fit(text(pool), pool["naics6"])
    train_s = time.time() - t0
    proba = clf.predict_proba(text(test))
    classes = clf.classes_
    items = []
    for i, item_id in enumerate(test["item_id"]):
        top = np.argsort(-proba[i])[:20]
        items.append({"item_id": item_id, "naics6": str(classes[top[0]]),
                      "candidates": [str(classes[j]) for j in top],
                      "candidate_scores": [round(float(proba[i, j]), 6) for j in top],
                      "confidence": round(float(proba[i, top[0]]), 6)})
    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"run": RUN_ID, "n": len(items), "train_rows": len(pool), "train_seconds": round(train_s, 1),
           "items": items}
    (RUNS / f"{RUN_ID}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved eval/runs/{RUN_ID}.json ({len(items)} items, trained on {len(pool)} rows, {train_s:.0f}s). "
          f"Score it with scripts/run_confidence_eval.py {RUN_ID}")


if __name__ == "__main__":
    main()