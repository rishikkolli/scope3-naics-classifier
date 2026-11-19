"""S0b ablations + few-shot and candidate-recall diagnostics (Simranjeet: baselines,
few-shot examples, query rewriting).

Everything here runs locally with no API calls, so it can be re-run for free:

  A. Feature / model ablations on the supervised baseline (S0b).
  B. Candidate recall@k of the logreg candidate generator -- the ceiling the LLM
     stages S1-S3 inherit (design.md: "the retriever is frozen before S1").
  C. Few-shot neighbour quality: how often the k nearest gov_pool rows carry the
     gold code. This is the headroom of the S2 few-shot block.
  D. Rule-based query normalization (a free, deterministic stand-in for the R5 LLM
     rewrite in src/classify.py:llm_rewrite, which is implemented but unbudgeted).

Usage:  python scripts/run_baseline_ablations.py
Writes: eval/runs/ablations_s0b.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_gov                        # noqa: E402
from src.emissions import FactorLookup, to_2022_usd  # noqa: E402
from src.evaluate import emissions_error, hierarchical_accuracy  # noqa: E402

RUNS = ROOT / "eval" / "runs"
FY_YEAR = 2024

# ---------------------------------------------------------------- D. query normalization
# Federal descriptions carry contracting boilerplate and bare part numbers that say
# nothing about the product. These are the two patterns error analysis flagged in the
# S0c zero-shot failures ("IGF::OT::IGF", "8503119313!SPACER,SLEEVE").
_BOILER = re.compile(r"\bIGF::[A-Z]{2}::IGF\b|::", re.I)
_PARTNO = re.compile(r"\b(?=\w*\d)(?=\w*[A-Z])[A-Z0-9][A-Z0-9\-/]{5,}\b")
_DELIVERY = re.compile(r"\b(delivery order|task order|idiq|bpa call|modification|mod\s*\d+)\b", re.I)


def normalize(text: str) -> str:
    """R5-lite: strip contracting boilerplate and bare part numbers, keep the nouns."""
    t = _BOILER.sub(" ", text)
    t = _DELIVERY.sub(" ", t)
    t = _PARTNO.sub(" ", t)
    t = re.sub(r"[!*|;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------- scoring
def metrics(gold, pred, test, lookup) -> dict:
    m = {f"acc@{d}": round(hierarchical_accuracy(gold, pred, d), 4) for d in (2, 4, 6)}
    groups = {c: lookup.group(c) for c in set(gold) | set(pred)}
    pairs = [(g, p) for g, p in zip(gold, pred) if groups.get(g) is not None]
    m["factor_equiv_acc"] = round(float(np.mean([groups[g] == groups.get(p) for g, p in pairs])), 4)

    amt = np.array([to_2022_usd(a, FY_YEAR) for a in test["amount"]])
    fg = np.array([lookup.factor(g) if lookup.factor(g) is not None else np.nan for g in gold])
    fp = np.array([lookup.factor(p) if lookup.factor(p) is not None else np.nan for p in pred])
    ok = ~np.isnan(fg)
    fp = np.where(np.isnan(fp), 0.0, fp)
    true_kg, pred_kg = amt[ok] * fg[ok], amt[ok] * fp[ok]
    m["emissions_rel_error"] = round(emissions_error(true_kg, pred_kg)["relative"], 4)

    # the single worst item dominates; report the error with it removed
    err = np.abs(pred_kg - true_kg)
    worst = int(np.argmax(err))
    keep = np.ones(len(err), bool)
    keep[worst] = False
    m["emissions_rel_error_excl_worst"] = round(
        emissions_error(true_kg[keep], pred_kg[keep])["relative"], 4)
    m["worst_item_row"] = int(np.flatnonzero(ok)[worst])
    m["worst_item_share"] = round(float(err[worst] / max(err.sum(), 1e-9)), 4)
    return m


def text_of(df, fields=("description", "recipient"), norm=False, amount_bucket=False):
    parts = []
    for f in fields:
        col = df[f].astype(str)
        parts.append(col.map(normalize) if norm else col)
    out = parts[0]
    for p in parts[1:]:
        out = out + " " + p
    if amount_bucket:
        buckets = np.floor(np.log10(np.maximum(df["amount"].to_numpy(dtype=float), 1.0)))
        out = out + " amtbucket" + buckets.astype(int).astype(str)
    return out.tolist()


def word_tfidf():
    return TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))


def char_tfidf():
    return TfidfVectorizer(sublinear_tf=True, min_df=2, analyzer="char_wb", ngram_range=(3, 5))


def logreg():
    return LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)


VARIANTS = {
    "A1_desc_only": dict(fields=("description",), vec=word_tfidf, clf=logreg),
    "A2_desc+recipient (= S0b)": dict(fields=("description", "recipient"), vec=word_tfidf, clf=logreg),
    "A3_desc+recipient, char 3-5": dict(fields=("description", "recipient"), vec=char_tfidf, clf=logreg),
    "A4_desc+recipient, word+char": dict(
        fields=("description", "recipient"),
        vec=lambda: FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())]), clf=logreg),
    "A5_+amount bucket": dict(fields=("description", "recipient"), vec=word_tfidf, clf=logreg,
                              amount_bucket=True),
    "A6_normalized text (R5-lite)": dict(fields=("description", "recipient"), vec=word_tfidf,
                                         clf=logreg, norm=True),
    "A7_LinearSVC": dict(fields=("description", "recipient"), vec=word_tfidf,
                         clf=lambda: LinearSVC(C=1.0)),
    "A8_kNN cosine (k=5)": dict(fields=("description", "recipient"), vec=word_tfidf,
                                clf=lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine")),
}


def main() -> None:
    pool, test = load_gov("gov_pool"), load_gov("gov_test")
    lookup = FactorLookup(ROOT / "data" / "raw")
    gold = test["naics6"].tolist()
    print(f"pool={len(pool)} rows / {pool['naics6'].nunique()} codes   test={len(test)} rows")

    results, probs_ref = {}, None
    for name, cfg in VARIANTS.items():
        kw = dict(fields=cfg["fields"], norm=cfg.get("norm", False),
                  amount_bucket=cfg.get("amount_bucket", False))
        Xtr, Xte = text_of(pool, **kw), text_of(test, **kw)
        t0 = time.time()
        clf = make_pipeline(cfg["vec"](), cfg["clf"]())
        clf.fit(Xtr, pool["naics6"])
        pred = list(clf.predict(Xte))
        m = metrics(gold, pred, test, lookup)
        m["fit_seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"{name:32s} acc@6={m['acc@6']:.4f} acc@2={m['acc@2']:.4f} "
              f"fe={m['factor_equiv_acc']:.4f} em={m['emissions_rel_error']:.4f} "
              f"({m['fit_seconds']}s)")
        if name.startswith("A2") and hasattr(clf[-1], "predict_proba"):
            probs_ref = (clf[-1].predict_proba(clf[0].transform(Xte)), clf[-1].classes_)

    # ------------------------------------------------- B. candidate recall@k (logreg)
    recall_at_k = {}
    if probs_ref is not None:
        P, classes = probs_ref
        order = np.argsort(-P, axis=1)
        ranked = classes[order]
        for k in (1, 3, 5, 10, 20, 50):
            hit = [g in set(ranked[i, :k]) for i, g in enumerate(gold)]
            recall_at_k[k] = round(float(np.mean(hit)), 4)
        print("candidate recall@k (logreg):", recall_at_k)

    # ------------------------------------------------- C. few-shot neighbour quality
    vec = word_tfidf()
    Ptr = vec.fit_transform(text_of(pool))
    Pte = vec.transform(text_of(test))
    sim = linear_kernel(Pte, Ptr)
    nn = np.argsort(-sim, axis=1)
    pool_codes = pool["naics6"].to_numpy()
    fewshot = {}
    for k in (1, 3, 5, 10, 20):
        hit = [gold[i] in set(pool_codes[nn[i, :k]]) for i in range(len(gold))]
        fewshot[k] = round(float(np.mean(hit)), 4)
    print("gold code present among k nearest pool rows:", fewshot)

    # same, on normalized text, to isolate the effect of the R5-lite rewrite
    vec_n = word_tfidf()
    Ptr_n = vec_n.fit_transform(text_of(pool, norm=True))
    Pte_n = vec_n.transform(text_of(test, norm=True))
    nn_n = np.argsort(-linear_kernel(Pte_n, Ptr_n), axis=1)
    fewshot_norm = {}
    for k in (1, 3, 5, 10, 20):
        hit = [gold[i] in set(pool_codes[nn_n[i, :k]]) for i in range(len(gold))]
        fewshot_norm[k] = round(float(np.mean(hit)), 4)
    print("same, normalized text:", fewshot_norm)

    # how much text the normalizer actually removes, and on how many rows
    raw = test["description"].astype(str).tolist()
    changed = [(a, normalize(a)) for a in raw if normalize(a) != a]
    norm_stats = {"rows_changed": len(changed), "rows": len(raw),
                  "mean_chars_removed": round(float(np.mean([len(a) - len(b) for a, b in changed])), 1)
                  if changed else 0.0,
                  "rows_emptied": sum(1 for _, b in changed if not b.strip())}
    print("normalizer:", norm_stats)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = {"variants": results, "candidate_recall_at_k": recall_at_k,
           "fewshot_neighbour_recall": fewshot,
           "fewshot_neighbour_recall_normalized": fewshot_norm,
           "normalizer": norm_stats,
           "pool_rows": len(pool), "pool_codes": int(pool["naics6"].nunique()), "test_rows": len(test)}
    (RUNS / "ablations_s0b.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RUNS / 'ablations_s0b.json'}")


if __name__ == "__main__":
    main()
