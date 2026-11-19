"""Sensitivity and error-structure analysis of the emissions calculation
(Simranjeet: emissions calculation).

kgCO2e = spend(2022 USD) x EPA v1.3 factor. Three modelling choices sit between a
predicted NAICS code and a kgCO2e number, and none of them had been measured:

  1. spend concentration   -- how much of the true total rides on a handful of awards
  2. the 2022->2017 concordance -- codes that split across several 2017 codes get the
     mean factor, so part of the error is irreducible from the mapping, not the classifier
  3. the deflator (CPI-U placeholder, src/emissions.py) and purchaser vs producer prices
     (margins on/off)

It also breaks the classifier's emissions error down by *confusion distance* (same
4-digit parent, same 2-digit sector, cross-sector) to show which mistakes are expensive.

No API calls. Usage:  python scripts/analyze_emissions_sensitivity.py
Writes: eval/runs/emissions_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_gov                                        # noqa: E402
from src.emissions import FactorLookup, load_factors, to_2022_usd    # noqa: E402

RUNS = ROOT / "eval" / "runs"
FY_YEAR = 2024
RAW = ROOT / "data" / "raw"


def main() -> None:
    pool, test = load_gov("gov_pool"), load_gov("gov_test")
    lk = FactorLookup(RAW)
    gold = test["naics6"].tolist()
    amt = np.array([to_2022_usd(a, FY_YEAR) for a in test["amount"]])
    out = {}

    # ---------------------------------------------- 1. spend / emissions concentration
    fg = np.array([lk.factor(g) if lk.factor(g) is not None else np.nan for g in gold])
    ok = ~np.isnan(fg)
    true_kg = np.where(ok, amt * np.nan_to_num(fg), 0.0)
    order = np.argsort(-true_kg)
    tot = true_kg.sum()
    out["factor_coverage"] = round(float(ok.mean()), 4)
    out["emissions_concentration"] = {
        f"top_{k}_share": round(float(true_kg[order[:k]].sum() / tot), 4)
        for k in (1, 5, 10, 25, 50)}
    out["spend_concentration_top10"] = round(float(np.sort(amt)[::-1][:10].sum() / amt.sum()), 4)
    out["largest_item"] = {"row": int(order[0]), "gold": gold[order[0]],
                           "amount_2022usd": round(float(amt[order[0]]), 2),
                           "share_of_true_emissions": round(float(true_kg[order[0]] / tot), 4)}

    # ---------------------------------------------- 2. concordance splits
    split_rows = [g for g in gold if lk.is_split(g)]
    spreads = []
    for g in set(split_rows):
        t = [c for c in lk.conc.get(g, []) if c in lk.factors.index]
        vals = lk.factors.loc[t].astype(float)
        spreads.append(float(vals.max() / max(vals.min(), 1e-12)))
    out["concordance"] = {
        "test_rows_whose_gold_code_splits": len(split_rows),
        "distinct_split_codes": len(set(split_rows)),
        "median_factor_ratio_within_split": round(float(np.median(spreads)), 3) if spreads else None,
        "max_factor_ratio_within_split": round(float(np.max(spreads)), 3) if spreads else None,
    }

    # ---------------------------------------------- 3. deflator and margins sensitivity
    raw_amt = test["amount"].to_numpy(dtype=float)
    nominal_kg = np.where(ok, raw_amt * np.nan_to_num(fg), 0.0).sum()
    out["deflator"] = {
        "total_kg_cpi_deflated": round(float(tot), 1),
        "total_kg_nominal_fy2024": round(float(nominal_kg), 1),
        "ratio": round(float(tot / nominal_kg), 4),
        "note": "every test award is FY2024, so deflation is one scalar: it moves absolute "
                "kgCO2e but leaves every relative error metric unchanged",
    }
    lk_nm = FactorLookup(RAW, use_margins=False)
    fg_nm = np.array([lk_nm.factor(g) if lk_nm.factor(g) is not None else np.nan for g in gold])
    ok2 = ok & ~np.isnan(fg_nm)
    out["margins"] = {
        "total_kg_with_margins": round(float((amt[ok2] * fg[ok2]).sum()), 1),
        "total_kg_without_margins": round(float((amt[ok2] * fg_nm[ok2]).sum()), 1),
        "ratio": round(float((amt[ok2] * fg[ok2]).sum() / (amt[ok2] * fg_nm[ok2]).sum()), 4),
    }
    f_all = load_factors(RAW).astype(float)
    out["factor_distribution"] = {
        "n_codes": int(f_all.shape[0]),
        "min": round(float(f_all.min()), 4), "median": round(float(f_all.median()), 4),
        "max": round(float(f_all.max()), 4),
        "p90_over_p10": round(float(f_all.quantile(0.9) / f_all.quantile(0.1)), 2),
    }

    # ---------------------------------------------- 4. error by confusion distance (S0b)
    def text(df):
        return (df["description"].astype(str) + " " + df["recipient"].astype(str)).tolist()

    clf = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
                        LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1))
    clf.fit(text(pool), pool["naics6"])
    pred = list(clf.predict(text(test)))

    fp = np.array([lk.factor(p) if lk.factor(p) is not None else 0.0 for p in pred])
    err = np.where(ok, np.abs(amt * fp - true_kg), 0.0)

    def bucket(g, p):
        if g == p:
            return "correct"
        if lk.group(g) is not None and lk.group(g) == lk.group(p):
            return "factor_equivalent"
        if g[:4] == p[:4]:
            return "same_4digit"
        if g[:2] == p[:2]:
            return "same_2digit"
        return "cross_sector"

    buckets = {}
    for i, (g, p) in enumerate(zip(gold, pred)):
        if not ok[i]:
            continue
        b = bucket(g, p)
        d = buckets.setdefault(b, {"n": 0, "abs_kg": 0.0, "factor_ratios": []})
        d["n"] += 1
        d["abs_kg"] += float(err[i])
        if fp[i] > 0 and fg[i] > 0:
            d["factor_ratios"].append(float(max(fp[i], fg[i]) / min(fp[i], fg[i])))
    total_err = sum(d["abs_kg"] for d in buckets.values())
    out["error_by_confusion_distance"] = {
        b: {"n": d["n"],
            "share_of_emissions_error": round(d["abs_kg"] / max(total_err, 1e-9), 4),
            "median_factor_ratio": round(float(np.median(d["factor_ratios"])), 2)
            if d["factor_ratios"] else None}
        for b, d in sorted(buckets.items(), key=lambda kv: -kv[1]["abs_kg"])}
    out["s0b_total_rel_error"] = round(float(total_err / tot), 4)

    # ---------------------------------------------- 5. cost of six-digit precision
    # If the classifier were right to 4 digits and we picked the most common 6-digit
    # child in the pool, how much emissions error would remain?
    common_child = (pool.groupby(pool["naics6"].str[:4])["naics6"]
                    .agg(lambda s: s.value_counts().idxmax()).to_dict())
    pred4 = [common_child.get(g[:4], g) for g in gold]
    f4 = np.array([lk.factor(p) if lk.factor(p) is not None else 0.0 for p in pred4])
    out["oracle_4digit_then_common_child"] = {
        "acc@6": round(float(np.mean([a == b for a, b in zip(gold, pred4)])), 4),
        "emissions_rel_error": round(float(np.abs(amt[ok] * f4[ok] - true_kg[ok]).sum() / tot), 4),
    }

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "emissions_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nsaved {RUNS / 'emissions_sensitivity.json'}")


if __name__ == "__main__":
    main()
