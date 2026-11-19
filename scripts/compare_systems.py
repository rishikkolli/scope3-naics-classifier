"""Compare two scored runs item by item and test a simple confidence-based combination.

Reads the <run>_S4_items.csv files written by run_confidence_eval.py (both runs must cover the
same gov_test rows in the same order) and reports:
  - accuracy of each run, of "either correct" (oracle ceiling) and of "both correct"
  - agreement rate, accuracy when the runs agree / disagree, AUROC of agreement as a confidence
  - pick-higher-confidence: per item, take the prediction whose cross-fit calibrated C4
    confidence is higher (exploratory: both C4 scores are cross-fit on gov_test itself)
  - emissions relative error with and without the single largest-error item

Usage:  python scripts/compare_systems.py S0b_tfidf_logreg_proba S3_logreg_full
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parents[1] / "eval" / "runs"


def main() -> None:
    ra, rb = sys.argv[1], sys.argv[2]
    a = pd.read_csv(RUNS / f"{ra}_S4_items.csv", dtype={"pred": str, "item_id": str})
    b = pd.read_csv(RUNS / f"{rb}_S4_items.csv", dtype={"pred": str, "item_id": str})
    if a["item_id"].tolist() != b["item_id"].tolist():
        sys.exit("runs do not cover the same rows in the same order")
    ca, cb = a["correct"].astype(bool).to_numpy(), b["correct"].astype(bool).to_numpy()
    agree = a["pred"].to_numpy() == b["pred"].to_numpy()
    pick_a = a["conf"].to_numpy() >= b["conf"].to_numpy()

    print(f"n = {len(a)}")
    print(f"acc@6  {ra} {ca.mean():.3f} | {rb} {cb.mean():.3f} | either {(ca | cb).mean():.3f} "
          f"| both {(ca & cb).mean():.3f}")
    print(f"agree on {agree.mean():.3f} of items; acc when agree {cb[agree].mean():.3f}; "
          f"when disagree {ra} {ca[~agree].mean():.3f}, {rb} {cb[~agree].mean():.3f}")
    print(f"AUROC of agreement as confidence for {rb}: {roc_auc_score(cb, agree):.3f}")
    print(f"pick-higher-confidence acc@6: {np.where(pick_a, ca, cb).mean():.3f} "
          f"({pick_a.mean():.2f} of items take {ra})")

    worst = a["abs_err_kg"].fillna(0).idxmax()
    for name, d in [(ra, a), (rb, b)]:
        ok = d["emis_ok"].astype(bool)
        rel = d.loc[ok, "abs_err_kg"].sum() / d.loc[ok, "true_kg"].sum()
        dd = d.drop(index=worst)
        ok2 = dd["emis_ok"].astype(bool)
        rel2 = dd.loc[ok2, "abs_err_kg"].sum() / dd.loc[ok2, "true_kg"].sum()
        print(f"emissions rel. error {name}: {rel:.3f} (excluding row {worst}: {rel2:.3f})")


if __name__ == "__main__":
    main()
