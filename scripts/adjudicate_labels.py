"""Merge the two annotators' messy labels, report agreement, emit the final label file.

Reads data/labeled/messy_to_label.csv (filled in by hand per docs/labeling_guidelines.md),
reports Cohen's kappa at 2/4/6 digits and evidence-level agreement, writes rows needing
adjudication to messy_disagreements.csv, and — once an `adjudicated` column resolves them —
writes data/labeled/messy_final.csv (one row per item: final naics6, evidence, split).

Usage:  python scripts/adjudicate_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "labeled"


def main() -> None:
    df = pd.read_csv(LAB / "messy_to_label.csv", dtype=str)
    done = df[df["naics6"].notna() & (df["naics6"].str.strip() != "")]
    n_items = df["item_id"].nunique()
    print(f"{len(done)}/{len(df)} label slots filled ({done['item_id'].nunique()}/{n_items} items)")
    pairs = done.pivot_table(index="item_id", columns=done.groupby("item_id").cumcount(),
                             values="naics6", aggfunc="first")
    both = pairs.dropna()
    if both.empty:
        sys.exit("No item has two labels yet - nothing to adjudicate.")
    a, b = both[0].str.strip(), both[1].str.strip()
    for d in (2, 4, 6):
        ad, bd = a.str[:d].where(a != "UNKNOWN", "UNK"), b.str[:d].where(b != "UNKNOWN", "UNK")
        print(f"kappa@{d}: {cohen_kappa_score(ad, bd):.3f}   raw agreement: {(ad == bd).mean():.3f}")
    ev = done.pivot_table(index="item_id", columns=done.groupby("item_id").cumcount(),
                          values="evidence", aggfunc="first").dropna()
    if not ev.empty:
        print(f"evidence-level agreement: {(ev[0] == ev[1]).mean():.3f}")

    dis = both[a != b]
    if len(dis):
        out = dis.rename(columns={0: "label_a", 1: "label_b"})
        out["adjudicated"] = ""
        meta = done.drop_duplicates("item_id").set_index("item_id")[
            ["split", "description", "merchant", "mcc", "amount_usd"]]
        out.join(meta).to_csv(LAB / "messy_disagreements.csv")
        print(f"{len(dis)} disagreements -> {LAB / 'messy_disagreements.csv'} "
              "(third annotator fills 'adjudicated')")

    adj_path = LAB / "messy_disagreements.csv"
    final = both[a == b][0].rename("naics6").to_frame()
    if adj_path.exists():
        adj = pd.read_csv(adj_path, dtype=str).set_index("item_id")
        resolved = adj[adj.get("adjudicated", pd.Series(dtype=str)).notna()
                       & (adj["adjudicated"].str.strip() != "")]["adjudicated"].rename("naics6")
        final = pd.concat([final, resolved.to_frame()])
    meta = done.drop_duplicates("item_id").set_index("item_id")[
        ["split", "description", "merchant", "mcc", "amount_usd", "evidence"]]
    final = final.join(meta)
    final.to_csv(LAB / "messy_final.csv")
    print(f"final labels for {len(final)} items -> {LAB / 'messy_final.csv'}")


if __name__ == "__main__":
    main()
