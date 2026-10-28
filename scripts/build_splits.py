"""Freeze evaluation splits (design.md section 4).

  data/interim/gov_test.csv      300 USAspending awards, stratified by sector
  data/interim/gov_pool.csv      remaining awards (few-shot pool, B1 training, gov calibration)
  data/labeled/messy_to_label.csv  300 P-card rows stratified by MCC -> messy_dev (100) + messy_test (200),
                                    each row assigned to two annotators (label_template columns)

Usage:  python scripts/build_splits.py
"""
from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
SEED = CFG["eval_split_seed"]
ANNOTATOR_PAIRS = [("rishik", "simranjeet"), ("simranjeet", "bineet"), ("bineet", "rishik")]


def stratified_sample(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """Proportional allocation with at least one row per stratum where possible."""
    frac = n / len(df)
    out = df.groupby(by, group_keys=False).apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED))
    return out.sample(min(n, len(out)), random_state=SEED)


def gov_splits() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/raw/usaspending/awards_fy2024.jsonl").open(encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["naics6"] = df["NAICS"].map(lambda x: (x or {}).get("code"))
    df = df.dropna(subset=["naics6"])
    test = stratified_sample(df, "_sector_query", CFG["usaspending"]["test_size"])
    pool = df.drop(test.index)
    out = ROOT / CFG["paths"]["interim"]
    test.to_csv(out / "gov_test.csv", index=False)
    pool.to_csv(out / "gov_pool.csv", index=False)
    print(f"gov_test {len(test)}  gov_pool {len(pool)}")


def messy_sample() -> None:
    p = pd.read_csv(ROOT / "data/raw/pcard/ok_pcard_202407.csv", dtype=str)
    p["AMOUNT"] = pd.to_numeric(p["AMOUNT"], errors="coerce")
    p = p[p["AMOUNT"] > 0]   # refunds are handled separately, not labeled
    n_dev, n_test = CFG["pcard"]["dev_size"], CFG["pcard"]["test_size"]
    s = stratified_sample(p, "MCC_DESCRIPTION", n_dev + n_test).reset_index(drop=True)
    s["split"] = ["messy_dev"] * n_dev + ["messy_test"] * (len(s) - n_dev)
    pairs = cycle(ANNOTATOR_PAIRS)
    rows = []
    for i, r in s.iterrows():
        for annotator in next(pairs):
            rows.append({"item_id": f"pcard_{r['ROWID']}", "split": r["split"], "description": r["ITEM_DESCR"],
                         "merchant": r["MERCHANT"], "mcc": r["MCC_DESCRIPTION"], "amount_usd": r["AMOUNT"],
                         "annotator": annotator, "naics6": "", "naics6_alt": "", "confidence": "",
                         "ambiguity_reason": "", "notes": ""})
    dest = ROOT / CFG["paths"]["labeled"] / "messy_to_label.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"messy sample {len(s)} rows -> {len(rows)} label slots -> {dest}")


if __name__ == "__main__":
    gov_splits()
    messy_sample()
