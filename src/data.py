"""Load the frozen evaluation splits (scripts/build_splits.py)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_gov(split: str) -> pd.DataFrame:
    """gov_test or gov_pool -> DataFrame[item_id, description, amount, naics6, sector]."""
    df = pd.read_csv(ROOT / "data" / "interim" / f"{split}.csv", dtype={"naics6": str})
    out = pd.DataFrame({
        "item_id": df["Award ID"].astype(str),
        "description": df["Description"].fillna("").astype(str),
        "recipient": df["Recipient Name"].fillna("").astype(str),
        "amount": pd.to_numeric(df["Award Amount"], errors="coerce").fillna(0.0),
        "naics6": df["naics6"].str.strip(),
        "sector": df["_sector_query"].astype(str),
    })
    return out[out["naics6"].str.fullmatch(r"\d{6}")].reset_index(drop=True)


def load_messy_labeled() -> pd.DataFrame | None:
    """Adjudicated messy labels once labeling is done; None until then."""
    path = ROOT / "data" / "labeled" / "messy_final.csv"
    return pd.read_csv(path, dtype=str) if path.exists() else None
