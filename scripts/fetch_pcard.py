"""Download public purchase-card (P-card) transactions for the `messy` eval split.

Source: State of Oklahoma, Office of Management and Enterprise Services (OMES),
published on data.ok.gov (CKAN). Transactions are short, merchant-driven
descriptions much closer to corporate expense data than federal contracts.
These are NOT labeled with NAICS; the team hand-labels a sample
(see docs/labeling_guidelines.md).

Files are published monthly (pcard_public_YYYYMM.csv, ~7 MB, ~34k rows).
Columns: CALENDAR_YEAR, CALENDAR_MONTH, AGENCYNBR, AGENCYNAME, LAST_NAME,
FIRST_INITIAL, ITEM_DESCR, AMOUNT, MERCHANT, TRANSACTION_DATE, POST_DATE,
MCC_DESCRIPTION, ROWID.  Cardholder name columns are dropped before saving.

Usage:  python scripts/fetch_pcard.py --month 202407
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

CKAN = "https://data.ok.gov/api/3/action/package_search"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pcard"
DROP = ["LAST_NAME", "FIRST_INITIAL"]


def find_resource(month: str) -> str:
    r = requests.get(CKAN, params={"q": "Purchase Card (PCard)", "rows": 50}, timeout=60)
    r.raise_for_status()
    target = f"pcard_public_{month}.csv"
    for pkg in r.json()["result"]["results"]:
        for res in pkg.get("resources", []):
            if res.get("url", "").endswith(target):
                return res["url"]
    raise RuntimeError(f"No resource named {target}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="202407")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    url = find_resource(args.month)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), dtype=str).drop(columns=DROP, errors="ignore")
    dest = OUT / f"ok_pcard_{args.month}.csv"
    df.to_csv(dest, index=False)
    print(f"saved {len(df):,} rows -> {dest}")


if __name__ == "__main__":
    main()
