"""Download public purchase-card (P-card) transactions for the `messy` eval split.

Source: State of Oklahoma, Office of Management and Enterprise Services (OMES),
published on data.ok.gov (CKAN). Transactions are short, merchant-driven
descriptions much closer to corporate expense data than federal contracts.
These are NOT labeled with NAICS; the team hand-labels a sample
(see docs/labeling_guidelines.md).

Usage:  python scripts/fetch_pcard.py --fy 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import requests

CKAN = "https://data.ok.gov/api/3/action/package_search"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pcard"


def find_resource(fy: int) -> str:
    q = f'title:"Purchase Card (PCard) Fiscal Year {fy}"'
    r = requests.get(CKAN, params={"q": q, "rows": 5}, timeout=60)
    r.raise_for_status()
    for pkg in r.json()["result"]["results"]:
        for res in pkg.get("resources", []):
            if res.get("format", "").lower() == "csv":
                return res["url"]
    raise RuntimeError(f"No CSV resource found for FY{fy}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fy", type=int, default=2024)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    url = find_resource(args.fy)
    dest = OUT / f"ok_pcard_fy{args.fy}.csv"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"saved {dest} from {url}")


if __name__ == "__main__":
    main()
