"""Pull labeled federal contract awards from the USAspending.gov API.

Each award has a free-text description, an obligated amount, and a NAICS code
(the label). We pull a sector-stratified pool so rare sectors are represented,
then split into: train/dev pool (few-shot + calibration) and a held-out test set.

API docs: https://api.usaspending.gov/docs/endpoints
Usage:    python scripts/fetch_usaspending.py --fy 2024 --per-sector 1000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
SECTORS = ["11", "21", "22", "23", "31", "32", "33", "42", "44", "45", "48", "49",
           "51", "52", "53", "54", "56", "61", "62", "71", "72", "81", "92"]
FIELDS = ["Award ID", "Recipient Name", "Description", "Award Amount",
          "NAICS", "PSC", "Awarding Agency", "Start Date"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "usaspending"


def fy_period(fy: int) -> dict:
    return {"start_date": f"{fy - 1}-10-01", "end_date": f"{fy}-09-30"}


def fetch_sector(sector: str, fy: int, per_sector: int, page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    page = 1
    while len(rows) < per_sector:
        body = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"],
                "time_period": [fy_period(fy)],
                "naics_codes": {"require": [sector]},
            },
            "fields": FIELDS,
            "page": page,
            "limit": page_size,
            "sort": "Start Date",
            "order": "desc",
        }
        r = requests.post(API, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        rows.extend(data.get("results", []))
        if not data.get("page_metadata", {}).get("hasNext"):
            break
        page += 1
        time.sleep(0.3)
    return rows[:per_sector]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fy", type=int, default=2024)
    ap.add_argument("--per-sector", type=int, default=1000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"awards_fy{args.fy}.jsonl"
    n = 0
    with dest.open("w", encoding="utf-8") as f:
        for s in SECTORS:
            rows = fetch_sector(s, args.fy, args.per_sector)
            for row in rows:
                row["_sector_query"] = s
                f.write(json.dumps(row) + "\n")
            n += len(rows)
            print(f"sector {s}: {len(rows)}")
    print(f"saved {n} awards -> {dest}")


if __name__ == "__main__":
    main()
