"""Download 2022 NAICS reference files from the U.S. Census Bureau.

Files:
  - 2-6 digit code list (titles)
  - full descriptions (definitions incl. cross-reference text)
  - index file (~20k short index entries -> NAICS-6)
  - cross-references
  - 2017 <-> 2022 concordances (USAspending labels mix versions)

Usage:  python scripts/fetch_naics.py
"""
from pathlib import Path
import requests

BASE = "https://www.census.gov/naics"
FILES = {
    "2-6 digit_2022_Codes.xlsx": f"{BASE}/2022NAICS/2-6%20digit_2022_Codes.xlsx",
    "2022_NAICS_Descriptions.xlsx": f"{BASE}/2022NAICS/2022_NAICS_Descriptions.xlsx",
    "2022_NAICS_Index_File.xlsx": f"{BASE}/2022NAICS/2022_NAICS_Index_File.xlsx",
    "2022_NAICS_Cross_References.xlsx": f"{BASE}/2022NAICS/2022_NAICS_Cross_References.xlsx",
    "2017_to_2022_NAICS.xlsx": f"{BASE}/concordances/2017_to_2022_NAICS.xlsx",
    "2022_to_2017_NAICS.xlsx": f"{BASE}/concordances/2022_to_2017_NAICS.xlsx",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "naics"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (research; scope3-naics-classifier)"}
    for name, url in FILES.items():
        dest = OUT / name
        if dest.exists():
            print(f"skip  {name}")
            continue
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"saved {name} ({len(r.content):,} bytes)")


if __name__ == "__main__":
    main()
