"""Download EPA Supply Chain GHG Emission Factors v1.3 by NAICS-6 (2022 USD).

Source: https://catalog.data.gov/dataset/supply-chain-greenhouse-gas-emission-factors-v1-3-by-naics-6
Units: kg CO2e per 2022 USD. Columns include factors without and with margins
(producer vs purchaser price).

Usage:  python scripts/fetch_epa_factors.py
"""
from pathlib import Path
import requests

BASE = "https://pasteur.epa.gov/uploads/10.23719/1531143"
FILES = [
    "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv",
    "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_byGHG_USD2022.csv",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "epa"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists():
            print(f"skip  {name}")
            continue
        r = requests.get(f"{BASE}/{name}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"saved {name} ({len(r.content):,} bytes)")


if __name__ == "__main__":
    main()
