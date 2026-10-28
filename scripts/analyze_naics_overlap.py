"""Quantify one-to-many mappings between NAICS text, codes and EPA factors (design.md section 1).

Reports:
  - index phrases (normalized) mapping to >1 NAICS-6
  - index head nouns (text before the first comma) spanning >1 NAICS-6
  - identical 6-digit definitions
  - cross-reference count
  - NAICS-6 codes per distinct EPA factor / USEEIO sector

Usage:  python scripts/analyze_naics_overlap.py
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.split().str.join(" ")


def main() -> None:
    ix = pd.read_excel(RAW / "naics" / "2022_NAICS_Index_File.xlsx", dtype={"NAICS22": str})
    text = ix["INDEX ITEM DESCRIPTION"]
    phrase_codes = ix.groupby(norm(text))["NAICS22"].nunique()
    head_codes = ix.groupby(text.astype(str).str.split(",").str[0].str.lower().str.strip())["NAICS22"].nunique()
    print(f"index entries: {len(ix):,} covering {ix.NAICS22.nunique():,} codes "
          f"(median {ix.groupby('NAICS22').size().median():.0f} per code)")
    print(f"phrases -> >1 code: {(phrase_codes > 1).sum():,} of {len(phrase_codes):,}")
    print(f"head nouns -> >1 code: {(head_codes > 1).sum():,} of {len(head_codes):,}")
    print(head_codes.sort_values(ascending=False).head(10).to_string())

    d = pd.read_excel(RAW / "naics" / "2022_NAICS_Descriptions.xlsx", dtype={"Code": str})
    d6 = d[d.Code.str.fullmatch(r"\d{6}")]
    desc = d6.Description.astype(str)
    print(f"6-digit definitions: {len(d6):,}; median {desc.str.split().str.len().median():.0f} words; "
          f"identical text across codes: {desc.str.lower().str.strip().duplicated(keep=False).sum()}")

    xr = pd.read_excel(RAW / "naics" / "2022_NAICS_Cross_References.xlsx")
    print(f"cross-references: {len(xr):,} on {xr.Code.nunique():,} codes")

    e = pd.read_csv(RAW / "epa" / "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv")
    f = e["Supply Chain Emission Factors with Margins"]
    sectors = e.groupby("Reference USEEIO Code").size()
    print(f"EPA codes: {len(e):,}; distinct factors: {f.nunique()}; USEEIO sectors: {len(sectors)} "
          f"({(sectors > 1).sum()} shared by >1 code, covering {sectors[sectors > 1].sum()} codes)")
    print(f"factor range: {f.min():.3f}-{f.max():.3f} kgCO2e/USD (median {f.median():.3f})")


if __name__ == "__main__":
    main()
