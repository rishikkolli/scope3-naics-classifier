"""Spend-based emissions: kgCO2e = spend (2022 USD) x EPA factor (kgCO2e / 2022 USD).

EPA v1.3 factors are keyed by 2017 NAICS; the classifier predicts 2022 codes, so lookups
go through the Census 2022->2017 concordance. 1->many splits use the mean target factor
and are flagged (design.md section 1).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FACTOR_FILE = "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv"

# Annual CPI-U averages (BLS). The design calls for the BEA PCE deflator; CPI-U is the
# placeholder until we swap the series — same mechanism, slightly different values.
# TODO(verify): re-check 2024/2025 once used for the report.
_CPI = {2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689, 2025: 322.0}


def load_factors(raw_dir: Path, use_margins: bool = True) -> pd.Series:
    """2017 NAICS-6 -> factor. `use_margins=True` gives purchaser-price factors."""
    df = pd.read_csv(raw_dir / "epa" / FACTOR_FILE, dtype={"2017 NAICS Code": str})
    col = ("Supply Chain Emission Factors with Margins" if use_margins
           else "Supply Chain Emission Factors without Margins")
    return df.set_index(df["2017 NAICS Code"].str.strip())[col]


def load_useeio_groups(raw_dir: Path) -> pd.Series:
    """2017 NAICS-6 -> USEEIO sector id (codes sharing a sector share a factor)."""
    df = pd.read_csv(raw_dir / "epa" / FACTOR_FILE, dtype={"2017 NAICS Code": str})
    return df.set_index(df["2017 NAICS Code"].str.strip())["Reference USEEIO Code"]


def load_concordance_2022_to_2017(raw_dir: Path) -> dict[str, list[str]]:
    """2022 NAICS-6 -> list of 2017 NAICS-6 (usually one; splits are flagged downstream)."""
    df = pd.read_excel(raw_dir / "naics" / "2022_to_2017_NAICS.xlsx", skiprows=2, dtype=str)
    c2022, c2017 = df.columns[0], df.columns[2]
    df = df.dropna(subset=[c2022, c2017])
    df[c2022] = df[c2022].str.strip()
    df[c2017] = df[c2017].str.strip()
    df = df[df[c2022].str.fullmatch(r"\d{6}") & df[c2017].str.fullmatch(r"\d{6}")]
    return df.groupby(c2022)[c2017].apply(lambda s: sorted(set(s))).to_dict()


class FactorLookup:
    """factor(naics6_2022) with concordance mapping, mean over splits, and group ids."""

    def __init__(self, raw_dir: Path, use_margins: bool = True):
        self.factors = load_factors(raw_dir, use_margins)
        self.useeio = load_useeio_groups(raw_dir)
        self.conc = load_concordance_2022_to_2017(raw_dir)

    def _targets(self, naics6_2022: str) -> list[str]:
        t = [c for c in self.conc.get(str(naics6_2022), []) if c in self.factors.index]
        if not t and str(naics6_2022) in self.factors.index:   # label was already a 2017 code
            t = [str(naics6_2022)]
        return t

    def factor(self, naics6_2022: str) -> float | None:
        t = self._targets(naics6_2022)
        return float(self.factors.loc[t].mean()) if t else None

    def is_split(self, naics6_2022: str) -> bool:
        return len(self._targets(naics6_2022)) > 1

    def group(self, naics6_2022: str) -> frozenset | None:
        """Factor-equivalence group: the set of USEEIO sectors the code maps to."""
        t = self._targets(naics6_2022)
        return frozenset(self.useeio.loc[t]) if t else None


def to_2022_usd(amount: float, year: int) -> float:
    """Deflate nominal spend to 2022 USD."""
    if year not in _CPI:
        year = min(_CPI, key=lambda y: abs(y - year))
    return amount * _CPI[2022] / _CPI[year]


def emissions(amount_2022: float, naics6: str, lookup: FactorLookup) -> float | None:
    f = lookup.factor(naics6)
    return None if f is None else amount_2022 * f


def expected_emissions_error(amount_2022: float, p_wrong: float,
                             candidate_factors: list[float]) -> float:
    """Priority score for human review (design.md section 3)."""
    if not candidate_factors:
        return 0.0
    spread = max(candidate_factors) - min(candidate_factors)
    return amount_2022 * p_wrong * spread
