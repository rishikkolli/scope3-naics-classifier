"""Spend-based emissions: kgCO2e = spend (2022 USD) x EPA factor (kgCO2e / 2022 USD)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_factors(raw_dir: Path, use_margins: bool = True) -> pd.Series:
    """NAICS-6 -> factor. `use_margins=True` gives purchaser-price factors."""
    raise NotImplementedError


def factor_groups(factors: pd.Series) -> pd.Series:
    """NAICS-6 -> group id; codes with identical factors share a group
    (used for factor-equivalent accuracy)."""
    raise NotImplementedError


def to_2022_usd(amount: float, year: int) -> float:
    """Deflate nominal spend to 2022 USD (BEA PCE price index)."""
    raise NotImplementedError


def emissions(amount_2022: float, naics6: str, factors: pd.Series) -> float:
    return amount_2022 * float(factors[naics6])


def expected_emissions_error(amount_2022: float, p_wrong: float,
                             candidate_factors: list[float]) -> float:
    """Priority score for human review (design.md section 3)."""
    if not candidate_factors:
        return 0.0
    spread = max(candidate_factors) - min(candidate_factors)
    return amount_2022 * p_wrong * spread
