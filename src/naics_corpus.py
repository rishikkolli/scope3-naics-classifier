"""Build the NAICS retrieval corpus.

One NAICS-6 code -> many chunks:
  - "title":      code title
  - "definition": full Census description (cross-ref sentences stripped)
  - "index":      each Census index entry (~20k short phrases, closest in length to queries)
  - "crossref":   cross-reference text, kept separately as "not this code, see Y" hints

Retrieval scores are aggregated *by code* downstream, so many chunks per code is fine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Chunk:
    chunk_id: str
    naics6: str
    kind: str   # title | definition | index | crossref
    text: str


def load_codes(raw_dir: Path) -> pd.DataFrame:
    """2-6 digit code list -> DataFrame[naics, title, level]."""
    raise NotImplementedError


def load_descriptions(raw_dir: Path) -> pd.DataFrame:
    raise NotImplementedError


def load_index_entries(raw_dir: Path) -> pd.DataFrame:
    raise NotImplementedError


def load_cross_references(raw_dir: Path) -> pd.DataFrame:
    raise NotImplementedError


def build_chunks(raw_dir: Path) -> list[Chunk]:
    raise NotImplementedError


def overlap_report(raw_dir: Path) -> dict:
    """Quantify one-to-many mappings (feedback item 1):
    - index phrases (normalized) mapping to >1 NAICS-6
    - definitions sharing near-duplicate text
    - cross-reference fan-out per code
    """
    raise NotImplementedError
