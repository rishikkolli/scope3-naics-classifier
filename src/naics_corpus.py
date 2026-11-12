"""Build the NAICS retrieval corpus.

One NAICS-6 code -> many chunks:
  - "title":      code title
  - "definition": full Census description (cross-ref sentences stripped)
  - "index":      each Census index entry (~20k short phrases, closest in length to queries)
  - "crossref":   cross-reference text, kept separately as "not this code, see Y" hints

Retrieval scores are aggregated *by code* downstream, so many chunks per code is fine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Chunk:
    chunk_id: str
    naics6: str
    kind: str   # title | definition | index | crossref
    text: str


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def load_codes(raw_dir: Path) -> pd.DataFrame:
    """2-6 digit code list -> DataFrame[naics, title, level]."""
    df = pd.read_excel(raw_dir / "naics" / "2-6 digit_2022_Codes.xlsx", dtype=str)
    df = df.rename(columns={"2022 NAICS US   Code": "naics", "2022 NAICS US Title": "title"})
    df = df.dropna(subset=["naics"])
    df["naics"] = df["naics"].str.replace(r"\.0$", "", regex=True).str.strip()
    # Census marks some titles with a trailing "T" footnote
    df["title"] = df["title"].map(_clean).str.replace(r"T$", "", regex=True).str.strip()
    df = df[df["naics"].str.fullmatch(r"\d{2,6}")]
    df["level"] = df["naics"].str.len()
    return df[["naics", "title", "level"]].reset_index(drop=True)


def load_descriptions(raw_dir: Path) -> pd.DataFrame:
    """6-digit code -> title + definition text (cross-reference tail stripped)."""
    df = pd.read_excel(raw_dir / "naics" / "2022_NAICS_Descriptions.xlsx", dtype=str)
    df["Code"] = df["Code"].str.strip()
    df = df[df["Code"].str.fullmatch(r"\d{6}")].copy()
    def strip_xref(text: str) -> str:
        return _clean(re.split(r"Cross-References\.", str(text))[0])
    df["definition"] = df["Description"].map(strip_xref)
    df["title"] = df["Title"].map(_clean).str.replace(r"T$", "", regex=True).str.strip()
    return df.rename(columns={"Code": "naics6"})[["naics6", "title", "definition"]].reset_index(drop=True)


def load_index_entries(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / "naics" / "2022_NAICS_Index_File.xlsx", dtype=str)
    df = df.rename(columns={"NAICS22": "naics6", "INDEX ITEM DESCRIPTION": "text"})
    df["naics6"] = df["naics6"].str.strip()
    df = df[df["naics6"].str.fullmatch(r"\d{6}")].copy()
    df["text"] = df["text"].map(_clean)
    return df[["naics6", "text"]].reset_index(drop=True)


def load_cross_references(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(raw_dir / "naics" / "2022_NAICS_Cross_References.xlsx", dtype=str)
    df = df.rename(columns={"Code": "naics6", "Cross-Reference": "text"})
    df["naics6"] = df["naics6"].str.strip()
    df = df[df["naics6"].str.fullmatch(r"\d{6}")].copy()
    df["text"] = df["text"].map(_clean)
    return df[["naics6", "text"]].reset_index(drop=True)


def build_chunks(raw_dir: Path, include_crossrefs: bool = False) -> list[Chunk]:
    """All retrievable chunks. Cross-refs are off by default: they describe what a code
    is NOT, so they are shown to the LLM as hints rather than retrieved as evidence."""
    chunks: list[Chunk] = []
    desc = load_descriptions(raw_dir)
    for _, r in desc.iterrows():
        chunks.append(Chunk(f"t_{r.naics6}", r.naics6, "title", r.title))
        if r.definition:
            chunks.append(Chunk(f"d_{r.naics6}", r.naics6, "definition", f"{r.title}. {r.definition}"))
    for i, r in load_index_entries(raw_dir).iterrows():
        chunks.append(Chunk(f"i_{i}", r.naics6, "index", r.text))
    if include_crossrefs:
        for i, r in load_cross_references(raw_dir).iterrows():
            chunks.append(Chunk(f"x_{i}", r.naics6, "crossref", r.text))
    return chunks


def crossref_hints(raw_dir: Path) -> dict[str, list[str]]:
    """naics6 -> its cross-reference sentences, for the LLM prompt."""
    xr = load_cross_references(raw_dir)
    return xr.groupby("naics6")["text"].apply(list).to_dict()
