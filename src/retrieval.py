"""Candidate-code retrieval.

Variants (each evaluated with recall@k, see design.md section 2):
  dense | dense_asym (e5/bge prefixes) | bm25 | hybrid (RRF) | + query rewrite | + rerank

Raw cosine similarity is NOT used as confidence; only rank and top-1/top-2 margin.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    naics6: str
    score: float           # aggregated over that code's chunks
    rank: int
    best_chunk_text: str


class Retriever:
    def __init__(self, chunks, embedder: str, hybrid: bool = True):
        raise NotImplementedError

    def build_index(self) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int = 20) -> list[Candidate]:
        """Retrieve chunks, aggregate scores by NAICS-6 (max), dedupe, return top-k codes."""
        raise NotImplementedError


def rewrite_query(description: str) -> str:
    """LLM rewrite: expand abbreviations / vendor jargon into a plain product or service phrase."""
    raise NotImplementedError


def rerank(query: str, candidates: list[Candidate], top_n: int = 50) -> list[Candidate]:
    """Cross-encoder rerank of the top-n candidates."""
    raise NotImplementedError


def recall_at_k(gold: list[str], ranked: list[list[str]], k: int) -> float:
    hits = sum(g in r[:k] for g, r in zip(gold, ranked))
    return hits / max(len(gold), 1)
