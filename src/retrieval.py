"""Candidate-code retrieval.

Priority-ordered variants (design.md section 2), each evaluated with recall@k:
  R1 bm25 | R2 dense (e5 query/passage prefixes) | R3 + index chunks (chunk-type choice)
  R4 hybrid RRF | R5 + query rewrite (LLM) | R6 rerank | R7 metadata boost

Raw cosine similarity is NOT used as confidence; only rank and top-1/top-2 margin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .naics_corpus import Chunk


@dataclass
class Candidate:
    naics6: str
    score: float           # aggregated over that code's chunks
    rank: int
    best_chunk_text: str


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _aggregate_by_code(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Candidate]:
    """Max chunk score per code -> top-k codes."""
    best: dict[str, tuple[float, str]] = {}
    for c, s in zip(chunks, scores):
        s = float(s)
        if c.naics6 not in best or s > best[c.naics6][0]:
            best[c.naics6] = (s, c.text)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
    return [Candidate(code, sc, i + 1, txt) for i, (code, (sc, txt)) in enumerate(ranked)]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 20) -> list[Candidate]:
        scores = self.bm25.get_scores(_tokenize(query))
        return _aggregate_by_code(self.chunks, scores, k)


class DenseRetriever:
    """Asymmetric embeddings. e5 models need 'query: ' / 'passage: ' prefixes."""

    def __init__(self, chunks: list[Chunk], model_name: str = "intfloat/e5-base-v2",
                 cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._prefix = "e5" in model_name.lower()
        self.emb = self._load_or_encode(cache_dir)

    def _load_or_encode(self, cache_dir: Path | None) -> np.ndarray:
        cache = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tag = re.sub(r"\W+", "_", f"{self.model_name}_{len(self.chunks)}")
            cache = cache_dir / f"emb_{tag}.npy"
            if cache.exists():
                return np.load(cache)
        texts = [("passage: " + c.text) if self._prefix else c.text for c in self.chunks]
        emb = self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                show_progress_bar=True)
        if cache is not None:
            np.save(cache, emb)
        return emb

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = [("query: " + q) if self._prefix else q for q in queries]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True)

    def search(self, query: str, k: int = 20) -> list[Candidate]:
        return self.search_encoded(self.encode_queries([query])[0], k)

    def search_encoded(self, q_emb: np.ndarray, k: int = 20) -> list[Candidate]:
        scores = self.emb @ q_emb
        return _aggregate_by_code(self.chunks, scores, k)


class HybridRetriever:
    """Reciprocal rank fusion of the per-code rankings of BM25 and dense (R4)."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k

    def search(self, query: str, k: int = 20, q_emb: np.ndarray | None = None) -> list[Candidate]:
        pool = 3 * k
        lists = [self.bm25.search(query, pool),
                 self.dense.search(query, pool) if q_emb is None
                 else self.dense.search_encoded(q_emb, pool)]
        fused: dict[str, float] = {}
        text: dict[str, str] = {}
        for cands in lists:
            for c in cands:
                fused[c.naics6] = fused.get(c.naics6, 0.0) + 1.0 / (self.rrf_k + c.rank)
                text.setdefault(c.naics6, c.best_chunk_text)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Candidate(code, sc, i + 1, text[code]) for i, (code, sc) in enumerate(ranked)]


def rewrite_query(description: str, merchant: str = "", mcc: str = "") -> str:
    """R5: LLM rewrite — expand abbreviations / vendor jargon into a plain product or
    service phrase. Implemented in src/classify.py (needs an API key)."""
    from .classify import llm_rewrite
    return llm_rewrite(description, merchant, mcc)


def recall_at_k(gold: list[str], ranked: list[list[str]], k: int) -> float:
    hits = sum(g in r[:k] for g, r in zip(gold, ranked))
    return hits / max(len(gold), 1)
