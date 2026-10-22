"""LLM classification stages.

  zero_shot      : description -> NAICS-6 directly (baseline)
  rag            : choose among retrieved top-k candidates
  rag_fewshot    : + nearest labeled training examples
Every call logs prompt, candidates, raw output, tokens, latency for error analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prediction:
    item_id: str
    naics6: str
    candidates: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)   # for self-consistency
    logprob: float | None = None
    verbal_conf: float | None = None
    rationale: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0


def zero_shot(description: str, amount: float | None = None) -> Prediction:
    raise NotImplementedError


def rag(description: str, candidates, amount: float | None = None) -> Prediction:
    raise NotImplementedError


def rag_fewshot(description: str, candidates, examples, amount: float | None = None) -> Prediction:
    raise NotImplementedError
