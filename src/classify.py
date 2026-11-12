"""LLM classification stages.

  zero_shot      : description -> NAICS-6 directly (baseline S0c)
  rag            : choose among retrieved top-k candidates (S1)
  rag_fewshot    : + nearest labeled training examples (S2; S3 adds amount/MCC context)
Every call logs candidates, raw samples, tokens and latency for error analysis.

Needs OPENAI_API_KEY in .env (model set in configs/config.yaml).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# gpt-4o-mini pricing, USD per 1M tokens, for cost logging
_PRICE_IN, _PRICE_OUT = 0.15, 0.60


@dataclass
class Prediction:
    item_id: str
    naics6: str
    candidates: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)   # for self-consistency (C1)
    logprob: float | None = None                       # C5
    verbal_conf: float | None = None                   # C3
    rationale: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0


def _client():
    from openai import OpenAI
    return OpenAI()


_SYSTEM = (
    "You classify purchase records into 2022 NAICS-6 industry codes for spend-based "
    "Scope 3 emissions accounting. Code the industry that PRODUCES the purchased good "
    "or service - not the buyer, and not the retailer unless the purchase is a retail "
    "service. Reply with JSON only: "
    '{"naics6": "######", "confidence": 0-100, "rationale": "one sentence"}'
)


def _record_text(description: str, merchant: str = "", mcc: str = "",
                 amount: float | None = None) -> str:
    parts = [f"Description: {description or '(empty)'}"]
    if merchant:
        parts.append(f"Merchant/recipient: {merchant}")
    if mcc:
        parts.append(f"Category code: {mcc}")
    if amount is not None:
        parts.append(f"Amount: ${amount:,.2f}")
    return "\n".join(parts)


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _call(user_msg: str, n_samples: int, item_id: str, candidates: list[str]) -> Prediction:
    client = _client()
    t0 = time.time()
    # one deterministic answer + (n_samples-1) sampled ones for self-consistency
    resp = client.chat.completions.create(
        model=MODEL, n=n_samples, temperature=0.7 if n_samples > 1 else 0.0,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user_msg}])
    latency = time.time() - t0
    parsed = [_parse(c.message.content or "") for c in resp.choices]
    samples = [str(p.get("naics6", "")) for p in parsed]
    best = max(set(samples), key=samples.count) if samples else ""
    first = next((p for p, s in zip(parsed, samples) if s == best), {})
    usage = resp.usage
    cost = (usage.prompt_tokens * _PRICE_IN + usage.completion_tokens * _PRICE_OUT) / 1e6
    return Prediction(item_id=item_id, naics6=best, candidates=candidates, samples=samples,
                      verbal_conf=first.get("confidence"), rationale=str(first.get("rationale", "")),
                      cost_usd=cost, latency_s=latency)


def zero_shot(description: str, item_id: str = "", amount: float | None = None,
              merchant: str = "", mcc: str = "", n_samples: int = 5) -> Prediction:
    msg = (_record_text(description, merchant, mcc, amount)
           + "\n\nAssign the single best 2022 NAICS-6 code.")
    return _call(msg, n_samples, item_id, [])


def _candidate_block(candidates) -> str:
    lines = [f"{c.naics6}: {c.best_chunk_text}" for c in candidates]
    return "Candidate codes (choose ONE of these):\n" + "\n".join(lines)


def rag(description: str, candidates, item_id: str = "", amount: float | None = None,
        merchant: str = "", mcc: str = "", n_samples: int = 5) -> Prediction:
    msg = (_record_text(description, merchant, mcc, amount) + "\n\n"
           + _candidate_block(candidates))
    return _call(msg, n_samples, item_id, [c.naics6 for c in candidates])


def rag_fewshot(description: str, candidates, examples, item_id: str = "",
                amount: float | None = None, merchant: str = "", mcc: str = "",
                n_samples: int = 5) -> Prediction:
    shots = "\n".join(f'"{d}" -> {code}' for d, code in examples)
    msg = (f"Labeled examples of similar records:\n{shots}\n\n"
           + _record_text(description, merchant, mcc, amount) + "\n\n"
           + _candidate_block(candidates))
    return _call(msg, n_samples, item_id, [c.naics6 for c in candidates])


def llm_rewrite(description: str, merchant: str = "", mcc: str = "") -> str:
    """R5: expand abbreviations / vendor jargon into a plain product or service phrase."""
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0,
        messages=[{"role": "user", "content":
                   "Rewrite this purchase record as a short plain-English phrase naming the "
                   "product or service bought (e.g. 'dell lat 5540 x10' -> 'laptop computers'). "
                   "Reply with the phrase only.\n\n"
                   + _record_text(description, merchant, mcc)}])
    return (resp.choices[0].message.content or description).strip()
