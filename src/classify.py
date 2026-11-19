"""LLM classification stages.

  zero_shot      : description -> NAICS-6 directly (baseline S0c)
  rag            : choose among retrieved top-k candidates (S1)
  rag_fewshot    : + nearest labeled training examples (S2; S3 adds amount/MCC context)
Every call logs candidates, raw samples, tokens and latency for error analysis.

Provider is picked from .env: LLM_PROVIDER=anthropic|openai, or auto-detected from
whichever of ANTHROPIC_API_KEY / OPENAI_API_KEY is set (Anthropic wins if both are).
LLM_MODEL overrides the default model for the provider.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

def _provider() -> str:
    p = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if p:
        return p
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"


PROVIDER = _provider()
_DEFAULTS = {
    # model, USD per 1M input tokens, USD per 1M output tokens (cost logging only)
    "openai": ("gpt-4o-mini", 0.15, 0.60),
    "anthropic": ("claude-haiku-4-5", 1.00, 5.00),
}
MODEL = os.environ.get("LLM_MODEL", _DEFAULTS[PROVIDER][0])
_PRICE_IN = float(os.environ.get("LLM_PRICE_IN", _DEFAULTS[PROVIDER][1]))
_PRICE_OUT = float(os.environ.get("LLM_PRICE_OUT", _DEFAULTS[PROVIDER][2]))


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
    if PROVIDER == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    from openai import OpenAI
    return OpenAI()


def _complete(system: str, user_msg: str, n: int, temperature: float,
              max_tokens: int = 300) -> tuple[list[str], int, int]:
    """Return (texts, input_tokens, output_tokens) for n completions on either provider."""
    client = _client()
    if PROVIDER == "anthropic":
        # Messages API has no n parameter, so sample n times
        texts, tin, tout = [], 0, 0
        for _ in range(n):
            kw = {"system": system} if system else {}
            # anthropic SDK >= 2 has no temperature argument; default sampling
            # already gives varied samples for the self-consistency vote
            r = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                       messages=[{"role": "user", "content": user_msg}], **kw)
            texts.append("".join(b.text for b in r.content if b.type == "text"))
            tin += r.usage.input_tokens
            tout += r.usage.output_tokens
        return texts, tin, tout
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": user_msg}]
    r = client.chat.completions.create(model=MODEL, n=n, temperature=temperature,
                                       max_tokens=max_tokens, messages=msgs)
    return ([c.message.content or "" for c in r.choices],
            r.usage.prompt_tokens, r.usage.completion_tokens)


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
    t0 = time.time()
    # n_samples sampled answers; the majority vote is the prediction (self-consistency, C1)
    texts, tin, tout = _complete(_SYSTEM, user_msg, n_samples,
                                 0.7 if n_samples > 1 else 0.0)
    latency = time.time() - t0
    parsed = [_parse(t) for t in texts]
    samples = [str(p.get("naics6", "")) for p in parsed]
    best = max(set(samples), key=samples.count) if samples else ""
    first = next((p for p, s in zip(parsed, samples) if s == best), {})
    cost = (tin * _PRICE_IN + tout * _PRICE_OUT) / 1e6
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
    texts, _, _ = _complete("", (
                   "Rewrite this purchase record as a short plain-English phrase naming the "
                   "product or service bought (e.g. 'dell lat 5540 x10' -> 'laptop computers'). "
                   "Reply with the phrase only.\n\n"
                   + _record_text(description, merchant, mcc)), 1, 0.0, max_tokens=60)
    return (texts[0] or description).strip()
