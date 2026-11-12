"""Confidence signals and abstention (mandatory component, design.md section 5)."""
from __future__ import annotations

from collections import Counter

import numpy as np


def self_consistency(samples: list[str]) -> float:
    if not samples:
        return 0.0
    return Counter(samples).most_common(1)[0][1] / len(samples)


def retrieval_margin(scores: list[float]) -> float:
    return float(scores[0] - scores[1]) if len(scores) > 1 else 0.0


def fit_combiner(features: np.ndarray, correct: np.ndarray):
    """C4: logistic regression over [self_consistency, margin, verbal_conf(, logprob)],
    fit on the dev split to predict correctness. Returns the fitted model."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(features, correct.astype(int))
    return model


def combine(features: np.ndarray, model) -> np.ndarray:
    """P(correct) from a combiner fit by fit_combiner."""
    return model.predict_proba(features)[:, 1]


def abstain(conf: np.ndarray, coverage: float) -> np.ndarray:
    """Boolean mask of items kept at the given coverage (highest confidence first)."""
    n_keep = int(round(coverage * len(conf)))
    keep = np.zeros(len(conf), dtype=bool)
    keep[np.argsort(-conf)[:n_keep]] = True
    return keep
