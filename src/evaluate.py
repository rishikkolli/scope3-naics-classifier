"""Evaluation metrics, reported separately for `gov` and `messy` splits."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def hierarchical_accuracy(gold: list[str], pred: list[str], digits: int) -> float:
    return float(np.mean([g[:digits] == p[:digits] for g, p in zip(gold, pred)]))


def factor_equivalent_accuracy(gold, pred, group_of) -> float:
    return float(np.mean([group_of.get(g) == group_of.get(p) for g, p in zip(gold, pred)]))


def emissions_error(true_kg: np.ndarray, pred_kg: np.ndarray) -> dict:
    abs_err = np.abs(pred_kg - true_kg)
    return {
        "total_abs_kg": float(abs_err.sum()),
        "relative": float(abs_err.sum() / max(true_kg.sum(), 1e-9)),
        "net_bias_kg": float((pred_kg - true_kg).sum()),
    }


def auroc(correct: np.ndarray, conf: np.ndarray) -> float:
    return float(roc_auc_score(correct.astype(int), conf))


def risk_coverage(correct: np.ndarray, conf: np.ndarray):
    order = np.argsort(-conf)
    errs = 1 - correct[order].astype(float)
    n = np.arange(1, len(errs) + 1)
    coverage = n / len(errs)
    risk = np.cumsum(errs) / n
    aurc = float(np.trapz(risk, coverage))
    return coverage, risk, aurc


def ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
