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


# =========================================================================== harness additions
# Below: S4 on the run files in eval/runs/ (see evaluate.load_run). The functions above
# (self_consistency, retrieval_margin, fit_combiner, combine, abstain) are unchanged.
import math  # noqa: E402

import pandas as pd  # noqa: E402

SIGNALS = ["C1_self_consistency", "C1_agree_4digit", "C1_entropy", "C3_verbal", "C2_retrieval_margin", "C2_pred_rank", "C5_logprob", "model_prob"]


def agreement_with_pred(samples, pred: str, digits: int = 6) -> float:
    """C1 measured against the final prediction (equals self_consistency when pred = majority)."""
    s = [str(x)[:digits] for x in (samples or []) if str(x)]
    return sum(x == str(pred)[:digits] for x in s) / len(s) if s else float("nan")


def sample_entropy_conf(samples) -> float:
    """1 - normalized entropy of the samples: 1 = all agree, 0 = all different."""
    s = [str(x) for x in (samples or []) if str(x)]
    if len(s) < 2:
        return float("nan")
    p = np.array(list(Counter(s).values()), float) / len(s)
    return float(1 + (p * np.log(p)).sum() / math.log(len(s)))


def extract_features(items: pd.DataFrame) -> pd.DataFrame:
    """One column per signal (NaN where the run did not log it), named after design.md section 5:
    C1_* self-consistency variants, C2_retrieval_margin (needs `candidate_scores` in the run
    file), C2_pred_rank = 1/rank of the prediction among the candidates, C3_verbal, C5_logprob,
    and model_prob = a probability the system logged itself (`confidence`, e.g. S0b_..._proba)."""
    def col(name):
        return items[name] if name in items else pd.Series([None] * len(items), index=items.index)
    pred = items["pred"].astype(str)
    samples, cands, scores = col("samples"), col("candidates"), col("candidate_scores")

    def margin(sc):
        return retrieval_margin(sc) if isinstance(sc, list) and len(sc) > 1 else np.nan

    def rank_sig(c, p):
        if not isinstance(c, list) or not c:
            return np.nan
        codes = [str(x.get("naics6", x.get("naics")) if isinstance(x, dict) else x) for x in c]
        return 1.0 / (codes.index(p) + 1) if p in codes else 0.0

    verbal = pd.to_numeric(col("verbal_conf"), errors="coerce")
    return pd.DataFrame({
        "C1_self_consistency": [agreement_with_pred(s, p) for s, p in zip(samples, pred)],
        "C1_agree_4digit": [agreement_with_pred(s, p, 4) for s, p in zip(samples, pred)],
        "C1_entropy": [sample_entropy_conf(s) for s in samples],
        "C3_verbal": np.where(verbal > 1, verbal / 100, verbal),
        "C2_retrieval_margin": [margin(s) for s in scores],
        "C2_pred_rank": [rank_sig(c, p) for c, p in zip(cands, pred)],
        "C5_logprob": np.exp(pd.to_numeric(col("logprob"), errors="coerce").clip(upper=0)),
        "model_prob": pd.to_numeric(col("confidence"), errors="coerce"),
    }, index=items.index)


def usable_signals(feats: pd.DataFrame, min_logged: float = 0.5) -> list[str]:
    """Signals logged for most items and not constant."""
    return [c for c in feats.columns
            if feats[c].notna().mean() >= min_logged and feats[c].nunique(dropna=True) > 1]


class ConfidenceModel:
    """C4 for real data: logistic regression over the usable signals, with missing values imputed
    (median + missing-indicator) and optional isotonic recalibration. Same idea as
    fit_combiner/combine, but takes the DataFrame from extract_features."""

    def __init__(self, signals: list[str] | None = None, isotonic: bool = True):
        self.signals, self.isotonic = signals, isotonic

    def _X(self, feats):
        X = feats[self.signals].astype(float)
        miss = X[self.flag].isna().astype(float).to_numpy() if self.flag else np.empty((len(X), 0))
        return np.hstack([X.fillna(self.med).to_numpy(), miss])

    def fit(self, feats: pd.DataFrame, correct) -> "ConfidenceModel":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.preprocessing import StandardScaler
        self.signals = self.signals or usable_signals(feats)
        if not self.signals:
            raise ValueError("no usable confidence signal in this run")
        y = np.asarray(correct).astype(int)
        self.med = feats[self.signals].median()
        self.flag = [s for s in self.signals if feats[s].isna().any()]
        X = self._X(feats)
        self.scaler = StandardScaler().fit(X)
        self.lr = fit_combiner(self.scaler.transform(X), y)
        if self.isotonic:
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                combine(self.scaler.transform(X), self.lr), y)
        return self

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        p = combine(self.scaler.transform(self._X(feats)), self.lr)
        return self.iso.predict(p) if self.isotonic else p

    def coefficients(self) -> dict:
        names = self.signals + [f"{s}_missing" for s in self.flag]
        return dict(zip(names, np.round(self.lr.coef_[0], 3)))


def cross_fit(feats: pd.DataFrame, correct, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold confidence when there is no separate dev split (gov_test has none):
    each item is scored by a model fit on the other folds, so nothing is scored by a model
    that saw its label."""
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(correct).astype(int)
    out = np.full(len(y), np.nan)
    k = max(2, min(folds, np.bincount(y).min()))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(feats, y):
        out[te] = ConfidenceModel().fit(feats.iloc[tr], y[tr]).predict(feats.iloc[te])
    return out


def threshold_for_risk(conf, correct, target_risk: float) -> float:
    """On dev data: lowest confidence threshold whose answered items have error <= target_risk."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    o = np.argsort(-conf, kind="stable")
    risk = np.cumsum(1 - correct[o]) / np.arange(1, len(o) + 1)
    ok = np.where(risk <= target_risk)[0]
    return float(conf[o][ok.max()]) if len(ok) else float("inf")


def review_priority(items: pd.DataFrame, conf, lookup, top_k: int = 5) -> np.ndarray:
    """E5: emissions.expected_emissions_error per item = 2022 spend x P(wrong) x spread of EPA
    factors over the prediction and the top-k candidates (median spread when none logged)."""
    from .emissions import expected_emissions_error, to_2022_usd
    spreads = []
    for _, r in items.iterrows():
        c = r.get("candidates") if isinstance(r.get("candidates"), list) else []
        codes = [str(r["pred"])] + [str(x.get("naics6", x.get("naics")) if isinstance(x, dict) else x) for x in c[:top_k]]
        f = [x for x in (lookup.factor(cd) for cd in codes) if x is not None]
        spreads.append(max(f) - min(f) if len(f) > 1 else np.nan)
    spreads = np.array(spreads, float)
    fill = np.nanmedian(spreads) if np.isfinite(spreads).any() else 1.0
    spreads = np.where(np.isnan(spreads), fill, spreads)
    amt = pd.to_numeric(items.get("amount_usd"), errors="coerce").fillna(0).clip(lower=0).to_numpy()
    years = items["year"].fillna(2024).astype(int).to_numpy() if "year" in items else np.full(len(items), 2024)
    return np.array([expected_emissions_error(to_2022_usd(a, y), 1 - c, [0.0, s])
                     for a, y, c, s in zip(amt, years, np.asarray(conf, float), spreads)])


def review_curve(err_kg, order):
    """Share of total emissions error removed after a human reviews items in `order`.
    Returns (share_reviewed, share_removed, area under the curve)."""
    err = np.nan_to_num(np.asarray(err_kg, float))
    tot = err.sum()
    removed = np.r_[0.0, np.cumsum(err[order]) / tot] if tot > 0 else np.zeros(len(order) + 1)
    x = np.r_[0.0, np.arange(1, len(order) + 1) / len(order)]
    return x, removed, float(np.sum(np.diff(x) * (removed[1:] + removed[:-1]) / 2))