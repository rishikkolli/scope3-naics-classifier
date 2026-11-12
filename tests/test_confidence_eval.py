"""Tests for src/evaluate.py and src/confidence.py.

Usage:  pip install pytest   then   python -m pytest -q tests
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import confidence as C
from src import evaluate as E

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lookup():
    from src.emissions import FactorLookup
    return FactorLookup(ROOT / "data" / "raw")


# ----------------------------------------------------------------------------- original functions
def test_original_functions_unchanged_behaviour():
    assert E.hierarchical_accuracy(["334111", "UNKNOWN"], ["334118", "x"], 4) == 1.0
    assert E.abstention_recall(["UNKNOWN", "334111"], np.array([True, False])) == 1.0
    assert E.emissions_error(np.array([10.0]), np.array([12.0]))["relative"] == pytest.approx(0.2)
    assert C.self_consistency(["a", "a", "b"]) == pytest.approx(2 / 3)
    assert C.retrieval_margin([0.5, 0.2]) == pytest.approx(0.3)
    assert C.abstain(np.array([0.2, 0.9]), 0.5).tolist() == [False, True]


def test_risk_coverage_runs_on_new_numpy_and_ece_bins():
    y = np.array([1, 1, 1, 0, 0])
    cov, risk, aurc = E.risk_coverage(y, np.array([.9, .8, .7, .2, .1]))
    assert risk[2] == 0 and risk[-1] == pytest.approx(0.4)
    assert aurc < E.risk_coverage(y, np.array([.1, .2, .3, .8, .9]))[2]
    assert E.ece(np.zeros(100), np.zeros(100)) == 0.0             # conf = 0 is no longer dropped
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=20000)
    assert E.ece(rng.uniform(size=20000) < conf, conf) < 0.02
    assert np.isnan(E.auroc(np.ones(4), np.arange(4)))


# ----------------------------------------------------------------------------- confidence
def test_signals_from_a_classify_prediction():
    df = pd.DataFrame([{"pred": "334111", "samples": ["334111"] * 4 + ["334118"], "verbal_conf": 80,
                        "candidates": ["334111", "334118"], "candidate_scores": [0.5, 0.3]}])
    f = C.extract_features(df).iloc[0]
    assert f["C1_self_consistency"] == pytest.approx(0.8) and f["C1_agree_4digit"] == 1.0
    assert f["C3_verbal"] == pytest.approx(0.8) and f["C2_retrieval_margin"] == pytest.approx(0.2) and f["C2_pred_rank"] == 1.0
    assert C.sample_entropy_conf(["a"] * 5) == pytest.approx(1.0)
    assert C.sample_entropy_conf(list("abcde")) == pytest.approx(0.0)


def test_confidence_model_and_cross_fit():
    rng = np.random.default_rng(1)
    n = 1500
    y = rng.uniform(size=n) < 0.5
    feats = pd.DataFrame({"C1_self_consistency": np.clip(y * 0.6 + rng.normal(0.2, 0.15, n), 0, 1),
                          "C3_verbal": rng.uniform(size=n)})
    feats.loc[:150, "C3_verbal"] = np.nan
    cm = C.ConfidenceModel().fit(feats[:1000], y[:1000])
    assert E.auroc(y[1000:], cm.predict(feats[1000:])) > 0.9
    assert abs(cm.coefficients()["C1_self_consistency"]) > abs(cm.coefficients()["C3_verbal"])
    oof = C.cross_fit(feats, y)
    assert not np.isnan(oof).any() and E.auroc(y, oof) > 0.9


def test_threshold_and_review_curve():
    assert C.threshold_for_risk(np.array([.1, .9, .5, .7]), np.array([0, 1, 0, 1]), 0.0) == pytest.approx(0.7)
    err = np.array([0, 0, 100, 1])
    _, removed, best = C.review_curve(err, np.argsort(-err))
    assert removed[1] == pytest.approx(100 / 101) and best > C.review_curve(err, np.argsort(err))[2]


def test_paired_bootstrap():
    a = pd.DataFrame({"item_id": range(300), "scorable": True, "acc6": np.r_[np.ones(150), np.zeros(150)]})
    b = a.copy()
    b.loc[150:239, "acc6"] = 1.0                              # loc is inclusive: 90 items fixed
    r = E.paired_bootstrap(a, b, "acc6", B=500)
    assert r["diff_b_minus_a"] == pytest.approx(0.3) and r["ci95"][0] > 0 and r["p_value"] < 0.01


# ----------------------------------------------------------------------------- real NAICS / EPA files
def test_gold_normalizer(lookup):
    n = E.GoldNormalizer(lookup)
    assert n("334111").note == "2022"
    assert n("UNKNOWN").note == "unknown"
    assert n("4238").note == "partial"
    assert n("443120").note == "legacy->2022"
    g = n("517311")                                           # 2017 code
    assert g.note == "2017->2022" and g.code in n.codes_2022


def test_score_items_evidence_levels_and_errors(lookup):
    df = pd.DataFrame([
        {"item_id": "1", "split": "t", "gold_naics": "334111", "pred": "334111", "amount_usd": 1000},
        {"item_id": "2", "split": "t", "gold_naics": "334111", "pred": "331110", "amount_usd": 1000,
         "candidates": ["331110", "334111"]},
        {"item_id": "3", "split": "t", "gold_naics": "334111", "pred": "541511", "amount_usd": 10,
         "candidates": ["541511"]},
        {"item_id": "4", "split": "t", "gold_naics": "4541", "pred": "454110", "amount_usd": 50, "evidence": "partial"},
        {"item_id": "5", "split": "t", "gold_naics": "UNKNOWN", "pred": "ABSTAIN", "amount_usd": 20,
         "evidence": "insufficient"},
        {"item_id": "6", "split": "t", "gold_naics": "517311", "pred": "517111", "amount_usd": 10},  # 2017 gold
    ])
    it = E.score_items(df, lookup).set_index("item_id")
    assert it.loc["1", "acc6"] and it.loc["1", "abs_err_kg"] == 0
    assert it.loc["2", "error_stage"] == "selection_error" and it.loc["2", "abs_err_kg"] > 100
    assert it.loc["3", "error_stage"] == "retrieval_miss"
    assert it.loc["4", "correct"] and it.loc["4", "evidence"] == "partial"
    assert not it.loc["5", "scorable"] and it.loc["5", "error_stage"] == "correct_abstention"
    assert it.loc["6", "acc6"]                                # 2022 prediction of a 2017 gold is right
    rep = E.evaluate_split(it.reset_index(), conf=np.array([.9, .3, .2, .8, .05, .7]), B=50)
    assert rep["accuracy"]["acc6"]["value"] == pytest.approx(0.5)   # items 1 and 6 right, of 1, 2, 3, 6
    assert rep["confidence"]["selective"][1]["abstain_rate_insufficient"] == 1.0


def test_load_run_formats(tmp_path):
    import json
    p = tmp_path / "S1.json"
    p.write_text(json.dumps({"run": "S1", "items": [{"item_id": "a", "naics6": "334111", "samples": ["334111"]}]}))
    q = tmp_path / "R2.json"
    q.write_text(json.dumps({"run": "R2", "items": [{"item_id": "a", "gold": "334111", "top20": ["334118", "334111"]}]}))
    assert E.load_run(p).loc[0, "pred"] == "334111"
    r = E.load_run(q)
    assert r.loc[0, "pred"] == "334118" and r.loc[0, "candidates"] == ["334118", "334111"]


def test_perfect_run_scores_100_percent(lookup, tmp_path):
    """gov_test item_ids repeat; gold must still be matched to the right item."""
    import json
    from src.data import load_gov
    test = load_gov("gov_test")
    p = tmp_path / "perfect.json"
    p.write_text(json.dumps({"run": "perfect", "items": [{"item_id": i, "naics6": g}
                                                         for i, g in zip(test.item_id, test.naics6)]}))
    it = E.score_items(E.attach_gold(E.load_run(p)), lookup)
    assert len(it) == len(test) and it["acc6"].mean() == 1.0