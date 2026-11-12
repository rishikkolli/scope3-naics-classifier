"""Evaluation metrics, reported separately for `gov` and `messy` splits.

Part 1 - metric functions (used by scripts/train_baseline.py and scripts/score_run.py).
Part 2 - the full harness (used by scripts/run_eval.py): per-item scoring with gold-label
         normalization, the `evidence` levels of design.md section 7, selective prediction,
         bootstrap confidence intervals, paired tests and the section 6 error analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


UNKNOWN = "UNKNOWN"
ABSTAIN = "ABSTAIN"


# =========================================================================== part 1: metrics
def hierarchical_accuracy(gold: list[str], pred: list[str], digits: int) -> float:
    """Over answerable rows only; UNKNOWN-gold rows never enter the denominator
    (design.md section 7). Partial-evidence gold prefixes score at their own length."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g != UNKNOWN]
    if not pairs:
        return float("nan")
    return float(np.mean([g[:digits] == p[:digits] for g, p in pairs if len(g) >= digits]))


def abstention_recall(gold: list[str], abstained: np.ndarray) -> float:
    """Share of unanswerable (gold=UNKNOWN) items the system abstained on —
    the only correct behavior for them."""
    mask = np.array([g == UNKNOWN for g in gold])
    return float(abstained[mask].mean()) if mask.any() else float("nan")


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
    correct = np.asarray(correct).astype(int)
    if correct.min() == correct.max():          # one class only: undefined
        return float("nan")
    return float(roc_auc_score(correct, conf))


def risk_coverage(correct: np.ndarray, conf: np.ndarray):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    errs = 1 - np.asarray(correct)[order].astype(float)
    n = np.arange(1, len(errs) + 1)
    coverage = n / len(errs)
    risk = np.cumsum(errs) / n
    aurc = float(np.mean(risk))   # was np.trapz, which NumPy >= 2.4 removed
    return coverage, risk, aurc


def reliability(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin confidence vs. accuracy (the reliability diagram behind ECE)."""
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    b = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)   # conf = 0 lands in bin 0
    return pd.DataFrame([{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": int((b == i).sum()),
                          "mean_conf": float(conf[b == i].mean()) if (b == i).any() else np.nan,
                          "accuracy": float(correct[b == i].mean()) if (b == i).any() else np.nan}
                         for i in range(n_bins)])


def ece(correct: np.ndarray, conf: np.ndarray, n_bins: int = 10) -> float:
    r = reliability(correct, conf, n_bins)
    r = r[r.n > 0]
    return float((r.n / r.n.sum() * (r.accuracy - r.mean_conf).abs()).sum())


# =========================================================================== part 2: harness
# Pre-2017 codes still found in USAspending FY2024 labels (9 of 299 gov_test rows). Not in the
# 2017<->2022 concordance, so mapped by hand. Check against census.gov/naics before relying on it.
LEGACY_TO_2022: dict[str, list[str]] = {
    "443120": ["449210"],            # computer & software stores (2012) -> electronics & appliance retailers
    "443130": ["449210"],            # camera & photographic supplies stores (2012)
    "517110": ["517111"],            # wired telecom carriers (2012)
    "541712": ["541714", "541715"],  # R&D physical/engineering/life sciences (2012), split in 2017
    "541710": ["541714", "541715"],
    "333319": ["333310"],            # other commercial & service machinery mfg (2012)
    "315999": ["315990"],
}


@dataclass
class Gold:
    raw: str
    code: str | None                 # primary 2022 code; None for UNKNOWN / partial / invalid
    acceptable: set[str] = field(default_factory=set)
    prefix: str | None = None        # partial-evidence label, e.g. "4238"
    note: str = ""                   # 2022 | 2017->2022 | legacy->2022 | partial | unknown | invalid


class GoldNormalizer:
    """Gold labels mix 2022, 2017 and older codes (37 of 299 gov_test rows are not 2022 codes).
    The classifier predicts 2022 codes, so a 2017 gold is converted and every successor
    2022 code counts as correct. Built on emissions.FactorLookup's concordance."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.codes_2022 = set(lookup.conc)
        self.to_2022: dict[str, list[str]] = {}
        for new, olds in lookup.conc.items():
            for o in olds:
                self.to_2022.setdefault(o, []).append(new)

    def __call__(self, label) -> Gold:
        raw = "" if label is None or (isinstance(label, float) and np.isnan(label)) else str(label).strip()
        if raw.upper() in {"", UNKNOWN, "NAN", "NONE"}:
            return Gold(raw, None, note="unknown")
        if raw.isdigit() and 2 <= len(raw) <= 5:
            return Gold(raw, None, prefix=raw, note="partial")
        if not (raw.isdigit() and len(raw) == 6):
            return Gold(raw, None, note="invalid")
        if raw in self.codes_2022:
            return Gold(raw, raw, {raw}, note="2022")
        new = sorted(set(self.to_2022.get(raw, [])))
        if new:
            return Gold(raw, new[0], set(new), note="2017->2022")
        legacy = [c for c in LEGACY_TO_2022.get(raw, []) if c in self.codes_2022]
        if legacy:
            return Gold(raw, legacy[0], set(legacy), note="legacy->2022")
        return Gold(raw, None, note="invalid")

    def true_factor(self, g: Gold) -> float | None:
        """A 2017 gold code uses its own EPA factor; otherwise the 2022 code's factor."""
        return self.lookup.factor(g.raw) or (self.lookup.factor(g.code) if g.code else None)


def load_run(path: Path) -> pd.DataFrame:
    """Read any eval/runs/<run>.json written by this repo into one row per item:
    S-series (classify.Prediction dicts), S0 baselines ({item_id, gold, pred}) and R-series
    (ranked codes only -> top-1 is the prediction). Extra keys are kept (e.g. candidate_scores,
    confidence)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for it in data["items"]:
        r = dict(it)
        r["pred"] = r.pop("naics6", r.get("pred"))
        if r.get("pred") is None:
            ranked = r.get("top20") or r.get("ranked") or []
            r["pred"] = ranked[0] if ranked else ""
            r.setdefault("candidates", ranked)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["pred"] = df["pred"].fillna("").astype(str).str.strip()
    df["run"] = data.get("run", Path(path).stem)
    return df


def attach_gold(df: pd.DataFrame, fy_year: int = 2024) -> pd.DataFrame:
    """Join gold label, amount, split and evidence: gov items from data/interim/gov_test.csv,
    messy items from data/labeled/messy_final.csv (scripts/adjudicate_labels.py).

    gov_test "Award ID" values repeat (251 distinct in 299 rows), so gov items are matched by
    position when the run lists them in gov_test order (every script here does, --limit keeps a
    prefix); only otherwise by item_id."""
    from .data import load_gov, load_messy_labeled
    df = df.reset_index(drop=True)
    gov = load_gov("gov_test").rename(columns={"naics6": "gold_naics", "amount": "amount_usd"})
    gov = gov.assign(split="gov_test", evidence="sufficient", year=fy_year)
    cols = ["split", "gold_naics", "amount_usd", "evidence", "year"]
    out = df.drop(columns=[c for c in cols + ["gold"] if c in df.columns])
    for c in cols:
        out[c] = None
    n = len(df)
    ids = df["item_id"].astype(str).tolist()
    if n <= len(gov) and ids == gov["item_id"].astype(str).head(n).tolist():
        out[cols] = gov[cols].head(n).to_numpy()
    else:
        g = gov.drop_duplicates("item_id", keep=False).set_index("item_id")
        hit = out["item_id"].isin(g.index)
        out.loc[hit, cols] = g.loc[out.loc[hit, "item_id"], cols].to_numpy()
    messy = load_messy_labeled()
    if messy is not None:
        m = messy.reset_index() if "item_id" not in messy.columns else messy
        m = m.rename(columns={"naics6": "gold_naics"}).assign(year=2024).set_index("item_id")
        m["amount_usd"] = pd.to_numeric(m["amount_usd"], errors="coerce")
        hit = out["item_id"].isin(m.index) & out["split"].isna()
        out.loc[hit, cols] = m.loc[out.loc[hit, "item_id"], cols].to_numpy()
    if "gold" in df.columns:                      # S0 baseline files carry their own gold
        out["gold_naics"] = out["gold_naics"].fillna(df["gold"])
    out["amount_usd"] = pd.to_numeric(out["amount_usd"], errors="coerce")
    return out


def _candidates(r) -> list[tuple[str, float | None]]:
    c = r.get("candidates")
    if not isinstance(c, list):
        return []
    s = r.get("candidate_scores") if isinstance(r.get("candidate_scores"), list) else []
    out = []
    for i, x in enumerate(c):
        if isinstance(x, dict):
            out.append((str(x.get("naics6", x.get("naics"))), x.get("score")))
        else:
            out.append((str(x), s[i] if i < len(s) else None))
    return out


def score_items(df: pd.DataFrame, lookup) -> pd.DataFrame:
    """Per-item columns used by every metric: gold_code, gold_note, scorable, abstained,
    correct, acc2/acc4/acc6, factor_equiv, gold_rank, true_kg, pred_kg, abs_err_kg, emis_ok,
    error_stage, error_type. Spend is deflated to 2022 USD with emissions.to_2022_usd."""
    from .emissions import to_2022_usd
    norm = GoldNormalizer(lookup)
    recs = []
    for _, r in df.iterrows():
        g, pred = norm(r.get("gold_naics")), str(r["pred"]).strip()
        abst = pred.upper() == ABSTAIN or pred == ""
        ev = r.get("evidence") if isinstance(r.get("evidence"), str) and r.get("evidence") else (
            {"unknown": "insufficient", "partial": "partial"}.get(g.note, "sufficient"))
        rec = {"gold_code": g.code, "gold_note": g.note, "evidence": ev, "abstained": abst}
        if g.code:
            # both sides in 2022 codes: a model trained on gov_pool also predicts 2017/older codes
            pv = norm(pred).acceptable | {pred}
            gv = g.acceptable | {g.raw}
            acc = {d: (not abst) and any(p[:d] == a[:d] for p in pv for a in gv) for d in (2, 4, 6)}
            gg, gp = lookup.group(g.code) or lookup.group(g.raw), lookup.group(pred)
            rec.update(scorable=True, correct=acc[6], acc2=acc[2], acc4=acc[4], acc6=acc[6],
                       factor_equiv=(not abst) and (acc[6] or (gg is not None and gg == gp)))
        elif g.prefix:
            ok = (not abst) and pred.startswith(g.prefix)
            rec.update(scorable=True, correct=ok, acc6=np.nan, factor_equiv=np.nan,
                       acc2=(not abst) and pred[:2] == g.prefix[:2],
                       acc4=((not abst) and pred[:4] == g.prefix[:4]) if len(g.prefix) >= 4 else np.nan)
        else:
            rec.update(scorable=False, correct=False, acc2=np.nan, acc4=np.nan, acc6=np.nan, factor_equiv=np.nan)

        cands = _candidates(r)
        rank = np.nan
        if cands and g.acceptable:
            rank = next((i for i, (c, _) in enumerate(cands, 1) if c in g.acceptable), np.inf)
        rec["gold_rank"] = rank

        amt = pd.to_numeric(r.get("amount_usd"), errors="coerce")
        f_true = norm.true_factor(g) if g.code else None
        ok = f_true is not None and pd.notna(amt) and amt > 0
        a22 = to_2022_usd(float(amt), int(r.get("year") or 2024)) if ok else np.nan
        f_pred = None if abst else lookup.factor(pred)
        rec["true_kg"] = a22 * f_true if ok else np.nan
        # abstained = no estimate until a human reviews it; an unmappable prediction = 0 kg (as train_baseline)
        rec["pred_kg"] = (a22 * (f_pred or 0.0)) if ok else np.nan
        rec["emis_ok"] = bool(ok)
        recs.append(rec)
    add = pd.DataFrame(recs)
    out = df.reset_index(drop=True).drop(columns=[c for c in add.columns if c in df.columns]).join(add)
    out["abs_err_kg"] = (out["pred_kg"] - out["true_kg"]).abs().where(out["emis_ok"])
    out["error_stage"], out["error_type"] = zip(*[_categorize(r) for _, r in out.iterrows()])
    return out


def _categorize(r) -> tuple[str, str]:
    """design.md section 6. Stage = where it went wrong; type = automatic first guess
    ('goods vs services' and 'label noise' still need a human pass over the items file)."""
    if r["evidence"] == "insufficient":
        return ("correct_abstention" if r["abstained"] else "should_abstain"), "vague_description"
    if not r["scorable"]:
        return "unscorable_gold", "label_noise"
    if r["abstained"]:
        return "abstained", ""
    if bool(r["correct"]):
        return "correct", ""
    rank = r["gold_rank"]
    stage = ("no_candidates_logged" if pd.isna(rank) else
             "retrieval_miss" if np.isinf(rank) else "selection_error")
    if r["evidence"] == "partial":
        typ = "vague_description"
    elif r["gold_note"] in ("2017->2022", "legacy->2022"):
        typ = "naics_version_mismatch"
    elif r["acc4"] is True:
        typ = "near_neighbor"
    else:
        typ = "other"
    return stage, typ


def error_analysis(items: pd.DataFrame) -> pd.DataFrame:
    """Each error stage/type by count AND by share of total emissions error."""
    wrong = items[~items["error_stage"].isin(["correct", "correct_abstention"])]
    tot = items["abs_err_kg"].sum()
    g = wrong.groupby(["error_stage", "error_type"], dropna=False)
    t = pd.DataFrame({"count": g.size(), "emissions_error_kg": g["abs_err_kg"].sum()})
    t["share_of_emissions_error"] = t["emissions_error_kg"] / tot if tot > 0 else np.nan
    return t.sort_values(["emissions_error_kg", "count"], ascending=False).reset_index()


def emissions_summary(items: pd.DataFrame) -> dict:
    """Relative error (sum |pred-true| / sum true), per-item median, net bias."""
    m = items[items["emis_ok"].astype(bool)]
    if m.empty:
        return {"n": 0, "relative": np.nan, "median_item": np.nan, "net_bias_rel": np.nan}
    t, p = m["true_kg"].to_numpy(float), m["pred_kg"].to_numpy(float)
    e = emissions_error(t, p)
    return {"n": int(len(m)), "relative": e["relative"], "net_bias_rel": e["net_bias_kg"] / max(t.sum(), 1e-9),
            "median_item": float(np.median(np.abs(p - t) / t))}


def recall_at_k(items: pd.DataFrame, ks=(1, 5, 10, 20, 50)) -> dict:
    r = items["gold_rank"].dropna()
    return {f"recall@{k}": float((r <= k).mean()) for k in ks} if len(r) else {}


def selective(items: pd.DataFrame, conf, coverage: float) -> dict:
    """Answer only the `coverage` share with highest confidence; the rest go to human review
    (assumed fixed). Accuracy of what is answered, emissions error left after review, and
    abstention rate by evidence level (insufficient should be high, sufficient low)."""
    from .confidence import abstain
    keep = abstain(np.asarray(conf, float), coverage)
    s = items["scorable"].astype(bool).to_numpy()
    e = items["emis_ok"].astype(bool).to_numpy()
    out = {"coverage": coverage,
           "acc_answered": float(items["correct"].to_numpy()[keep & s].mean()) if (keep & s).any() else np.nan}
    if e.any():
        err = items["abs_err_kg"].to_numpy(float)
        out["emissions_error_after_review"] = float(np.where(keep, err, 0)[e].sum() / items["true_kg"][e].sum())
    ev = items["evidence"].to_numpy()
    for lvl in ("sufficient", "partial", "insufficient"):
        if (ev == lvl).any():
            out[f"abstain_rate_{lvl}"] = float((~keep[ev == lvl]).mean())
    return out


def _point(items: pd.DataFrame, conf, metric: str) -> float:
    s = items[items["scorable"].astype(bool)]
    if metric in ("acc2", "acc4", "acc6", "factor_equiv", "correct"):
        v = pd.to_numeric(s[metric], errors="coerce").dropna()
        return float(v.mean()) if len(v) else np.nan
    if metric == "emissions_relative":
        return emissions_summary(items)["relative"]
    if metric == "auroc":
        return auroc(items["correct"].to_numpy(), conf)
    if metric == "aurc":
        return risk_coverage(items["correct"].to_numpy(), conf)[2]
    if metric == "ece":
        return ece(items["correct"].to_numpy(), conf)
    raise KeyError(metric)


def bootstrap_ci(items: pd.DataFrame, metric: str, conf=None, B: int = 1000, seed: int = 0) -> dict:
    """Point value and 95% percentile bootstrap interval over items."""
    rng, n = np.random.default_rng(seed), len(items)
    conf = None if conf is None else np.asarray(conf, float)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(_point(items.iloc[i], None if conf is None else conf[i], metric))
    return {"value": _point(items, conf, metric),
            "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = "acc6",
                     B: int = 2000, seed: int = 0) -> dict:
    """Run B minus run A on the SAME items (joined on item_id): difference, 95% CI, p-value.
    Use this to claim that run N improved on run N-1 (design.md Stages)."""
    # key = item_id + occurrence number, since gov_test item_ids can repeat
    key = lambda d: d["item_id"].astype(str) + "#" + d.groupby("item_id").cumcount().astype(str)  # noqa: E731
    a, b = a.assign(_key=key(a)).set_index("_key"), b.assign(_key=key(b)).set_index("_key")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids].reset_index(drop=True), b.loc[ids].reset_index(drop=True)
    rng, n, diffs = np.random.default_rng(seed), len(ids), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        diffs.append(_point(b.iloc[i], None, metric) - _point(a.iloc[i], None, metric))
    d = _point(b, None, metric) - _point(a, None, metric)
    diffs = np.array(diffs, float)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"metric": metric, "n_items": int(n), "diff_b_minus_a": float(d),
            "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
            "p_value": float(min(p, 1.0))}


def evaluate_split(items: pd.DataFrame, conf=None, coverages=(0.7, 0.8, 0.9), B: int = 1000) -> dict:
    s = items[items["scorable"].astype(bool)]
    rep = {"n_items": int(len(items)), "n_scorable": int(len(s)),
           "gold_notes": items["gold_note"].value_counts().to_dict(),
           "evidence": items["evidence"].value_counts().to_dict(),
           "accuracy": {m: bootstrap_ci(items, m, B=B) for m in ("acc2", "acc4", "acc6", "factor_equiv")},
           "emissions": emissions_summary(items) | {"relative_ci95": bootstrap_ci(items, "emissions_relative", B=B)["ci95"]},
           "retrieval": recall_at_k(items),
           "error_analysis": error_analysis(items).to_dict("records")}
    if "cost_usd" in items:
        rep["cost_usd_total"] = float(pd.to_numeric(items["cost_usd"], errors="coerce").sum())
    if conf is not None:
        conf = np.asarray(conf, float)
        fe = pd.to_numeric(items["factor_equiv"], errors="coerce").to_numpy()
        m = ~np.isnan(fe)
        rep["confidence"] = {
            "auroc": bootstrap_ci(items, "auroc", conf, B=B),
            "auroc_factor_equivalent": auroc(fe[m], conf[m]) if m.any() else np.nan,
            "aurc": risk_coverage(items["correct"].to_numpy(), conf)[2],
            "ece": ece(items["correct"].to_numpy(), conf),
            "abstention_recall_at_80": selective(items, conf, 0.8).get("abstain_rate_insufficient", np.nan),
            "selective": [selective(items, conf, c) for c in coverages]}
    return rep