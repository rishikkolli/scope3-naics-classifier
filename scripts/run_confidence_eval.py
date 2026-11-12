"""S4: confidence & abstention (C-series) and full evaluation of a saved run.

Reads any eval/runs/<run>.json written by run_classifier.py, train_baseline.py,
train_baseline_proba.py or run_retrieval_eval.py, and reports per split:
  - acc@2/4/6 and factor-equivalent accuracy with 95% bootstrap CIs, emissions error
  - E4: AUROC of each confidence signal (C1, C2, C3, C5) alone and combined (C4)
  - selective prediction at 70/80/90% coverage (configs/config.yaml abstention.coverage_targets),
    abstention recall on `evidence` = insufficient items (design.md section 7)
  - E5: review queue by P(wrong) vs. by expected emissions error
  - error analysis by stage and type (design.md section 6)

C4 is fit on messy_dev and applied to messy_test when both exist; otherwise 5-fold
cross-fitting inside the split (gov_test has no dev split).

Writes eval/runs/<run>_S4.json (a row for summarize_runs.py), <run>_S4.md,
<run>_S4_items.csv and <run>_S4_<split>.png.

Usage:  python scripts/run_confidence_eval.py S0b_tfidf_logreg_proba [S1 S2 ...]
        python scripts/run_confidence_eval.py --compare S1 S2     (paired bootstrap, acc@6)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import confidence as C                    # noqa: E402
from src import evaluate as E                      # noqa: E402
from src.emissions import FactorLookup             # noqa: E402

RUNS = ROOT / "eval" / "runs"
COVERAGES = (0.7, 0.8, 0.9)   # configs/config.yaml abstention.coverage_targets


def md_table(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load_items(run: str, lookup: FactorLookup) -> pd.DataFrame:
    df = E.attach_gold(E.load_run(RUNS / f"{run}.json"))
    return E.score_items(df[df["split"].notna()], lookup)


def c4_confidence(items: pd.DataFrame) -> tuple[np.ndarray, dict]:
    feats = C.extract_features(items)
    signals = C.usable_signals(feats)
    conf, info = np.full(len(items), np.nan), {}
    if not signals:
        return conf, {"note": "no confidence signal logged in this run"}
    y = items["correct"].astype(int).to_numpy()
    dev, test = (items.split == "messy_dev").to_numpy(), (items.split == "messy_test").to_numpy()
    if dev.sum() >= 30 and test.any() and len(set(y[dev])) == 2:
        cm = C.ConfidenceModel(signals).fit(feats[dev], y[dev])
        conf[dev | test] = cm.predict(feats[dev | test])
        info["messy_test"] = {"method": f"fit on messy_dev (n={int(dev.sum())})", "coefficients": cm.coefficients()}
    for split in items.split.unique():
        m = (items.split == split).to_numpy() & np.isnan(conf)
        if m.sum() >= 20 and len(set(y[m])) == 2:
            conf[m] = C.cross_fit(feats[m][signals], y[m])
            info[split] = {"method": "5-fold cross-fit within split",
                           "coefficients": C.ConfidenceModel(signals).fit(feats[m][signals], y[m]).coefficients()}
    return conf, info


def e4_signals(items: pd.DataFrame, conf: np.ndarray, m: np.ndarray) -> pd.DataFrame:
    feats, y = C.extract_features(items[m]), items.loc[m, "correct"].astype(int).to_numpy()
    rows = []
    for s in C.usable_signals(feats):
        v = feats[s].to_numpy(float)
        ok = ~np.isnan(v)
        rows.append({"signal": s, "logged": float(ok.mean()), "auroc": E.auroc(y[ok], v[ok])})
    rows.append({"signal": "C4_combined", "logged": 1.0, "auroc": E.auroc(y, conf[m])})
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def e5_review(items: pd.DataFrame, conf: np.ndarray, m: np.ndarray, lookup: FactorLookup) -> dict:
    m = m & items.emis_ok.astype(bool).to_numpy()
    if not m.any():
        return {}
    sub, c, err = items[m], conf[m], items.loc[m, "abs_err_kg"].to_numpy(float)
    orders = {"by P(wrong)": np.argsort(c, kind="stable"),
              "by expected emissions error": np.argsort(-C.review_priority(sub, c, lookup), kind="stable"),
              "random": np.random.default_rng(0).permutation(len(sub))}
    out = {}
    for name, o in orders.items():
        x, removed, area = C.review_curve(err, o)
        out[name] = {"removed@10%": float(np.interp(.1, x, removed)),
                     "removed@20%": float(np.interp(.2, x, removed)), "area": area, "curve": (x, removed)}
    return out


def plot(items: pd.DataFrame, conf: np.ndarray, m: np.ndarray, e5: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y, c = items.loc[m, "correct"].astype(float).to_numpy(), conf[m]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cov, risk, aurc = E.risk_coverage(y, c)
    ax[0].plot(cov, risk, color="#2a6fdb")
    ax[0].axhline(1 - y.mean(), ls="--", color="grey", lw=1, label="no abstention")
    ax[0].set(xlabel="coverage (share answered)", ylabel="error rate of answered items",
              title=f"Risk-coverage (AURC {aurc:.3f})", ylim=(0, 1))
    ax[0].legend(frameon=False)
    r = E.reliability(y, c).dropna()
    ax[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax[1].scatter(r.mean_conf, r.accuracy, s=np.clip(r.n * 3, 10, 300), color="#2a6fdb")
    ax[1].set(xlabel="confidence", ylabel="accuracy", title=f"Reliability (ECE {E.ece(y, c):.3f})",
              xlim=(0, 1), ylim=(0, 1))
    for (name, v), col in zip(e5.items(), ("#2a6fdb", "#d9480f", "grey")):
        ax[2].plot(*v["curve"], color=col, label=f"{name} (area {v['area']:.2f})")
    ax[2].set(xlabel="share of items reviewed", ylabel="share of emissions error removed",
              title="E5 review queue", xlim=(0, 1), ylim=(0, 1))
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ci(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def evaluate_run(run: str, lookup: FactorLookup, B: int) -> None:
    items = load_items(run, lookup)
    conf, conf_info = c4_confidence(items)
    items["conf"] = conf
    items.drop(columns=[c for c in ("candidates", "samples", "top20", "candidate_scores") if c in items]
               ).to_csv(RUNS / f"{run}_S4_items.csv", index=False)
    md, metrics = [f"# S4 evaluation: {run}", ""], {}
    for split in sorted(items.split.unique()):
        m = (items.split == split).to_numpy()
        has_conf = not np.isnan(conf[m]).any()
        rep = E.evaluate_split(items[m], conf[m] if has_conf else None, COVERAGES, B=B)
        a, em = rep["accuracy"], rep["emissions"]
        md += [f"## {split}  (n={rep['n_items']}, scorable={rep['n_scorable']})", "",
               f"Gold code versions: {rep['gold_notes']}; evidence: {rep['evidence']}", "",
               "| Metric | Value [95% CI] |", "|---|---|",
               f"| acc@2 / acc@4 / acc@6 | {ci(a['acc2'])} / {ci(a['acc4'])} / {ci(a['acc6'])} |",
               f"| factor_equiv_acc | {ci(a['factor_equiv'])} |",
               f"| emissions_rel_error | {em['relative']:.3f} [{em['relative_ci95'][0]:.3f}, {em['relative_ci95'][1]:.3f}] |",
               f"| median_item_factor_rel_error | {em['median_item']:.3f} |",
               f"| emissions_net_bias_rel | {em['net_bias_rel']:+.3f} |"]
        md += [f"| {k} | {v:.3f} |" for k, v in rep["retrieval"].items()]
        pre = "" if split == "gov_test" else f"{split}:"
        metrics.update({f"{pre}acc@2": round(a["acc2"]["value"], 4), f"{pre}acc@4": round(a["acc4"]["value"], 4),
                        f"{pre}acc@6": round(a["acc6"]["value"], 4),
                        f"{pre}factor_equiv_acc": round(a["factor_equiv"]["value"], 4),
                        f"{pre}emissions_rel_error": round(em["relative"], 4),
                        f"{pre}median_item_factor_rel_error": round(em["median_item"], 4)})
        if has_conf:
            c = rep["confidence"]
            md += [f"| C4 AUROC | {ci(c['auroc'])} |",
                   f"| C4 AUROC, factor-equivalent correctness | {c['auroc_factor_equivalent']:.3f} |",
                   f"| AURC (lower is better) | {c['aurc']:.3f} |", f"| ECE | {c['ece']:.3f} |", "",
                   f"C4: {conf_info.get(split, {}).get('method')}; "
                   f"coefficients {conf_info.get(split, {}).get('coefficients')}", "",
                   md_table(pd.DataFrame([{"coverage": f"{s['coverage']:.0%}", "acc_answered": s["acc_answered"],
                                           "emissions_error_after_review": s.get("emissions_error_after_review", np.nan),
                                           "abstention_recall": s.get("abstain_rate_insufficient", np.nan)}
                                          for s in c["selective"]])),
                   "", "**E4: each confidence signal alone (AUROC for predicting a correct answer)**", "",
                   md_table(e4_signals(items, conf, m))]
            e5 = e5_review(items, conf, m, lookup)
            if e5:
                md += ["", "**E5: review queue order (share of emissions error removed)**", "",
                       md_table(pd.DataFrame([{"order": k, "10% reviewed": v["removed@10%"],
                                               "20% reviewed": v["removed@20%"], "area": v["area"]}
                                              for k, v in e5.items()]))]
                png = RUNS / f"{run}_S4_{split}.png"
                plot(items, conf, m, e5, png, f"{run} - {split}")
                md += ["", f"![S4 figures]({png.name})"]
            metrics.update({f"{pre}auroc": round(c["auroc"]["value"], 4), f"{pre}aurc": round(c["aurc"], 4),
                            f"{pre}ece": round(c["ece"], 4),
                            f"{pre}acc@80%cov": round(c["selective"][1]["acc_answered"], 4)})
        md += ["", "**Error analysis (design.md section 6)**", "", md_table(pd.DataFrame(rep["error_analysis"])), ""]
    (RUNS / f"{run}_S4.md").write_text("\n".join(md), encoding="utf-8")
    (RUNS / f"{run}_S4.json").write_text(json.dumps(
        {"run": f"{run}+S4", "n": int(len(items)), "metrics": metrics, "c4": conf_info},
        indent=1, default=float), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved eval/runs/{run}_S4.md (+ .json, _items.csv, figures)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--metric", default="acc6", choices=["acc2", "acc4", "acc6", "factor_equiv"])
    ap.add_argument("--B", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args()
    if not args.runs and not args.compare:
        ap.error("give run names (e.g. S1) or --compare A B")
    lookup = FactorLookup(ROOT / "data" / "raw")
    if args.compare:
        a, b = (load_items(r, lookup) for r in args.compare)
        for split in sorted(set(a.split) & set(b.split)):
            r = E.paired_bootstrap(a[a.split == split], b[b.split == split], args.metric, B=args.B)
            print(f"{split}: {args.compare[1]} - {args.compare[0]} on {args.metric} = {r['diff_b_minus_a']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p = {r['p_value']:.3f}, n = {r['n_items']}")
    for run in args.runs:
        evaluate_run(run, lookup, args.B)


if __name__ == "__main__":
    main()