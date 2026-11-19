# Baseline, few-shot and emissions results

Milestone-3 runs for the non-LLM baselines, the few-shot example selector, query
rewriting and the emissions calculation. Everything here is local and free to re-run —
no API calls.

These scripts only need pandas, numpy, scikit-learn and openpyxl — not the retrieval or
LLM stack in `requirements.txt`. `reportlab` is needed only to rebuild the report PDF:

```bash
python -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn scipy reportlab openpyxl
```

```bash
python scripts/analyze_emissions_sensitivity.py       # emissions structure       (~1 min)
python scripts/run_baseline_ablations.py              # A1-A8 + recall ceilings   (~5 min)
python scripts/run_baseline_ablations.py --combined   # A9/A10 + paired bootstrap (~3 min)
```

`scripts/build_milestone3_pdf.py` quotes these numbers but hardcodes them, like the rest
of the Milestone 3 report, so it does not need the runs above to build.

Frozen outputs land in `eval/runs/ablations_s0b.json` and
`eval/runs/emissions_sensitivity.json`. Those paths are gitignored, so this file is the
committed record. All numbers are on the 299-item `gov_test` split, trained on the
6,601-row / 436-code `gov_pool`.

## S0b ablations

| variant | acc@2 | acc@4 | acc@6 | factor_equiv | emissions_rel_error | excl. worst item | fit |
|---|---|---|---|---|---|---|---|
| A1 description only | 0.7324 | 0.6823 | 0.6589 | 0.6960 | 0.0541 | 0.0122 | 16s |
| **A2 description + recipient (= published S0b)** | 0.8194 | 0.7559 | 0.7291 | 0.7509 | 0.2549 | 0.0133 | 21s |
| A3 char_wb 3-5 only | 0.8127 | 0.7559 | 0.7157 | 0.7436 | 0.0104 | 0.0073 | 120s |
| A4 word + char_wb 3-5 | 0.8495 | 0.7926 | 0.7625 | 0.7766 | 0.2546 | 0.0130 | 114s |
| A5 + log-amount bucket | 0.8294 | 0.7726 | 0.7458 | 0.7692 | 0.0525 | 0.0105 | 19s |
| A6 normalized text (R5-lite) | 0.8161 | 0.7559 | 0.7291 | 0.7473 | 0.0771 | 0.0088 | 17s |
| A7 LinearSVC (word) | 0.8662 | 0.7860 | 0.7492 | 0.7729 | 0.0117 | 0.0043 | 1s |
| A8 kNN cosine, k=5 | 0.7726 | 0.6756 | 0.6421 | 0.6740 | 0.0200 | 0.0087 | 0.2s |

A2 reproduces the published S0b exactly (`docs/results.md`), so the deltas are like-for-like.

## Stacked variants and significance

| system | acc@2 | acc@4 | acc@6 | factor_equiv | emissions_rel_error | fit | acc@6 vs S0b (paired bootstrap, 5k) |
|---|---|---|---|---|---|---|---|
| A2 (= S0b) | 0.8194 | 0.7559 | 0.7291 | 0.7509 | 0.2549 | 19s | — |
| A9 word+char, LinearSVC | 0.8662 | 0.8094 | 0.7692 | 0.7875 | 0.0065 | 7s | **+0.040 [+0.007, +0.074]**, p = 0.024, 20W/8L |
| **A10 word+char, LinearSVC, +amount** | **0.8696** | **0.8127** | **0.7759** | **0.7949** | 0.0078 | 8s | **+0.047 [+0.013, +0.084]**, p = 0.011, 22W/8L |

For reference, `S3_logreg_full` is acc@6 0.722 — **below A10**. A10 should
become the baseline, but `LinearSVC` has no `predict_proba`, so it needs Platt scaling
before the C-series confidence model can consume it.

## Ceilings the LLM stages inherit

| k | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| gold code in logreg top-k (candidate ceiling, S1–S3) | 0.7291 | 0.8227 | 0.8462 | 0.8696 | 0.8930 | 0.9197 |
| gold code among k nearest pool rows (few-shot ceiling, S2–S3) | 0.7425 | 0.7993 | 0.8361 | 0.8796 | 0.8963 | — |
| same, on R5-lite normalized text | 0.7258 | 0.7893 | 0.8194 | 0.8528 | 0.8863 | — |

The few-shot index in `scripts/run_classifier.py` is fit on `"<desc> <recipient>"` but
queried with `"<desc> (supplier: <recipient>)"`. Checked: the mismatch changes nothing
(identical recall at every k, because the extra token is constant across queries). Dropping
the recipient from the query *does* matter — recall@5 falls 0.8361 → 0.7592.

## Query rewriting (R5-lite): negative result

Deterministic normalizer (strip `IGF::OT::IGF`, delivery/task-order phrases, bare
alphanumeric part numbers): rewrote 133 / 299 descriptions, −15.2 characters on average,
emptied 3 descriptions entirely. acc@6 unchanged (0.7291), acc@2 slightly worse
(0.8161 vs 0.8194), few-shot recall@5 worse (0.8194 vs 0.8361).

Cause: IDF already discounts text that appears in every record, so removing it adds no
information to a TF-IDF model, and the part-number regex deletes real product tokens.
The real `llm_rewrite` (implemented, unrun) is still worth testing on the messy split,
where the noise is abbreviation rather than repetition.

## Emissions structure

- **Concentration.** One award (row 46, $11.7B LOGCAP IV, 561210) is **93.6%** of the split's
  true emissions; top 5 = 96.9%, top 10 = 98.5%. It is classified correctly by every system,
  so it sits in the denominator. Numerator: row 102 ($631M TRICARE pharmacy, gold 524114,
  S0b predicts 562910, factor 0.033 → 0.988) is **94.8%** of S0b's error.
  `emissions_rel_error` on `gov_test` is therefore a two-item statistic.
- **Error by confusion distance (S0b).** cross-sector 47 items → 99.4% of emissions error
  (median factor ratio 1.76×); same-2-digit 16 items → 0.5% (1.20×); same-4-digit 5 items →
  0.02% (1.11×); factor-equivalent 5 items → 0%. Sector accuracy is what matters; the last
  two digits barely do.
- **4-digit oracle.** Correct 4-digit parent + most common 6-digit child in the pool: acc@6
  0.7759, emissions error 0.002. Motivates hierarchical decoding.
- **Ruled out.** The 2022→2017 concordance affects 4 test rows, max factor ratio within a
  split 1.38×. Margins on/off changes the total by 0.07%. Every award is FY2024, so the
  deflator is one scalar (0.933) that cancels out of all relative metrics — the CPI→PCE
  `TODO` in `src/emissions.py` is low priority.
- **Coverage.** 8.7% of gold codes have no EPA factor and drop out of the emissions metrics.
  1,016 codes have one; factor p90/p10 = 8.0, min 0.029, max 3.924.

## Baseline failure structure

- Predicts 90 distinct codes where gold uses 128 (of 436 available); top-5 classes take
  26.4% of predictions vs 17.4% of gold.
- acc@6 **0.000** on items whose gold code has ≤2 pool examples (9 items); 0.813 where it has
  >10. 7 items (2.3%) have a gold code absent from the pool — unreachable by construction.
- Mean predicted factor is 10.0% above the gold mean; high-factor classes (562910) act as
  magnets. This is the upward bias behind `emissions_net_bias_rel` in `docs/results.md`.
- Short text hurts: acc@6 0.630 on descriptions of ≤3 tokens (15.4% of rows) vs 0.747 on
  longer ones. 12.7% of descriptions contain `IGF::`.
- A majority vote over the same five neighbours the LLM sees gets 0.642 (A8), against S3's
  0.722 — the LLM does use those examples better than nearest-neighbour voting.
