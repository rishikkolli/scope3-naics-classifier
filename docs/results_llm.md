# LLM stage results (first 50 gov_test rows)

Model: `claude-haiku-4-5`, 5 self-consistency samples per item, majority vote, k=10 candidates.
All rows are scored on the **same first 50 rows** of `gov_test`:

```
python scripts/score_run.py S0a_retrieval_top1 S0b_tfidf_logreg S0c S1_bm25 \
    S1_logreg S2_logreg S3_logreg --first-n 50
```

`--retriever logreg` = top-k classes of the S0b TF-IDF + logistic-regression model trained on
`gov_pool` (6,601 rows, 436 codes). `bm25` was used because the e5 encoder was not available in
the run environment; hybrid (BM25 + e5) is still to be run.

| run | candidates | acc@2 | acc@4 | acc@6 | factor_equiv_acc | emissions_rel_error | cost (USD) |
|---|---|---|---|---|---|---|---|
| S0a retrieval top-1 | dense e5 | 0.32 | 0.18 | 0.18 | 0.196 | 1.952 | 0 |
| S0b TF-IDF + logreg | - | 0.82 | 0.78 | 0.76 | 0.761 | 0.008 | 0 |
| S0c LLM zero-shot | none | 0.38 | 0.28 | 0.18 | 0.239 | 0.006 | 0.13 |
| S1 LLM picks from top-10 | BM25 | 0.30 | 0.18 | 0.16 | 0.196 | 0.079 | 0.19 |
| S1 LLM picks from top-10 | logreg | 0.62 | 0.52 | 0.52 | 0.587 | 0.001 | 0.40 |
| **S2 + 5 nearest labeled examples** | logreg | **0.88** | **0.84** | **0.80** | **0.870** | 0.001 | 0.42 |
| S3 + amount & recipient | logreg | 0.86 | 0.84 | 0.80 | 0.848 | 0.001 | 0.43 |

## Diagnostics

| run | correct / 50 | gold in candidates | correct when gold in candidates | AUROC vote agreement | AUROC verbal conf. | mean verbal conf. | acc@70% / 80% / 90% coverage (verbal) |
|---|---|---|---|---|---|---|---|
| S0c | 9 | - | - | 0.690 | 0.705 | 80.3 | |
| S1 bm25 | 8 | 15 | 7 | 0.473 | 0.762 | 71.1 | |
| S1 logreg | 26 | 47 | 26 | 0.630 | 0.819 | 78.8 | 0.63 / 0.60 / 0.58 |
| S2 logreg | 40 | 47 | 40 | 0.641 | 0.842 | 89.9 | 0.91 / 0.90 / 0.87 |
| S3 logreg | 40 | 47 | 40 | 0.589 | 0.870 | 89.6 | 0.94 / 0.93 / 0.89 |

Bootstrap (5,000 resamples of the 50 items): S2 acc@6 0.80 [0.68, 0.90]; S2 minus S0b
+0.04 [-0.08, +0.16]; S1-logreg minus S0b -0.24 [-0.40, -0.06].

Takeaways:

- Candidate recall was the bottleneck. Logreg candidates contain the gold code for 47/50
  items vs 15/50 for BM25, and S1 accuracy rises from 0.16 to 0.52.
- Without examples the LLM overrides the logreg top-1 with a "textbook" reading of the
  definitions (17 items logreg had right, S1 got wrong). Five labeled neighbors (S2) fix this:
  40/47 when the gold code is available, 4 losses vs 6 wins against logreg.
- Adding amount and recipient (S3) does not change accuracy but gives the best-ranked verbal
  confidence (AUROC 0.87).
- Verbal confidence is overconfident in level (mean 79-90) but ranks errors well; vote
  agreement is a weak signal because most items get 5/5 agreement.
- `emissions_rel_error` on 50 rows is dominated by a few very large awards.
- Scoring note: `item_id` is not unique in `gov_test` (award IDs like `0001` repeat), so
  `score_run.py` matches items by row position.
- Total Claude API spend for all runs above: $1.56.

# Full gov_test run (299 items)

`S3_logreg_full`: S3 (5 nearest labeled examples + amount and supplier), logreg top-10
candidates, `claude-haiku-4-5`, **3** samples per item, $1.57. Scored with the full S4 harness,
which also accepts 2017-to-2022 concordance equivalents as correct:

```
python scripts/run_classifier.py --stage S3 --retriever logreg --run-id S3_logreg_full --n-samples 3
python scripts/run_confidence_eval.py S3_logreg_full S0b_tfidf_logreg_proba
python scripts/run_confidence_eval.py --compare S0b_tfidf_logreg_proba S3_logreg_full
python scripts/compare_systems.py S0b_tfidf_logreg_proba S3_logreg_full
```

| metric | S0b TF-IDF + logreg | S3 LLM (logreg top-10) |
|---|---|---|
| acc@2 / acc@4 / acc@6 | 0.819 / 0.756 / 0.739 | 0.819 / 0.769 / 0.722 |
| factor_equiv_acc | 0.749 | 0.749 |
| emissions_rel_error | 0.256 | 0.022 |
| emissions_rel_error, excluding row 102 | 0.016 | 0.010 |
| C4 AUROC [95% CI] | 0.894 [0.851, 0.930] | 0.917 [0.878, 0.949] |
| AURC / ECE | 0.070 / 0.042 | 0.070 / 0.031 |
| acc at 70% / 80% / 90% coverage | 0.923 / 0.845 / 0.796 | 0.928 / 0.854 / 0.788 |
| review 10% by expected emissions error: share of emissions error removed | 0.998 | 0.979 |

S3 minus S0b on acc@6 (paired bootstrap): **-0.017 [-0.064, +0.023], p = 0.49**.

Findings:

- The +0.04 gain on the first 50 items did not hold on all 299: S3 ties the supervised baseline
  on accuracy (difference not significant).
- The large emissions-error gap is one item: row 102 is a $676M award coded 524114 (health
  insurance carriers); S0b predicts 562910 (remediation, ~30x the factor) and alone causes 94% of
  S0b's emissions error. Excluding it, the errors are 1.6% (S0b) and 1.0% (S3).
- S3 confidence is better ranked and better calibrated (AUROC 0.917 vs 0.894, ECE 0.031 vs 0.042;
  CIs overlap). The C4 model leans on verbal confidence (0.875 alone) and retrieval margin
  (0.856); self-consistency over 3 samples adds little (0.563).
- The two systems err on different items: they agree on 72.6% of items (89.4% accurate when they
  agree; agreement alone has AUROC 0.811), and at least one is right on 80.6%.
- Exploratory combination, taking per item the prediction with the higher cross-fit C4
  confidence: acc@6 **0.759**, +0.020 [0.000, +0.040] over S0b (8 wins, 2 losses). Both
  confidences are cross-fit on gov_test itself, so this needs confirming on a held-out split.
- Errors (S3, design.md section 6): retrieval misses (gold code not among the candidates,
  42 items, 11 of them NAICS-version mismatches) cause 86% of emissions error; selection
  errors (41 items, 10 near-neighbor) cause 13%.

Total Claude API spend so far: about $3.15.
