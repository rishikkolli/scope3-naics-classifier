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
