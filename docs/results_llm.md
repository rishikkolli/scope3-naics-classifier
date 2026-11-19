# LLM stage results (first 50 gov_test rows)

Model: `claude-haiku-4-5`, 5 self-consistency samples per item, majority vote.
All rows below are scored on the **same first 50 rows** of `gov_test`
(`python scripts/score_run.py S0c S1_bm25 S0a_retrieval_top1 S0b_tfidf_logreg --first-n 50`).

S1 uses BM25 candidates (`--retriever bm25`, k=10) because the e5 encoder was not
available in the run environment; the hybrid retriever is the planned S1.

| run | acc@2 | acc@4 | acc@6 | factor_equiv_acc | emissions_rel_error | median_item_factor_rel_error | cost (USD) |
|---|---|---|---|---|---|---|---|
| S0a retrieval top-1 | 0.32 | 0.18 | 0.18 | 0.196 | 1.952 | 0.449 | 0 |
| S0b TF-IDF + logreg | 0.82 | 0.78 | 0.76 | 0.761 | 0.008 | 0.000 | 0 |
| S0c LLM zero-shot | 0.38 | 0.28 | 0.18 | 0.239 | 0.006 | 0.322 | 0.13 |
| S1 LLM + BM25 top-10 | 0.30 | 0.18 | 0.16 | 0.196 | 0.079 | 0.433 | 0.19 |

## Diagnostics

| run | correct / 50 | gold code in candidates | correct when gold in candidates | AUROC vote agreement | AUROC verbal confidence | mean verbal confidence |
|---|---|---|---|---|---|---|
| S0c | 9 | - | - | 0.690 | 0.705 | 80.3 |
| S1_bm25 | 8 | 15 / 50 | 7 / 15 | 0.473 | 0.762 | 71.1 |

Takeaways:

- S1 is capped by retrieval: the gold code is in the BM25 top-10 for only 15 of 50 items,
  and when it is there the LLM picks it about half the time (7/15).
- The supervised baseline wins by a wide margin on government data because it learns
  agency- and recipient-specific coding conventions from `gov_pool`; the LLM sees none of that.
- Verbal confidence is badly overconfident (mean 71-80 vs 16-18% accuracy) but still ranks
  errors better than chance (AUROC 0.71-0.76).
- `emissions_rel_error` on 50 rows is dominated by a few very large awards, so it can look
  small even when per-item factors are far off; the median per-item factor error is the
  steadier signal at this sample size.
- Scoring note: `item_id` is not unique in `gov_test` (award IDs like `0001` repeat), so
  `score_run.py` now matches items by row position.
