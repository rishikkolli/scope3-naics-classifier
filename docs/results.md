# Run results

Regenerate with `python scripts/summarize_runs.py`.

| run | n | recall@1 | recall@5 | recall@10 | recall@20 | recall@50 | acc@2 | acc@4 | acc@6 | factor_equiv_acc | gold_factor_coverage | emissions_rel_error | emissions_net_bias_rel | median_item_factor_rel_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_bm25_all_chunks | 299 | 0.0569 | 0.2074 | 0.2843 | 0.3712 | 0.4482 |  |  |  |  |  |  |  |  |
| R1a_bm25_defs_only | 299 | 0.0669 | 0.1605 | 0.2441 | 0.3211 | 0.4013 |  |  |  |  |  |  |  |  |
| R2_dense_e5 | 299 | 0.1505 | 0.311 | 0.3779 | 0.4482 | 0.5753 |  |  |  |  |  |  |  |  |
| R4_hybrid_rrf | 299 | 0.1137 | 0.301 | 0.3913 | 0.4482 | 0.5518 |  |  |  |  |  |  |  |  |
| S0a_retrieval_top1 | 299 |  |  |  |  |  | 0.2809 | 0.1839 | 0.1371 | 0.1978 | 0.913 | 1.9078 | 1.8893 | 0.4818 |
| S0b_tfidf_logreg | 299 |  |  |  |  |  | 0.8194 | 0.7559 | 0.7291 | 0.7509 | 0.913 | 0.2549 | 0.2476 | 0.0 |
