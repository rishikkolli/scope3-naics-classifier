# Design Notes

How the system is engineered, organized around the proposal feedback.

---

## 1. One definition ↔ multiple codes

Many-to-many relationships between text and codes need to be checked against the source files (`scripts/analyze_naics_overlap.py`, TODO). The places we expect them:

| Source of overlap | Effect | Handling |
|---|---|---|
| **Census NAICS Index File** — ~20k short index entries, each → one NAICS-6; near-identical phrases can appear under different codes | Retrieval returns several codes for the same phrase | Index each entry as its own chunk; aggregate chunk scores **by code** (max and sum); dedupe before passing candidates to the LLM |
| **Cross-references** ("…except X, see 333xxx") | Definitions of neighboring codes share vocabulary | Store cross-refs as separate, negatively-labeled text; show them to the LLM as "not this code — see Y" hints |
| **EPA factors are sector-level** — many NAICS-6 codes share one factor | Some 6-digit errors cost 0 kgCO2e | Report a *factor-equivalent* accuracy level; emissions error naturally handles this |
| **NAICS 2017 vs 2022** — USAspending labels mix versions, and **EPA v1.3 factors are keyed by 2017 NAICS** (verified from the CSV header) | Label/index in 2022, factors in 2017 | Classify in 2022; map labels to 2022 and predictions to 2017 for factor lookup via the Census concordances; flag 1→many splits (use mean factor of targets, report how many items are affected) |

Deliverable: a table of (a) # index phrases mapping to >1 code, (b) # NAICS-6 codes per unique EPA factor, (c) # 2017→2022 splits.

## 2. Similarity scores and retrieval quality

Short messy queries vs. long formal definitions produce **low, compressed cosine scores** (we expect a narrow band; to be measured). Absolute similarity is therefore not meaningful — **rank** is. So:

- **Measure retrieval with recall@k** on labeled data (k ∈ {1, 5, 10, 20, 50}); choose k from the recall curve.
- **Never use raw similarity as a confidence score.** Use rank margins (top-1 vs top-2 gap) only as one calibrated signal.
- Plot the similarity distribution for correct vs. incorrect top-1 retrievals.

Ablations (each reported as recall@k):

1. Dense embeddings of NAICS definitions (baseline)
2. Asymmetric embedding model with query/passage prefixes (e5 / bge)
3. Index-entry chunks in addition to definitions (shorter docs ≈ closer to query length)
4. Hybrid BM25 + dense (reciprocal rank fusion)
5. LLM query rewriting ("dell lat 5540 x10" → "laptop computers")
6. Cross-encoder reranking of top-50
7. *(stretch)* Contrastively fine-tuned embedder on USAspending (description, code) pairs

## 3. Using the dollar amount

1. **Emissions** — `kgCO2e = spend × factor`. Convert spend to 2022 USD (BEA PCE / CPI deflator) and use the purchaser-price factor (with margins) for spend recorded at purchaser price.
2. **Review prioritization** — rank abstained items by *expected emissions error*:
   `E[err] ≈ spend × P(wrong) × spread(factor over top-k candidates)`.
   A $500 ambiguous item between two similar factors is low priority; a $5M item between steel and consulting is high priority.
3. *(optional)* **Classification feature** — amount as context ("$40 software" ≈ license; "$4M software" ≈ custom development services).

## 4. Evaluation data: clean + messy

Federal contract descriptions are cleaner and more structured than corporate expense data. The test set mixes:

| Split | Source | n | Labels |
|---|---|---|---|
| `gov` | USAspending contracts | ~350 | Award NAICS; **audit a 50-item sample** by hand (contracting officers sometimes pick codes for size-standard reasons) and report agreement |
| `messy` | Public purchase-card transactions (merchant + short description) | ~150 | **Hand-labeled** NAICS-6 by ≥2 team members; report inter-annotator agreement (Cohen's κ) |

All metrics are reported per split. A terse, synthetic "corporate-style" rewrite of gov descriptions is a robustness check only — never the main messy set.

## 5. Confidence-based abstention (mandatory)

**Confidence signals** (per item):

- Self-consistency: agreement rate across N sampled predictions
- Token log-probability of the chosen code
- Retrieval margin: score gap between the top two candidate codes
- Verbalized confidence (0–100)
- Combined: logistic regression over the above, fit on a dev split

**Metrics** — does confidence identify what is truly wrong?

- **AUROC** of confidence for predicting correctness (6-digit and factor-equivalent)
- **Risk–coverage curve** and AURC
- **ECE** (expected calibration error) + reliability diagram
- **Selective emissions error** at 70/80/90% coverage
- Same metrics with abstention ranked by expected emissions error (§3)

## 6. Error analysis

Every stage logs per-item outputs so errors can be categorized:

- Retrieval miss (true code not in top-k) vs. selection error (in top-k, wrong pick)
- Vague description · goods vs. services · near-neighbor code · NAICS version mismatch · label noise

Each category is reported by count **and** by share of total emissions error.

## Stages

1. Zero-shot baseline (LLM → NAICS-6 directly)
2. Retrieval-augmented (choose among top-k)
3. + Few-shot examples retrieved from training data
4. + Confidence and abstention
5. *(stretch)* hierarchical classification; small vs. large vs. fine-tuned model comparison
