# Design Notes

How the system is engineered, organized around the proposal feedback.

---

## 1. One definition ↔ multiple codes

Checked with `scripts/analyze_naics_overlap.py` (2026-10-28):

- All 1,012 six-digit definitions are distinct, and **0 of 20,398** normalized index phrases map to more than one code. Definitions themselves are not ambiguous.
- The ambiguity is on the query side: **1,076 of 17,778** index head nouns (text before the first comma) span several codes, e.g. *tapes* (11), *pipe*, *bags*, *boxes*, *furniture* (10 each). A two-word P-card line ("pipe", "bags") has the same problem.
- 4,601 cross-references link neighboring codes.
- EPA v1.3: 1,016 codes but only **281 distinct factor values** across 386 USEEIO sectors; 815 codes share a sector with another code. Factors range 0.029–3.924 kgCO2e/USD.

Where this matters:

| Source of overlap | Effect | Handling |
|---|---|---|
| **Census NAICS Index File** — ~20k short index entries, each → one NAICS-6; near-identical phrases can appear under different codes | Retrieval returns several codes for the same phrase | Index each entry as its own chunk; aggregate chunk scores **by code** (max and sum); dedupe before passing candidates to the LLM |
| **Cross-references** ("…except X, see 333xxx") | Definitions of neighboring codes share vocabulary | Store cross-refs as separate, negatively-labeled text; show them to the LLM as "not this code — see Y" hints |
| **EPA factors are sector-level** — many NAICS-6 codes share one factor | Some 6-digit errors cost 0 kgCO2e | Report a *factor-equivalent* accuracy level; emissions error naturally handles this |
| **NAICS 2017 vs 2022** — USAspending labels mix versions, and **EPA v1.3 factors are keyed by 2017 NAICS** (verified from the CSV header) | Label/index in 2022, factors in 2017 | Classify in 2022; map labels to 2022 and predictions to 2017 for factor lookup via the Census concordances; flag 1→many splits (use mean factor of targets, report how many items are affected) |

Deliverable: a table of (a) # index phrases mapping to >1 code, (b) # NAICS-6 codes per unique EPA factor, (c) # 2017→2022 splits.

## 2. Similarity scores and retrieval quality

Short messy queries vs. long formal definitions produce **compressed, uninformative cosine
scores** — now measured (E1, gov_test, e5-base-v2): the gold code's best chunk averages
cosine 0.800 (p10–p90: 0.763–0.837) while the best *non-gold* chunk averages 0.825; the
mean margin is **negative** (−0.024) and the gold code outscores all others on raw cosine
only 17% of the time. Absolute similarity is therefore not meaningful — **rank** is. So:

- **Measure retrieval with recall@k** on labeled data (k ∈ {1, 5, 10, 20, 50}); choose k from the recall curve.
- **Never use raw similarity as a confidence score.** Use rank margins (top-1 vs top-2 gap) only as one calibrated signal.
- Plot the similarity distribution for correct vs. incorrect top-1 retrievals.

Ablations in **strict priority order** (milestone-2 feedback: if time runs out, the bottom
of the list is dropped, never the top). Each is added *on top of* the previous one and
reported as recall@k, so any gain is attributable to exactly one technique:

| Priority | Technique | Rationale for its rank |
|---|---|---|
| R1 | BM25 over all chunk types | zero-cost floor; exact-word matches matter for product nouns |
| R2 | Dense, asymmetric prefixes (e5 `query:`/`passage:`) | biggest expected jump; built for short→long matching |
| R3 | Index-entry chunks + score aggregation by code | closes the query/document length gap (2 vs 47 words) |
| R4 | Hybrid BM25 + dense (reciprocal rank fusion) | cheap once R1+R2 exist |
| R5 | LLM query rewriting of messy descriptions | needs API budget; helps mainly the messy split |
| R6 | Cross-encoder reranking of top-50 | latency cost; only worth it if R1–R4 recall plateaus below target |
| R7 | MCC/PSC soft metadata boost | split-specific, smaller expected gain |
| R8 *(stretch)* | Contrastively fine-tuned embedder on USAspending pairs | largest effort, needed only if R2–R6 underdeliver |

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
| `gov` | USAspending contracts | 300 | Award NAICS; **audit a 50-item sample** by hand (contracting officers sometimes pick codes for size-standard reasons) and report agreement |
| `messy` | Public purchase-card transactions (merchant + short description) | 200 test + 100 dev | **Hand-labeled** NAICS-6 by ≥2 team members; report inter-annotator agreement (Cohen's κ) |

All metrics are reported per split. A terse, synthetic "corporate-style" rewrite of gov descriptions is a robustness check only — never the main messy set.

## 5. Confidence-based abstention (mandatory)

**Confidence signals** (per item), in **strict priority order** — implemented top-down,
lowest priorities dropped first if time runs out:

| Priority | Signal | Rationale for its rank |
|---|---|---|
| C1 | Self-consistency: agreement rate across N sampled predictions | strongest known signal; no extra plumbing |
| C2 | Retrieval margin: score gap between the top two candidate codes | free — computed during retrieval |
| C3 | Verbalized confidence (0–100) | one extra JSON field in the same call |
| C4 | Combined: logistic regression over C1–C3 (+C5 if present), fit on `messy_dev` | the headline deliverable; needs C1–C3 first |
| C5 | Token log-probability of the chosen code | provider-dependent (logprobs not always exposed); add last |

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

## Stages: baseline first, one change per run

Milestone-2 feedback: establish the baseline first, then add improvements **one at a time**,
so every gain is attributable to a specific technique. Each run writes a frozen result file
`eval/runs/<run_id>.json` (config + per-item outputs + metrics); run N is always compared
against run N−1 on the same items.

| Run | System | What the delta measures |
|---|---|---|
| S0a | B0 retrieval-only (best retriever's top-1 code, no LLM) | floor |
| S0b | B1 TF-IDF + logistic regression on gov_pool | is generative AI needed at all? |
| S0c | B2 zero-shot LLM | parametric knowledge alone |
| S1 | + retrieval: LLM chooses among top-k (frozen retriever from R-series) | value of grounding in NAICS text |
| S2 | + few-shot neighbors from gov_pool | value of labeled examples |
| S3 | + amount & MCC/PSC in the prompt | value of structured context |
| S4 | + confidence & abstention (C-series) | selective-prediction value |
| S5 *(stretch)* | hierarchical; small vs. large model | — |

The retriever is tuned first (R1–R8, frozen before S1); S1–S4 then never touch retrieval,
so classifier deltas are never confounded with retrieval changes.

## 7. Low-information items: insufficient evidence ≠ wrong prediction

Milestone-2 feedback: for purchase descriptions with little information (39% of P-card rows
are "GENERAL PURCHASE" etc.), distinguish *incorrect predictions* from *items where the
information is insufficient for a reliable classification*.

**Labeling.** Annotators assign one of three evidence levels (see labeling guidelines):
`sufficient` (codeable from description+merchant+MCC), `partial` (codeable only to 2–4
digits; record the finest honest prefix), `insufficient` (`naics6=UNKNOWN`). UNKNOWN rows
are kept — they are the abstention test set, not noise.

**Scoring.** Three-way outcome per item:
- *correct* — prediction matches gold (at the digit level being scored);
- *wrong* — evidence was sufficient and the system still erred (a real model failure);
- *unanswerable* — gold is UNKNOWN: the only correct behavior is to abstain.

Accuracy is computed over sufficient-evidence rows only; UNKNOWN rows are scored as
**abstention recall** (share of unanswerable items the system abstained on) and never enter
the accuracy denominator. `partial` rows are scored at their honest digit level. The error
analysis then separates "model picked a wrong code" from "nobody could have known," and the
confidence study (§5) checks that low confidence concentrates on both wrong *and*
unanswerable items.
