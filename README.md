# Scope 3 Emissions Estimation from Procurement Text

Rishik Kolli (rk5129) · Simranjeet Singh (xs2797) · Bineet Singh Chadha (bsc6177)
NYU — Generative AI, Final Project

## Task

Given a purchase record (free-text description + dollar amount):

1. assign a NAICS-6 industry code,
2. compute spend-based emissions (`spend_USD2022 × EPA factor [kgCO2e/USD]`),
3. output a calibrated confidence and **abstain** (route to human review) when confidence is low.

## What this project is (and isn't)

The goal is *not* a prompt-engineering exercise. Effort goes into:

- **Retrieval quality** — how well short, messy purchase text retrieves the right NAICS code from formal definitions (measured with recall@k, not raw similarity).
- **Confidence-based abstention (required)** — showing that the system's confidence actually tracks which predictions are wrong.
- **Error analysis** — categorized errors at every stage, weighted by emissions impact.

See [`docs/design.md`](docs/design.md) for the full design and how it addresses proposal feedback.

## Pipeline

```
purchase text ─► query rewrite (optional) ─► hybrid retrieval (BM25 + dense)
              ─► rerank ─► LLM picks among top-k candidates (+ few-shot)
              ─► confidence signals ─► accept / abstain
              ─► emissions = spend (2022 USD) × factor
```

## Evaluation

| Split | Source | Size | Labels |
|---|---|---|---|
| `gov` | USAspending.gov federal contracts | ~350 | NAICS from award record (audited subset) |
| `messy` | Public purchase-card (P-card) transactions | ~150 | Hand-labeled by team |

Metrics (reported per split):

- **Primary:** emissions-weighted error (|kgCO2e_pred − kgCO2e_true|, summed and relative)
- NAICS accuracy at 2/4/6 digits, plus *factor-equivalent* accuracy (codes sharing an EPA factor)
- Retrieval recall@k
- **Abstention:** AUROC (confidence vs. correctness), risk–coverage curve / AURC, ECE, emissions error at 70/80/90% coverage
- Cost and latency per item

## Repo layout

```
configs/      run configuration
data/raw/     downloaded source data (NAICS, EPA factors, USAspending, P-card)
data/interim/ cleaned / joined tables
data/labeled/ hand-labeling templates and labels
docs/         design notes, labeling guidelines
scripts/      data fetch scripts
src/          pipeline modules
eval/         evaluation outputs
notebooks/    exploration and error analysis
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys
```
