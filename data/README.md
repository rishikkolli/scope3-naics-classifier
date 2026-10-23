# Data

Pulled 2026-10-22.

| Path | Source | Notes |
|---|---|---|
| `raw/naics/` | [Census NAICS 2022](https://www.census.gov/naics/) | Codes, descriptions, index file (~20k entries), cross-references, 2017↔2022 concordances |
| `raw/epa/` | [EPA Supply Chain GHG Emission Factors v1.3](https://catalog.data.gov/dataset/supply-chain-greenhouse-gas-emission-factors-v1-3-by-naics-6) | kg CO2e / 2022 USD. **Keyed by 2017 NAICS** → map via concordance |
| `raw/usaspending/awards_fy2024.jsonl` | [USAspending API](https://api.usaspending.gov) `spending_by_award` | 6,900 FY2024 contract awards (types A–D), 300 per NAICS sector × 23 sectors, 443 unique NAICS-6 codes. Fields: Award ID, Recipient, Description, Award Amount, NAICS{code,description}, PSC, Agency, Start Date |
| `raw/pcard/ok_pcard_202407.csv` | [Oklahoma OMES P-card, data.ok.gov](https://data.ok.gov/) | 34,138 transactions, July 2024. Short item text + merchant + MCC + amount. **Unlabeled** — source for the hand-labeled `messy` split. Cardholder name columns removed |

## Observations so far

- USAspending descriptions mix clean product phrases ("SHIELDING CUBE FOR RADIO SCIENCE RESEARCH") with contract boilerplate ("CONTRACT W58RGZ-17-D-0055, TASK ORDER 0002 - …") and a few near-empty strings (11 with < 3 chars).
- P-card text is truncated and merchant-driven ("Bar Audio Wired In-Ear Hea PCE" at "AMZN Mktp US", MCC "BOOK STORES"; "ONG NMB" at "OKLAHOMANATURALGAS"). The MCC is often misleading for Amazon purchases.
- Award amounts span from under $1k to over $1B; emissions error will be dominated by large awards unless also reported per-item.

## Splits (to build)

- `gov_test` — 350 awards, stratified by sector, held out
- `gov_pool` — remainder, for few-shot retrieval + confidence calibration (dev)
- `messy_test` — 150 P-card rows, hand-labeled per `docs/labeling_guidelines.md`
