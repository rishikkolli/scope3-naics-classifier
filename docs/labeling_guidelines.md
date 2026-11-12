# Hand-Labeling Guidelines

Used for (a) the `messy` P-card split and (b) the audit sample of `gov` labels.

## What to label

For each row, assign the **NAICS-6 (2022)** code of the industry that *produces* the purchased good or service — not the industry of the buyer, and not the merchant's retail category unless the purchase is a retail service.

| Field | Meaning |
|---|---|
| `naics6` | Your best NAICS-6 code, a shorter prefix (see `evidence`), or `UNKNOWN` |
| `naics6_alt` | Second plausible code, if genuinely ambiguous |
| `evidence` | `sufficient` / `partial` / `insufficient` — see below |
| `confidence` | `high` / `medium` / `low` |
| `ambiguity_reason` | `vague`, `goods_vs_services`, `near_neighbor`, `multi_item`, `other` |
| `notes` | Free text |

## Evidence level (required)

Low-information rows must be separated from codeable ones, so that the evaluation can
distinguish a wrong prediction from an item nobody could classify (design.md §7):

- `sufficient` — description + merchant + MCC pin down a NAICS-6. Label it.
- `partial` — you can only get to 2 or 4 digits honestly (e.g. "GENERAL PURCHASE" at
  Grainger → industrial supplies wholesale, but not which one). Put the finest honest
  prefix in `naics6` (e.g. `4238`) and say in `notes` what is missing.
- `insufficient` — no real signal (e.g. "GENERAL PURCHASE" at a big-box retailer).
  Set `naics6=UNKNOWN`. Do not guess: a guessed label would punish the system for
  correctly abstaining. These rows become the abstention test set.

## Rules

1. **Goods vs. retail:** a laptop bought at Best Buy → the *manufacturer* code (334111), since spend-based factors attribute emissions to production. Note the retailer in `notes`.
2. **Services:** label the service performed (e.g. "HVAC repair" → 238220, not the equipment manufacturer).
3. **Multi-item purchases:** label the dominant item by dollar value; set `ambiguity_reason=multi_item`.
4. **Unknowable:** if the text gives no product signal (e.g. "Amazon purchase"), set `naics6=UNKNOWN`, `evidence=insufficient`. Keep these rows — they test abstention.
5. Use the Census NAICS search (census.gov/naics) and the definitions in `data/raw/naics`.

## Process

- Each item is labeled independently by **two** team members.
- Compute Cohen's κ at 2-, 4-, and 6-digit levels before adjudication.
- Disagreements are adjudicated by the third member; store both raw labels and the final label.

## Gov label audit

Draw 50 random items from the `gov` test split. Label blind (without seeing the award NAICS), then compare. Report agreement rate and the categories of disagreement.
