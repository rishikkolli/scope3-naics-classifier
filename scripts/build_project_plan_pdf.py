"""Render the 2-page project plan to docs/project_plan.pdf.

Edit the text below and re-run:  python scripts/build_project_plan_pdf.py
Numbers quoted in the plan come from scripts/analyze_naics_overlap.py and a profile
of data/raw (pulled 2026-10-22).
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "docs" / "project_plan.pdf"

INK = colors.HexColor("#1b1f24")
ACCENT = colors.HexColor("#1f5f8b")
RULE = colors.HexColor("#c9d3dc")
SHADE = colors.HexColor("#eef3f7")

base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.1, leading=9.9, textColor=INK)
small = ParagraphStyle("small", parent=base, fontSize=7.5, leading=9.0)
cell = ParagraphStyle("cell", parent=base, fontSize=7.4, leading=8.8)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
h1 = ParagraphStyle("h1", parent=base, fontName="Helvetica-Bold", fontSize=13.5, leading=16,
                    alignment=TA_CENTER, textColor=INK)
sub = ParagraphStyle("sub", parent=base, fontSize=8.3, alignment=TA_CENTER, textColor=colors.HexColor("#4a5560"))
h2 = ParagraphStyle("h2", parent=base, fontName="Helvetica-Bold", fontSize=9.3, leading=11,
                    textColor=ACCENT, spaceBefore=5, spaceAfter=1.5)
bullet = ParagraphStyle("bullet", parent=base, leftIndent=9, bulletIndent=1, spaceAfter=0.6)

story = []


def H(text):
    story.append(Paragraph(text, h2))


def P(text, style=base):
    story.append(Paragraph(text, style))


def B(items):
    for t in items:
        story.append(Paragraph(t, bullet, bulletText="•"))


def T(rows, widths, header=True):
    data = [[Paragraph(c, cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SHADE))
    t.setStyle(TableStyle(style))
    story.append(t)


# ---------------------------------------------------------------- header
P("Scope 3 Emissions Estimation from Procurement Text: Project Plan", h1)
P("Rishik Kolli (rk5129) · Simranjeet Singh (xs2797) · Bineet Singh Chadha (bsc6177) · "
  "NYU Generative AI · revised after proposal feedback", sub)
story.append(Spacer(1, 3))

# ---------------------------------------------------------------- 1
H("1. Refined problem and task")
P("<b>Input:</b> one purchase record: free-text description, merchant/recipient, optional structured "
  "field (P-card MCC or federal PSC), dollar amount, date. <b>Output:</b> (i) a NAICS-6 (2022) code, "
  "(ii) emissions = spend in 2022 USD × EPA v1.3 factor (kgCO2e/USD, purchaser price), (iii) a confidence "
  "score and an <b>accept / abstain</b> decision, with abstained items ranked for human review by "
  "<i>expected emissions error</i>. Two changes from the proposal: abstention is now a <b>core</b> deliverable, "
  "and evaluation mixes clean federal contracts with hand-labeled messy purchase-card data. Most of the effort "
  "goes into retrieval quality, calibration and error analysis, not prompt wording.")

# ---------------------------------------------------------------- feedback
H("2. How the plan answers the proposal feedback")
T([
    ["Feedback", "What we found or will do"],
    ["1. Can one definition map to several codes?",
     "<b>Checked on the Census files.</b> Every 6-digit definition is unique (1,012 codes), and none of the 20,398 "
     "index phrases maps to more than one code. The many-to-many problem is on the <i>query</i> side: "
     "1,076 head nouns span several codes (<i>pipe, bags, boxes, furniture</i>: 10 codes each), and 4,601 "
     "cross-references (\"…except X, see 3329xx\") link neighboring codes. We index each index phrase as its own "
     "chunk, <b>aggregate scores by code</b>, and show the LLM cross-refs as \"not this code\" hints. In the other "
     "direction, EPA factors collapse 1,016 codes into <b>281 distinct factor values</b>, so we also report "
     "<i>factor-equivalent</i> accuracy."],
    ["2. Similarity between messy text and formal definitions is low; what is retrieval quality?",
     "We expect low, compressed cosine scores, so <b>we will never threshold on raw similarity or use it as confidence</b>. "
     "Experiment E1 measures the score distribution (gold vs. top non-gold) per split and chunk type, and judges "
     "retrieval by <b>recall@k</b> (k = 1–50). P-card text has a median of <b>2 words</b> vs. 47 for a definition, so "
     "we add short index-phrase chunks, BM25 hybrid search, LLM query rewriting and a cross-encoder reranker (E2)."],
    ["3. How is the dollar amount used?",
     "(a) Emissions after deflating to 2022 USD. (b) <b>Review priority</b> = spend × P(wrong) × factor spread across "
     "top candidates (factors range 0.03–3.92 kgCO2e/$, 135×). (c) A tested input feature (a $40 vs. $4M "
     "\"software\" line is a license vs. custom development). (d) Data handling: 1,281 negative P-card rows "
     "(refunds), and awards up to $12.6B that would dominate a summed error."],
    ["4. Gov data is too clean",
     "Test set = <b>300 gov + 200 hand-labeled P-card</b> rows, every metric reported per split. Calibration uses a "
     "separate 100-row messy dev split, so the abstention threshold is not fit only on clean data."],
], [1.55, 5.75])

# ---------------------------------------------------------------- 3
H("3. Simplest credible baselines")
B([
    "<b>B0 Retrieval only:</b> top-1 code from BM25 / dense search over NAICS text, with no LLM. This is the floor the LLM must beat.",
    "<b>B1 Supervised:</b> TF-IDF + logistic regression trained on ~6,400 USAspending (description, NAICS) pairs. "
    "This tests whether generative AI is needed at all.",
    "<b>B2 Zero-shot LLM:</b> description → NAICS-6 directly (the proposal's baseline). <b>B3 MCC lookup</b> (messy only): "
    "hand map of the most common MCC categories to NAICS.",
])

# ---------------------------------------------------------------- 4
H("4. Proposed architecture")
P("<font face='Courier' size='7.4'>record → normalize (strip contract boilerplate, expand abbreviations) → [query rewrite] → "
  "hybrid retrieval over chunks {title, definition, index phrase, cross-ref} → aggregate by code → rerank top-50 → "
  "LLM selects among top-k with retrieved few-shot examples → confidence signals → accept / abstain → "
  "2022→2017 concordance → emissions</font>")
B([
    "<b>Corpus</b> (<font face='Courier'>src/naics_corpus.py</font>): ~1,012 codes × (title + definition + ~13 index phrases) plus cross-refs. "
    "<b>Retriever</b> (<font face='Courier'>src/retrieval.py</font>): e5/bge with query/passage prefixes + BM25, fused with reciprocal rank fusion. "
    "MCC/PSC used as a soft metadata boost (the metadata-filtered RAG pattern from week 7), tested as an ablation.",
    "<b>Selector</b> (<font face='Courier'>src/classify.py</font>): structured JSON output {code, rationale, confidence}; few-shot examples are "
    "nearest neighbors from the gov pool. <b>Confidence</b> (<font face='Courier'>src/confidence.py</font>): self-consistency over 5 samples, "
    "code log-probability, top-1/top-2 retrieval margin, verbalized confidence, and a logistic-regression combination fit on dev data. "
    "<b>Emissions</b> (<font face='Courier'>src/emissions.py</font>): map 2022 codes to 2017 factors via the concordance; 1→many splits use the mean factor and are flagged.",
    "Every stage logs its inputs and outputs per item (candidates, gold rank, samples, tokens, latency), so each error "
    "can be traced to the stage that caused it.",
])

# ---------------------------------------------------------------- 5
H("5. Evaluation data and test cases")
T([
    ["Split", "Source", "n", "Labels / purpose"],
    ["gov_pool", "USAspending FY2024 (have 6,900; 23 sectors × 300)", "~6,400", "Few-shot pool, B1 training, gov calibration dev"],
    ["gov_test", "USAspending, sector-stratified, held out", "300",
     "Award NAICS; <b>50 audited blind</b> by the team (officers sometimes choose codes for size-standard reasons)"],
    ["messy_dev / messy_test", "OK P-card, July 2024 (34,138 rows), stratified by MCC", "100 / 200",
     "<b>Hand-labeled by 2 annotators</b>, adjudicated by the third; Cohen's κ at 2/4/6 digits; annotator confidence recorded"],
    ["challenge", "Curated from both sources", "~40", "Unit-style cases for the edge cases in §8; run on every change"],
], [1.05, 2.45, 0.6, 3.2])

# ---------------------------------------------------------------- 6
H("6. Success metrics and methodology")
B([
    "<b>Primary:</b> emissions error, Σ|kgCO2e<sub>pred</sub> − kgCO2e<sub>true</sub>| / Σ kgCO2e<sub>true</sub>, plus a "
    "per-item median so a few $1B awards do not decide the result. <b>Secondary:</b> accuracy at 2/4/6 digits, factor-equivalent "
    "accuracy, retrieval recall@k, and cost and latency per item.",
    "<b>Abstention (core):</b> AUROC of confidence for predicting a wrong answer; risk–coverage curve and AURC; ECE with a reliability diagram; "
    "emissions error at 70/80/90% coverage. <b>Does confidence track real errors?</b> On the human-labeled messy set we also report "
    "the Spearman correlation between model confidence and annotator agreement, and the share of human-marked UNKNOWN rows the system abstains on.",
    "All comparisons use 95% bootstrap CIs and paired bootstrap tests on the same items. Success means beating B1/B2 on emissions error on "
    "<i>both</i> splits, and AUROC ≥ 0.75 on messy_test.",
])

# ---------------------------------------------------------------- 7
H("7. Outcome vs. trajectory evaluation")
P("<b>Outcome:</b> final code, emissions and the accept/abstain decision (§6). <b>Trajectory:</b> the pipeline is a fixed chain, "
  "so we score each stage, following the trajectory logging from the week 8 agent assignment. Was the gold code retrieved (gold rank)? "
  "Did query rewriting move it up or down? Given that it was in the top-k, did the LLM pick it? Did confidence flag the miss? This splits "
  "every error into <b>retrieval miss / selection error / label noise / missed abstention</b>, with each category reported by count and by share "
  "of emissions error. If the stretch agentic variant is built (the LLM can re-query or walk the NAICS hierarchy), we also score tool-call "
  "count and redundant steps.")

# ---------------------------------------------------------------- 8
H("8. Important edge cases")
T([
    ["Case", "Evidence in our data", "Handling"],
    ["No product signal", "39% of P-card rows are GENERAL PURCHASE / PAYMENT ON ACCOUNT", "Fall back to merchant + MCC; expected to abstain; label UNKNOWN if unknowable"],
    ["Retailer ≠ producer", "18% of P-card rows are Amazon; Walmart and Sam's Club also common", "Label the producing industry; the MCC is often misleading"],
    ["Multi-item / truncated", "\"PATIO RENTAL…|MEAL VOUCHERS\", \"Hea PCE\"", "Dominant-dollar item; rewrite step expands abbreviations"],
    ["Refunds, extreme spend", "1,281 negative rows; awards up to $12.6B", "Net refunds against purchases; report per-item and dollar-weighted error"],
    ["Contract boilerplate", "\"CONTRACT W58RGZ-17-D-0055, TASK ORDER…\"; 11 near-empty", "Strip IDs; empty text must abstain"],
    ["Goods vs. services; NAICS 2017↔2022", "\"software\", \"HVAC\"; mixed code versions in labels", "Amount as feature; concordance with split flags"],
], [1.45, 3.0, 2.85])

# ---------------------------------------------------------------- 9
H("9. Planned experiments")
B([
    "<b>E1 Similarity:</b> cosine distributions for gold vs. hardest negative, per split and chunk type. <b>E2 Retrieval ablations</b> (recall@k): "
    "dense → asymmetric prefixes → + index chunks → + BM25 hybrid → + rewrite → + rerank → + MCC/PSC boost.",
    "<b>E3 Classifier stages:</b> B0–B3 vs. RAG vs. RAG + few-shot; k ∈ {5, 10, 20}; with and without amount and structured field. "
    "<b>E4 Confidence:</b> each signal alone vs. combined; calibrated on gov-dev vs. messy-dev. <b>E5 Review queue:</b> abstain by P(wrong) vs. "
    "by expected emissions error (emissions error recovered per item reviewed). <b>E6 (stretch):</b> small vs. large model.",
])

# ---------------------------------------------------------------- 10
H("10. Core and stretch goals")
P("<b>Core:</b> NAICS corpus and overlap report; hybrid retriever with E1/E2; B0–B3 and the RAG + few-shot classifier; "
  "<b>confidence-based abstention with AUROC, AURC and ECE</b>; 300 double-labeled P-card rows with κ; stage-attributed error analysis weighted by emissions. "
  "<b>Stretch:</b> contrastively fine-tuned embedder on gov_pool pairs; LoRA-tuned small selector model (week 9); hierarchical "
  "(2→4→6 digit) classification; agentic re-query; a small review-queue demo.")

# ---------------------------------------------------------------- 11
H("11. Team ownership (proposed)")
T([
    ["Member", "Owns", "Also"],
    ["Rishik Kolli", "Corpus, overlap report, retriever, E1/E2, repo and data pipeline", "Labels messy rows; adjudicates Bineet/Simranjeet disagreements"],
    ["Simranjeet Singh", "LLM selector, few-shot, query rewrite, emissions module, B1–B3, E3/E6", "Labels messy rows; gov audit (25)"],
    ["Bineet Singh Chadha", "Confidence signals, abstention, eval harness and bootstrap, E4/E5", "Labels messy rows; gov audit (25)"],
], [1.25, 3.7, 2.35])
P("Labeling load: 300 messy rows × 2 labels = 600 labels, about 200 per person, plus the 50-row gov audit. "
  "Everyone writes the error-analysis section for their own stage.", small)

# ---------------------------------------------------------------- 12
H("12. Next steps")
B([
    "<b>Week 1:</b> freeze splits (<font face='Courier'>scripts/build_splits.py</font>); draw the 300-row P-card sample; label 30 rows together to align on guidelines; build the corpus.",
    "<b>Week 2:</b> E1 + E2 retrieval ablations; B0–B2 on gov_test; first labeling round with κ. "
    "<b>Week 3:</b> RAG + few-shot (E3); finish labeling; gov audit. <b>Week 4:</b> confidence and abstention (E4/E5). "
    "<b>Week 5:</b> error analysis, stretch goals, report.",
])

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                        title="Scope 3 NAICS Classifier: Project Plan",
                        author="Rishik Kolli, Simranjeet Singh, Bineet Singh Chadha")
doc.build(story)
print(f"saved {OUT}")
