"""Render the Milestone 3 report (Results, Failures & Iteration) to docs/milestone3_report.pdf.

Same style as docs/project_plan.pdf (scripts/build_project_plan_pdf.py).

Edit the text below and re-run:  python scripts/build_milestone3_pdf.py
Numbers come from docs/results.md, eval/runs/S0b_tfidf_logreg_proba_S4.md,
eval/runs/E1_similarity.json and docs/results_llm.md.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "docs" / "milestone3_report.pdf"

# Times New Roman (metric-compatible with the proposal's Liberation Serif); falls back to built-in Times.
FONTS = Path("C:/Windows/Fonts")
if (FONTS / "times.ttf").exists():
    for name, file in [("Serif", "times.ttf"), ("Serif-Bold", "timesbd.ttf"),
                       ("Serif-Italic", "timesi.ttf"), ("Serif-BoldItalic", "timesbi.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                                  italic="Serif-Italic", boldItalic="Serif-BoldItalic")
    SERIF, SERIF_BOLD = "Serif", "Serif-Bold"
else:
    SERIF, SERIF_BOLD = "Times-Roman", "Times-Bold"

SIZE = 9.6          # body size; the report must fit three pages
LEAD = SIZE * 1.15

base = ParagraphStyle("base", fontName=SERIF, fontSize=SIZE, leading=LEAD, alignment=TA_JUSTIFY,
                      textColor=colors.black, spaceAfter=2)
cell = ParagraphStyle("cell", parent=base, fontSize=SIZE - 0.5, leading=(SIZE - 0.5) * 1.13, alignment=0,
                      spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName=SERIF_BOLD)
h1 = ParagraphStyle("h1", parent=base, fontName=SERIF_BOLD, fontSize=15, leading=18, alignment=TA_CENTER)
sub = ParagraphStyle("sub", parent=base, fontSize=SIZE + 0.6, leading=LEAD + 1, alignment=TA_CENTER)
h2 = ParagraphStyle("h2", parent=base, fontName=SERIF_BOLD, fontSize=SIZE + 0.9, leading=LEAD + 1,
                    alignment=0, spaceBefore=4, spaceAfter=1.5)

story = []


def H(text):
    story.append(Paragraph(text, h2))


def P(text, style=base):
    story.append(Paragraph(text, style))


def T(rows, widths):
    """Plain ruled table: rule above and below the header and below the last row."""
    data = [[Paragraph(c, cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story.append(KeepTogether([t]))   # never split a table across pages
    story.append(Spacer(1, 3))



# ---------------------------------------------------------------- header
P("Scope 3 Emissions Estimation from Procurement Text<br/>Milestone 3: Results, Failures and Iteration", h1)
P("Rishik Kolli (rk5129), Simranjeet Singh (xs2797), Bineet Singh Chadha (bsc6177)", sub)
story.append(Spacer(1, 3))

H("1. Current System")
P("Each purchase record (description, supplier, amount) is mapped to a 2022 NAICS-6 code, and emissions are "
  "estimated as spend in 2022 dollars times the matching EPA v1.3 supply-chain factor (keyed by 2017 NAICS, so "
  "predictions pass through the Census concordance). The pipeline is implemented end to end: a corpus of 22,397 "
  "NAICS chunks (titles, definitions and index phrases); BM25, dense (e5-base-v2) and hybrid retrievers; two "
  "non-LLM baselines plus an ablated family of supervised variants; four LLM stages that run on Claude Haiku 4.5 or GPT-4o-mini (S0c zero-shot, S1 choose from "
  "top-k candidates, S2 plus the five nearest labeled examples, S3 plus amount and supplier), with candidates "
  "from BM25, the hybrid retriever, the supervised model, or a union of the supervised model and BM25; a confidence model that "
  "combines retrieval margin and model probability; and an evaluation harness reporting hierarchical accuracy, "
  "factor-equivalent accuracy, emissions error, recall@k, AUROC, AURC, ECE and accuracy at fixed coverage, with "
  "bootstrap confidence intervals. Evaluation so far uses the frozen federal test split: 299 USAspending contracts "
  "with award NAICS codes. A 6,601-row disjoint pool (436 codes) is used for training and few-shot examples. The "
  "300-item purchase-card set is drawn and formatted for labeling but not yet labeled.")

H("2. Baseline and Current Results")
P("<b>Retrieval (299 items).</b> We first tested how well formal NAICS text can be retrieved from contract "
  "descriptions. Dense retrieval is the best single retriever, and hybrid fusion did not beat it at depth:")
T([["Retriever", "recall@1", "recall@5", "recall@10", "recall@20", "recall@50"],
   ["R1 BM25, all chunks", "0.057", "0.207", "0.284", "0.371", "0.448"],
   ["R1a BM25, definitions only", "0.067", "0.161", "0.244", "0.321", "0.401"],
   ["R2 dense e5", "0.151", "0.311", "0.378", "0.448", "0.575"],
   ["R4 hybrid (RRF)", "0.114", "0.301", "0.391", "0.448", "0.552"]],
  [2.2, 0.95, 0.95, 0.95, 0.95, 0.95])
P("<b>Baselines (299 items).</b> Taking the top retrieved code (S0a) gives 0.137 six-digit accuracy and a 191% "
  "relative emissions error. A TF-IDF + logistic-regression classifier trained on the pool (S0b) gives 0.739 "
  "six-digit accuracy, 0.819 two-digit accuracy, 0.749 factor-equivalent accuracy and a 25.6% emissions error. "
  "Because S0b is the bar the generative pipeline has to clear, we ablated it to find out which part of it works. "
  "The variants below are scored with the same metric functions as the baseline script, which do not credit "
  "2017-to-2022 concordance equivalents, so S0b reads 0.729 rather than the 0.739 the S4 harness gives it; the "
  "deltas are therefore like-for-like. Three changes each help on their own and hold up together:")
T([["Supervised baseline variant", "acc@2", "acc@4", "acc@6", "Factor-eq.", "acc@6 vs S0b"],
   ["Description only, no supplier name", "0.732", "0.682", "0.659", "0.696", "-0.070"],
   ["Description + supplier (= S0b)", "0.819", "0.756", "0.729", "0.751", "-"],
   ["+ char 3-5 n-grams", "0.850", "0.793", "0.762", "0.777", "+0.033"],
   ["+ log-amount bucket", "0.829", "0.773", "0.746", "0.769", "+0.017"],
   ["Hinge loss (LinearSVC) instead of logistic", "0.866", "0.786", "0.749", "0.773", "+0.020"],
   ["<b>All three combined</b>", "<b>0.870</b>", "<b>0.813</b>", "<b>0.776</b>", "<b>0.795</b>",
    "<b>+0.047 [+0.013, +0.084]</b>"]],
  [2.45, 0.6, 0.6, 0.6, 0.8, 1.65])
P("The combined baseline reaches 0.776 acc@6 (p = 0.011 by paired bootstrap over the 299 items, 22 wins / 8 "
  "losses) and trains in eight seconds. The supplier name is the most valuable single field, worth seven points "
  "alone, and character n-grams help because part numbers and clipped abbreviations carry no reusable word "
  "tokens. On like-for-like scoring this raises the bar for the LLM stages by nearly five points.")
P("<b>Confidence and abstention on S0b (299 items).</b> Confidence separates right from wrong predictions well: "
  "AUROC 0.894 [0.851, 0.930], AURC 0.070 and ECE 0.042. Answering only the 70% most confident items raises "
  "accuracy to 0.923 (0.845 at 80% coverage, 0.796 at 90%). Ordering the review queue by expected emissions error "
  "removes 99.8% of the emissions error after reviewing 10% of items, versus 1.0% when ordering by probability of "
  "being wrong.")
P("<b>LLM stages (first 50 test items, Claude Haiku 4.5, five samples per item, k = 10).</b> All rows below are "
  "scored on the same 50 items; total API cost was $1.56. Because S1-S3 can only choose from the candidate list "
  "they are handed, their ceiling is fixed before the model is called: the gold code is in the supervised top-10 "
  "for 87.0% of items and among the five few-shot neighbours for 83.6%.")
T([["Run", "Candidates", "acc@2", "acc@6", "Factor-eq.", "Gold in top-10", "AUROC (verbal conf.)"],
   ["S0b TF-IDF + logreg", "-", "0.82", "0.76", "0.76", "-", "-"],
   ["S0c zero-shot", "none", "0.38", "0.18", "0.24", "-", "0.71"],
   ["S1 choose from top-10", "BM25", "0.30", "0.16", "0.20", "15 / 50", "0.76"],
   ["S1 choose from top-10", "logreg top-10", "0.62", "0.52", "0.59", "47 / 50", "0.82"],
   ["<b>S2 + 5 labeled examples</b>", "logreg top-10", "<b>0.88</b>", "<b>0.80</b>", "<b>0.87</b>", "47 / 50", "0.84"],
   ["S3 + amount, supplier", "logreg top-10", "0.86", "0.80", "0.85", "47 / 50", "0.87"]],
  [1.75, 1.05, 0.6, 0.6, 0.9, 1.0, 1.1])
P("<b>Full test set (299 items).</b> We then ran S3 on all 299 items (three samples per item, $1.57) and scored "
  "both systems with the full harness, which also accepts 2017-to-2022 concordance equivalents:")
T([["System", "acc@2", "acc@6", "Factor-eq.", "C4 AUROC [95% CI]", "ECE", "acc at 70 / 80 / 90% cov."],
   ["S0b TF-IDF + logreg", "0.819", "0.739", "0.749", "0.894 [0.851, 0.930]", "0.042", "0.923 / 0.845 / 0.796"],
   ["S3 LLM, logreg top-10", "0.819", "0.722", "0.749", "0.917 [0.878, 0.949]", "0.031", "0.928 / 0.854 / 0.788"],
   ["Pick higher-confidence system", "-", "0.759", "-", "-", "-", "-"]],
  [1.75, 0.5, 0.5, 0.8, 1.45, 0.45, 1.55])
P("The 4-point S2 gain on the first 50 items did not hold: on all 299, S3 minus S0b is -0.017 [-0.064, +0.023] "
  "(paired bootstrap, p = 0.49), so the LLM only matches the original supervised baseline on accuracy, and it "
  "falls behind the strengthened one, 0.722 against 0.776 - a conservative gap, since the 0.722 carries the S4 "
  "harness's concordance credit and the 0.776 does not. Its confidence is better ranked and better calibrated, "
  "although the intervals overlap, and taking per item the prediction whose calibrated confidence is higher gives "
  "0.759, +0.020 [0.000, +0.040] over S0b. The plainest summary of where we stand is that the cheapest change of "
  "this milestone - different features and a different loss, eight seconds of training - moved accuracy more than "
  "the whole generative pipeline did.")

H("3. What Worked and What Failed")
P("<b>Raw similarity is not a usable signal (E1).</b> For 262 items, the best-scoring chunk of the correct code has a "
  "mean cosine of 0.800, while the best wrong code scores 0.825. The correct code outscores every wrong code for only "
  "17% of items. This confirms the concern raised in the proposal feedback: short contract text compared against long "
  "formal definitions produces compressed, uninformative scores. We use rank and the top-1/top-2 margin instead, and "
  "the margin alone predicts correctness with AUROC 0.909.")
P("<b>Retrieval misses dominate the emissions error.</b> Error analysis of S0b assigns 95.3% of the total emissions "
  "error to 25 retrieval misses, where the correct code never reached the candidate list. The 38 selection errors "
  "(correct code retrieved but not chosen) account for only 3.3%, and NAICS version mismatches for 1.0%. Near-neighbor "
  "confusions (for example 811210 vs 811219, both repair codes) are frequent but cheap, because the two codes usually "
  "share an emission factor. The same pattern holds for the full S3 run: retrieval misses (42 items) cause 86% of "
  "its emissions error.")
P("<b>The LLM fails without candidates or examples.</b> Zero-shot, the model produces plausible but wrong codes for "
  "terse government strings such as \"IGF::OT::IGF\" or part numbers (\"8503119313!SPACER,SLEEVE\"), which carry "
  "almost no product information. With BM25 candidates it is limited by retrieval: the gold code was available for "
  "only 15 of 50 items, and the model chose it 7 times.")
P("<b>The LLM overrides good candidates with \"textbook\" readings.</b> Given the logistic-regression top-10 but no "
  "examples (S1), the model lost 17 items that the baseline had right. It picked the code whose definition best "
  "matches the literal wording, while federal contracting practice codes by the contractor's industry: a camera system "
  "for wildland fire support is coded as forestry support (115310), not engineering services. Five labeled neighbors "
  "(S2) fix most of this: 40 of the 47 items with the gold code available, 6 wins and 4 losses against the baseline.")
P("<b>Confidence levels are inflated, but rankings are useful.</b> Mean self-reported confidence was 79 to 90 at "
  "16% to 80% accuracy, so the raw numbers cannot be used as probabilities. Their ranking still predicts errors "
  "(AUROC 0.71 to 0.87), and becomes more informative as the system improves. Self-consistency vote agreement is a "
  "weak signal (AUROC 0.47 to 0.69) because most items receive 5/5 agreement.")
P("<b>Small-sample gains were misleading, but the errors are complementary.</b> The 50-item comparison suggested "
  "the LLM beat the baseline; the full set did not confirm it. The two systems do fail on different items, though. "
  "They agree on 73% of items and are 89% accurate when they agree, and at least one of them is right on 81%. "
  "Agreement alone predicts correctness with AUROC 0.81.")
P("<b>The emissions metric is a two-item statistic.</b> One $676M health-insurance contract (524114) that the "
  "baseline codes as remediation services (562910, a factor about 30 times higher) causes 94% of the baseline's "
  "25.6% emissions error; S3 codes it closer (2.2% total error), and excluding it the errors are 1.6% and 1.0%. "
  "The denominator is just as concentrated: a single $11.7B logistics award (561210), which every system codes "
  "correctly, is 93.6% of the true emissions of the whole split, and the top five items are 96.9%. Our primary "
  "metric is therefore close to one item divided by another, which is why it moved from 25.6% to 2.2% between two "
  "systems whose accuracy is indistinguishable. We now report it with and without the largest item and treat the "
  "federal figure as a sanity check rather than a comparison. Three suspected sources of error are ruled out: the "
  "2022-to-2017 concordance affects four test rows with at most a 1.4x factor spread, purchaser- versus "
  "producer-price factors move the total by 0.07%, and since every award is FY2024 the deflator is one scalar "
  "that cancels out of every relative metric.")
P("<b>Expensive and cheap errors separate cleanly.</b> Grouping the baseline's errors by how far the predicted "
  "code sits from the gold code, cross-sector confusions (47 items, median factor ratio 1.8x) carry 99.4% of the "
  "emissions error, same-2-digit errors (16 items) carry 0.5%, and same-4-digit errors (5 items, ratio 1.1x) "
  "carry 0.02%. Sector accuracy is what the emissions estimate rests on; the last two digits barely matter. An "
  "oracle given the correct 4-digit parent and then the most common 6-digit child in the pool reaches 0.776 acc@6 "
  "at a 0.2% emissions error, which is a concrete case for hierarchical decoding.")
P("<b>Query normalization did not help (negative result).</b> Because the zero-shot errors concentrate on "
  "boilerplate and part numbers, we tested a deterministic normalizer that strips contracting boilerplate, "
  "delivery- and task-order phrases and bare part numbers, as a free stand-in for the R5 LLM rewrite. It rewrote "
  "133 of 299 descriptions and changed six-digit accuracy by nothing at all, while making two-digit accuracy and "
  "few-shot neighbour recall slightly worse. IDF already discounts text that appears in every record, and the "
  "part-number pattern removed genuine product tokens, emptying three descriptions outright. Rewriting can only "
  "pay where the model cannot discount the noise itself, so R5 stays open for the purchase-card split, where the "
  "noise is abbreviation rather than repetition.")
P("<b>The supervised baseline's ceiling is class coverage.</b> It predicts 90 distinct codes where the gold "
  "labels use 128 of the 436 available, and its top five classes take 26.4% of predictions against 17.4% of "
  "gold. On items whose gold code has at most two training examples it scores 0.000, against 0.813 where the "
  "code has more than ten, and 7 items (2.3%) have a gold code absent from the pool altogether. It also leans "
  "high: its mean predicted factor is 10% above the gold mean, because high-factor classes such as 562910 act as "
  "magnets. None of these limits binds a retrieval-plus-LLM system, which can emit any of the 1,016 codes that "
  "have a factor - the strongest reason to expect the LLM to win on purchase-card text even though it does not "
  "win here.")
P("<b>An evaluation bug.</b> Award IDs are not unique in the test split (\"0001\" appears 27 times), so our first "
  "LLM scoring matched predictions to the wrong rows. We now match by row position and check the order explicitly.")

story.append(CondPageBreak(0.9 * inch))   # keep the heading with its table
H("4. Iterations and Improvements")
T([["Change", "Why", "Effect"],
   ["Chunk-level index with max-per-code aggregation; dense and hybrid retrieval",
    "One-to-many head nouns (pipe, bags) and short queries", "recall@50 0.448 (BM25) to 0.575 (dense)"],
   ["Rank and margin instead of raw cosine", "E1 showed compressed, overlapping scores",
    "Margin alone: AUROC 0.909"],
   ["Supervised baseline before any LLM work", "Need a credible non-LLM bar",
    "acc@6 0.137 (S0a) to 0.739 (S0b)"],
   ["Combined confidence model and cost-weighted review queue", "Abstention is a required deliverable",
    "AUROC 0.894; 10% review removes 99.8% of emissions error"],
   ["LLM over BM25 candidates instead of zero-shot", "Constrain the output space", "acc@6 0.18 to 0.16 (no gain)"],
   ["Logreg top-10 as LLM candidates", "Gold code missing from BM25 candidates for 35/50",
    "Gold in candidates 15 to 47; acc@6 0.16 to 0.52"],
   ["Add 5 nearest labeled examples (S2)", "LLM ignored coding conventions", "acc@6 0.52 to 0.80"],
   ["Add amount and supplier (S3)", "Disambiguate goods vs services", "acc@6 unchanged; conf. AUROC 0.84 to 0.87"],
   ["Score by row position", "Duplicate award IDs", "Corrected all LLM metrics"],
   ["Run S3 on all 299 items (3 samples)", "Check the 50-item result", "acc@6 0.722 vs 0.739 (n.s.); AUROC 0.917"],
   ["Pick the higher-confidence system per item", "Errors are complementary", "acc@6 0.759 (+0.020 vs S0b)"],
   ["Union of supervised and BM25 candidates", "Codes unseen in the federal pool", "Federal recall@10 0.866 to 0.873"],
   ["Char n-grams, hinge loss and an amount bucket in the baseline", "Needed to know which part of S0b works",
    "acc@6 0.729 to 0.776, +0.047 [+0.013, +0.084]"],
   ["Deterministic query normalization", "Boilerplate and part numbers dominated the zero-shot errors",
    "no change in acc@6; few-shot recall slightly worse"]],
  [2.45, 2.3, 2.25])

H("5. Remaining Gaps")
P("<b>No results on messy data yet.</b> Our main claim concerns corporate-style purchase-card text, and every "
  "number above comes from federal contracts. Hand labels (two annotators, Cohen's kappa) are the main blocker. "
  "<b>The LLM alone does not beat the supervised baseline on federal data, and now trails it</b> - 0.722 "
  "against 0.776. Its value so far is better confidence and complementary errors; the combination gain (+0.020) "
  "is small and was measured with confidence models cross-fit on the same split, so it needs a held-out "
  "confirmation. <b>Our primary metric is not measurable on this split</b> and should become median per-item "
  "factor error plus factor-equivalent accuracy, with the total reported with and without the largest item. "
  "<b>Promoting the stronger baseline would cost the confidence model an input,</b> because LinearSVC exposes no "
  "probability and needs Platt scaling first. <b>Candidate generation depends on federal training data.</b> The logistic-regression candidates "
  "only cover the 436 codes seen in the pool and will transfer poorly to purchase cards, where the definition-based "
  "retrievers have only 0.39 recall@10. <b>Calibration is in-split.</b> Confidence models are cross-fit within the "
  "federal test set; the planned purchase-card calibration split does not exist until labeling is done. "
  "<b>Federal labels are not audited.</b> The planned 50-item audit of award NAICS codes has not been done.")

H("6. Next Steps (prioritized)")
P("1. <b>Label the purchase-card set</b> (200 test, 100 calibration; two annotators per item), adjudicate "
  "disagreements and report inter-annotator agreement. 2. <b>Run S3 with union candidates on the purchase-card "
  "test set</b> and compare against S0b with paired bootstrap intervals; this is where the LLM should help most, "
  "since the supervised model has not seen corporate-style text. 3. <b>Validate the combined system</b> (pick the "
  "higher-confidence prediction, or use agreement as a signal) with confidence models fit on the purchase-card "
  "calibration split and tested on the held-out split. 4. <b>Report abstention on both splits</b>: AUROC, ECE, and "
  "accuracy and emissions error at 70/80/90% coverage. 5. <b>Error analysis on messy data</b> "
  "(merchant-category miscoding, ambiguous retailer purchases) and the 50-item federal audit. 6. <b>Promote the "
  "strengthened baseline</b> with Platt scaling so the confidence model keeps a probability input, and re-report "
  "every delta against it. 7. <b>Test hierarchical 2-to-4-to-6 decoding</b>, which the error structure now "
  "justifies directly. 8. <b>Run the real R5 LLM rewrite on the messy split</b>, where the free proxy's failure "
  "does not apply. 9. Final report and presentation.")

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.68 * inch, rightMargin=0.68 * inch,
                        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                        title="Scope 3 NAICS Classifier: Milestone 3",
                        author="Rishik Kolli, Simranjeet Singh, Bineet Singh Chadha")
doc.build(story)
print(f"saved {OUT}")
