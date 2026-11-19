"""Render Simranjeet's individual Milestone 3 report to docs/milestone3_simranjeet.pdf.

Scope: the components I own under the project plan's division of work -- the LLM
classifier stages (S0c-S3), few-shot example selection, query rewriting, the emissions
calculation, and the non-LLM baselines. Retrieval (R-series) and the confidence model
(C-series) are Rishik's and Bineet's and are referenced only where my components depend
on them.

Style matches docs/project_plan.pdf (scripts/build_project_plan_pdf.py).

Tables are filled from frozen result files so the report cannot drift from the runs:
  eval/runs/ablations_s0b.json        (scripts/run_baseline_ablations.py)
  eval/runs/emissions_sensitivity.json (scripts/analyze_emissions_sensitivity.py)
LLM-stage numbers come from docs/results_llm.md (runs made before the API budget ran out).

Usage:  python scripts/build_milestone3_simranjeet_pdf.py
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "milestone3_simranjeet.pdf"
RUNS = ROOT / "eval" / "runs"

def _load(name: str, produced_by: str) -> dict:
    f = RUNS / name
    if not f.exists():
        raise SystemExit(f"missing {f.relative_to(ROOT)} -- run `{produced_by}` first")
    return json.loads(f.read_text())


AB = _load("ablations_s0b.json",
           "python scripts/run_baseline_ablations.py && "
           "python scripts/run_baseline_ablations.py --combined")
EM = _load("emissions_sensitivity.json", "python scripts/analyze_emissions_sensitivity.py")
if "combined" not in AB:
    raise SystemExit("eval/runs/ablations_s0b.json has no stacked variants -- run "
                     "`python scripts/run_baseline_ablations.py --combined` first")

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

SIZE = 8.9
LEAD = SIZE * 1.15

base = ParagraphStyle("base", fontName=SERIF, fontSize=SIZE, leading=LEAD, alignment=TA_JUSTIFY,
                      textColor=colors.black, spaceAfter=2)
cell = ParagraphStyle("cell", parent=base, fontSize=SIZE - 0.5, leading=(SIZE - 0.5) * 1.13,
                      alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName=SERIF_BOLD)
h1 = ParagraphStyle("h1", parent=base, fontName=SERIF_BOLD, fontSize=14, leading=17,
                    alignment=TA_CENTER)
sub = ParagraphStyle("sub", parent=base, fontSize=SIZE + 0.4, leading=LEAD + 1, alignment=TA_CENTER)
h2 = ParagraphStyle("h2", parent=base, fontName=SERIF_BOLD, fontSize=SIZE + 0.9, leading=LEAD + 1,
                    alignment=0, spaceBefore=4, spaceAfter=1.5)

story = []


def H(text):
    story.append(Paragraph(text, h2))


def P(text, style=base):
    story.append(Paragraph(text, style))


def T(rows, widths):
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
    story.append(KeepTogether([t]))
    story.append(Spacer(1, 3))


def f(x, nd=3):
    return f"{x:.{nd}f}"


V = AB["variants"]
REF = "A2_desc+recipient (= S0b)"

# ---------------------------------------------------------------- header
P("Scope 3 Emissions Estimation from Procurement Text<br/>"
  "Milestone 3: Results, Failures and Iteration", h1)
P("Simranjeet Singh (xs2797)<br/>"
  "<i>My components: LLM classifier stages, few-shot examples, query rewriting, "
  "emissions calculation, baselines</i>", sub)
story.append(Spacer(1, 4))

# ---------------------------------------------------------------- 1
H("1. Current System (the parts I own)")
P("Our system maps a purchase record (description, supplier, amount) to a 2022 NAICS-6 code and "
  "estimates emissions as spend in 2022 USD times the matching EPA v1.3 supply-chain factor, with "
  "a confidence score that lets it abstain. This report covers only the four pieces I own under "
  "our division of work; retrieval (R-series) and the confidence model (C-series) are Rishik's "
  "and Bineet's and appear only where my components depend on them.")
P("<b>Baselines.</b> Two non-LLM reference systems that the generative pipeline has to beat: "
  "<i>S0a</i> takes the top-1 code from the best retriever, and <i>S0b</i> is a TF-IDF + "
  f"logistic-regression classifier trained on the {AB['pool_rows']:,}-row federal pool "
  f"({AB['pool_codes']} codes) and evaluated on the frozen {AB['test_rows']}-item federal test "
  "split (<tt>scripts/train_baseline.py</tt>).")
P("<b>LLM classifier stages.</b> Four stages, each adding exactly one thing so the delta is "
  "attributable (<tt>src/classify.py</tt>): <i>S0c</i> zero-shot, <i>S1</i> choose one of the "
  "top-k retrieved candidates, <i>S2</i> plus the five nearest labeled examples, <i>S3</i> plus "
  "amount and supplier. Each call returns JSON with the code, a 0-100 verbalized confidence and a "
  "rationale, is sampled N times for a self-consistency vote, and logs candidates, samples, "
  "tokens, cost and latency. Either provider works (Claude Haiku 4.5 or GPT-4o-mini).")
P("<b>Few-shot example selection.</b> The S2/S3 examples are the five nearest gov_pool rows by "
  "TF-IDF cosine over description + supplier, injected as <tt>\"text\" -&gt; code</tt> pairs.")
P("<b>Emissions calculation.</b> <tt>src/emissions.py</tt>: spend is deflated to 2022 USD, the "
  "predicted 2022 code is mapped through the Census 2022-to-2017 concordance (EPA factors are "
  "keyed by 2017 NAICS; 1-to-many splits take the mean target factor), and the purchaser-price "
  "factor is applied. It also exposes the factor-equivalence groups and the "
  "expected-emissions-error score that orders the review queue.")
P("<b>Query rewriting (R5).</b> <tt>llm_rewrite</tt> is implemented but has not been run: it "
  "costs one extra LLM call per item and the design ranks it below R1-R4. For this milestone I "
  "tested a free deterministic stand-in instead (below).")

# ---------------------------------------------------------------- 2
H("2. Baseline and Current Results")
P("All numbers below are on the frozen 299-item federal test split. Retrieval-only top-1 (S0a) "
  "reaches 0.137 six-digit accuracy and a 191% relative emissions error; the supervised baseline "
  "(S0b) reaches 0.739 and 25.5%. S0b is therefore the bar, and this milestone was mostly spent "
  "finding out <i>why</i> it is hard to beat. I ran eight new local ablations of it "
  "(<tt>scripts/run_baseline_ablations.py</tt>, no API cost):")

ROWS = [["Variant of the supervised baseline", "acc@2", "acc@4", "acc@6", "Factor-eq.",
         "Emis. rel. err.", "excl. worst item"]]
for name, m in V.items():
    label = name.split("_", 1)[1]
    if name == REF:
        label = f"<b>{label}</b>"
    ROWS.append([label, f(m["acc@2"]), f(m["acc@4"]), f(m["acc@6"]), f(m["factor_equiv_acc"]),
                 f(m["emissions_rel_error"]), f(m["emissions_rel_error_excl_worst"], 4)])
T(ROWS, [2.35, 0.62, 0.62, 0.62, 0.78, 0.95, 1.06])

P(f"<b>The desc+recipient row reproduces the published S0b exactly</b> (acc@6 {V[REF]['acc@6']}, "
  f"emissions error "
  f"{V[REF]['emissions_rel_error']}), so these are like-for-like deltas. Four results matter. "
  f"(i) <b>The supplier name is the most valuable single field:</b> dropping it costs "
  f"{(V[REF]['acc@6'] - V['A1_desc_only']['acc@6'])*100:.1f} points of acc@6 "
  f"({V['A1_desc_only']['acc@6']} vs {V[REF]['acc@6']}). The baseline is partly learning <i>who "
  f"sold it</i> rather than <i>what was bought</i> -- fine on federal contracts, and a real "
  f"transfer risk on purchase cards where the merchant is a retailer, not the producer. "
  f"(ii) <b>Character n-grams help:</b> adding char 3-5 features to the word features gives "
  f"{V['A4_desc+recipient, word+char']['acc@6']} acc@6 "
  f"(+{(V['A4_desc+recipient, word+char']['acc@6'] - V[REF]['acc@6'])*100:.1f} points), which fits the "
  f"data -- part numbers and clipped abbreviations have no reusable word tokens. (iii) <b>Hinge "
  f"loss is better and far cheaper:</b> LinearSVC gives the best two-digit accuracy of all "
  f"({V['A7_LinearSVC']['acc@2']} vs {V[REF]['acc@2']}) and fits in "
  f"{V['A7_LinearSVC']['fit_seconds']}s against {V[REF]['fit_seconds']}s, though it gives no "
  f"probabilities, which the team's confidence model needs. (iv) <b>The dollar amount helps the "
  f"discriminative model but not the LLM:</b> a log-magnitude bucket adds "
  f"+{(V['A5_+amount bucket']['acc@6'] - V[REF]['acc@6'])*100:.1f} points here, while putting the "
  f"same amount into the prompt (S2 to S3) changed accuracy by exactly 0.000.")

P("<b>What the LLM stages depend on.</b> S1-S3 can only pick from the candidate list they are "
  "given, and S2-S3 only help if the few-shot neighbours carry useful labels, so I measured both "
  "ceilings directly (also free -- they need no LLM call):")
T([["k", "1", "3", "5", "10", "20", "50"],
   ["Gold code in logreg top-k (candidate ceiling for S1-S3)"]
   + [f(AB["candidate_recall_at_k"].get(str(k), AB["candidate_recall_at_k"].get(k, float('nan'))))
      for k in (1, 3, 5, 10, 20, 50)],
   ["Gold code among the k nearest pool rows (few-shot ceiling)"]
   + [f(AB["fewshot_neighbour_recall"].get(str(k), AB["fewshot_neighbour_recall"].get(k, float('nan'))))
      for k in (1, 3, 5, 10, 20)] + ["-"]],
  [3.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
P(f"At the k = 10 we actually use, the gold code is in the candidate list for "
  f"{float(AB['candidate_recall_at_k'].get('10', AB['candidate_recall_at_k'].get(10))):.1%} of "
  f"items and among the five few-shot neighbours for "
  f"{float(AB['fewshot_neighbour_recall'].get('5', AB['fewshot_neighbour_recall'].get(5))):.1%}. "
  "Those two numbers, not the prompt, set the LLM's ceiling.")

P("<b>LLM stages.</b> The stage runs were made with Claude Haiku 4.5 before the API budget ran "
  "out (total spend $3.15 for every LLM run in the project). The staged comparison below is on "
  "the first 50 test items, five samples per item, k = 10; S3 was then re-run on all 299:")
T([["Run", "Candidates", "acc@2", "acc@6", "Factor-eq.", "Gold in top-10", "AUROC (verbal conf.)"],
   ["S0b TF-IDF + logreg", "-", "0.82", "0.76", "0.76", "-", "-"],
   ["S0c zero-shot", "none", "0.38", "0.18", "0.24", "-", "0.71"],
   ["S1 pick from top-10", "BM25", "0.30", "0.16", "0.20", "15 / 50", "0.76"],
   ["S1 pick from top-10", "logreg", "0.62", "0.52", "0.59", "47 / 50", "0.82"],
   ["<b>S2 + 5 labeled examples</b>", "logreg", "<b>0.88</b>", "<b>0.80</b>", "<b>0.87</b>",
    "47 / 50", "0.84"],
   ["S3 + amount, supplier", "logreg", "0.86", "0.80", "0.85", "47 / 50", "0.87"],
   ["S3 on all 299 items (3 samples)", "logreg", "0.819", "0.722", "0.749", "257 / 299", "0.875"]],
  [1.9, 0.85, 0.6, 0.6, 0.78, 0.95, 1.2])
P("On the first 50 items S2 looked like a 4-point win over S0b. It did not survive the full "
  "split: S3 minus S0b on acc@6 is <b>-0.017 [-0.064, +0.023]</b> (paired bootstrap, p = 0.49). "
  "<b>My LLM stages tie the supervised baseline on accuracy and do not beat it.</b> They are "
  "better on the emissions metric (2.2% vs 25.5% relative error) and their confidence ranks "
  "errors better (AUROC 0.917 vs 0.894, ECE 0.031 vs 0.042), but section 3 explains why the "
  "emissions gap is almost entirely one item.")

CB = AB["combined"]
BT = AB["combined_bootstrap_vs_s0b"]
A10 = "A10_word+char, LinearSVC, +amount"
P("<b>Stacking the three changes that each helped</b> gives a baseline that is significantly "
  "better than the one we have been reporting, and better than my own LLM pipeline:")
T([["System", "acc@2", "acc@4", "acc@6", "Factor-eq.", "Emis. rel. err.", "Fit", "acc@6 vs S0b [95% CI]"],
   ["S0b (reported so far)", f(CB['A2_ref (= S0b)']['acc@2']), f(CB['A2_ref (= S0b)']['acc@4']),
    f(CB['A2_ref (= S0b)']['acc@6']), f(CB['A2_ref (= S0b)']['factor_equiv_acc']),
    f(CB['A2_ref (= S0b)']['emissions_rel_error'], 4), f"{CB['A2_ref (= S0b)']['fit_seconds']:.0f}s", "-"],
   ["S3 LLM (logreg top-10)", "0.819", "0.769", "0.722", "0.749", "0.0220", "-",
    "-0.017 [-0.064, +0.023]"],
   ["word+char, LinearSVC", f(CB['A9_word+char, LinearSVC']['acc@2']),
    f(CB['A9_word+char, LinearSVC']['acc@4']), f(CB['A9_word+char, LinearSVC']['acc@6']),
    f(CB['A9_word+char, LinearSVC']['factor_equiv_acc']),
    f(CB['A9_word+char, LinearSVC']['emissions_rel_error'], 4),
    f"{CB['A9_word+char, LinearSVC']['fit_seconds']:.0f}s",
    f"+{BT['A9_word+char, LinearSVC']['delta_acc6']:.3f} "
    f"[+{BT['A9_word+char, LinearSVC']['ci95'][0]:.3f}, +{BT['A9_word+char, LinearSVC']['ci95'][1]:.3f}]"],
   [f"<b>{A10.split('_',1)[1]}</b>", f"<b>{f(CB[A10]['acc@2'])}</b>", f"<b>{f(CB[A10]['acc@4'])}</b>",
    f"<b>{f(CB[A10]['acc@6'])}</b>", f"<b>{f(CB[A10]['factor_equiv_acc'])}</b>",
    f(CB[A10]['emissions_rel_error'], 4), f"{CB[A10]['fit_seconds']:.0f}s",
    f"<b>+{BT[A10]['delta_acc6']:.3f} [+{BT[A10]['ci95'][0]:.3f}, +{BT[A10]['ci95'][1]:.3f}]</b>"]],
  [1.72, 0.52, 0.52, 0.52, 0.72, 0.85, 0.42, 1.73])
P(f"The combined baseline reaches <b>{f(CB[A10]['acc@6'])}</b> acc@6, "
  f"+{BT[A10]['delta_acc6']*100:.1f} points over S0b with a paired-bootstrap interval that "
  f"excludes zero ([+{BT[A10]['ci95'][0]:.3f}, +{BT[A10]['ci95'][1]:.3f}], p = "
  f"{BT[A10]['p_two_sided']:.3f}, {BT[A10]['wins']} wins / {BT[A10]['losses']} losses), and it "
  f"trains in {CB[A10]['fit_seconds']:.0f} seconds. It also happens to code row 102 correctly, "
  f"which drops the emissions error from {f(CB['A2_ref (= S0b)']['emissions_rel_error'], 4)} to "
  f"{f(CB[A10]['emissions_rel_error'], 4)} -- another reminder of how little that metric measures "
  f"here. The uncomfortable summary is that <b>the cheapest change of the milestone (eight "
  f"seconds of scikit-learn) moved accuracy more than the entire LLM pipeline did,</b> and it "
  f"raises the bar my S1-S3 stages have to clear from 0.739 to {f(CB[A10]['acc@6'])}.")

# ---------------------------------------------------------------- 3
H("3. What Worked and What Failed")
conc = EM["emissions_concentration"]
big = EM["largest_item"]
ebc = EM["error_by_confusion_distance"]
P(f"<b>The headline emissions metric is a two-item statistic (my biggest finding).</b> I "
  f"profiled the emissions calculation item by item (<tt>scripts/analyze_emissions_sensitivity.py"
  f"</tt>). One award -- row {big['row']}, an ${big['amount_2022usd']/1e9:.1f}B logistics "
  f"contract (LOGCAP IV, code {big['gold']}) -- is {conc['top_1_share']:.1%} of the true "
  f"emissions of the whole split, and the top five items are {conc['top_5_share']:.1%}. That "
  f"award is classified <i>correctly</i> by both systems, so it sits almost entirely in the "
  f"denominator. The numerator is dominated by a different single item: row 102, a $631M TRICARE "
  f"pharmacy contract labeled 524114 (health insurance carriers, factor 0.033) which S0b codes as "
  f"562910 (remediation services, factor 0.988, a 30x jump), causing 94.8% "
  f"of S0b's total emissions error on its own. So 'emissions relative error' on this split is one "
  f"item divided by another, which is why it moved from 25.5% to 2.2% between my two systems "
  f"while accuracy did not move at all. We now report the metric with and without the worst item, "
  f"and I treat the federal emissions number as a sanity check rather than a comparison metric.")
P(f"<b>Cheap errors and expensive errors are cleanly separable.</b> Grouping S0b's errors by how "
  f"far the predicted code sits from the gold code: near misses are almost free and cross-sector "
  f"confusions are everything. Same-4-digit errors ({ebc['same_4digit']['n']} items, median factor "
  f"ratio {ebc['same_4digit']['median_factor_ratio']}x) carry "
  f"{ebc['same_4digit']['share_of_emissions_error']:.2%} of the emissions error; same-2-digit "
  f"errors ({ebc['same_2digit']['n']} items) carry {ebc['same_2digit']['share_of_emissions_error']:.1%}; "
  f"cross-sector errors ({ebc['cross_sector']['n']} items, median ratio "
  f"{ebc['cross_sector']['median_factor_ratio']}x) carry "
  f"{ebc['cross_sector']['share_of_emissions_error']:.1%}. This says plainly what to optimise: "
  f"getting the 2-digit sector right matters, and the last two digits barely do. I confirmed the "
  f"upside with an oracle -- given the correct 4-digit parent and then simply the most common "
  f"6-digit child in the pool, acc@6 is "
  f"{EM['oracle_4digit_then_common_child']['acc@6']:.3f} (above the real baseline's 0.739) with a "
  f"{EM['oracle_4digit_then_common_child']['emissions_rel_error']:.1%} emissions error.")
P("<b>The LLM fails without candidates, and fails differently with them.</b> Zero-shot (S0c) "
  "collapses to 0.18 acc@6 because federal descriptions carry no product information: 12.7% of "
  "test descriptions contain the contracting boilerplate \"IGF::OT::IGF\" and the median "
  "description is 7 tokens. Given BM25 candidates it was capped by retrieval (gold available for "
  "15 of 50). Given good candidates but no examples (S1 + logreg top-10), it lost 17 items the "
  "baseline had right: it picks the code whose formal definition best matches the literal "
  "wording, while federal practice codes by the contractor's line of business -- a camera system "
  "for wildland fire support is coded 115310 (forestry support), not engineering services. The "
  "five labeled neighbours in S2 are what fixed this (0.52 to 0.80 acc@6), which is the clearest "
  "positive result I have: <b>the examples, not the definitions, teach the coding convention.</b>")
P("<b>Verbalized confidence is badly scaled but well ordered.</b> Mean self-reported confidence "
  "was 79-90 at 16-80% accuracy, so it is unusable as a probability, but its ranking predicts "
  "errors (AUROC 0.71 to 0.87, rising as the system improves) and it is the strongest single "
  "input to the team's combined confidence model (0.875 alone). Self-consistency over samples is "
  "a weak signal (AUROC 0.47-0.69) because most items return unanimous votes -- a negative result "
  "that saved us from paying for five samples per item.")
nz = AB["normalizer"]
fs, fsn = AB["fewshot_neighbour_recall"], AB["fewshot_neighbour_recall_normalized"]
P(f"<b>Query normalization failed, and the reason was instructive (negative result).</b> Since "
  f"the zero-shot errors were concentrated on boilerplate and part numbers, I expected cleaning "
  f"the text to help, and built a deterministic normalizer (strip <tt>IGF::OT::IGF</tt>, delivery- "
  f"and task-order phrases, and bare alphanumeric part numbers) as a free stand-in for the R5 LLM "
  f"rewrite. It rewrote {nz['rows_changed']} of {nz['rows']} descriptions, removing "
  f"{nz['mean_chars_removed']:.0f} characters on average, and changed six-digit accuracy by "
  f"nothing at all ({V['A6_normalized text (R5-lite)']['acc@6']} vs {V[REF]['acc@6']}) while "
  f"making two-digit accuracy and the few-shot neighbours slightly <i>worse</i> (gold code among "
  f"the 5 nearest rows: {fsn['5'] if '5' in fsn else fsn[5]} vs {fs['5'] if '5' in fs else fs[5]}). "
  f"Two causes, both avoidable in hindsight: IDF already discounts boilerplate that appears in "
  f"every record, so removing it adds no information to a TF-IDF model; and my part-number regex "
  f"deleted genuine product tokens, emptying {nz['rows_emptied']} descriptions completely. The "
  f"lesson is that rewriting can only pay off where the <i>model</i> cannot discount noise -- a "
  f"prompted LLM reading the raw string, or the purchase-card split where the noise is "
  f"abbreviation rather than repetition -- not in front of a bag-of-words baseline.")
P("<b>The baseline's ceiling is class coverage, which is exactly where the LLM should win.</b> "
  "S0b predicts only 90 distinct codes where the gold labels use 128 (of 436 available), and its "
  "top five classes absorb 26.4% of predictions against 17.4% of gold. On items whose gold code "
  "has at most two pool examples it scores <b>0.000</b> acc@6 (9 items), against 0.813 where the "
  "code has more than ten; 7 items (2.3%) have a gold code absent from the pool entirely and are "
  "unreachable by construction. It also leans high: mean predicted factor is 10.0% above the "
  "gold mean, because high-factor classes such as 562910 act as magnets. None of these limits "
  "applies to a retrieval-plus-LLM system, which can emit any of the 1,016 codes with a factor. "
  "Measured against a majority vote over the same five neighbours the LLM already uses (0.642 "
  "acc@6), S3's 0.722 shows it does extract more from those examples than nearest-neighbour "
  "voting does; it just does not yet beat a model trained on 6,601 in-domain rows.")
P("<b>Label noise sits on the one item that drives the metric.</b> Row 102, the $631M item "
  "behind 94.8% of the emissions error, is a TRICARE <i>pharmacy services</i> contract to Express "
  "Scripts labeled 524114, direct health and medical insurance carriers. That is a defensible "
  "contracting choice but not obviously the industry that produced the service, the convention our "
  "prompt states, so I cannot call this a model error -- the strongest argument yet for the "
  "50-item federal label audit I share with Bineet.")
P(f"<b>Two suspected error sources ruled out, cheaply.</b> The 2022-to-2017 concordance affects "
  f"only {EM['concordance']['test_rows_whose_gold_code_splits']} test rows, and the factor spread "
  f"within those splits is at most "
  f"{EM['concordance']['max_factor_ratio_within_split']:.2f}x, so averaging over split targets is "
  f"not a meaningful source of error. Purchaser- versus producer-price factors differ by "
  f"{(EM['margins']['ratio']-1)*100:.2f}% in total, and because every test award is FY2024 the "
  f"deflator is a single scalar ({EM['deflator']['ratio']:.3f}) that cancels out of every relative "
  f"metric. Both were on my list to fix; both are now parked with evidence.")

# ---------------------------------------------------------------- 4
H("4. Iterations and Improvements")
T([["Change I made", "Why", "Effect"],
   ["Supervised baseline before any LLM work", "Milestone-2 feedback: need a credible non-LLM bar",
    "acc@6 0.137 (S0a) to 0.739 (S0b)"],
   ["LLM over BM25 candidates instead of zero-shot", "Constrain the output space to real codes",
    "acc@6 0.18 to 0.16: no gain, retrieval-capped"],
   ["Supervised top-10 as the candidate list", "Gold code missing from BM25 top-10 for 35/50 items",
    "gold in candidates 15 to 47; acc@6 0.16 to 0.52"],
   ["Five nearest labeled examples (S2)", "S1 ignored federal coding convention",
    "acc@6 0.52 to 0.80 (50 items)"],
   ["Amount and supplier in the prompt (S3)", "Disambiguate goods from services",
    "acc@6 unchanged; conf. AUROC 0.84 to 0.87"],
   ["Five samples per item down to three", "Vote agreement was a weak signal (AUROC 0.47-0.69)",
    "cost per item -40%; the signal dropped was the weak one"],
   ["Score by row position, not award ID", "Award IDs repeat (\"0001\" 27 times)",
    "corrected every LLM metric"],
   ["Re-ran S3 on all 299 items", "Check the 50-item gain",
    "0.722 vs 0.739, -0.017 [-0.064, +0.023]: gain was noise"],
   ["Emissions error reported with and without the worst item", "One item was 94.8% of it",
    "25.5% to 1.3% excluding row 102"],
   ["Char 3-5 n-grams added to the baseline", "Part numbers have no reusable word tokens",
    f"acc@6 {V[REF]['acc@6']} to {V['A4_desc+recipient, word+char']['acc@6']}"],
   ["Hinge loss instead of logistic regression", "436 classes, sparse features, 1-vs-rest margin",
    f"acc@2 {V[REF]['acc@2']} to {V['A7_LinearSVC']['acc@2']}; fit 21s to 1s"],
   ["Log-magnitude amount bucket as a feature", "The LLM ignored the amount; test if a model can use it",
    f"acc@6 +{(V['A5_+amount bucket']['acc@6']-V[REF]['acc@6'])*100:.1f} points"],
   ["<b>All three stacked</b>", "Each helped alone; test together",
    f"<b>acc@6 {f(CB[A10]['acc@6'])}, +{BT[A10]['delta_acc6']:.3f} "
    f"[+{BT[A10]['ci95'][0]:.3f}, +{BT[A10]['ci95'][1]:.3f}], p = {BT[A10]['p_two_sided']:.3f}</b>"],
   ["Deterministic query normalization (R5-lite)", "Boilerplate and part numbers dominated zero-shot errors",
    "no change in acc@6; few-shot recall slightly worse"]],
  [2.35, 2.3, 2.35])

# ---------------------------------------------------------------- 5
H("5. Remaining Gaps")
P("We defined success as beating the supervised and zero-shot baselines on emissions error for "
  "<i>both</i> data types and reaching AUROC 0.75 on the purchase-card test set. Against that: "
  "<b>(i) the purchase-card split does not exist yet</b>, so the criterion is untested on the "
  "data the project is actually about, and labeling is the blocker. <b>(ii) My LLM stages do not "
  "beat the supervised baseline on federal data</b>, and the gap widened this milestone rather "
  "than closing: 0.722 against a baseline that is now "
  f"{f(CB[A10]['acc@6'])}. <b>(iii) The emissions-error criterion is not measurable on this "
  "split</b> -- 93.6% of the true total is one correctly-classified award, so the ratio is a "
  "two-item statistic and swings by an order of magnitude on a single prediction. I think the "
  "criterion needs revising to median per-item factor error plus factor-equivalent accuracy, with "
  "the total reported with and without the largest item. <b>(iv) Query rewriting is still "
  "unanswered</b>, not answered negatively: the free proxy failed for a reason that does not "
  "apply to an LLM rewrite on messy text, and the real R5 run needs API budget. <b>(v) The "
  "few-shot block is unablated</b>: k = 5 with one selection strategy, although the neighbour "
  f"ceiling rises from {fs['5'] if '5' in fs else fs[5]} at k = 5 to "
  f"{fs['10'] if '10' in fs else fs[10]} at k = 10, so there is measurable headroom I have not "
  "tested. <b>(vi) No cross-model comparison</b>: the GPT-4o-mini path is implemented and never "
  f"run, so I cannot speak to cost-quality trade-offs. <b>(vii) {(1-EM['factor_coverage'])*100:.1f}% "
  "of gold codes have no EPA factor</b> and those rows drop silently out of every emissions "
  "metric. <b>(viii) If we adopt the stronger baseline, it returns no probabilities</b>, which "
  "the team's confidence model uses as an input, so it needs Platt scaling before it can be used "
  "end to end -- a dependency I created and have not yet resolved.")

# ---------------------------------------------------------------- 6
H("6. Next Steps (prioritized, two weeks)")
P("<b>1. Finish my share of the purchase-card labeling</b> (about 200 double labels) and my half "
  "of the 50-item federal audit. Everything below is blocked on the first and the row-102 "
  "ambiguity is blocked on the second. <b>2. Run S2/S3 with union candidates on the purchase-card "
  "test set</b> against the strengthened baseline, with paired bootstrap intervals. This is the "
  "experiment the whole project turns on: the supervised model has never seen corporate text, its "
  "coverage limits are now measured (0.000 accuracy on codes with two or fewer training examples, "
  "7 unreachable items), and that is precisely the weakness an LLM over retrieved definitions "
  "should not share. <b>3. Promote the word+char LinearSVC baseline</b> and Platt-calibrate it so "
  "Bineet's confidence model keeps a probability input; re-report all deltas against it. <b>4. "
  "Run the real R5 LLM rewrite on the messy split only</b> (about $1), scored on accuracy and "
  "few-shot recall, since that is where the noise is abbreviation rather than repetition. "
  "<b>5. Ablate the few-shot block</b>: k in {0, 3, 5, 10}, label-diverse versus nearest-only. <b>6. Try hierarchical 2-to-4-to-6 decoding</b>, which the error "
  "structure now justifies directly: cross-sector confusions carry 99.4% of the emissions error "
  "while same-4-digit confusions carry 0.02%, and a 4-digit oracle plus the most common child "
  "reaches 0.776 acc@6 at 0.2% emissions error. <b>7. Switch the headline metric</b> as above. <b>8. Cost-quality comparison</b> (GPT-4o-mini versus "
  "Haiku 4.5), if budget allows.")

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.68 * inch, rightMargin=0.68 * inch,
                        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                        title="Scope 3 NAICS Classifier: Milestone 3 (Simranjeet Singh)",
                        author="Simranjeet Singh")
doc.build(story)
print(f"saved {OUT}")
