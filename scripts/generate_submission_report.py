"""Generate professional submission report PDF from paper draft + paper_figures/.

Run:  .venv/bin/python scripts/generate_submission_report.py
Output: LEMC_submission_report.pdf
"""
from __future__ import annotations

import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "paper_figures")
OUT = os.path.join(ROOT, "LEMC_submission_report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=16, leading=20, spaceAfter=10))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#444"), spaceAfter=14))
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=13, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a3d5c")))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2c5282")))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, spaceAfter=6, leading=13))
styles.add(ParagraphStyle(name="Abstract", parent=styles["Normal"], fontSize=9.5, leading=13, leftIndent=12, rightIndent=12, spaceAfter=10, backColor=colors.HexColor("#f7f9fb"), borderPadding=10))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#555"), spaceAfter=10, leading=11))
styles.add(ParagraphStyle(name="Finding", parent=styles["Normal"], fontSize=9.5, leading=13, backColor=colors.HexColor("#e8f5e9"), borderPadding=8, spaceAfter=10))

story: list = []


def h1(t): story.append(Paragraph(t, styles["H1"]))
def h2(t): story.append(Paragraph(t, styles["H2"]))
def body(t): story.append(Paragraph(t, styles["Body"]))
def cap(t): story.append(Paragraph(t, styles["Caption"]))
def finding(t): story.append(Paragraph(t, styles["Finding"]))


def fig(rel_path: str, caption: str, width: float = 6.3 * inch):
    path = os.path.join(FIG, rel_path)
    if not os.path.exists(path):
        body(f"<i>[Figure missing: {rel_path}]</i>")
        return
    img = Image(path)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = min(width * ratio, 4.2 * inch)
    story.append(img)
    cap(caption)


def tbl(data, col_widths=None, highlight: int | None = None, font_size=8):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    sty = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#bbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    if highlight is not None:
        sty += [
            ("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#e8f5e9")),
            ("FONTNAME", (0, highlight), (-1, highlight), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(sty))
    story.append(t)
    story.append(Spacer(1, 0.12 * inch))


# ═══════════════════════════════════════════════════════════════════════════════
# Title page
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 0.35 * inch))
story.append(Paragraph(
    "Fixed Geometry, Not Learned Residuals:<br/>"
    "Exposing Ego-Motion Shortcuts in Sensor-Free<br/>"
    "Pedestrian Trajectory and Intent Prediction",
    styles["Title2"],
))
story.append(Paragraph("Submission report · Anonymous · blind review", styles["Subtitle"]))
story.append(Paragraph(f"Generated {date.today().strftime('%B %d, %Y')} · PIE + JAAD · protocol τ=16, η=45, TTE 30–60", styles["Subtitle"]))

story.append(Paragraph(
    "<b>Abstract.</b> Egocentric pedestrian tracks confound pedestrian motion with camera motion. "
    "Zhang et al. decompose the two using ego-speed as input and soft supervision; LIM deletes speed "
    "to avoid causal confusion. We ask whether classical offline geometry can recover camera motion "
    "without any speed sensor, and whether a learned module can reproduce that geometry from tracks alone. "
    "We introduce a standing-pedestrian variance gate and show that three learned LEMC variants "
    "(unsupervised, OBD-magnitude, ORB-regression) all fail it, while full-affine ORB compensation "
    "evaluated at each agent's image position passes (93.1% PIE, 96.9% JAAD). Fixed ORB is "
    "trajectory-neutral on PIE (ADE 60.5 vs 60.8 px) and trajectory-improving on JAAD "
    "(62.7 vs 83.1 px, −25%). Intent AUC rises monotonically with ego-motion information available "
    "(compensated 0.80 &lt; baseline 0.89 &lt; GT speed 0.93), suggesting a substantial share of "
    "reported intent accuracy reflects an ego-motion shortcut.",
    styles["Abstract"],
))

finding(
    "<b>Three linked findings:</b> (1) Fixed ORB passes the physical variance gate; learned residuals do not. "
    "(2) JAAD closes the sensor-free gap (−25% ADE). (3) Intent accuracy tracks available ego-motion signal, "
    "not pedestrian-specific reasoning alone."
)

# ═══════════════════════════════════════════════════════════════════════════════
# I. Introduction & Method
# ═══════════════════════════════════════════════════════════════════════════════
h1("I. Introduction &amp; Method")
body(
    "A camera on a moving vehicle makes every tracked object's image-plane trajectory a mixture of "
    "pedestrian motion and camera motion. Zhang et al. use ego-speed as input and soft supervisor; "
    "LIM removes speed to avoid causal confusion. Neither approach runs natively on JAAD, which ships "
    "no continuous ego-motion signal."
)
body(
    "<b>Fixed ORB-affine (main method).</b> Offline ORB + RANSAC partial affine on consecutive frames "
    "with pedestrian boxes masked. Transition target at centre (c<sub>x</sub>, c<sub>y</sub>) is "
    "M[c<sub>x</sub>, c<sub>y</sub>, 1]<sup>T</sup> − [c<sub>x</sub>, c<sub>y</sub>] — not global translation. "
    "Translation-only ORB flattened ~21% of standing+moving windows; per-agent affine ~89% (diagnostic, n=300)."
)
body(
    "<b>Learned LEMC (ablation).</b> Multi-agent displacement statistics → GRU → cumsum residual, "
    "with three supervision regimes: none, OBD magnitude, ORB regression."
)
body(
    "<b>Variance gate.</b> For standing pedestrians while the vehicle moves: "
    "Var(compensated) &lt; Var(raw) over the observation window. Independent of downstream labels."
)

fig("orb/paper_fig2_architecture.png",
    "Figure 1. Fixed ORB-affine pipeline: offline geometry compensates ego track before GRU prediction (sensor-free at inference).",
    width=6.0 * inch)
fig("orb/paper_fig1_teaser.png",
    "Figure 2. Standing pedestrian (mean OBD 18.6 km/h): raw ego-view track drifts; ORB-affine compensated track stays near-stationary.",
    width=4.8 * inch)

story.append(PageBreak())
h1("II. Experimental Setup")
tbl([
    ["Setting", "Value"],
    ["Observation τ / prediction η", "16 / 45 frames"],
    ["Time-to-event", "30–60 frames, 50% overlap"],
    ["PIE test windows", "1,773 (augmented store, median 8 agents/frame)"],
    ["JAAD test windows", "295 (124 scenes with ORB affines)"],
    ["Backbone", "Linear embed → 2-layer GRU → traj + intent heads"],
    ["Seeds", "0, 1, 2 (mean ± std)"],
], col_widths=[2.0 * inch, 4.4 * inch])

fig("frames/pie_multi_agent_detection.png",
    "Figure 3. PIE multi-agent scene: annotated bounding boxes used for track-based prediction (ego in orange).",
    width=5.8 * inch)
fig("frames/jaad_multi_agent_detection.png",
    "Figure 4. JAAD multi-agent scene: same bbox-only protocol, no OBD/GPS available.",
    width=5.8 * inch)

# ═══════════════════════════════════════════════════════════════════════════════
# III. Results — PIE
# ═══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
h1("III. Results — PIE")
h2("A. Main grid")
fig("metrics/pie_grid_ade_fde.png",
    "Figure 5. PIE test ADE and FDE (mean ± std, 3 seeds). Fixed ORB matches baseline; all learned LEMC variants are worse.",
    width=6.2 * inch)

tbl([
    ["Method", "ADE", "FDE", "AUC", "F1"],
    ["Baseline (bbox-only)", "60.81±3.31", "121.75±9.87", "0.89±0.01", "0.74±0.03"],
    ["GT OBD speed", "66.96±1.28", "136.89±3.02", "0.93±0.01", "0.85±0.02"],
    ["LEMC, no aux.", "81.68±5.13", "159.91±2.39", "0.77±0.05", "0.64±0.01"],
    ["LEMC + OBD aux.", "85.77±11.19", "164.25±11.33", "0.80±0.01", "0.60±0.03"],
    ["LEMC + ORB-reg.", "81.73±4.61", "163.54±7.55", "0.81±0.01", "0.63±0.00"],
    ["Fixed ORB affine", "60.49±1.83", "130.28±1.31", "0.80±0.01", "0.60±0.01"],
], col_widths=[1.45*inch, 0.85*inch, 0.85*inch, 0.75*inch, 0.75*inch], highlight=6)

h2("B. Standing-pedestrian variance gate")
fig("orb/pie_sign_gate.png",
    "Figure 6. Sign gate on PIE (standing + moving): learned LEMC+OBD 0.8% vs fixed ORB 93.1%.",
    width=4.2 * inch)
tbl([
    ["Method", "Dataset", "All windows", "Standing + moving"],
    ["LEMC + OBD aux.", "PIE", "0.7%", "0.8% — FAIL"],
    ["Fixed ORB affine", "PIE", "76.7%", "93.1% — PASS"],
    ["Fixed ORB affine", "JAAD", "71.9%", "96.9% — PASS"],
], col_widths=[1.4*inch, 0.7*inch, 1.0*inch, 1.5*inch], highlight=3)

h2("C. ORB geometric validation")
fig("orb/paper_fig4_orb_obd_scatter.png",
    "Figure 7. ORB |t| vs OBD speed (R²=0.41, n=75,268). Independent sanity check — ORB never consumes speed.",
    width=5.0 * inch)

h2("D. Stratified by ego-vehicle state")
fig("metrics/pie_stratified_ade.png",
    "Figure 8. ADE by ego state: fixed ORB best when accelerating; GT speed worst when accelerating/decelerating (LIM-aligned).",
    width=5.8 * inch)
tbl([
    ["Ego state", "n", "Baseline", "GT speed", "Fixed ORB"],
    ["Accelerating", "251", "58.5 / 127.8", "88.6 / 173.2", "52.2 / 122.6"],
    ["Constant", "1364", "64.5 / 130.6", "62.2 / 120.3", "59.4 / 130.1"],
    ["Decelerating", "158", "65.0 / 139.8", "82.0 / 174.7", "67.5 / 151.5"],
], col_widths=[1.0*inch, 0.45*inch, 1.15*inch, 1.15*inch, 1.15*inch])

# ═══════════════════════════════════════════════════════════════════════════════
# IV. Results — JAAD & Intent
# ═══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
h1("IV. Results — JAAD &amp; Intent")
h2("A. Cross-dataset trajectory")
fig("metrics/cross_dataset_ade.png",
    "Figure 9. Cross-dataset ADE: PIE neutral (60.8→60.5), JAAD −25% (83.1→62.7).",
    width=5.5 * inch)
fig("metrics/jaad_grid_metrics.png",
    "Figure 10. JAAD test metrics (3 seeds): ADE, FDE, ARB, FRB all improve with fixed ORB.",
    width=6.2 * inch)

tbl([
    ["Method", "ADE", "FDE", "ARB", "FRB", "AUC", "F1"],
    ["Baseline", "83.10±2.00", "156.25±4.31", "44.76±2.15", "71.23±2.08", "0.47±0.05", "0.90±0.00"],
    ["Fixed ORB", "62.68±2.34", "120.11±2.82", "34.80±2.25", "54.17±0.91", "0.44±0.00", "0.90±0.00"],
], col_widths=[0.9*inch, 0.75*inch, 0.75*inch, 0.7*inch, 0.7*inch, 0.65*inch, 0.6*inch], highlight=2, font_size=7.5)

body(
    "JAAD intent sits at or below chance (AUC 0.47 / 0.44); high F1 (0.90) reflects majority-class imbalance. "
    "Contribution on JAAD is trajectory-only."
)

h2("B. Trajectory overlays on real frames")
fig("trajectory/pie_compare_0.png",
    "Figure 11. PIE: baseline vs fixed ORB predictions on a real frame (observed, GT, predicted).",
    width=6.4 * inch)
fig("trajectory/jaad_compare_0.png",
    "Figure 12. JAAD: same comparison — fixed ORB closer to GT future trajectory.",
    width=6.4 * inch)

fig("trajectory/paper_fig3_qualitative.png",
    "Figure 13. Raw vs ORB-compensated ego tracks across ego-speed percentiles.",
    width=6.2 * inch)

h2("C. Intent: ego-motion shortcut")
fig("intent/pie_intent_auc_f1.png",
    "Figure 14. PIE intent AUC and F1 by method: GT speed best, baseline middle, compensated lowest.",
    width=5.5 * inch)
fig("intent/pie_roc_curves.png",
    "Figure 15. PIE intent ROC: baseline vs fixed ORB (test set).",
    width=4.5 * inch)

body(
    "<b>Ego-motion shortcut pattern:</b> AUC rises monotonically with ego-motion information: "
    "fixed ORB (0.80) &lt; baseline (0.89) &lt; GT speed (0.93). F1 follows the same order "
    "(0.60 &lt; 0.74 &lt; 0.85). Consistent with LIM's causal-confusion account."
)

fig("trajectory/lemc_qualitative_pie_lemc_augmented_auxobd_seed0_test.png",
    "Figure 16. Learned LEMC+OBD: compensated track shows non-physical loops (failed gate).",
    width=6.0 * inch)
fig("intent/lemc_obd_scatter_pie_lemc_augmented_auxobd_seed0_test.png",
    "Figure 17. Learned LEMC residual vs OBD: superficial R²=0.30 correlation does not imply valid compensation.",
    width=5.0 * inch)

# ═══════════════════════════════════════════════════════════════════════════════
# V. Discussion & Conclusion
# ═══════════════════════════════════════════════════════════════════════════════
story.append(PageBreak())
h1("V. Discussion &amp; Conclusion")
body(
    "<b>Why learned residuals fail.</b> A single shared 2D shift cannot represent rotation/scale components "
    "of camera motion whose apparent effect depends on image position. Multi-agent statistics average over "
    "a target that is not actually shared — structural mismatch, not insufficient data. ORB-regression "
    "supervision does not close the gap (81.7 vs 60.5 px), confirming architectural limitation."
)
body(
    "<b>Limitations.</b> Bbox-only GRU backbone (not comparable to Zhang I3D ADE 17.41). ORB depth confound. "
    "Hold-last for missing future affines. JAAD intent at chance. Ego-motion-shortcut finding from grid "
    "observation, not dedicated ablation."
)
finding(
    "<b>Conclusion.</b> Position-dependent ORB affine passes a physical variance test that three learned "
    "alternatives fail. Trajectory-neutral on PIE, −25% ADE on JAAD, zero sensor dependence. "
    "Intent AUC tracks available ego-motion signal — sensor-free geometric compensation is viable for "
    "trajectory; intent remains open."
)

h2("Reproducibility")
body(
    "<code>python scripts/09b_extract_orb_affine.py</code> → "
    "<code>python scripts/12_fixed_orb_affine.py --seed 0</code> → "
    "<code>bash scripts/10_run_grid.sh</code> → "
    "<code>python scripts/19_make_paper_figures.py</code> → "
    "<code>python scripts/generate_submission_report.py</code>"
)
body("Figures: <code>results/paper_figures/</code> · Checkpoints: <code>checkpoints/</code> · Configs: <code>configs/</code>")

h2("References")
refs = [
    "[1] Rasouli et al., PIE, ICCV 2019.",
    "[2] Kotseruba et al., JAAD 2.0.",
    "[3] Zhang et al., Decouple ego-view motions, IEEE TIP 2024.",
    "[4] LIM, Causal confusion in crossing intention, IEEE TITS 2025.",
    "[5] Rublee et al., ORB, ICCV 2011.",
    "[6] Fischler & Bolles, RANSAC, CACM 1981.",
]
for r in refs:
    body(r)

# ═══════════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    OUT,
    pagesize=letter,
    topMargin=0.55 * inch,
    bottomMargin=0.55 * inch,
    leftMargin=0.65 * inch,
    rightMargin=0.65 * inch,
    title="LEMC Submission Report",
)
doc.build(story)
print(f"saved to {OUT}")
