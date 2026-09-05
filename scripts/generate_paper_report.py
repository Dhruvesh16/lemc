"""Build LEMC path-to-paper PDF report from latest results and figures."""
from __future__ import annotations

import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
OUT_PATH = os.path.join(ROOT, "LEMC_path_to_paper_report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=16, spaceAfter=8))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1a3d5c")))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=7, leading=14))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey, spaceAfter=12, leading=11))
styles.add(ParagraphStyle(name="Finding", parent=styles["Normal"], spaceAfter=10, leading=14, backColor=colors.HexColor("#e8f5e9"), borderPadding=8))
styles.add(ParagraphStyle(name="Warn", parent=styles["Normal"], spaceAfter=10, leading=14, backColor=colors.HexColor("#fdecea"), borderPadding=8))

story: list = []


def h1(t: str):
    story.append(Paragraph(t, styles["H1"]))


def h2(t: str):
    story.append(Paragraph(t, styles["H2"]))


def body(t: str):
    story.append(Paragraph(t, styles["Body"]))


def finding(t: str):
    story.append(Paragraph(t, styles["Finding"]))


def warn(t: str):
    story.append(Paragraph(t, styles["Warn"]))


def fig(name: str, caption: str, width: float = 6.4 * inch):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        body(f"<i>[Figure missing: {name}]</i>")
        return
    img = Image(path)
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * ratio
    story.append(img)
    story.append(Paragraph(caption, styles["Caption"]))


def table(data, col_widths=None, highlight_row: int | None = None):
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#e8f5e9")))
        style.append(("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))


# --- Title page content
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("LEMC: Sensor-Free Ego-Motion Compensation", styles["Title"]))
story.append(Paragraph("Path-to-Paper Report", styles["Heading2"]))
story.append(Spacer(1, 0.1 * inch))
body(f"Generated {date.today().isoformat()} · PIE · Zhang protocol (τ=16, η=45, TTE 30–60)")
story.append(Spacer(1, 0.25 * inch))

finding(
    "<b>Bottom line:</b> Learned LEMC with OBD-magnitude supervision fails the standing-pedestrian "
    "variance gate and worsens ADE. <b>Fixed full-affine ORB compensation</b> passes the gate "
    "(93.1% on PIE, 96.9% on JAAD) and is <b>ADE-neutral on PIE</b> (60.5 vs 60.8 px) and "
    "<b>ADE-helpful on JAAD</b> (62.7 vs 83.1 px, −25%). GT ego speed helps intent (AUC 0.93) "
    "but hurts trajectory ADE — consistent with LIM."
)

# --- Executive summary
h1("1. Executive Summary")
body(
    "Egocentric pedestrian tracks mix pedestrian motion with camera motion. Zhang et al. (TIP 2024) "
    "decompose using ego speed as input and soft supervisor; LIM (TITS 2025) deletes speed to avoid "
    "causal confusion. This report documents the full path-to-paper run: protocol alignment, sign/variance "
    "gate, main experiment grid (3 seeds), ORB extraction, and the pivot from learned residuals to "
    "fixed geometric compensation."
)
warn(
    "<b>Critical finding (Steps 1–2):</b> Attempt-3 LEMC-on (OBD aux, old protocol) — sign gate "
    "0.75% standing flatten; ADE 65.9 vs baseline 43.9. Magnitude–speed correlation (R²=0.297) is "
    "not valid compensation."
)

# --- Protocol
h1("2. Protocol & Setup")
table(
    [
        ["Setting", "Value"],
        ["Observation τ", "16 frames"],
        ["Prediction η", "45 frames"],
        ["Time-to-event", "30–60 frames"],
        ["Overlap", "50%"],
        ["PIE test windows", "1773"],
        ["Track store", "Augmented (median 8 agents/frame)"],
        ["Seeds", "0, 1, 2 (mean ± std reported)"],
    ],
    col_widths=[2.2 * inch, 4.2 * inch],
)

# --- Method
h1("3. Method")
body(
    "<b>Learned LEMC:</b> Multi-agent displacement statistics (ego excluded) → GRU → per-transition "
    "residual → cumsum → subtract from ego track. Supervision variants: none, OBD magnitude correlation, "
    "learned ORB regression."
)
body(
    "<b>Fixed ORB affine (main result):</b> Offline ORB + RANSAC partial affine on consecutive frames "
    "with pedestrian boxes masked. Transition target at ego centre (cx,cy) is "
    "<code>M@[cx,cy,1] − [cx,cy]</code> — not global translation. Observation and future tracks are "
    "compensated into a stabilized frame; GRU predicts; ego-view ADE adds camera shift back."
)
finding(
    "Translation-only ORB flattened ~21% of standing+moving windows; full affine on the ego point "
    "flattened ~89% in a diagnostic (n=300). This is why global (tx,ty) supervision failed."
)
fig("fig2_architecture.png", "Figure 1. Fixed ORB-affine → GRU pipeline (sensor-free at inference).", width=6.2 * inch)

# --- Main results
story.append(PageBreak())
h1("4. Main Results (PIE test)")
fig("fig_main_grid_ade.png", "Figure 2. Main grid — ADE (mean ± std, 3 seeds). Fixed ORB matches baseline; learned LEMC variants are worse.")

table(
    [
        ["#", "Method", "ADE", "FDE", "AUC", "F1"],
        ["1", "Baseline", "60.81±3.31", "121.75±9.87", "0.89±0.01", "0.74±0.03"],
        ["2", "GT OBD speed", "66.96±1.28", "136.89±3.02", "0.93±0.01", "0.85±0.02"],
        ["3", "LEMC + OBD aux", "85.77±11.19", "164.25±11.33", "0.80±0.01", "0.60±0.03"],
        ["4", "LEMC + ORB (learned)", "81.73±4.61", "163.54±7.55", "0.81±0.01", "0.63±0.00"],
        ["5", "LEMC, no aux", "81.68±5.13", "159.91±2.39", "0.77±0.05", "0.64±0.01"],
        ["6", "Fixed ORB affine", "60.49±1.83", "130.28±1.31", "0.80±0.01", "0.60±0.01"],
    ],
    col_widths=[0.35 * inch, 1.5 * inch, 1.05 * inch, 1.05 * inch, 0.85 * inch, 0.85 * inch],
    highlight_row=6,
)
body("Baseline ARB/FRB (seed 0): 35.1 / 61.3 px at 30-frame horizon. Zhang et al. report ADE 17.41 on PIE with I3D vision — gap expected for bbox-only GRU.")

# --- Sign gate
h1("5. Sign / Variance Gate")
fig("fig_sign_gate.png", "Figure 3. Fraction of windows where compensated track variance &lt; raw variance (standing + moving). Gate: &gt;50% required.")

table(
    [
        ["Method", "Flatten (all)", "Standing + moving"],
        ["Attempt-3 OBD aux (old proto)", "0.7%", "0.8% — FAIL"],
        ["Fixed ORB affine v2", "76.7%", "93.1% — PASS"],
    ],
    col_widths=[2.5 * inch, 1.5 * inch, 2.0 * inch],
    highlight_row=3,
)

# --- Qualitative + ORB validation
h1("6. Qualitative & ORB Validation")
fig("fig1_teaser.png", "Figure 4. Teaser — standing pedestrian while car moves: raw ego-view track vs ORB-affine compensated.", width=4.8 * inch)
fig("fig3_qualitative.png", "Figure 5. Raw vs compensated tracks across ego-speed percentiles.", width=6.4 * inch)
fig("fig4_orb_obd_scatter.png", "Figure 6. ORB |t| vs OBD speed (R²≈0.41, n=75k transitions). Independent geometric sanity check.", width=5.0 * inch)

# --- Stratified
story.append(PageBreak())
h1("7. Stratified Analysis (seed 0)")
fig("fig5_stratified_ade.png", "Figure 7. ADE by ego-vehicle state — baseline vs GT speed vs fixed ORB.")

table(
    [
        ["Ego state", "n", "Baseline ADE/FDE", "GT speed ADE/FDE", "Fixed ORB ADE/FDE"],
        ["Accelerating", "251", "58.5 / 127.8", "88.6 / 173.2", "52.2 / 122.6"],
        ["Constant", "1364", "64.5 / 130.6", "62.2 / 120.3", "59.4 / 130.1"],
        ["Decelerating", "158", "65.0 / 139.8", "82.0 / 174.7", "67.5 / 151.5"],
    ],
    col_widths=[1.1 * inch, 0.5 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch],
)
body(
    "Fixed ORB helps most when accelerating (geometry-dominated). GT speed is worst on accelerating/"
    "decelerating for ADE while best for intent overall — speed encodes driver reaction more than "
    "pedestrian dynamics."
)

# --- Positioning
h1("8. Positioning vs Related Work")
table(
    [
        ["", "Sensor at inference?", "Route", "JAAD-ready?"],
        ["Zhang et al. (TIP 2024)", "Yes (speed/action)", "Two-tower + action-aware loss", "Behaviour re-encoding"],
        ["LIM (TITS 2025)", "No (deleted)", "Skeleton + adversarial scrub", "Yes"],
        ["Fixed ORB (ours)", "No", "Offline ORB affine → compensate", "Yes"],
        ["Learned LEMC", "No", "Multi-agent residual", "Failed gate here"],
    ],
    col_widths=[1.6 * inch, 1.5 * inch, 2.0 * inch, 1.3 * inch],
    highlight_row=3,
)

# --- Limitations & next steps
h1("9. JAAD Results (sensor-free, no OBD)")
body(
    "JAAD 2.0: 323 videos, 336/52/295 train/val/test windows (TTE 30–60, 50% overlap). "
    "No OBD speed — sensor-free only. ORB affines extracted from 124 scenes with images available. "
    "3 seeds run on CPU (training ~10s/seed; small dataset)."
)
table(
    [
        ["Method", "ADE", "FDE", "ARB", "FRB", "AUC", "F1", "Sign gate (s+m)"],
        ["Baseline GRU", "83.10±2.00", "156.25±4.31", "44.76±2.15", "71.23±2.08", "0.47±0.05", "0.90±0.00", "—"],
        ["Fixed ORB affine", "62.68±2.34", "120.11±2.82", "34.80±2.25", "54.17±0.91", "0.44±0.00", "0.90±0.00", "96.9%"],
    ],
    col_widths=[1.3*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.6*inch, 0.8*inch],
    highlight_row=2,
)
finding(
    "JAAD ADE drops <b>25%</b> (83.1 → 62.7 px). Unlike PIE (ADE-neutral), JAAD benefits "
    "from affine compensation — consistent with stronger and more variable camera motion across "
    "JAAD's diverse clip conditions. Sign gate 96.9% standing+moving (all windows 71.9%)."
)

story.append(PageBreak())
h1("10. Limitations & Next Steps")
body(
    "Single GRU backbone; no on-vehicle latency study beyond CPU note (1.58 ms/seq); "
    "learned LEMC does not match ORB under joint training; ORB depth confound; "
    "hold-last for missing future affines."
)
body("<b>Next:</b> wire \\cite{} calls into paper/main.tex; compile blind PDF; "
     "outsider read; submit to IEEE IV 2026.")

h1("11. Reproducibility")
body(
    "<code>python scripts/09b_extract_orb_affine.py</code> → "
    "<code>python scripts/12_fixed_orb_affine.py --seed 0</code> → "
    "<code>bash scripts/10_run_grid.sh</code> → "
    "<code>python scripts/13_make_figures.py</code> → "
    "<code>python scripts/generate_paper_report.py</code>"
)

doc = SimpleDocTemplate(
    OUT_PATH,
    pagesize=letter,
    topMargin=0.55 * inch,
    bottomMargin=0.55 * inch,
    leftMargin=0.65 * inch,
    rightMargin=0.65 * inch,
    title="LEMC Path to Paper Report",
)
doc.build(story)
print(f"saved to {OUT_PATH}")
