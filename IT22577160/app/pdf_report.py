"""
pdf_report.py  –  AURA CSU Decision Support System
Clinical PDF report builder (ReportLab Platypus).

Produces a clean, non-overlapping A4 PDF with:
  • Header letterhead with system branding & generation timestamp
  • Case / Patient metadata block
  • AI Prediction summary box  (predicted drug group, confidence, step)
  • Top-3 prediction table
  • UAS7 severity block (if present)
  • Step alignment commentary
  • Lab values table
  • Clinical features table
  • Extracted OCR labs table
  • Visual analysis images  (original skin, Grad-CAM, redness map)
  • EAACI guideline step detail
  • Disclaimer / footer
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ──────────────────────────────────────────────────────────────────────────────
# Palette  (clinical blues / greys)
# ──────────────────────────────────────────────────────────────────────────────
C_HEADER_BG    = colors.HexColor("#1A3A5C")   # deep navy
C_HEADER_TEXT  = colors.white
C_ACCENT       = colors.HexColor("#2E7DBF")   # mid blue
C_ACCENT_LIGHT = colors.HexColor("#D7E8F5")   # pale blue tint
C_SECTION_BG   = colors.HexColor("#EEF4FA")   # very light blue
C_TABLE_HEAD   = colors.HexColor("#2E7DBF")
C_TABLE_ALT    = colors.HexColor("#F4F8FC")
C_BORDER       = colors.HexColor("#B0C8E0")
C_WARNING_BG   = colors.HexColor("#FFF3CD")
C_WARNING_BDR  = colors.HexColor("#FFC107")
C_OK_BG        = colors.HexColor("#D4EDDA")
C_OK_BDR       = colors.HexColor("#28A745")
C_TEXT         = colors.HexColor("#1C2B3A")
C_LIGHT_TEXT   = colors.HexColor("#5A6A7A")
C_DIVIDER      = colors.HexColor("#C5D8EC")

PAGE_W, PAGE_H = A4
MARGIN         = 1.8 * cm
CONTENT_W      = PAGE_W - 2 * MARGIN


# ──────────────────────────────────────────────────────────────────────────────
# Display label helpers  (match the frontend DRUG_META / STEP_META mappings)
# ──────────────────────────────────────────────────────────────────────────────

_DRUG_LABELS: Dict[str, str] = {
    "H1_ANTIHISTAMINE": "H1 Antihistamine",
    "LTRA":             "LTRA",
    "ADVANCED_THERAPY": "Advanced Therapy",
    "OTHER":            "Other Therapy",
}

_STEP_LABELS: Dict[str, str] = {
    "STEP_1": "Step 1 — Standard-dose 2nd-gen H1-AH",
    "STEP_2": "Step 2 — Up-dosed H1-AH (×4)",
    "STEP_3": "Step 3 — Add Omalizumab (anti-IgE)",
    "STEP_4": "Step 4 — Ciclosporin / Immunosuppressant",
}


def _drug_label(raw: str) -> str:
    """Convert raw drug group enum (e.g. H1_ANTIHISTAMINE) to display label."""
    return _DRUG_LABELS.get(raw, raw.replace("_", " ").title() if raw else "N/A")


def _step_label(raw: str) -> str:
    """Convert raw step enum (e.g. STEP_1) to display label."""
    return _STEP_LABELS.get(raw, raw.replace("_", " ").title() if raw else "N/A")


# ──────────────────────────────────────────────────────────────────────────────
# Style sheet
# ──────────────────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()
    def S(name, **kw):
        return ParagraphStyle(name=name, **kw)

    return {
        "title": S("RPT_title",
            fontSize=20, fontName="Helvetica-Bold",
            textColor=C_HEADER_TEXT, alignment=TA_CENTER,
            spaceAfter=2),
        "subtitle": S("RPT_subtitle",
            fontSize=10, fontName="Helvetica",
            textColor=C_HEADER_TEXT, alignment=TA_CENTER,
            spaceAfter=0),
        "section": S("RPT_section",
            fontSize=11, fontName="Helvetica-Bold",
            textColor=C_ACCENT, spaceBefore=10, spaceAfter=4),
        "body": S("RPT_body",
            fontSize=9, fontName="Helvetica",
            textColor=C_TEXT, leading=13),
        "body_bold": S("RPT_body_bold",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=C_TEXT, leading=13),
        "small": S("RPT_small",
            fontSize=7.5, fontName="Helvetica",
            textColor=C_LIGHT_TEXT, leading=11),
        "disclaimer": S("RPT_disclaimer",
            fontSize=7, fontName="Helvetica-Oblique",
            textColor=C_LIGHT_TEXT, alignment=TA_CENTER, leading=10),
        "th": S("RPT_th",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER),
        "td": S("RPT_td",
            fontSize=9, fontName="Helvetica",
            textColor=C_TEXT, alignment=TA_LEFT),
        "td_c": S("RPT_td_c",
            fontSize=9, fontName="Helvetica",
            textColor=C_TEXT, alignment=TA_CENTER),
        "summary_label": S("RPT_sum_lbl",
            fontSize=8, fontName="Helvetica",
            textColor=C_LIGHT_TEXT, alignment=TA_CENTER),
        "summary_value": S("RPT_sum_val",
            fontSize=14, fontName="Helvetica-Bold",
            textColor=C_ACCENT, alignment=TA_CENTER),
        "summary_sub": S("RPT_sum_sub",
            fontSize=8.5, fontName="Helvetica",
            textColor=C_TEXT, alignment=TA_CENTER),
        "caption": S("RPT_caption",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=C_LIGHT_TEXT, alignment=TA_CENTER, spaceAfter=4),
        "footer": S("RPT_footer",
            fontSize=7.5, fontName="Helvetica",
            textColor=C_LIGHT_TEXT, alignment=TA_RIGHT),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Page template helpers  (header letterhead + footer on every page)
# ──────────────────────────────────────────────────────────────────────────────

def _make_page_template(doc, styles):
    """Return a PageTemplate that draws the letterhead header + footer."""

    header_h = 2.6 * cm
    footer_h = 0.9 * cm
    body_top  = PAGE_H - header_h - MARGIN * 0.6
    body_bot  = footer_h + MARGIN * 0.4

    frame = Frame(
        MARGIN, body_bot,
        CONTENT_W, body_top - body_bot,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
        id="body",
    )

    def _on_page(canvas, doc):
        canvas.saveState()

        # ── Header band ──────────────────────────────────────────────────────
        canvas.setFillColor(C_HEADER_BG)
        canvas.rect(0, PAGE_H - header_h, PAGE_W, header_h, stroke=0, fill=1)

        # Accent stripe
        canvas.setFillColor(C_ACCENT)
        canvas.rect(0, PAGE_H - header_h - 3, PAGE_W, 3, stroke=0, fill=1)

        # Title text
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(MARGIN, PAGE_H - header_h + 1.55 * cm,
                          "AURA  ·  CSU Clinical Decision Support System")

        canvas.setFont("Helvetica", 8.5)
        canvas.drawString(MARGIN, PAGE_H - header_h + 0.85 * cm,
                          "Chronic Spontaneous Urticaria  ·  AI-Assisted Pharmacotherapy Recommendation")

        # Page number (right side)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - header_h + 1.1 * cm,
                               f"Page {doc.page}")

        # Generated date (right)
        canvas.setFont("Helvetica", 7.5)
        ts = datetime.now().strftime("%d %b %Y  %H:%M")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - header_h + 0.55 * cm,
                               f"Generated: {ts}")

        # ── Footer band ──────────────────────────────────────────────────────
        canvas.setFillColor(C_SECTION_BG)
        canvas.rect(0, 0, PAGE_W, footer_h + 0.3 * cm, stroke=0, fill=1)
        canvas.setFillColor(C_DIVIDER)
        canvas.rect(0, footer_h + 0.3 * cm, PAGE_W, 0.06 * cm, stroke=0, fill=1)

        canvas.setFillColor(C_LIGHT_TEXT)
        canvas.setFont("Helvetica-Oblique", 6.5)
        canvas.drawCentredString(PAGE_W / 2, 0.4 * cm,
            "CONFIDENTIAL · FOR AUTHORISED CLINICAL USE ONLY · "
            "AI output is decision-support, not a substitute for clinical judgement.")

        canvas.restoreState()

    return PageTemplate(id="clinical", frames=[frame], onPage=_on_page)


# ──────────────────────────────────────────────────────────────────────────────
# Utility flowables
# ──────────────────────────────────────────────────────────────────────────────

def _divider(color=C_DIVIDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=4, spaceBefore=4)


def _section_title(text: str, styles) -> Paragraph:
    return Paragraph(f"<b>{text.upper()}</b>", styles["section"])


def _colored_box_table(rows, col_widths, head_row=True):
    """Render a styled table with alternating row colours."""
    tbl = Table(rows, colWidths=col_widths, repeatRows=(1 if head_row else 0))
    style_cmds = [
        ("BACKGROUND",  (0, 0), (-1, 0 if not head_row else 0), C_TABLE_HEAD if head_row else C_SECTION_BG),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white if head_row else C_TEXT),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0), 9),
        ("ALIGN",       (0, 0), (-1, 0), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_TABLE_ALT]),
        ("GRID",        (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _pil_to_reportlab_image(pil_img: PILImage.Image,
                             max_w: float, max_h: float) -> Image:
    """Convert a PIL image to a ReportLab Image flowable (no temp file)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    return Image(buf, width=w * scale, height=h * scale)


def _summary_card(label: str, value: str, sub: str, styles, bg=C_ACCENT_LIGHT,
                  bdr=C_ACCENT) -> Table:
    """Single metric card (label / big-value / sub-value)."""
    data = [
        [Paragraph(label, styles["summary_label"])],
        [Paragraph(value, styles["summary_value"])],
        [Paragraph(sub,   styles["summary_sub"])],
    ]
    t = Table(data, colWidths=[4.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0),(-1, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.2, bdr),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────────────

def _section_meta(patient_meta: Dict[str, Any], prediction: Dict[str, Any],
                  styles) -> list:
    """Case information block."""
    items = []
    ts = datetime.now().strftime("%d %B %Y  %H:%M")
    case_id = patient_meta.get("Case ID", "N/A")
    step    = _step_label(prediction.get("mapped_guideline_step", "N/A"))

    _ml = ParagraphStyle("_meta_lbl", fontSize=8.5, fontName="Helvetica-Bold",
                          textColor=C_TEXT, leading=12)
    _mv = ParagraphStyle("_meta_val", fontSize=8.5, fontName="Helvetica",
                          textColor=C_TEXT, leading=12)

    # 4 cols: label | value | label | value  — widths sum to CONTENT_W
    LW, VW = 3.5*cm, CONTENT_W/2 - 3.5*cm   # ≈ 5.2 cm each value col
    abstain_txt = "ABSTAINED" if prediction.get("abstain") else "Prediction made"

    info_rows = [
        [Paragraph("Case / Session ID", _ml), Paragraph(case_id,      _mv),
         Paragraph("Report Date",       _ml), Paragraph(ts,            _mv)],
        [Paragraph("Guideline Version",  _ml), Paragraph("EAACI 2022", _mv),
         Paragraph("Model Version",      _ml), Paragraph("AURA v1.0",  _mv)],
        [Paragraph("Treatment Step",     _ml), Paragraph(step,         _mv),
         Paragraph("Abstain Flag",        _ml), Paragraph(abstain_txt,  _mv)],
    ]

    tbl = Table(info_rows, colWidths=[LW, VW, LW, VW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), C_SECTION_BG),
        ("BACKGROUND",    (2, 0), (2, -1), C_SECTION_BG),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
    ]))
    items += [_section_title("Case Information", styles), _divider(), tbl, Spacer(1, 6)]
    return items


def _section_prediction_summary(prediction: Dict[str, Any], styles) -> list:
    """High-visibility AI prediction summary cards + OOD flag."""
    items = [_section_title("AI Prediction Summary", styles), _divider()]

    drug   = _drug_label(prediction.get("predicted_drug_group", "N/A"))
    conf   = prediction.get("confidence", 0.0)
    step   = _step_label(prediction.get("mapped_guideline_step", "N/A"))
    ood    = prediction.get("ood_flag", False)
    uas7   = prediction.get("uas7_score")
    align  = prediction.get("guideline_step_alignment")

    conf_pct = f"{conf * 100:.1f}%"
    conf_sub = "Confidence"

    # Choose card colour based on confidence level
    def _conf_color():
        if conf >= 0.75: return C_OK_BG, C_OK_BDR
        if conf >= 0.55: return C_ACCENT_LIGHT, C_ACCENT
        return C_WARNING_BG, C_WARNING_BDR

    conf_bg, conf_bdr = _conf_color()

    card_drug  = _summary_card("Predicted Drug Group",  drug,      "EAACI Pharmacotherapy", styles)
    card_conf  = _summary_card("Model Confidence",      conf_pct,  conf_sub, styles, conf_bg, conf_bdr)
    card_step  = _summary_card("Guideline Step",        step,      "Mapped EAACI Step",     styles)

    cards_row  = Table([[card_drug, Spacer(0.4*cm, 1), card_conf,
                         Spacer(0.4*cm, 1), card_step]],
                       colWidths=[4.5*cm, 0.4*cm, 4.5*cm, 0.4*cm, 4.5*cm])
    cards_row.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                                    ("LEFTPADDING",  (0,0), (-1,-1), 0),
                                    ("RIGHTPADDING", (0,0), (-1,-1), 0)]))
    items.append(cards_row)
    items.append(Spacer(1, 8))

    # OOD warning row
    if ood:
        ood_text = (
            "<b>⚠  Out-of-Distribution Alert</b>  ·  The input data falls outside the model's training "
            "distribution (OOD z-score: {:.2f}).  Treat this prediction with heightened caution and "
            "defer to clinical assessment.".format(prediction.get("ood_z", 0.0))
        )
        ood_tbl = Table([[Paragraph(ood_text, styles["body"])]],
                        colWidths=[CONTENT_W])
        ood_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_WARNING_BG),
            ("BOX",           (0,0), (-1,-1), 1.0, C_WARNING_BDR),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items += [ood_tbl, Spacer(1, 4)]

    # Abstain notice
    if prediction.get("abstain"):
        abs_tbl = Table([[Paragraph(
            "<b>Model Abstained</b>  ·  Prediction confidence is below the configured threshold.  "
            "The system has declined to commit to a recommendation.  Manual clinical review required.",
            styles["body"]
        )]], colWidths=[CONTENT_W])
        abs_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_WARNING_BG),
            ("BOX",           (0,0), (-1,-1), 1.0, C_WARNING_BDR),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items += [abs_tbl, Spacer(1, 4)]

    # Step alignment commentary
    if align:
        align_map = {
            "aligned":      ("✓  Aligned",      C_OK_BG,      C_OK_BDR,      "Model step and UAS7-recommended step are in agreement."),
            "model_higher": ("↑  Model Higher",  C_WARNING_BG, C_WARNING_BDR, "Model recommends a higher treatment step than UAS7 alone suggests."),
            "model_lower":  ("↓  Model Lower",   C_ACCENT_LIGHT, C_ACCENT,    "Model recommends a lower treatment step than UAS7 alone suggests."),
        }
        lbl, bg, bdr, desc = align_map.get(align, (align, C_ACCENT_LIGHT, C_ACCENT, ""))
        al_tbl = Table([[Paragraph(
            f"<b>Step Alignment ({lbl})</b>  ·  {desc}", styles["body"]
        )]], colWidths=[CONTENT_W])
        al_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), bg),
            ("BOX",           (0,0), (-1,-1), 1.0, bdr),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        items += [al_tbl, Spacer(1, 4)]

    items.append(Spacer(1, 4))
    return items


def _section_top3(prediction: Dict[str, Any], styles) -> list:
    """Top-3 predicted drug groups table."""
    top3: List[List[Any]] = prediction.get("top3", [])
    if not top3:
        return []

    items = [_section_title("Top-3 Candidate Drug Groups", styles), _divider()]
    header = [
        Paragraph("Rank",         styles["th"]),
        Paragraph("Drug Group",   styles["th"]),
        Paragraph("Confidence",   styles["th"]),
        Paragraph("Probability",  styles["th"]),
    ]
    rows = [header]
    medal = ["①", "②", "③"]
    for i, entry in enumerate(top3[:3]):
        drug_name = entry[0] if len(entry) > 0 else "N/A"
        prob      = float(entry[1]) if len(entry) > 1 else 0.0
        rows.append([
            Paragraph(medal[i] if i < 3 else str(i+1), styles["td_c"]),
            Paragraph(_drug_label(str(drug_name)), styles["td"]),
            Paragraph(f"{prob * 100:.2f}%", styles["td_c"]),
            _confidence_bar(prob),
        ])

    tbl = Table(rows, colWidths=[1.5*cm, 8.0*cm, 3.0*cm, CONTENT_W - 12.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_TABLE_HEAD),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_TABLE_ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",         (2, 0), (2, -1),  "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    items += [tbl, Spacer(1, 6)]
    return items


def _confidence_bar(value: float) -> Table:
    """Inline mini percentage bar rendered as a 2-cell table."""
    fill_w = max(int(value * 40), 1)   # max 40 pts
    empty_w = 40 - fill_w
    color = C_OK_BDR if value >= 0.75 else (C_ACCENT if value >= 0.55 else C_WARNING_BDR)
    inner = Table(
        [["", ""]],
        colWidths=[fill_w, max(empty_w, 0.1)],
        rowHeights=[8],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0, 0), color),
        ("BACKGROUND",    (1,0), (1, 0), C_BORDER),
        ("BOX",           (0,0), (-1,-1), 0.4, C_BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    return inner


def _section_uas7(prediction: Dict[str, Any], styles) -> list:
    """UAS7 score and severity interpretation."""
    score = prediction.get("uas7_score")
    interp: Dict[str, Any] = prediction.get("uas7_interpretation") or {}
    if score is None:
        return []

    items = [_section_title("Urticaria Activity Score (UAS7)", styles), _divider()]

    severity  = interp.get("severity", interp.get("severity_label", interp.get("label", "N/A")))
    rec_step  = interp.get("recommended_step", "N/A")
    category  = interp.get("category",  "")

    # Colour band by severity
    sev_colors = {
        "none":            (C_OK_BG,      C_OK_BDR),
        "minimal":         (C_OK_BG,      C_OK_BDR),
        "well-controlled": (C_OK_BG,      C_OK_BDR),
        "mild":            (C_ACCENT_LIGHT, C_ACCENT),
        "moderate":        (C_WARNING_BG,   C_WARNING_BDR),
        "severe":          (colors.HexColor("#F8D7DA"), colors.HexColor("#DC3545")),
        "very severe":     (colors.HexColor("#F8D7DA"), colors.HexColor("#DC3545")),
    }
    bg, bdr = sev_colors.get((severity or "").lower().strip(),
                              (C_ACCENT_LIGHT, C_ACCENT))

    rows = [
        ["UAS7 Score",   f"{score:.1f} / 42",  "Severity",         category or severity or "N/A"],
        ["Activity Band", interp.get("range", "N/A"), "Recommended Step", str(rec_step)],
    ]
    tbl = Table(rows, colWidths=[3.5*cm, 5.0*cm, 3.5*cm, 6.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), bg),
        ("BACKGROUND",    (2, 0), (2, -1), bg),
        ("BOX",           (0, 0), (-1, -1), 1.0, bdr),
        ("INNERGRID",     (0, 0), (-1, -1), 0.4, C_BORDER),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME",      (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    items += [tbl, Spacer(1, 6)]
    return items


def _section_lab_values(prediction: Dict[str, Any], styles) -> list:
    """Lab values used by the model."""
    used: Dict[str, float] = prediction.get("used_features", {})
    lab_keys = ["CRP", "FT4", "IgE", "VitD", "Age"]
    rows = [[
        Paragraph("Biomarker",          styles["th"]),
        Paragraph("Value (model input)", styles["th"]),
        Paragraph("Reference note",      styles["th"]),
    ]]
    ref_notes = {
        "CRP":  "< 5 mg/L  (normal)",
        "FT4":  "9–25 pmol/L",
        "IgE":  "< 100 IU/mL  (adult)",
        "VitD": "30–100 ng/mL  (sufficient)",
        "Age":  "years",
    }
    for k in lab_keys:
        val = used.get(k, 0.0)
        rows.append([
            Paragraph(k, styles["td_c"]),
            Paragraph(f"{val:.2f}", styles["td_c"]),
            Paragraph(ref_notes.get(k, "—"), styles["td"]),
        ])
    if len(rows) == 1:
        return []

    items = [_section_title("Laboratory Values", styles), _divider()]
    tbl = _colored_box_table(rows, [4.0*cm, 5.0*cm, CONTENT_W - 9.0*cm])
    items += [tbl, Spacer(1, 6)]
    return items


def _section_clinical_features(prediction: Dict[str, Any], styles) -> list:
    """Clinical / patient features."""
    used: Dict[str, float] = prediction.get("used_features", {})
    lab_keys = {"CRP", "FT4", "IgE", "VitD", "Age"}
    clin_items = [(k, v) for k, v in used.items() if k not in lab_keys and v != 0.0]
    if not clin_items:
        return []

    rows = [[
        Paragraph("Clinical Variable", styles["th"]),
        Paragraph("Value",             styles["th"]),
    ]]
    for k, v in clin_items:
        rows.append([
            Paragraph(str(k).replace("\n", " "), styles["td"]),
            Paragraph(f"{v:.2f}", styles["td_c"]),
        ])

    items = [_section_title("Clinical Features", styles), _divider()]
    tbl = _colored_box_table(rows, [CONTENT_W * 0.75, CONTENT_W * 0.25])
    items += [tbl, Spacer(1, 6)]
    return items


def _section_extracted_labs(extracted_labs: Dict[str, Any], styles) -> list:
    """OCR-extracted lab values."""
    if not extracted_labs:
        return []

    flat = {k: v for k, v in extracted_labs.items()
            if not isinstance(v, dict) and k != "flags"}
    if not flat:
        return []

    rows = [[
        Paragraph("Parameter", styles["th"]),
        Paragraph("Extracted Value", styles["th"]),
    ]]
    for k, v in flat.items():
        rows.append([
            Paragraph(str(k), styles["td"]),
            Paragraph(str(v) if v is not None else "—", styles["td_c"]),
        ])

    items = [_section_title("OCR-Extracted Lab Results", styles), _divider()]
    tbl = _colored_box_table(rows, [CONTENT_W * 0.55, CONTENT_W * 0.45])
    items += [tbl, Spacer(1, 6)]
    return items


def _section_modality_weights(prediction: Dict[str, Any], styles) -> list:
    """Multimodal gate weights."""
    weights: List[float] = prediction.get("modality_gate_weights", [])
    if not weights:
        return []

    labels = ["Image", "Lab Values", "Clinical"]
    display_labels = labels[:len(weights)]

    rows = [[Paragraph(lbl, styles["th"]) for lbl in display_labels]]
    rows.append([Paragraph(f"{w*100:.1f}%", styles["td_c"]) for w in weights])

    n = len(weights)
    col_w = CONTENT_W / n
    tbl = _colored_box_table(rows, [col_w] * n)

    items = [_section_title("Modality Contribution Weights", styles), _divider()]
    items += [
        Paragraph("The multimodal gate determines how much each data source influenced this prediction:",
                  styles["body"]),
        Spacer(1, 4),
        tbl,
        Spacer(1, 6),
    ]
    return items


def _section_images(images: Dict[str, PILImage.Image], styles) -> list:
    """Visual analysis section with three side-by-side images."""
    if not images:
        return []

    items = [_section_title("Visual Analysis", styles), _divider()]
    items.append(Paragraph(
        "Left: Original skin image.  Centre: Grad-CAM saliency map (regions influencing the prediction).  "
        "Right: Redness / erythema distribution map.",
        styles["small"],
    ))
    items.append(Spacer(1, 5))

    img_keys    = ["skin", "gradcam", "redness"]
    captions    = ["Original Image", "Grad-CAM Saliency", "Redness Map"]
    n_cols      = sum(1 for k in img_keys if k in images)
    cell_w      = (CONTENT_W - (n_cols - 1) * 0.3 * cm) / n_cols
    img_h       = 5.0 * cm
    gap         = Spacer(0.3 * cm, 1)

    img_cells  = []
    cap_cells  = []
    col_widths = []

    for idx, (k, cap) in enumerate(zip(img_keys, captions)):
        if k not in images:
            continue
        pil_img = images[k]
        rl_img  = _pil_to_reportlab_image(pil_img, cell_w, img_h)

        # Wrap image in a centred single-cell table with border
        img_box = Table([[rl_img]], colWidths=[cell_w])
        img_box.setStyle(TableStyle([
            ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 3),
            ("RIGHTPADDING",  (0,0), (-1,-1), 3),
            ("BACKGROUND",    (0,0), (-1,-1), colors.white),
        ]))

        img_cells.append(img_box)
        cap_cells.append(Paragraph(cap, styles["caption"]))
        col_widths.append(cell_w)

        if idx < n_cols - 1:
            img_cells.append(gap)
            cap_cells.append(Paragraph("", styles["caption"]))
            col_widths.append(0.3 * cm)

    if img_cells:
        grid = Table([img_cells, cap_cells], colWidths=col_widths)
        grid.setStyle(TableStyle([
            ("VALIGN",  (0,0), (-1,-1), "TOP"),
            ("ALIGN",   (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",   (0,0), (-1,-1), 0),
            ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ("TOPPADDING",    (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ]))
        items.append(grid)
    items.append(Spacer(1, 8))
    return items


def _section_guideline(prediction: Dict[str, Any], styles) -> list:
    """EAACI guideline step detail."""
    detail: Dict[str, Any] = prediction.get("guideline_step_detail", {})
    if not detail:
        return []

    items = [_section_title("EAACI Guideline Step Details", styles), _divider()]

    step  = prediction.get("mapped_guideline_step", "N/A")
    label = detail.get("label", step)
    ind   = detail.get("indication", "")
    drugs = detail.get("drugs", [])
    dur   = detail.get("duration", "")
    notes = detail.get("notes", [])

    # Header row
    hdr_style = ParagraphStyle("_gl_hdr", fontSize=9.5, fontName="Helvetica-Bold",
                                textColor=colors.white)
    hdr_tbl = Table(
        [[Paragraph(f"{step}  ·  {label}", hdr_style)]],
        colWidths=[CONTENT_W],
    )
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    items.append(hdr_tbl)

    # Column widths: label 3.2 cm, value fills the rest
    LBL_W = 3.2 * cm
    VAL_W = CONTENT_W - LBL_W

    lbl_sty = ParagraphStyle("_gl_lbl", fontSize=8.5, fontName="Helvetica-Bold",
                              textColor=C_TEXT, leading=12)
    val_sty = ParagraphStyle("_gl_val", fontSize=8.5, fontName="Helvetica",
                              textColor=C_TEXT, leading=13)
    drug_sty = ParagraphStyle("_gl_drug", fontSize=8.5, fontName="Helvetica",
                               textColor=C_TEXT, leading=14, leftIndent=6,
                               bulletIndent=0)

    detail_rows = []
    if ind:
        detail_rows.append([
            Paragraph("Indication",  lbl_sty),
            Paragraph(str(ind),      val_sty),
        ])
    if dur:
        detail_rows.append([
            Paragraph("Duration",    lbl_sty),
            Paragraph(str(dur),      val_sty),
        ])
    if drugs:
        # Each drug on its own line as a bullet
        drug_paras = [Paragraph(f"• {d}", drug_sty) for d in drugs]
        detail_rows.append([
            Paragraph("Recommended<br/>Drugs", lbl_sty),
            drug_paras,
        ])
    if notes:
        note_list = notes if isinstance(notes, list) else [notes]
        for ni, note in enumerate(note_list):
            lbl = f"Note {ni + 1}" if len(note_list) > 1 else "Note"
            detail_rows.append([
                Paragraph(lbl,       lbl_sty),
                Paragraph(str(note), val_sty),
            ])

    if detail_rows:
        tbl = Table(detail_rows, colWidths=[LBL_W, VAL_W])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, -1), C_SECTION_BG),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.white, C_TABLE_ALT]),
            ("BACKGROUND",    (0, 0), (0, -1), C_SECTION_BG),  # label col always tinted
            ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        items.append(tbl)

    items.append(Spacer(1, 6))
    return items


def _cu_metric_card(label: str, big_value: str, sub: str,
                    severity: str, styles) -> Table:
    """
    Compact metric card: label / big value / sub-line.
    severity in: 'high','moderate','mild','low','none','na'
    """
    _pal = {
        "high":     (colors.HexColor("#F8D7DA"), colors.HexColor("#DC3545")),
        "moderate": (C_WARNING_BG,               C_WARNING_BDR),
        "mild":     (C_ACCENT_LIGHT,             C_ACCENT),
        "low":      (C_OK_BG,                    C_OK_BDR),
        "none":     (C_OK_BG,                    C_OK_BDR),
        "na":       (C_SECTION_BG,               C_BORDER),
    }
    bg, bdr = _pal.get(severity.lower(), (C_ACCENT_LIGHT, C_ACCENT))

    val_style = ParagraphStyle("_cu_val", fontSize=13, fontName="Helvetica-Bold",
                               textColor=bdr, alignment=TA_CENTER)
    lbl_style = ParagraphStyle("_cu_lbl", fontSize=7.5, fontName="Helvetica",
                               textColor=C_LIGHT_TEXT, alignment=TA_CENTER)
    sub_style = ParagraphStyle("_cu_sub", fontSize=8, fontName="Helvetica",
                               textColor=C_TEXT, alignment=TA_CENTER)

    badge_bg  = bdr
    badge_txt = ParagraphStyle("_cu_bdg", fontSize=7, fontName="Helvetica-Bold",
                               textColor=colors.white, alignment=TA_CENTER)

    data = [
        [Paragraph(label,     lbl_style)],
        [Paragraph(big_value, val_style)],
        [Paragraph(severity.upper(), badge_txt)],
        [Paragraph(sub,       sub_style)],
    ]
    t = Table(data, colWidths=[3.9 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("BACKGROUND",    (0, 2), (-1, 2),  badge_bg),
        ("BOX",           (0, 0), (-1, -1), 1.0, bdr),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _cu_gauge_row(label: str, value_str: str, fill_frac: float,
                  bar_color, styles) -> Table:
    """
    Single horizontal gauge row:
      [label 5.0cm] [value 2.2cm] [████░░░░  9.2cm fill bar]
    fill_frac in [0, 1].
    """
    bar_total = 9.2 * cm
    fill_w  = max(bar_total * min(fill_frac, 1.0), 0.05 * cm)
    empty_w = max(bar_total - fill_w, 0.05 * cm)

    bar_inner = Table([["", ""]],
                      colWidths=[fill_w, empty_w], rowHeights=[9])
    bar_inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), bar_color),
        ("BACKGROUND",    (1, 0), (1, 0), C_BORDER),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    row = Table(
        [[Paragraph(label, styles["td"]),
          Paragraph(f"<b>{value_str}</b>", styles["td_c"]),
          bar_inner]],
        colWidths=[5.0 * cm, 2.2 * cm, bar_total],
    )
    row.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return row


def _section_cu_characteristics(cu_chars: dict, styles) -> list:
    """
    CU image-derived characteristics — redesigned as scorecards + gauges + geometry table.
    """
    if not cu_chars:
        return []

    # ── helpers ───────────────────────────────────────────────────────────────
    def _sev_redness(v):
        if v >= 0.65: return "high"
        if v >= 0.40: return "moderate"
        if v >= 0.20: return "mild"
        return "low"

    def _sev_coverage(v):
        if v >= 40: return "high"
        if v >= 20: return "moderate"
        if v >= 5:  return "mild"
        return "low"

    def _sev_ery_idx(v):
        if v >= 20: return "high"
        if v >= 8:  return "moderate"
        if v >= 2:  return "mild"
        return "low"

    def _sev_circ(v):
        # higher circularity = more CU-typical (good) → low severity label
        if v >= 0.75: return "low"
        if v >= 0.50: return "moderate"
        if v >  0.0:  return "high"
        return "na"

    def _diam_sev(v):
        if v >= 15: return "high"
        if v >= 6:  return "moderate"
        if v >  0:  return "mild"
        return "na"

    def _color_for_sev(sev):
        return {
            "high":     colors.HexColor("#DC3545"),
            "moderate": C_WARNING_BDR,
            "mild":     C_ACCENT,
            "low":      C_OK_BDR,
            "none":     C_OK_BDR,
            "na":       C_BORDER,
        }.get(sev, C_ACCENT)

    redness_mean = cu_chars.get("redness_mean_score",     0.0)
    redness_max  = cu_chars.get("redness_max_score",      0.0)
    coverage     = cu_chars.get("redness_coverage_pct",   0.0)
    ery_idx      = cu_chars.get("erythema_index",         0.0)
    wheal_count  = cu_chars.get("wheal_count",            0)
    avg_d        = cu_chars.get("wheal_avg_diameter_pct", 0.0)
    max_d        = cu_chars.get("wheal_max_diameter_pct", 0.0)
    circ         = cu_chars.get("wheal_mean_circularity", 0.0)
    ar           = cu_chars.get("wheal_mean_aspect_ratio",0.0)
    distribution = cu_chars.get("distribution_pattern",  "N/A")
    shape_desc   = cu_chars.get("shape_description",     "N/A")

    # ── Section header ────────────────────────────────────────────────────────
    items = [
        _section_title("Image Analysis  ·  CU Morphological Characteristics", styles),
        _divider(),
        Paragraph(
            "Auto-derived from the skin photograph using CIE LAB a*-channel erythema analysis "
            "and contour-based wheal detection (OpenCV morphological pipeline). "
            "All size values are image-diagonal–normalised (no physical scale calibration).",
            styles["small"],
        ),
        Spacer(1, 8),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1 — Erythema / Redness scorecard (4 metric cards)
    # ══════════════════════════════════════════════════════════════════════════
    items.append(Paragraph(
        "<b>● ERYTHEMA  /  REDNESS  PROFILE</b>  "
        "<font size='8' color='#5A6A7A'>(CIE LAB a*-channel · 0–1 normalised)</font>",
        styles["body_bold"],
    ))
    items.append(Spacer(1, 5))

    card_gap   = 0.3 * cm
    card_w     = 3.9 * cm
    n_cards    = 4
    total_card = n_cards * card_w + (n_cards - 1) * card_gap   # 16.5cm < 17.4cm ✓

    card_mean  = _cu_metric_card(
        "Mean Erythema", f"{redness_mean:.3f}",
        "avg a* across image", _sev_redness(redness_mean), styles)
    card_peak  = _cu_metric_card(
        "Peak Erythema", f"{redness_max:.3f}",
        "max local a* value", _sev_redness(redness_max), styles)
    card_cov   = _cu_metric_card(
        "Redness Coverage", f"{coverage:.1f}%",
        "% area above threshold", _sev_coverage(coverage), styles)
    card_eidx  = _cu_metric_card(
        "Erythema Index", f"{ery_idx:.2f}",
        "mean × coverage (au)", _sev_ery_idx(ery_idx), styles)

    gap = Spacer(card_gap, 1)
    card_row = Table(
        [[card_mean, gap, card_peak, gap, card_cov, gap, card_eidx]],
        colWidths=[card_w, card_gap, card_w, card_gap, card_w, card_gap, card_w],
    )
    card_row.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    items.append(card_row)
    items.append(Spacer(1, 8))

    # Erythema gauge bars
    items.append(Paragraph(
        "<font size='8' color='#5A6A7A'><i>Quantitative gauges  — bar shows proportion of scale maximum</i></font>",
        styles["small"],
    ))
    items.append(Spacer(1, 3))

    gauge_bg = colors.HexColor("#F0F6FB")
    gauge_outer = Table(
        [[
            _cu_gauge_row("Mean Erythema Score (0–1)",
                          f"{redness_mean:.3f}",
                          redness_mean,
                          _color_for_sev(_sev_redness(redness_mean)), styles),
        ],[
            _cu_gauge_row("Peak Erythema Score (0–1)",
                          f"{redness_max:.3f}",
                          redness_max,
                          _color_for_sev(_sev_redness(redness_max)), styles),
        ],[
            _cu_gauge_row("Redness Coverage (0–100%)",
                          f"{coverage:.1f}%",
                          coverage / 100.0,
                          _color_for_sev(_sev_coverage(coverage)), styles),
        ],[
            _cu_gauge_row("Composite Erythema Index (0–50 approx.)",
                          f"{ery_idx:.2f}",
                          min(ery_idx / 50.0, 1.0),
                          _color_for_sev(_sev_ery_idx(ery_idx)), styles),
        ]],
        colWidths=[CONTENT_W],
    )
    gauge_outer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), gauge_bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    items.append(gauge_outer)
    items.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2 — Wheal detection scorecards + geometry table
    # ══════════════════════════════════════════════════════════════════════════
    items.append(Paragraph(
        f"<b>● WHEAL DETECTION  &amp;  MORPHOLOGY</b>  "
        f"<font size='8' color='#5A6A7A'>({wheal_count} region"
        f"{'s' if wheal_count != 1 else ''} detected)</font>",
        styles["body_bold"],
    ))
    items.append(Spacer(1, 5))

    # 3 wheal summary cards
    wheal_sev = ("none" if wheal_count == 0 else
                 "mild" if wheal_count <= 2 else
                 "moderate" if wheal_count <= 6 else "high")
    card_wcount = _cu_metric_card(
        "Detected Wheals",   str(wheal_count),
        distribution,   wheal_sev, styles)
    card_avgd   = _cu_metric_card(
        "Avg Wheal Diam.", f"{avg_d:.1f}%",
        "% of image diag.", _diam_sev(avg_d), styles)
    card_circ_v = _cu_metric_card(
        "Wheal Circularity", f"{circ:.3f}",
        "round = 1.0", _sev_circ(circ), styles)

    card_row2 = Table(
        [[card_wcount, gap, card_avgd, gap, card_circ_v, Spacer(card_gap, 1), Spacer(card_w, 1)]],
        colWidths=[card_w, card_gap, card_w, card_gap, card_w, card_gap, card_w],
    )
    card_row2.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    items.append(card_row2)
    items.append(Spacer(1, 8))

    # Detailed geometry table  (3 columns: Characteristic | Value | CU Significance)
    C_W = [4.4*cm, 3.8*cm, CONTENT_W - 8.2*cm]   # sums to CONTENT_W

    def _badge_cell(text: str, sev: str) -> Table:
        """Coloured pill badge for severity/finding."""
        bg  = _color_for_sev(sev)
        sty = ParagraphStyle("_bdg2", fontSize=8, fontName="Helvetica-Bold",
                             textColor=colors.white, alignment=TA_CENTER)
        t = Table([[Paragraph(text, sty)]], colWidths=[3.6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("BOX",           (0,0),(-1,-1), 0.5, bg),
            ("TOPPADDING",    (0,0),(-1,-1), 2),
            ("BOTTOMPADDING", (0,0),(-1,-1), 2),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ]))
        return t

    def _sig(text): return Paragraph(text, styles["small"])

    geo_header = [
        Paragraph("CU Characteristic",  styles["th"]),
        Paragraph("Measured Value",      styles["th"]),
        Paragraph("Clinical Significance in CSU", styles["th"]),
    ]

    def _ar_txt(v):
        if v <= 1.3:  return "Near-circular (1:1)"
        if v <= 2.0:  return f"{v:.2f} – slightly elongated"
        return f"{v:.2f} – linear / elongated"

    geo_data = [
        geo_header,
        [
            Paragraph("Wheal Count", styles["td"]),
            _badge_cell(str(wheal_count), wheal_sev),
            _sig("Wheals are transient raised lesions; >6 simultaneously suggests "
                 "moderate–severe CSU activity. Isolated single wheals may indicate "
                 "pressure/dermatographic urticaria."),
        ],
        [
            Paragraph("Distribution Pattern", styles["td"]),
            _badge_cell(distribution, "na"),
            _sig("Focal lesions suggest triggerable CSU; diffuse/scattered pattern "
                 "correlates with higher UAS7, systemic mast-cell activation."),
        ],
        [
            Paragraph("Avg. Wheal Diameter\n(% of image diagonal)", styles["td"]),
            _badge_cell(f"{avg_d:.1f}%", _diam_sev(avg_d)),
            _sig("Urticarial wheals typically 1–10 cm. Large, confluent wheals may "
                 "indicate angioedema risk.  Size relative to image — no absolute "
                 "calibration applied."),
        ],
        [
            Paragraph("Max. Wheal Diameter\n(% of image diagonal)", styles["td"]),
            _badge_cell(f"{max_d:.1f}%", _diam_sev(max_d)),
            _sig("Largest single lesion.  Disproportionate max vs avg suggests "
                 "mixed morphology or confluence."),
        ],
        [
            Paragraph("Mean Circularity\n(4πA / P²)", styles["td"]),
            _badge_cell(f"{circ:.3f}", _sev_circ(circ)),
            _sig("Classic CSU wheals are round/oval (circularity ≥ 0.75). "
                 "Linear or serpiginous lesions suggest dermographism or "
                 "pressure urticaria variants."),
        ],
        [
            Paragraph("Wheal Shape (derived)", styles["td"]),
            _badge_cell(shape_desc.split("/")[0].strip() if shape_desc != "N/A" else "N/A", "na"),
            _sig(f"Full description: {shape_desc}.  Round-oval = classic CSU; "
                 "irregular = cholinergic, pressure, or mixed-type."),
        ],
        [
            Paragraph("Aspect Ratio\n(long / short axis)", styles["td"]),
            _badge_cell(_ar_txt(ar), "na"),
            _sig("AR close to 1.0 = circular wheal (classic). Elongated AR "
                 "(>2) may suggest factitious urticaria / dermographism."),
        ],
    ]

    geo_tbl = Table(geo_data, colWidths=C_W, repeatRows=1)
    geo_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_TABLE_ALT]),
        ("BACKGROUND",    (0, 1), (0, -1), C_SECTION_BG),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.4, C_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 1), (1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    items.append(geo_tbl)
    items.append(Spacer(1, 8))

    # Wheal circularity gauge
    items.append(Paragraph(
        "<font size='8' color='#5A6A7A'><i>Morphometric gauges</i></font>",
        styles["small"],
    ))
    items.append(Spacer(1, 3))
    morph_gauge = Table(
        [[_cu_gauge_row("Wheal Circularity (0–1, higher = more round/CU-typical)",
                        f"{circ:.3f}", circ,
                        _color_for_sev(_sev_circ(circ)), styles)],
         [_cu_gauge_row("Avg. Wheal Diameter (0–30% diag. scale)",
                        f"{avg_d:.1f}%", min(avg_d / 30.0, 1.0),
                        _color_for_sev(_diam_sev(avg_d)), styles)],
         [_cu_gauge_row("Largest Wheal Diameter (0–30% diag. scale)",
                        f"{max_d:.1f}%", min(max_d / 30.0, 1.0),
                        _color_for_sev(_diam_sev(max_d)), styles)]],
        colWidths=[CONTENT_W],
    )
    morph_gauge.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), gauge_bg),
        ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    items.append(morph_gauge)
    items.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 3 — CU Phenotype summary box
    # ══════════════════════════════════════════════════════════════════════════
    sev_word = {
        "high": "High-grade",
        "moderate": "Moderate",
        "mild": "Mild",
        "low": "Low-grade",
        "none": "No significant",
    }.get(_sev_ery_idx(ery_idx), "Indeterminate")

    shape_short = shape_desc.split("/")[0].strip() if shape_desc != "N/A" else "indeterminate"

    phenotype_lines = [
        f"<b>Erythema:</b>  {sev_word} (Index {ery_idx:.2f}); "
        f"mean score {redness_mean:.3f}, coverage {coverage:.1f}% of image area.",

        f"<b>Wheals:</b>  {wheal_count} region{'s' if wheal_count != 1 else ''} detected — "
        f"{distribution.lower()}.  Avg. diameter {avg_d:.1f}% of image diagonal; "
        f"largest {max_d:.1f}%.",

        f"<b>Morphology:</b>  {shape_short} shape (circularity {circ:.3f}, aspect ratio {ar:.2f}).  "
        f"{'Classic urticarial wheal profile.' if circ >= 0.70 else 'Irregular morphology — consider dermographic or pressure subtype.'}",
    ]

    summary_inner = Table(
        [[Paragraph("<b>CU PHENOTYPE SUMMARY  ·  IMAGE-DERIVED</b>",
                    ParagraphStyle("_ph_hdr", fontSize=8.5, fontName="Helvetica-Bold",
                                   textColor=C_HEADER_TEXT))]]+
        [[Paragraph(line, ParagraphStyle("_ph_body", fontSize=8.5, fontName="Helvetica",
                                         textColor=C_TEXT, leading=13, spaceBefore=2))]
         for line in phenotype_lines],
        colWidths=[CONTENT_W - 0.8*cm],
    )
    summary_inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), C_HEADER_BG),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#EEF4FA")),
        ("BOX",           (0, 0), (-1, -1), 1.0, C_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))

    summary_wrap = Table([[summary_inner]], colWidths=[CONTENT_W])
    summary_wrap.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))
    items.append(KeepTogether([
        Paragraph("<b>● CU PHENOTYPE SUMMARY</b>", styles["body_bold"]),
        Spacer(1, 4),
        summary_wrap,
        Spacer(1, 6),
    ]))

    items.append(Paragraph(
        "<i>Diameter values are image-diagonal–normalised. "
        "No physical scale marker applied; absolute mm estimates require in-frame ruler calibration.</i>",
        styles["small"],
    ))
    items.append(Spacer(1, 6))
    return items


def _section_disclaimer(styles) -> list:
    text = (
        "This report has been generated by the AURA Chronic Spontaneous Urticaria (CSU) Clinical Decision "
        "Support System.  The AI-derived recommendations are intended to assist, not replace, the judgement "
        "of a qualified healthcare professional.  All treatment decisions must be made in the context of the "
        "individual patient's full clinical picture, local formulary, contraindications, and current evidence-"
        "based guidelines.  This output does not constitute a medical prescription or legally binding clinical "
        "advice.  The system has not been validated as a standalone diagnostic or prescribing tool."
    )
    items = [
        Spacer(1, 10),
        _divider(C_ACCENT, thickness=0.8),
        Paragraph("<b>DISCLAIMER &amp; LIMITATIONS</b>", styles["section"]),
        Paragraph(text, styles["disclaimer"]),
        Spacer(1, 6),
    ]
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────────────────────────

def build_pdf_report(
    *,
    patient_meta: Dict[str, Any],
    prediction: Dict[str, Any],
    extracted_labs: Dict[str, Any],
    images: Optional[Dict[str, PILImage.Image]] = None,
    cu_characteristics: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Build a clinical A4 PDF report and return its bytes.

    Parameters
    ----------
    patient_meta   : dict   – case / patient metadata (Case ID, etc.)
    prediction     : dict   – full AnalyzeResponse payload
    extracted_labs : dict   – OCR-extracted lab values
    images         : dict   – PIL images keyed 'skin', 'gradcam', 'redness'

    Returns
    -------
    bytes  – the PDF file content, ready to stream as application/pdf
    """
    buf    = io.BytesIO()
    styles = _build_styles()

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="AURA CSU Clinical Decision Support Report",
        author="AURA System v1.0",
        subject="CSU Pharmacotherapy Recommendation",
    )
    doc.addPageTemplates([_make_page_template(doc, styles)])

    # ── Assemble story ──────────────────────────────────────────────────────
    story: list = []

    story += _section_meta(patient_meta, prediction, styles)
    story.append(Spacer(1, 8))

    story += _section_prediction_summary(prediction, styles)
    story += _section_top3(prediction, styles)
    story += _section_uas7(prediction, styles)
    story += _section_modality_weights(prediction, styles)

    story.append(Spacer(1, 4))
    story += _section_lab_values(prediction, styles)
    story += _section_clinical_features(prediction, styles)
    story += _section_extracted_labs(extracted_labs, styles)

    # Images on a fresh area (keep together so they don't split across pages)
    img_section = _section_images(images or {}, styles)
    if img_section:
        story.append(KeepTogether(img_section))

    # CU image-analysis parameters (redness, wheals, shape)
    if cu_characteristics:
        story += _section_cu_characteristics(cu_characteristics, styles)

    story += _section_guideline(prediction, styles)
    story += _section_disclaimer(styles)

    doc.build(story)
    buf.seek(0)
    return buf.read()
