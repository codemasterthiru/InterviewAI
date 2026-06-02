"""
pdf_generator.py
Generates a formatted PDF report from the output folder's report.json and
screenshot JPG files produced by the Screen Analyzer.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
_DARK    = colors.HexColor("#1E293B")
_ACCENT  = colors.HexColor("#3B82F6")
_DANGER  = colors.HexColor("#EF4444")
_WARN    = colors.HexColor("#F59E0B")
_OK      = colors.HexColor("#22C55E")
_LIGHT   = colors.HexColor("#F1F5F9")
_BORDER  = colors.HexColor("#CBD5E1")
_WHITE   = colors.white

# Severity colour per violation type
_SEVERITY: Dict[str, colors.Color] = {
    "screen_dimmed_overlay": _WARN,
    "extra_screen_detected": _DANGER,
    "popup_detected":        _WARN,
    "screen_switch":         _DANGER,
    "face_not_visible":      _DANGER,
    "multiple_faces":        _DANGER,
    "gaze_away":             _WARN,
    "overlay_detected":      _WARN,
}

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm


# ── Page template with header/footer ─────────────────────────────────────────

def _make_doc(pdf_path: Path) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.4 * cm,
        bottomMargin=2.0 * cm,
    )

    def _header_footer(canvas, doc):
        canvas.saveState()
        # Top rule
        canvas.setStrokeColor(_ACCENT)
        canvas.setLineWidth(2)
        canvas.line(MARGIN, PAGE_H - 1.4 * cm, PAGE_W - MARGIN, PAGE_H - 1.4 * cm)
        # Header text
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(_DARK)
        canvas.drawString(MARGIN, PAGE_H - 1.1 * cm, "Interview Proctoring — Violation Report")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_BORDER)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.1 * cm,
                               datetime.now().strftime("%d %b %Y"))
        # Bottom rule + page number
        canvas.setStrokeColor(_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_BORDER)
        canvas.drawCentredString(PAGE_W / 2, 0.9 * cm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(MARGIN, 2.0 * cm, PAGE_W - 2 * MARGIN, PAGE_H - 4.4 * cm, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_header_footer)])
    return doc


# ── Style helpers ─────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    title   = s("RTitle",  fontName="Helvetica-Bold", fontSize=22,
                textColor=_DARK,   spaceAfter=6,  alignment=TA_LEFT)
    sub     = s("RSub",    fontName="Helvetica",      fontSize=11,
                textColor=colors.HexColor("#64748B"), spaceAfter=14)
    h2      = s("RH2",     fontName="Helvetica-Bold", fontSize=13,
                textColor=_ACCENT, spaceBefore=14, spaceAfter=6)
    h3      = s("RH3",     fontName="Helvetica-Bold", fontSize=10,
                textColor=_DARK,   spaceBefore=8,  spaceAfter=4)
    body    = s("RBody",   fontName="Helvetica",      fontSize=9,
                textColor=_DARK,   spaceAfter=3,  leading=14)
    small   = s("RSmall",  fontName="Helvetica",      fontSize=8,
                textColor=colors.HexColor("#64748B"), spaceAfter=2)
    badge   = s("RBadge",  fontName="Helvetica-Bold", fontSize=8,
                textColor=_WHITE,  alignment=TA_CENTER)
    return dict(title=title, sub=sub, h2=h2, h3=h3, body=body, small=small, badge=badge)


# ── Reusable flowables ────────────────────────────────────────────────────────

def _rule(color=_BORDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=4)


def _kv_table(rows: List, col_widths=None) -> Table:
    """Two-column key/value table."""
    avail = PAGE_W - 2 * MARGIN
    cw = col_widths or [avail * 0.38, avail * 0.62]
    t = Table(rows, colWidths=cw)
    t.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), _DARK),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#475569")),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",(0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_WHITE, _LIGHT]),
        ("GRID",      (0, 0), (-1, -1), 0.3, _BORDER),
        ("ROUNDEDCORNERS", [3]),
    ]))
    return t


def _summary_table(violation_types: Dict[str, int]) -> Table:
    """Coloured summary table of violation counts."""
    avail = PAGE_W - 2 * MARGIN
    header = [
        Paragraph("<b>Violation Type</b>", ParagraphStyle("th", fontName="Helvetica-Bold",
                  fontSize=9, textColor=_WHITE)),
        Paragraph("<b>Count</b>", ParagraphStyle("thc", fontName="Helvetica-Bold",
                  fontSize=9, textColor=_WHITE, alignment=TA_CENTER)),
    ]
    data = [header]
    for vtype, cnt in violation_types.items():
        col = _SEVERITY.get(vtype, _ACCENT)
        label = vtype.replace("_", " ").title()
        data.append([
            Paragraph(label, ParagraphStyle("td", fontName="Helvetica", fontSize=9, textColor=_DARK)),
            Paragraph(f"<b>{cnt}</b>", ParagraphStyle("tdc", fontName="Helvetica-Bold",
                      fontSize=9, textColor=col, alignment=TA_CENTER)),
        ])
    t = Table(data, colWidths=[avail * 0.75, avail * 0.25])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  _DARK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, _BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


# ── Screenshot embed ──────────────────────────────────────────────────────────

_MAX_IMG_W = PAGE_W - 2 * MARGIN - 0.4 * cm
_MAX_IMG_H = 7 * cm


def _embed_image(screenshot_path: str, output_dir: Path) -> Image | None:
    """Return a scaled ReportLab Image or None if the file is not found."""
    p = Path(screenshot_path)
    if not p.exists():
        # Try relative to output_dir
        p = output_dir / p.name
    if not p.exists():
        return None
    try:
        img = Image(str(p))
        ratio = img.imageWidth / img.imageHeight
        w = min(_MAX_IMG_W, img.imageWidth)
        h = w / ratio
        if h > _MAX_IMG_H:
            h = _MAX_IMG_H
            w = h * ratio
        img.drawWidth  = w
        img.drawHeight = h
        return img
    except Exception:
        return None


# ── Main public function ───────────────────────────────────────────────────────

def generate_pdf(output_dir: Path, json_filename: str = "report.json") -> Path:
    """
    Build a PDF from *output_dir/json_filename* and the JPG screenshots
    also stored in *output_dir*.  Returns the path of the created PDF.
    """
    json_path = output_dir / json_filename
    if not json_path.exists():
        raise FileNotFoundError(f"JSON report not found: {json_path}")

    with open(json_path, encoding="utf-8") as fh:
        data: Dict = json.load(fh)

    pdf_stem = Path(json_filename).stem
    pdf_path = output_dir / f"{pdf_stem}.pdf"

    doc   = _make_doc(pdf_path)
    st    = _styles()
    story = []

    # ── Cover / Title ────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Proctoring Violation Report", st["title"]))
    story.append(Paragraph(
        f"Generated: {data.get('generated_at', 'N/A')} &nbsp;|&nbsp; "
        f"Status: <b>{data.get('status', 'N/A').upper()}</b>",
        st["sub"],
    ))
    story.append(_rule(_ACCENT, 1.5))
    story.append(Spacer(1, 0.3 * cm))

    # ── Video Metadata ───────────────────────────────────────────────────────
    story.append(Paragraph("Video Information", st["h2"]))
    vmeta = data.get("video_metadata", {})
    meta_rows = [
        ["Video File",    data.get("video_file", "N/A")],
        ["Resolution",    f"{vmeta.get('width', '?')} × {vmeta.get('height', '?')} px"],
        ["Frame Rate",    f"{vmeta.get('fps', '?')} fps"],
        ["Duration",      f"{vmeta.get('duration_seconds', '?')} s"],
        ["Total Frames",  str(vmeta.get("total_frames", "?"))],
    ]
    story.append(_kv_table(meta_rows))
    story.append(Spacer(1, 0.4 * cm))

    # ── Summary ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Summary", st["h2"]))
    summary = data.get("summary", {})
    sum_rows = [
        ["Total Incidents",              str(summary.get("total_incidents", 0))],
        ["Total Violation Duration",     f"{summary.get('total_violation_duration_secs', 0)} s"],
    ]
    story.append(_kv_table(sum_rows))
    story.append(Spacer(1, 0.3 * cm))

    vtypes = summary.get("violation_types", {})
    if vtypes:
        story.append(Paragraph("Violation Breakdown", st["h3"]))
        story.append(_summary_table(vtypes))

    story.append(Spacer(1, 0.5 * cm))
    story.append(PageBreak())

    # ── Violation Details ────────────────────────────────────────────────────
    violations: List[Dict] = data.get("violations", [])
    if violations:
        story.append(Paragraph("Violation Details", st["h2"]))
        story.append(_rule(_ACCENT, 1))
        story.append(Spacer(1, 0.2 * cm))

        for idx, v in enumerate(violations, start=1):
            vtype   = v.get("type", "unknown")
            sev_col = _SEVERITY.get(vtype, _ACCENT)
            label   = vtype.replace("_", " ").title()

            # Incident heading with coloured type label
            hex_col = "%02x%02x%02x" % (
                round(sev_col.red * 255),
                round(sev_col.green * 255),
                round(sev_col.blue * 255),
            )
            story.append(Paragraph(
                f'Incident #{idx} \u2014 '
                f'<font color="#{hex_col}"><b>{label}</b></font>',
                st["h3"],
            ))

            detail_rows = [
                ["Start Time",  v.get("start_time", "N/A")],
                ["End Time",    v.get("end_time",   "N/A")],
                ["Duration",    v.get("duration",   "N/A")],
                ["Confidence",  f"{float(v.get('confidence', 0)):.0%}"],
                ["Event Count", str(v.get("event_count", "N/A"))],
                ["Details",     v.get("details", "—")],
            ]
            if v.get("screenshot"):
                detail_rows.append(["Screenshot", Path(v["screenshot"]).name])

            story.append(_kv_table(detail_rows))

            # Embed screenshot
            sc = v.get("screenshot")
            if sc:
                img = _embed_image(sc, output_dir)
                if img:
                    story.append(Spacer(1, 0.2 * cm))
                    story.append(img)

            story.append(Spacer(1, 0.4 * cm))
            story.append(_rule())

    else:
        story.append(Paragraph("No violations recorded.", st["body"]))

    # ── Build PDF ────────────────────────────────────────────────────────────
    doc.build(story)
    return pdf_path
