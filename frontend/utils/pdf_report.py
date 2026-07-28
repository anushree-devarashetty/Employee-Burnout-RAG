"""
frontend/utils/pdf_report.py — ReportLab PDF generation (Person 4)
==============================================================================

Renders the JSON from ``POST /report`` into a downloadable PDF.

WHY THE PDF IS BUILT CLIENT-SIDE
    The API returns structured JSON, not a binary file. Document styling then
    lives here rather than in the backend, and one response feeds both the
    on-screen panel and the download without a second request.

RETURNS BYTES, NOT A FILE PATH
    `st.download_button` takes bytes directly. Building into a `BytesIO`
    avoids writing temp files that would accumulate in the repository and need
    cleaning up — and avoids a permissions problem on locked-down machines.

TWO THINGS THIS DOCUMENT DOES ON PURPOSE

  1. The confidence note is rendered as a bordered callout immediately after
     the summary — not buried on the last page. This report will be forwarded
     to people who never saw the dashboard, and its limitations have to travel
     with it. A reader deciding what to do about a named employee must see the
     caveat before the recommendations, not after.

  2. A degraded report is marked as such in a red banner at the top, and the
     provenance table records which provider and model produced it. A reader
     must never be unable to tell whether an analysis came from a language
     model or from a fallback.
==============================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from frontend.utils.data_utils import RISK_COLOURS, summarise_for_pdf

logger = logging.getLogger(__name__)

_PAGE_WIDTH, _PAGE_HEIGHT = A4
_MARGIN = 18 * mm
_CONTENT_WIDTH = _PAGE_WIDTH - 2 * _MARGIN

_INK = colors.HexColor("#1F2933")
_MUTED = colors.HexColor("#6B7280")
_RULE = colors.HexColor("#D8DEE6")
_PANEL = colors.HexColor("#F4F6F9")


def _styles() -> Dict[str, ParagraphStyle]:
    """Paragraph styles for the document."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=19, leading=24,
            textColor=_INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10, leading=14,
            textColor=_MUTED, alignment=TA_CENTER, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=13, leading=17,
            textColor=_INK, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=10, leading=15,
            textColor=_INK, alignment=TA_LEFT, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontSize=10, leading=15,
            textColor=_INK, spaceAfter=3,
        ),
        "caveat": ParagraphStyle(
            "caveat", parent=base["Normal"], fontSize=9.5, leading=14,
            textColor=_INK,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8.5, leading=12,
            textColor=_MUTED,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontSize=8.5, leading=11,
            textColor=_INK,
        ),
    }


def _escape(text: Any) -> str:
    """Escape XML metacharacters.

    ReportLab parses paragraph text as mini-XML, so an employee message
    containing ``<`` or ``&`` would raise a parse error and fail the whole
    document. Real messages contain both.
    """
    return (
        str(text if text is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _page_furniture(canvas, doc) -> None:
    """Draw the footer rule, page number and confidentiality notice."""
    canvas.saveState()
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.5)
    canvas.line(_MARGIN, 14 * mm, _PAGE_WIDTH - _MARGIN, 14 * mm)

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_MUTED)
    canvas.drawString(
        _MARGIN, 9.5 * mm,
        "Confidential — contains employee wellbeing data. Handle under your "
        "organisation's data protection policy.",
    )
    canvas.drawRightString(
        _PAGE_WIDTH - _MARGIN, 9.5 * mm, f"Page {doc.page}"
    )
    canvas.restoreState()


def _bullets(items: Sequence[str], style: ParagraphStyle) -> Any:
    """Render a list of strings as a bulleted list."""
    if not items:
        return Paragraph(
            "<i>None identified from the available evidence.</i>", style
        )
    return ListFlowable(
        [ListItem(Paragraph(_escape(item), style), leftIndent=10)
         for item in items],
        bulletType="bullet",
        bulletColor=_MUTED,
        bulletFontSize=7,
        leftIndent=12,
        spaceBefore=2,
    )


def _callout(text: str, style: ParagraphStyle, accent: colors.Color) -> Table:
    """A bordered panel used for the confidence note and degraded banner."""
    table = Table([[Paragraph(text, style)]], colWidths=[_CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _PANEL),
                ("BOX", (0, 0), (-1, -1), 0.75, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def build_pdf(
    report: Dict[str, Any],
    statistics: Optional[Dict[str, Any]] = None,
    organisation: str = "",
) -> bytes:
    """Render a ``POST /report`` payload into PDF bytes.

    Args:
        report: The `ReportResponse` payload.
        statistics: Aggregates. Falls back to ``report["statistics"]``.
        organisation: Optional name for the cover line.

    Returns:
        The complete PDF as bytes, ready for `st.download_button`.
    """
    style = _styles()
    analysis: Dict[str, Any] = report.get("analysis", {}) or {}
    stats: Dict[str, Any] = statistics or report.get("statistics", {}) or {}
    sources: List[Dict[str, Any]] = report.get("sources", []) or []
    degraded = bool(report.get("degraded"))

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN, bottomMargin=22 * mm,
        title=report.get("title", "Employee Burnout Analysis Report"),
        author="Explainable Employee Burnout Detection (ML + RAG)",
    )

    story: List[Any] = []

    # --- header -----------------------------------------------------------
    story.append(
        Paragraph(
            _escape(report.get("title", "Employee Burnout Analysis Report")),
            style["title"],
        )
    )
    generated = report.get("generated_at") or datetime.now().isoformat()
    story.append(
        Paragraph(
            f"Explainable Employee Burnout Detection &mdash; "
            f"Machine Learning, Sentiment Analysis and RAG<br/>"
            + (f"{_escape(organisation)}<br/>" if organisation else "")
            + f"Generated {_escape(str(generated)[:19].replace('T', ' '))}",
            style["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.75, color=_RULE))
    story.append(Spacer(1, 8))

    # --- degraded banner --------------------------------------------------
    # Placed before anything else: a reader must know immediately whether a
    # language model was involved in producing what follows.
    if degraded:
        story.append(
            _callout(
                "<b>DEGRADED REPORT.</b> The language model was unavailable. "
                "The analysis below was assembled directly from the machine "
                "learning pipeline's rule-based output, with no generated "
                "interpretation. Re-run this report once the model service is "
                "available.",
                style["caveat"],
                colors.HexColor(RISK_COLOURS["High"]),
            )
        )
        story.append(Spacer(1, 10))

    # --- key figures ------------------------------------------------------
    if stats:
        story.append(Paragraph("Key Figures", style["h2"]))
        rows = [["Metric", "Value"]] + [
            [Paragraph(label, style["cell"]), Paragraph(value, style["cell"])]
            for label, value in summarise_for_pdf(stats)
        ]
        table = Table(rows, colWidths=[_CONTENT_WIDTH * 0.55,
                                       _CONTENT_WIDTH * 0.45])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _INK),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, _PANEL]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(table)

    # --- summary ----------------------------------------------------------
    story.append(Paragraph("Burnout Summary", style["h2"]))
    story.append(
        Paragraph(
            _escape(analysis.get("burnout_summary", "No summary available.")),
            style["body"],
        )
    )

    risk = analysis.get("risk_level", "Unknown")
    story.append(
        Paragraph(
            f"<b>Assessed risk level:</b> "
            f"<font color='{RISK_COLOURS.get(str(risk), '#8A8A8A')}'>"
            f"<b>{_escape(risk)}</b></font>",
            style["body"],
        )
    )

    # --- confidence note, kept high in the document -----------------------
    note = analysis.get("confidence_note")
    if note:
        story.append(Spacer(1, 4))
        story.append(
            _callout(
                f"<b>Confidence and limitations.</b> {_escape(note)}",
                style["caveat"],
                colors.HexColor(RISK_COLOURS["Medium"]),
            )
        )

    # --- narrative sections -----------------------------------------------
    story.append(Paragraph("Emotional Analysis", style["h2"]))
    story.append(
        Paragraph(
            _escape(
                analysis.get("emotional_analysis", "No analysis available.")
            ),
            style["body"],
        )
    )

    for heading, key in (
        ("Possible Causes", "possible_causes"),
        ("Suggested Interventions", "suggested_interventions"),
        ("HR Recommendations", "hr_recommendations"),
        ("Mental Wellness Suggestions", "mental_wellness_suggestions"),
    ):
        # KeepTogether stops a heading being orphaned at a page break with its
        # list stranded overleaf.
        story.append(
            KeepTogether(
                [
                    Paragraph(heading, style["h2"]),
                    _bullets(analysis.get(key, []) or [], style["bullet"]),
                ]
            )
        )

    # --- evidence ---------------------------------------------------------
    if sources:
        story.append(PageBreak())
        story.append(Paragraph("Supporting Evidence", style["h2"]))
        story.append(
            Paragraph(
                f"The {len(sources)} record(s) below are the evidence this "
                f"analysis was based on. Scores and risk levels are produced "
                f"by the machine learning pipeline and are reproduced here "
                f"unchanged.",
                style["small"],
            )
        )
        story.append(Spacer(1, 6))

        header = ["Employee", "Risk", "Score", "Sentiment", "Message",
                  "Why flagged"]
        rows: List[List[Any]] = [
            [Paragraph(f"<b>{h}</b>", style["cell"]) for h in header]
        ]
        for record in sources:
            rows.append(
                [
                    Paragraph(_escape(record.get("employee_id", "")),
                              style["cell"]),
                    Paragraph(_escape(record.get("risk_level", "")),
                              style["cell"]),
                    Paragraph(f"{float(record.get('burnout_score', 0)):g}",
                              style["cell"]),
                    Paragraph(_escape(record.get("sentiment", "")),
                              style["cell"]),
                    Paragraph(_escape(str(record.get("clean_text", ""))[:150]),
                              style["cell"]),
                    Paragraph(_escape(str(record.get("explanation", ""))[:150]),
                              style["cell"]),
                ]
            )

        widths = [w * _CONTENT_WIDTH for w in (0.13, 0.09, 0.07, 0.12, 0.32, 0.27)]
        table = Table(rows, colWidths=widths, repeatRows=1)
        style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), _INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        # Tint the risk cell so a scanning reader finds the flagged rows
        # without reading every line.
        for index, record in enumerate(sources, start=1):
            level = str(record.get("risk_level", "Unknown"))
            if level in ("Medium", "High"):
                style_commands.append(
                    (
                        "BACKGROUND", (1, index), (1, index),
                        colors.HexColor(RISK_COLOURS[level]).clone(alpha=0.18)
                        if hasattr(colors.HexColor(RISK_COLOURS[level]), "clone")
                        else colors.HexColor(RISK_COLOURS[level]),
                    )
                )
        table.setStyle(TableStyle(style_commands))
        story.append(table)

    # --- provenance -------------------------------------------------------
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE))
    story.append(
        Paragraph(
            "<b>How this report was produced.</b> Sentiment from a RoBERTa "
            "classifier; emotions from NRC Emotion Lexicon word counts; "
            "burnout scores from a transparent rule-based formula. Narrative "
            "sections were "
            + (
                "assembled from the pipeline's own output (no language model)."
                if degraded
                else f"generated by {_escape(report.get('llm_provider', 'n/a'))}"
                     f" / {_escape(report.get('llm_model', 'n/a'))}."
            )
            + f" Records reviewed: {report.get('record_count', len(sources))}. "
              f"Scope: {_escape(report.get('scope', 'all'))}.",
            style["small"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "Burnout risk inferred from workplace text is an indicative "
            "signal, not a clinical or diagnostic finding. Use it to start "
            "supportive conversations, never as the sole basis for a decision "
            "affecting an individual.",
            style["small"],
        )
    )

    document.build(
        story, onFirstPage=_page_furniture, onLaterPages=_page_furniture
    )
    buffer.seek(0)
    return buffer.getvalue()


def build_filename(
    title: str = "burnout_report", scope: str = "all"
) -> str:
    """Timestamped, filesystem-safe download filename."""
    safe = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in title.lower()
    ).strip("_") or "burnout_report"
    return f"{safe}_{scope}_{datetime.now():%Y%m%d_%H%M}.pdf"


__all__ = ["build_pdf", "build_filename"]
