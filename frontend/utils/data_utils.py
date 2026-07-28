"""
frontend/utils/data_utils.py — Dashboard data access (Person 4)
==============================================================================

WHY THIS DELEGATES TO THE BACKEND INSTEAD OF READING THE CSV ITSELF

    Every function here calls into `backend.data_loader`. That is deliberate.
    If the dashboard did its own `pd.read_csv` and computed its own
    percentages, the gauge on screen and the numbers the LLM reasons over
    would be two independent implementations — and they would diverge the
    first time anyone adjusted a threshold. The dashboard would then display
    figures the backend disagreed with, which is far worse than a slightly
    tighter coupling between two halves of the same repository.

    So `backend.data_loader` is the single source of truth, and this module is
    a thin presentation layer on top of it: Streamlit caching, display
    formatting, and the filtering the sidebar controls drive.

RESILIENCE
    `load_records()` prefers the local read (fast, no HTTP) and falls back to
    the API only if the direct import fails. The dashboard therefore still
    renders its charts when the backend process is down — only the RAG panel,
    which genuinely needs the server, is unavailable.
==============================================================================
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import pandas as pd

logger = logging.getLogger(__name__)

#: Colours reused by charts, tables and the PDF so one record is the same
#: colour everywhere. Green/amber/red is the conventional reading for risk.
RISK_COLOURS: Dict[str, str] = {
    "Low": "#2E9E5B",
    "Medium": "#E8A33D",
    "High": "#D64545",
    "Unknown": "#8A8A8A",
}

SENTIMENT_COLOURS: Dict[str, str] = {
    "Positive": "#2E9E5B",
    "Neutral": "#7B8794",
    "Negative": "#D64545",
    "Unknown": "#8A8A8A",
}

#: Display order: worst first, because that is what an HR user came to see.
RISK_ORDER: Tuple[str, ...] = ("High", "Medium", "Low", "Unknown")

#: Columns shown in the records table, in order, with readable headings.
TABLE_COLUMNS: Dict[str, str] = {
    "employee_id": "Employee",
    "risk_level": "Risk",
    "burnout_score": "Score",
    "burnout_percent": "Score %",
    "sentiment": "Sentiment",
    "confidence": "Confidence",
    "dominant_emotion": "Top emotion",
    "clean_text": "Message",
    "explanation": "Why flagged",
}


# ==============================================================================
# Loading
# ==============================================================================
def load_records(force_reload: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load the enriched burnout records.

    Returns:
        ``(dataframe, error)``. Exactly one is None.
    """
    try:
        from backend.data_loader import load_burnout_dataframe

        return load_burnout_dataframe(force_reload=force_reload), None

    except FileNotFoundError as exc:
        return None, str(exc)
    except ImportError as exc:
        logger.warning("Direct load unavailable (%s); trying the API.", exc)
        return _load_via_api()
    except Exception as exc:
        return None, f"Could not load burnout data: {exc}"


def _load_via_api() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fallback path: reconstruct a frame from ``GET /statistics``.

    Only the top-risk records are available this way, so this is a degraded
    view. Used when the dashboard runs without the repository on its path.
    """
    from frontend.utils.api_client import get_client

    result = get_client().statistics()
    if not result:
        return None, result.error or "The backend is unavailable."

    records = result.get("top_high_risk", [])
    if not records:
        return None, "The backend returned no records."
    return pd.DataFrame(records), None


def load_statistics(
    frame: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Compute aggregate metrics using the backend's own function."""
    try:
        from backend.data_loader import compute_statistics

        if frame is None:
            frame, error = load_records()
            if error:
                return None, error
        return compute_statistics(frame), None
    except Exception as exc:
        return None, f"Could not compute statistics: {exc}"


# ==============================================================================
# Filtering — driven by the sidebar controls
# ==============================================================================
def filter_records(
    frame: pd.DataFrame,
    risk_levels: Optional[Sequence[str]] = None,
    sentiments: Optional[Sequence[str]] = None,
    search: str = "",
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """Apply the dashboard's filters.

    Search is a literal substring match across employee ID, message text and
    explanation. Deliberately not semantic: a user typing ``EMP_0007`` wants
    that record, not the five most conceptually similar ones. Semantic search
    is what the RAG panel is for.
    """
    result = frame

    if risk_levels:
        result = result[result["risk_level"].isin(list(risk_levels))]

    if sentiments:
        result = result[result["sentiment"].isin(list(sentiments))]

    if min_score is not None:
        result = result[result["burnout_score"] >= float(min_score)]

    term = (search or "").strip().lower()
    if term:
        # regex=False so a user typing "C++" or "(a|b)" gets a literal match
        # instead of a regex error that would blank the table.
        mask = result["employee_id"].str.lower().str.contains(term, regex=False)
        for column in ("clean_text", "explanation"):
            if column in result.columns:
                mask = mask | result[column].astype(str).str.lower().str.contains(
                    term, regex=False
                )
        result = result[mask]

    return result


def top_high_risk(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Highest-risk records first.

    Sorted by raw score with employee_id as a tiebreak, so the ordering is
    stable across reruns — a table that reshuffles on every interaction is
    disorienting when several records share a score, which is exactly the case
    in the current dataset (five records tie at 2).
    """
    return frame.sort_values(
        by=["burnout_score", "employee_id"], ascending=[False, True]
    ).head(limit)


def sorted_risk_levels(frame: pd.DataFrame) -> List[str]:
    """Risk levels present, ordered worst-first for the filter widget."""
    present = set(frame["risk_level"].astype(str).unique())
    ordered = [level for level in RISK_ORDER if level in present]
    return ordered + sorted(present - set(ordered))


# ==============================================================================
# Presentation
# ==============================================================================
def format_table(frame: pd.DataFrame, max_message_chars: int = 90) -> pd.DataFrame:
    """Prepare a frame for display: readable headings, truncated text."""
    columns = [c for c in TABLE_COLUMNS if c in frame.columns]
    display = frame[columns].copy()

    for column in ("clean_text", "explanation"):
        if column in display.columns:
            display[column] = display[column].astype(str).apply(
                lambda text: (
                    text[:max_message_chars] + "…"
                    if len(text) > max_message_chars
                    else text
                )
            )

    if "confidence" in display.columns:
        display["confidence"] = display["confidence"].apply(
            lambda v: f"{float(v):.0%}" if pd.notna(v) and float(v) > 0 else "—"
        )
    if "burnout_percent" in display.columns:
        display["burnout_percent"] = display["burnout_percent"].apply(
            lambda v: f"{float(v):.0f}%"
        )
    if "burnout_score" in display.columns:
        display["burnout_score"] = display["burnout_score"].apply(
            lambda v: f"{float(v):g}"
        )

    return display.rename(columns=TABLE_COLUMNS)


def risk_colour(level: str) -> str:
    """Hex colour for a risk level."""
    return RISK_COLOURS.get(str(level).title(), RISK_COLOURS["Unknown"])


def sentiment_colour(label: str) -> str:
    """Hex colour for a sentiment label."""
    return SENTIMENT_COLOURS.get(str(label).title(), SENTIMENT_COLOURS["Unknown"])


def gauge_band(percent: float) -> Tuple[str, str]:
    """Map the headline percentage to a label and colour.

    Thresholds mirror the intent of Person 2's score bands: a workforce with
    more than a third of its communication flagged at Medium or High risk is
    a materially different situation from one with a tenth.
    """
    if percent >= 60:
        return "Critical", RISK_COLOURS["High"]
    if percent >= 33:
        return "Elevated", RISK_COLOURS["Medium"]
    if percent >= 15:
        return "Watch", "#D9C441"
    return "Healthy", RISK_COLOURS["Low"]


def summarise_for_pdf(stats: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Flatten statistics into label/value pairs for the PDF summary table."""
    risk = stats.get("risk_distribution", {}) or {}
    sentiment = stats.get("sentiment_distribution", {}) or {}
    return [
        ("Records analysed", str(stats.get("total_records", 0))),
        (
            "Overall burnout (Medium + High share)",
            f"{stats.get('overall_burnout_percent', 0)}%",
        ),
        ("Mean normalised score", f"{stats.get('mean_burnout_percent', 0)}%"),
        ("Mean raw score", str(stats.get("mean_burnout_score", 0))),
        ("At-risk records", str(stats.get("at_risk_count", 0))),
        (
            "Risk distribution",
            ", ".join(f"{k}: {v}" for k, v in risk.items()) or "—",
        ),
        (
            "Sentiment distribution",
            ", ".join(f"{k}: {v}" for k, v in sentiment.items()) or "—",
        ),
        ("Most frequent emotion", str(stats.get("dominant_emotion", "—"))),
    ]


__all__ = [
    "RISK_COLOURS",
    "SENTIMENT_COLOURS",
    "RISK_ORDER",
    "TABLE_COLUMNS",
    "load_records",
    "load_statistics",
    "filter_records",
    "top_high_risk",
    "sorted_risk_levels",
    "format_table",
    "risk_colour",
    "sentiment_colour",
    "gauge_band",
    "summarise_for_pdf",
]
