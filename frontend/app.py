"""
frontend/app.py — Streamlit dashboard (Person 4)
==============================================================================

    Explainable Employee Burnout Detection using AI + RAG

RUN (from the repository root):
    streamlit run frontend/app.py

LAYOUT
    Sidebar   Backend status, Ask Question, Refresh Database, Generate Report,
              risk/sentiment filters, employee search
    Main      Overall Burnout % gauge, sentiment pie, emotion bar, risk
              distribution, score histogram, score-vs-confidence scatter,
              top high-risk table, searchable record table, RAG panel,
              PDF download

TWO STRUCTURAL DECISIONS

1.  CHARTS DO NOT NEED THE BACKEND
        Data is read through `backend.data_loader` directly, so the dashboard
        still renders every chart when the API process is down. Only the RAG
        panel and PDF report — which genuinely require the server — become
        unavailable, and the sidebar says so. A dead backend degrades one
        panel instead of blanking the page.

2.  RESULTS LIVE IN st.session_state
        Streamlit reruns this entire script on every widget interaction. A RAG
        answer held in a local variable would vanish the moment the user moved
        a filter — after a 60-second wait for a local model. Answers are
        therefore stored in session state and survive until explicitly
        replaced.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts frontend/ on sys.path, not the repo root. This must run before
# any project import, so it precedes them deliberately.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from frontend.utils.api_client import get_client
from frontend.utils.charts import (
    burnout_gauge,
    emotion_bar,
    employee_risk_scatter,
    risk_distribution_bar,
    risk_donut,
    score_histogram,
    sentiment_pie,
)
from frontend.utils.data_utils import (
    RISK_COLOURS,
    filter_records,
    format_table,
    gauge_band,
    load_records,
    load_statistics,
    sorted_risk_levels,
    top_high_risk,
)
from frontend.utils.pdf_report import build_filename, build_pdf
from frontend.utils import theme

APP_TITLE = "Explainable Employee Burnout Detection using AI + RAG"

st.set_page_config(
    page_title="Employee Burnout Detection — AI + RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(theme.css(), unsafe_allow_html=True)


# ==============================================================================
# Session state
# ==============================================================================
for key, default in (
    ("rag_answer", None),
    ("report_payload", None),
    ("last_question", ""),
    ("force_reload", False),
    ("uploads", None),
):
    st.session_state.setdefault(key, default)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_records(nonce: int):
    """Load records, cached briefly so filter changes do not re-read the CSV.

    `nonce` is part of the cache key purely so "Refresh Database" can bust the
    cache by incrementing it — Streamlit has no targeted invalidation.
    """
    frame, error = load_records(force_reload=True)
    if error:
        return None, error
    stats, stats_error = load_statistics(frame)
    return (frame, stats), stats_error


# ==============================================================================
# Sidebar
# ==============================================================================
def render_sidebar(frame: pd.DataFrame | None):
    """Sidebar controls. Returns the filter selections."""
    client = get_client()
    online = client.is_online()
    # Status only. The four-line "here is how to start uvicorn" block that used
    # to sit here was troubleshooting documentation living permanently in the
    # chrome — see theme.status() for the reasoning.
    st.sidebar.markdown(theme.status(online, client.base_url),
                        unsafe_allow_html=True)

    # --- Ask Question -----------------------------------------------------
    st.sidebar.markdown(theme.side_group("Ask a question", "ask"),
                        unsafe_allow_html=True)
    question = st.sidebar.text_area(
        "Question",
        value="Which employees show the strongest signs of burnout, and why?",
        height=90,
        label_visibility="collapsed",
    )
    col_a, col_b = st.sidebar.columns(2)
    top_k = col_a.number_input("Records", 1, 25, 5)
    include_sources = col_b.checkbox("Show sources", value=True)

    if st.sidebar.button("Ask", type="primary", width="stretch",
                         disabled=not online):
        with st.spinner("Retrieving records and generating an explanation…"):
            result = client.query(
                question=question, top_k=int(top_k),
                include_sources=include_sources,
            )
        st.session_state.rag_answer = result
        st.session_state.last_question = question

    # --- Refresh Database -------------------------------------------------
    st.sidebar.markdown(theme.side_group("Data", "data"), unsafe_allow_html=True)
    if st.sidebar.button("Refresh", width="stretch", disabled=not online):
        with st.spinner("Rebuilding the vector store…"):
            result = client.refresh(force=True)
        if result:
            st.sidebar.success(
                f"Indexed {result.get('records_indexed', 0)} records "
                f"in {result.get('elapsed_seconds', 0):.1f}s."
            )
            _cached_records.clear()
            st.session_state.force_reload = True
        else:
            st.sidebar.error(result.error or "Refresh failed.")

    # --- Generate Report --------------------------------------------------
    st.sidebar.markdown(theme.side_group("Report", "report"), unsafe_allow_html=True)
    scope = st.sidebar.selectbox(
        "Scope",
        ["all", "at_risk", "high_only", "employee"],
        format_func=lambda s: {
            "all": "All records",
            "at_risk": "At risk (Medium + High)",
            "high_only": "High risk only",
            "employee": "Single employee",
        }[s],
    )
    report_employee = None
    if scope == "employee" and frame is not None:
        report_employee = st.sidebar.selectbox(
            "Employee", sorted(frame["employee_id"].unique())
        )
    report_title = st.sidebar.text_input(
        "Report title", value="Employee Burnout Analysis Report"
    )
    focus = st.sidebar.text_input(
        "Focus (optional)", placeholder="e.g. workload distribution"
    )

    if st.sidebar.button("Generate", width="stretch", disabled=not online):
        with st.spinner("Generating the report…"):
            result = client.report(
                scope=scope, title=report_title,
                employee_id=report_employee, focus=focus or None,
            )
        st.session_state.report_payload = result

    # --- Upload documents -------------------------------------------------
    st.sidebar.markdown(theme.side_group("Upload communication", "upload"),
                        unsafe_allow_html=True)
    uploaded = st.sidebar.file_uploader(
        "Choose files",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    add_to_index = st.sidebar.checkbox(
        "Make searchable by the RAG panel",
        value=True,
    )

    if st.sidebar.button(
        "Analyse Uploads", width="stretch",
        disabled=not online or not uploaded,
    ):
        results = []
        progress = st.sidebar.progress(0.0, text="Analysing…")
        for position, upload in enumerate(uploaded, start=1):
            progress.progress(
                position / len(uploaded), text=f"Analysing {upload.name}…"
            )
            results.append(
                client.ingest(
                    upload.getvalue(), upload.name, index=add_to_index,
                    id_prefix="DOC",
                )
            )
        progress.empty()
        st.session_state.uploads = results

    if not online and uploaded:
        st.sidebar.caption("Start the backend to analyse uploads.")

    # --- Filters ----------------------------------------------------------
    st.sidebar.markdown(theme.side_group("Filters", "filter"), unsafe_allow_html=True)
    if frame is None or frame.empty:
        return {"risk": [], "sentiment": [], "search": ""}

    risk = st.sidebar.multiselect(
        "Risk level",
        sorted_risk_levels(frame),
        default=sorted_risk_levels(frame),
    )
    sentiment = st.sidebar.multiselect(
        "Sentiment",
        sorted(frame["sentiment"].unique()),
        default=sorted(frame["sentiment"].unique()),
    )
    search = st.sidebar.text_input(
        "Search employee or message", placeholder="e.g. EMP_0006 or 'deadline'"
    )

    return {"risk": risk, "sentiment": sentiment, "search": search}


# ==============================================================================
# RAG panel
# ==============================================================================
def render_analysis(analysis: dict, degraded: bool, notes: list):
    """Render the eight-key analysis returned by the pipeline."""
    if degraded:
        st.markdown(
            theme.callout(
                "The language model was unreachable, so this was assembled "
                "directly from the ML pipeline's rule-based output with no "
                "generated interpretation.",
                tone="alert", label="Degraded answer.",
            ),
            unsafe_allow_html=True,
        )

    risk = analysis.get("risk_level", "Unknown")
    st.markdown(
        theme.analysis_block("Burnout summary",
                             analysis.get("burnout_summary", "—"))
        + f'<div class="analysis-body" style="margin-top:.55rem">'
          f'Assessed risk level &nbsp;{theme.pill(risk)}</div>',
        unsafe_allow_html=True,
    )

    # Placed directly after the summary, before any recommendation — the same
    # ordering as the PDF, and for the same reason.
    note = analysis.get("confidence_note")
    if note:
        st.markdown(
            theme.callout(note, tone="caution",
                          label="Confidence and limitations."),
            unsafe_allow_html=True,
        )

    st.markdown(
        theme.analysis_block("Emotional analysis",
                             analysis.get("emotional_analysis", "—")),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    for column, (heading, key) in zip(
        (left, right, left, right),
        (
            ("Possible causes", "possible_causes"),
            ("Suggested interventions", "suggested_interventions"),
            ("HR recommendations", "hr_recommendations"),
            ("Mental wellness suggestions", "mental_wellness_suggestions"),
        ),
    ):
        with column:
            st.markdown(
                theme.analysis_block(heading, items=analysis.get(key) or []),
                unsafe_allow_html=True,
            )

    if notes:
        with st.expander("Retrieval notes"):
            for note_text in notes:
                st.caption(f"• {note_text}")




def _chart(figure, caption: str = "") -> None:
    """Render one chart inside a bordered panel.

    `st.container(border=True)` rather than CSS on the chart wrapper. Styling
    `[data-testid="stPlotlyChart"]` with a border and padding changed the box
    Streamlit had already measured, so the figure overflowed by exactly that
    amount and grew a scrollbar. The native container draws the border OUTSIDE
    the measured area, which is why it does not.
    """
    with st.container(border=True):
        st.plotly_chart(figure, width="stretch")
        if caption:
            st.caption(caption)


def _table_config(frame: pd.DataFrame) -> dict:
    """Column formatting shared by every record table.

    `st.column_config` is used instead of pre-formatting strings in pandas
    because it keeps the underlying values numeric — so the user can still sort
    by score — while displaying them as progress bars. Formatting into strings
    would make "10" sort before "9".
    """
    config = {
        "Employee": st.column_config.TextColumn("Employee", width="small"),
        "Risk": st.column_config.TextColumn("Risk", width="small"),
        "Message": st.column_config.TextColumn("Message", width="large"),
        # "Why flagged" was cut mid-word ("Negative sentiment detected, Sadness-").
        # It is the column that explains the score, so it gets real room.
        "Why flagged": st.column_config.TextColumn("Why flagged", width="large"),
        "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
        "Top emotion": st.column_config.TextColumn("Top emotion", width="small"),
    }
    if "Score" in frame.columns:
        top = max(1.0, float(pd.to_numeric(frame["Score"], errors="coerce").max()))
        config["Score"] = st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=top, format="%.0f", width="small",
        )
    if "Confidence" in frame.columns:
        # max_value=100, not 1.
        #
        # ProgressColumn runs printf on the RAW value, so a confidence of 0.987
        # with format "%.0f%%" rendered as "1%" on every row — the bar was
        # correct, the label was nonsense. The column is scaled to 0-100 in
        # _numeric_table so the number and the bar agree.
        config["Confidence"] = st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100,
            format="%.0f%%", width="small",
        )
    return config


def _numeric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for display but keep numbers numeric for the bars."""
    columns = {
        "employee_id": "Employee", "risk_level": "Risk", "burnout_score": "Score",
        "sentiment": "Sentiment", "confidence": "Confidence",
        "dominant_emotion": "Top emotion", "clean_text": "Message",
        "explanation": "Why flagged",
    }
    keep = [c for c in columns if c in frame.columns]
    display = frame[keep].rename(columns=columns).copy()

    # Confidence arrives as 0-1 from the model; ProgressColumn formats the raw
    # value, so it is scaled here rather than in the column config.
    if "Confidence" in display.columns:
        display["Confidence"] = pd.to_numeric(
            display["Confidence"], errors="coerce"
        ).fillna(0) * 100

    return display


def render_uploads() -> None:
    """Render the analysis of any uploaded documents.

    Kept visually separate from the main dataset throughout — different
    section, `DOC_` prefixed IDs, its own metrics. Uploaded messages are not
    part of Person 2's dataset and must never be mistaken for it, or a user
    would draw conclusions about the organisation from a single conversation
    they happened to paste in.
    """
    uploads = st.session_state.get("uploads") or []
    if not uploads:
        return

    st.subheader("Uploaded Communication")

    for result in uploads:
        if not result:
            st.error(result.error or "Upload failed.")
            if result.hint:
                st.code(result.hint)
            continue

        data = result.data or {}
        name = data.get("filename", "document")
        count = data.get("message_count", 0)

        with st.expander(f"**{name}** — {count} message(s)", expanded=True):
            if count == 0:
                st.warning("No messages could be extracted from this file.")
                for warning in data.get("warnings", []):
                    st.caption(f"• {warning}")
                continue

            stats = data.get("statistics") or {}
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Messages", count)
            col2.metric("At risk", stats.get("at_risk_count", 0))
            col3.metric(
                "Burnout share", f"{stats.get('overall_burnout_percent', 0)}%"
            )
            col4.metric(
                "Read via",
                "OCR" if data.get("ocr_used") else data.get("extraction_method", "—"),
            )

            # OCR quality is the single most important caveat on an uploaded
            # screenshot, so it is a visible warning rather than a footnote.
            if data.get("ocr_used"):
                st.markdown(
                    theme.callout(
                        "Character recognition errors (for example <code>I</code> "
                        "read as <code>|</code>) reduce sentiment and emotion "
                        "accuracy. Treat these scores as approximate.",
                        tone="caution", label="Read with OCR.",
                    ),
                    unsafe_allow_html=True,
                )

            if data.get("split_strategy") in ("paragraph", "whole-document"):
                st.info(
                    f"No email or chat structure was detected, so the document "
                    f"was split by **{data['split_strategy']}**. Burnout signals "
                    f"are diluted in long blocks — results improve when sender "
                    f"names and timestamps are present."
                )

            messages = data.get("messages", [])
            if messages:
                frame = pd.DataFrame(messages)
                # Sender and timestamp are deliberately not shown. They are
                # document metadata, not analysis: a transcript with no
                # timestamps renders an entirely blank column, which reads as
                # a defect rather than as absent data.
                columns = [
                    c for c in
                    ["employee_id", "risk_level",
                     "burnout_score", "sentiment", "confidence",
                     "dominant_emotion", "clean_text", "explanation"]
                    if c in frame.columns
                ]
                display = frame[columns].rename(columns={
                    "employee_id": "Employee", "risk_level": "Risk",
                    "burnout_score": "Score", "sentiment": "Sentiment",
                    "confidence": "Confidence",
                    "dominant_emotion": "Top emotion",
                    "clean_text": "Message", "explanation": "Why flagged",
                })
                st.dataframe(display, width="stretch", hide_index=True,
                             column_config=_table_config(display))

            if stats:
                left, right = st.columns(2)
                with left:
                    _chart(risk_distribution_bar(
                        stats.get("risk_distribution", {})))
                with right:
                    _chart(sentiment_pie(
                        stats.get("sentiment_distribution", {})))

            if data.get("indexed"):
                st.success(
                    f"{data.get('indexed_count', 0)} message(s) added to the "
                    f"search index — you can now ask the RAG panel about this "
                    f"conversation. Refresh Database clears them."
                )

            for warning in data.get("warnings", []):
                st.caption(f"• {warning}")

    st.divider()


def render_sources(sources: list):
    """Show the records the answer was grounded in."""
    if not sources:
        return
    st.markdown("#### Evidence used")
    frame = pd.DataFrame(sources)
    # Score, sentiment and retrieval similarity are omitted by request. The
    # explainability claim still holds: the record itself and the reason it
    # was flagged are both shown, which is what a reader needs in order to
    # check the answer rather than trust it.
    columns = [
        c for c in
        ["employee_id", "risk_level", "clean_text", "explanation"]
        if c in frame.columns
    ]
    display = frame[columns].rename(columns={
        "employee_id": "Employee", "risk_level": "Risk",
        "clean_text": "Message", "explanation": "Why flagged",
    })
    st.dataframe(display, width="stretch", hide_index=True,
                 column_config=_table_config(display))


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    st.markdown(
        theme.masthead(
            APP_TITLE,
            "Continuous analysis of workplace communication — RoBERTa sentiment, "
            "NRC emotion signals, behavioural features and Retrieval-Augmented "
            "Generation.",
        ),
        unsafe_allow_html=True,
    )

    nonce = int(st.session_state.force_reload)
    payload, error = _cached_records(nonce)

    if payload is None:
        st.error(error or "Could not load burnout data.")
        st.info(
            "The dashboard reads `outputs/burnout_scores.csv`, produced by the "
            "ML pipeline. Generate it from the repository root:\n\n"
            "```\npython models/burnout/burnout_score.py\n```"
        )
        render_sidebar(None)
        return

    frame, stats = payload
    filters = render_sidebar(frame)
    filtered = filter_records(
        frame,
        risk_levels=filters["risk"],
        sentiments=filters["sentiment"],
        search=filters["search"],
    )

    # --- headline metrics -------------------------------------------------
    band, band_colour = gauge_band(stats["overall_burnout_percent"])
    risk = stats["risk_distribution"]
    st.markdown(
        theme.kpi_row([
            ("Records analysed", f"{stats['total_records']:,}",
             f"from {filters.get('source_label', 'the dataset')}", None),
            ("Overall burnout", f"{stats['overall_burnout_percent']}%",
             f"{band} · Medium + High share", band_colour),
            ("At risk", f"{stats['at_risk_count']}",
             f"{risk.get('High', 0)} high · {risk.get('Medium', 0)} medium",
             theme.RISK["Medium"] if stats["at_risk_count"] else theme.RISK["Low"]),
            ("Dominant emotion", str(stats["dominant_emotion"]).title(),
             "most frequent NRC signal", None),
        ]),
        unsafe_allow_html=True,
    )

    # --- charts -----------------------------------------------------------
    st.markdown(
        theme.section("Overview", "Burnout and sentiment profile",
                      "Aggregate view across every analysed record.", "chart"),
        unsafe_allow_html=True,
    )
    # Both columns carry a caption of the same length so the two panels end at
    # the same y. A caption on one side only left the rows visibly out of step.
    left, right = st.columns([1, 1])
    with left:
        _chart(
            burnout_gauge(stats["overall_burnout_percent"]),
            f"Share of records at Medium or High risk. "
            f"Mean normalised score: {stats['mean_burnout_percent']}%.",
        )
    with right:
        _chart(
            sentiment_pie(stats["sentiment_distribution"]),
            "RoBERTa sentiment labels across every analysed record.",
        )

    left, right = st.columns([1, 1])
    with left:
        _chart(emotion_bar(stats["emotion_totals"]),
               "NRC lexicon word matches. Counts, not intensities.")
    with right:
        _chart(risk_distribution_bar(stats["risk_distribution"]),
               "Thresholds from the ML pipeline: Low \u2264 2, Medium \u2264 5.")

    left, right = st.columns([1, 1])
    with left:
        _chart(risk_donut(stats["risk_distribution"]),
               "The same risk counts as proportions.")
    with right:
        _chart(score_histogram(frame),
               "Shape matters: risk concentrated in a few people is a "
               "different problem from risk spread thinly.")

    _chart(employee_risk_scatter(frame),
           "A high score on a low-confidence prediction is weaker evidence. "
           "Points toward the lower right deserve a human check first.")

    # --- top high risk ----------------------------------------------------
    st.markdown(
        theme.section("Priority", "Highest-risk records",
                      "Ranked by the burnout score from the ML pipeline. Ties "
                      "break by employee ID, so the ordering is stable between "
                      "reruns.", "shield"),
        unsafe_allow_html=True,
    )
    top = _numeric_table(top_high_risk(frame, 10))
    st.dataframe(top, width="stretch", hide_index=True,
                 column_config=_table_config(top))

    # --- searchable records ----------------------------------------------
    st.markdown(
        theme.section("Explore", "All records",
                      f"Showing {len(filtered)} of {len(frame)}. Search matches "
                      f"employee ID, message text and explanation literally — "
                      f"not semantically.", "search"),
        unsafe_allow_html=True,
    )
    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        table = _numeric_table(filtered)
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config=_table_config(table))

    # --- uploaded documents ----------------------------------------------
    render_uploads()

    # --- RAG panel --------------------------------------------------------
    st.markdown(
        theme.section("Explainable AI", "Retrieval-augmented analysis",
                      "Grounded in the specific records shown as evidence "
                      "below each answer.", "sparkle"),
        unsafe_allow_html=True,
    )
    answer = st.session_state.rag_answer

    if answer is None:
        st.info(
            "Ask a question in the sidebar to generate an explanation grounded "
            "in the records above."
        )
    elif not answer:
        st.error(answer.error or "The query failed.")
        if answer.hint:
            st.code(answer.hint)
    else:
        data = answer.data or {}
        # The provider/model/latency/retrieval-count line was removed: it is
        # run metadata for a developer, not information a reader of the
        # analysis needs. It remains available on GET /health.
        render_analysis(
            data.get("answer", {}) or {},
            bool(data.get("degraded")),
            data.get("notes", []) or [],
        )
        render_sources(data.get("sources", []) or [])

    # --- report + PDF -----------------------------------------------------
    st.markdown(
        theme.section("Deliverable", "Structured report",
                      "Generated from the sidebar. Exports to PDF with the "
                      "confidence note ahead of the recommendations.", "download"),
        unsafe_allow_html=True,
    )
    report = st.session_state.report_payload

    if report is None:
        st.info("Generate a report from the sidebar to enable the PDF download.")
    elif not report:
        st.error(report.error or "Report generation failed.")
        if report.hint:
            st.code(report.hint)
    else:
        data = report.data or {}
        st.markdown(f"**{data.get('title', 'Report')}** — "
                    f"{data.get('record_count', 0)} record(s)")
        render_analysis(
            data.get("analysis", {}) or {},
            bool(data.get("degraded")),
            data.get("notes", []) or [],
        )
        try:
            pdf_bytes = build_pdf(data, statistics=stats)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=build_filename(
                    data.get("title", "burnout_report"),
                    str(data.get("scope", "all")),
                ),
                mime="application/pdf",
                type="primary",
            )
        except Exception as exc:
            # A PDF failure must not take the whole dashboard down with it.
            st.error(f"Could not build the PDF: {exc}")

    st.divider()
    st.caption(
        "Burnout risk inferred from workplace text is an indicative signal, "
        "not a clinical or diagnostic finding. Use it to start supportive "
        "conversations, never as the sole basis for a decision affecting an "
        "individual."
    )


if __name__ == "__main__":
    main()
