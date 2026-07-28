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

APP_TITLE = "Explainable Employee Burnout Detection using AI + RAG"

st.set_page_config(
    page_title="Employee Burnout Detection — AI + RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .caveat {
        background:#F4F6F9; border-left:4px solid #E8A33D;
        padding:0.85rem 1rem; border-radius:4px; font-size:0.9rem;
        line-height:1.5; margin:0.5rem 0 1rem 0;
      }
      .degraded {
        background:#FBE3E3; border-left:4px solid #D64545;
        padding:0.85rem 1rem; border-radius:4px; font-size:0.9rem;
        margin:0.5rem 0 1rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Session state
# ==============================================================================
for key, default in (
    ("rag_answer", None),
    ("report_payload", None),
    ("last_question", ""),
    ("force_reload", False),
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
    st.sidebar.title("Controls")

    # --- backend status ---------------------------------------------------
    client = get_client()
    online = client.is_online()
    st.sidebar.markdown(
        f"**Backend:** {'🟢 online' if online else '🔴 offline'}  \n"
        f"<span style='font-size:0.8rem;color:#6B7280'>{client.base_url}</span>",
        unsafe_allow_html=True,
    )
    if not online:
        st.sidebar.caption(
            "Charts below still work — they read the CSV directly. "
            "Start the API for questions and reports:\n\n"
            "`uvicorn backend.app:app --reload --port 8000`"
        )

    st.sidebar.divider()

    # --- Ask Question -----------------------------------------------------
    st.sidebar.subheader("Ask a Question")
    question = st.sidebar.text_area(
        "Question",
        value="Which employees show the strongest signs of burnout, and why?",
        height=90,
        label_visibility="collapsed",
    )
    col_a, col_b = st.sidebar.columns(2)
    top_k = col_a.number_input("Records", 1, 25, 5, help="How many records to retrieve")
    include_sources = col_b.checkbox("Show sources", value=True)

    if st.sidebar.button("Ask", type="primary", use_container_width=True,
                         disabled=not online):
        with st.spinner("Retrieving records and generating an explanation…"):
            result = client.query(
                question=question, top_k=int(top_k),
                include_sources=include_sources,
            )
        st.session_state.rag_answer = result
        st.session_state.last_question = question

    st.sidebar.divider()

    # --- Refresh Database -------------------------------------------------
    st.sidebar.subheader("Refresh Database")
    st.sidebar.caption(
        "Re-reads Person 2's CSV and rebuilds the vector index. Run this after "
        "the ML pipeline has been re-run."
    )
    if st.sidebar.button("Refresh", use_container_width=True, disabled=not online):
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

    st.sidebar.divider()

    # --- Generate Report --------------------------------------------------
    st.sidebar.subheader("Generate Report")
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

    if st.sidebar.button("Generate", use_container_width=True, disabled=not online):
        with st.spinner("Generating the report…"):
            result = client.report(
                scope=scope, title=report_title,
                employee_id=report_employee, focus=focus or None,
            )
        st.session_state.report_payload = result

    st.sidebar.divider()

    # --- Filters ----------------------------------------------------------
    st.sidebar.subheader("Filters")
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
            "<div class='degraded'><b>Degraded answer.</b> The language model "
            "was unreachable, so this was assembled directly from the ML "
            "pipeline's rule-based output with no generated interpretation."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Burnout Summary")
    st.write(analysis.get("burnout_summary", "—"))

    risk = analysis.get("risk_level", "Unknown")
    st.markdown(
        f"**Assessed risk level:** "
        f"<span style='color:{RISK_COLOURS.get(risk, '#8A8A8A')};"
        f"font-weight:700'>{risk}</span>",
        unsafe_allow_html=True,
    )

    # Placed directly after the summary, before any recommendation — the same
    # ordering as the PDF, and for the same reason.
    note = analysis.get("confidence_note")
    if note:
        st.markdown(
            f"<div class='caveat'><b>Confidence and limitations.</b> {note}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Emotional Analysis")
    st.write(analysis.get("emotional_analysis", "—"))

    left, right = st.columns(2)
    for column, (heading, key) in zip(
        (left, right, left, right),
        (
            ("Possible Causes", "possible_causes"),
            ("Suggested Interventions", "suggested_interventions"),
            ("HR Recommendations", "hr_recommendations"),
            ("Mental Wellness Suggestions", "mental_wellness_suggestions"),
        ),
    ):
        with column:
            st.markdown(f"#### {heading}")
            items = analysis.get(key) or []
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.caption("None identified from the available evidence.")

    if notes:
        with st.expander("Retrieval notes"):
            for note_text in notes:
                st.caption(f"• {note_text}")


def render_sources(sources: list):
    """Show the records the answer was grounded in."""
    if not sources:
        return
    st.markdown("#### Evidence used")
    st.caption(
        "These records grounded the answer above. Relevance is cosine "
        "similarity between the question and the record, from 0 to 1."
    )
    frame = pd.DataFrame(sources)
    columns = [
        c for c in
        ["employee_id", "risk_level", "burnout_score", "sentiment",
         "similarity", "clean_text", "explanation"]
        if c in frame.columns
    ]
    st.dataframe(
        frame[columns].rename(
            columns={
                "employee_id": "Employee", "risk_level": "Risk",
                "burnout_score": "Score", "sentiment": "Sentiment",
                "similarity": "Relevance", "clean_text": "Message",
                "explanation": "Why flagged",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    st.title(APP_TITLE)
    st.caption(
        "Continuous analysis of workplace communication using RoBERTa "
        "sentiment, NRC emotion signals, behavioural features and "
        "Retrieval-Augmented Generation."
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
    band, _ = gauge_band(stats["overall_burnout_percent"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records analysed", stats["total_records"])
    col2.metric(
        "Overall burnout", f"{stats['overall_burnout_percent']}%", band,
        delta_color="off",
    )
    col3.metric("At risk (Med + High)", stats["at_risk_count"])
    col4.metric("Most frequent emotion", str(stats["dominant_emotion"]).title())

    st.divider()

    # --- charts -----------------------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            burnout_gauge(
                stats["overall_burnout_percent"], stats["mean_burnout_percent"]
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            sentiment_pie(stats["sentiment_distribution"]),
            use_container_width=True,
        )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            emotion_bar(stats["emotion_totals"]), use_container_width=True
        )
    with right:
        st.plotly_chart(
            risk_distribution_bar(stats["risk_distribution"]),
            use_container_width=True,
        )

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(risk_donut(stats["risk_distribution"]),
                        use_container_width=True)
    with right:
        st.plotly_chart(score_histogram(frame), use_container_width=True)

    st.plotly_chart(employee_risk_scatter(frame), use_container_width=True)

    st.divider()

    # --- top high risk ----------------------------------------------------
    st.subheader("Top High-Risk Employees")
    st.caption(
        "Ranked by the burnout score from the ML pipeline. Ties are broken by "
        "employee ID so the ordering is stable between reruns."
    )
    st.dataframe(
        format_table(top_high_risk(frame, 10)),
        use_container_width=True,
        hide_index=True,
    )

    # --- searchable records ----------------------------------------------
    st.subheader("All Records")
    st.caption(
        f"Showing {len(filtered)} of {len(frame)} records. "
        "Search matches employee ID, message text and explanation literally."
    )
    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        st.dataframe(
            format_table(filtered), use_container_width=True, hide_index=True
        )

    st.divider()

    # --- RAG panel --------------------------------------------------------
    st.subheader("RAG Recommendations")
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
        st.caption(
            f"**Question:** {data.get('question', '')}  \n"
            f"{data.get('retrieved_count', 0)} record(s) retrieved · "
            f"{data.get('llm_provider', '')}/{data.get('llm_model', '')} · "
            f"{data.get('elapsed_seconds', 0):.1f}s"
        )
        render_analysis(
            data.get("answer", {}) or {},
            bool(data.get("degraded")),
            data.get("notes", []) or [],
        )
        render_sources(data.get("sources", []) or [])

    st.divider()

    # --- report + PDF -----------------------------------------------------
    st.subheader("Report")
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
