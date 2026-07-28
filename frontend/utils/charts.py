"""
frontend/utils/charts.py — Plotly figures (Person 4)
==============================================================================

Every chart specified for the dashboard:

    burnout_gauge()             Overall burnout %          (gauge)
    sentiment_pie()             Sentiment distribution     (pie)
    emotion_bar()               Emotion distribution       (bar)
    risk_distribution_bar()     Risk level distribution    (bar)
    risk_donut()                Risk mix                   (donut)
    score_histogram()           Score spread               (histogram)
    employee_risk_scatter()     Score vs confidence        (scatter)

CONVENTIONS APPLIED THROUGHOUT

  - Colour is consistent across every chart, the tables and the PDF, driven by
    `data_utils.RISK_COLOURS`. One record is the same colour everywhere, so a
    reader is never asked to re-learn the palette per panel.

  - Colour is never the only channel. Every chart labels its values directly,
    because roughly 1 in 12 men has a colour vision deficiency and red/green
    is the worst possible pairing for them — which is also the conventional
    palette for risk. Text labels make the charts readable regardless.

  - Every function handles the empty case explicitly and returns a figure with
    a readable message. Plotly renders an empty frame as a blank rectangle,
    which looks like a bug rather than an absence of data.

  - `template="plotly_white"` throughout so charts stay legible when exported
    into the PDF report on a white page.
==============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go

from frontend.utils.data_utils import (
    RISK_COLOURS,
    RISK_ORDER,
    SENTIMENT_COLOURS,
    gauge_band,
    risk_colour,
    sentiment_colour,
)

#: Applied to every figure so panels align on the dashboard grid.
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=40),
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
    hoverlabel=dict(font_size=13),
)


def _empty_figure(message: str, height: int = 300) -> go.Figure:
    """A figure that explains why it is empty, instead of a blank rectangle."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14, color="#6B7280"),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    figure.update_layout(
        **_LAYOUT,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return figure


# ==============================================================================
# Gauge — Overall Burnout %
# ==============================================================================
def burnout_gauge(
    percent: float,
    mean_percent: Optional[float] = None,
    height: int = 320,
) -> go.Figure:
    """Headline gauge: share of records at Medium or High risk.

    The value is a *risk share*, not a mean score, because it stays meaningful
    if Person 2 retunes the point weights. `mean_percent` is shown as a delta
    beneath it so both figures are visible without implying they are the same
    measurement.

    Coloured steps mark the bands from `gauge_band`, and a threshold line sits
    at 33% — the point at which a third of analysed communication is flagged
    and the situation stops being individual and becomes organisational.
    """
    percent = max(0.0, min(100.0, float(percent or 0)))
    label, colour = gauge_band(percent)

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta" if mean_percent is not None else "gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 44}},
            delta=(
                {
                    "reference": float(mean_percent),
                    "increasing": {"color": RISK_COLOURS["High"]},
                    "decreasing": {"color": RISK_COLOURS["Low"]},
                    "valueformat": ".1f",
                }
                if mean_percent is not None
                else None
            ),
            title={
                "text": f"<b>Overall Burnout</b><br>"
                        f"<span style='font-size:0.85em;color:{colour}'>{label}</span>"
                        f"<br><span style='font-size:0.7em;color:#6B7280'>"
                        f"share of records at Medium or High risk</span>",
                "font": {"size": 16},
            },
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": colour, "thickness": 0.72},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 15], "color": "#EAF6EE"},
                    {"range": [15, 33], "color": "#FBF6DC"},
                    {"range": [33, 60], "color": "#FDF0DC"},
                    {"range": [60, 100], "color": "#FBE3E3"},
                ],
                "threshold": {
                    "line": {"color": RISK_COLOURS["Medium"], "width": 3},
                    "thickness": 0.8,
                    "value": 33,
                },
            },
        )
    )
    figure.update_layout(**_LAYOUT, height=height)
    return figure


# ==============================================================================
# Pie — Sentiment distribution
# ==============================================================================
def sentiment_pie(distribution: Dict[str, int], height: int = 340) -> go.Figure:
    """Sentiment mix from Person 2's RoBERTa classifier."""
    data = {k: v for k, v in (distribution or {}).items() if v}
    if not data:
        return _empty_figure("No sentiment data available.", height)

    labels = list(data.keys())
    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=list(data.values()),
            marker=dict(
                colors=[sentiment_colour(label) for label in labels],
                line=dict(color="#FFFFFF", width=2),
            ),
            # Label + percentage + count directly on each slice, so the chart
            # is readable without relying on colour or the legend.
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent}",
            hovertemplate="<b>%{label}</b><br>%{value} records<br>%{percent}"
                          "<extra></extra>",
            hole=0.42,
            sort=False,
        )
    )
    figure.update_layout(
        **_LAYOUT,
        height=height,
        title="Sentiment Distribution",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5,
                    xanchor="center"),
    )
    return figure


# ==============================================================================
# Bar — Emotion distribution
# ==============================================================================
def emotion_bar(totals: Dict[str, float], height: int = 340) -> go.Figure:
    """NRC emotion word counts across all records.

    Horizontal bars because the eight category names are long enough to
    overlap when rotated on a vertical axis.

    Positive-valence emotions (joy, trust, anticipation, surprise) are coloured
    green and negative ones red. Emotion counts are not themselves good or bad
    — but an HR reader scanning this chart is looking for the balance between
    the two, and colouring by valence is what makes that balance visible at a
    glance.
    """
    if not totals or not any(totals.values()):
        return _empty_figure(
            "The NRC lexicon detected no emotion words in these records.", height
        )

    ordered = sorted(totals.items(), key=lambda item: item[1])
    names = [name.title() for name, _ in ordered]
    values = [value for _, value in ordered]

    positive = {"Joy", "Trust", "Anticipation", "Surprise"}
    colours = [
        RISK_COLOURS["Low"] if name in positive else RISK_COLOURS["High"]
        for name in names
    ]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colours),
            text=[f"{v:g}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:g} word matches<extra></extra>",
        )
    )
    figure.update_layout(
        **_LAYOUT,
        height=height,
        title="Emotion Distribution (NRC lexicon word counts)",
        xaxis_title="Word matches",
        yaxis_title=None,
        showlegend=False,
    )
    return figure


# ==============================================================================
# Bar — Risk level distribution
# ==============================================================================
def risk_distribution_bar(
    distribution: Dict[str, int], height: int = 340
) -> go.Figure:
    """Record counts per risk level.

    Zero-count levels are kept rather than dropped: "High: 0" is a meaningful,
    reassuring statement, and silently omitting the bar would leave a reader
    unsure whether the category exists at all.
    """
    if not distribution:
        return _empty_figure("No risk data available.", height)

    levels = [lvl for lvl in RISK_ORDER if lvl in distribution]
    levels += [lvl for lvl in distribution if lvl not in levels]
    counts = [int(distribution.get(level, 0)) for level in levels]

    if not any(counts):
        return _empty_figure("No records to classify.", height)

    figure = go.Figure(
        go.Bar(
            x=levels,
            y=counts,
            marker=dict(color=[risk_colour(level) for level in levels]),
            text=[str(count) for count in counts],
            textposition="outside",
            hovertemplate="<b>%{x} risk</b><br>%{y} records<extra></extra>",
        )
    )
    figure.update_layout(
        **_LAYOUT,
        height=height,
        title="Risk Level Distribution",
        xaxis_title=None,
        yaxis_title="Records",
        showlegend=False,
        yaxis=dict(rangemode="tozero"),
    )
    return figure


def risk_donut(distribution: Dict[str, int], height: int = 340) -> go.Figure:
    """Risk mix as proportions, complementing the count bar chart."""
    data = {k: v for k, v in (distribution or {}).items() if v}
    if not data:
        return _empty_figure("No risk data available.", height)

    labels = [lvl for lvl in RISK_ORDER if lvl in data]
    labels += [lvl for lvl in data if lvl not in labels]

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=[data[label] for label in labels],
            marker=dict(
                colors=[risk_colour(label) for label in labels],
                line=dict(color="#FFFFFF", width=2),
            ),
            hole=0.55,
            textinfo="label+percent",
            hovertemplate="<b>%{label} risk</b><br>%{value} records<br>"
                          "%{percent}<extra></extra>",
            sort=False,
        )
    )
    figure.update_layout(
        **_LAYOUT, height=height, title="Risk Mix", showlegend=False
    )
    return figure


# ==============================================================================
# Histogram — score spread
# ==============================================================================
def score_histogram(frame: pd.DataFrame, height: int = 320) -> go.Figure:
    """Distribution of raw burnout scores.

    Reveals shape the summary statistics hide: whether risk is concentrated in
    a few people or spread thinly across everyone. Those two situations call
    for completely different responses, and they can share a mean.
    """
    if frame is None or frame.empty or "burnout_score" not in frame.columns:
        return _empty_figure("No score data available.", height)

    figure = go.Figure(
        go.Histogram(
            x=frame["burnout_score"],
            marker=dict(color="#4C78A8", line=dict(color="#FFFFFF", width=1)),
            xbins=dict(size=1),
            hovertemplate="Score %{x}<br>%{y} records<extra></extra>",
        )
    )
    # Person 2's actual thresholds, drawn so a reader can see where the bands
    # fall rather than having to read burnout_score.py.
    figure.add_vline(
        x=2.5, line_dash="dash", line_color=RISK_COLOURS["Medium"],
        annotation_text="Medium ≥3", annotation_position="top",
    )
    figure.add_vline(
        x=5.5, line_dash="dash", line_color=RISK_COLOURS["High"],
        annotation_text="High ≥6", annotation_position="top",
    )
    figure.update_layout(
        **_LAYOUT,
        height=height,
        title="Burnout Score Distribution",
        xaxis_title="Burnout score (raw points)",
        yaxis_title="Records",
        bargap=0.08,
        showlegend=False,
    )
    return figure


# ==============================================================================
# Scatter — score vs model confidence
# ==============================================================================
def employee_risk_scatter(frame: pd.DataFrame, height: int = 360) -> go.Figure:
    """Burnout score against sentiment-model confidence, coloured by risk.

    Surfaces a caveat the other charts cannot: a high score resting on a
    low-confidence sentiment prediction is weaker evidence than the same score
    at high confidence. Points toward the lower right deserve a human check
    before anyone acts on them.
    """
    if frame is None or frame.empty:
        return _empty_figure("No records to plot.", height)

    if "confidence" not in frame.columns:
        return _empty_figure("Model confidence is not present in the data.", height)

    figure = go.Figure()
    for level in RISK_ORDER:
        subset = frame[frame["risk_level"] == level]
        if subset.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=subset["burnout_score"],
                y=subset["confidence"] * 100,
                mode="markers",
                name=f"{level} risk",
                marker=dict(
                    size=14, color=risk_colour(level),
                    line=dict(color="#FFFFFF", width=1.5), opacity=0.85,
                ),
                customdata=subset[["employee_id", "clean_text"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Score: %{x:g}<br>Confidence: %{y:.0f}%<br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        **_LAYOUT,
        height=height,
        title="Burnout Score vs Sentiment Confidence",
        xaxis_title="Burnout score (raw points)",
        yaxis_title="Sentiment model confidence (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5,
                    xanchor="center"),
    )
    return figure


__all__ = [
    "burnout_gauge",
    "sentiment_pie",
    "emotion_bar",
    "risk_distribution_bar",
    "risk_donut",
    "score_histogram",
    "employee_risk_scatter",
]
