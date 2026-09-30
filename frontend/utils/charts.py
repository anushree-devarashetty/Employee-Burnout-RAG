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

from frontend.utils.data_utils import RISK_ORDER, gauge_band

# Chart colours come from the THEME, not from data_utils.
#
# data_utils holds the light palette, which is correct for the PDF report —
# that always prints on white paper. On screen the palette must follow the
# active theme, because a colour tuned for white vibrates against a dark
# surface. Two consumers, two requirements, one deliberate split.
from frontend.utils import theme as _theme


def risk_colour(level: str) -> str:
    """Risk colour for the ACTIVE theme."""
    return _theme.RISK.get(str(level).title(), _theme.RISK["Unknown"])


def sentiment_colour(label: str) -> str:
    """Sentiment colour for the ACTIVE theme."""
    return _theme.SENTIMENT.get(str(label).title(), _theme.SENTIMENT["Unknown"])


RISK_COLOURS = _theme.RISK
SENTIMENT_COLOURS = _theme.SENTIMENT

#: Applied to every figure so panels align on the dashboard grid.
#
# `template="none"` rather than "plotly_white": a fixed white template paints an
# opaque white canvas, which sits as a glaring block inside a dark-mode page.
# With no template the figure background stays transparent and inherits the
# surrounding surface, so one chart definition works in both Streamlit themes.
#
# Colours here are chosen to read acceptably on either background: mid-tone
# greys for structure, saturated hues only for data.
def _is_dark() -> bool:
    """Whether Streamlit is rendering dark, read from its own setting."""
    try:
        from frontend.utils.theme import active_is_dark

        return active_is_dark()
    except Exception:
        return False


_DARK = _is_dark()

# Axis and label colours have to differ per theme. A single mid-grey that reads
# well on white is too dim against a dark surface — legible in a screenshot,
# unreadable on a projector.
_GRID = "rgba(150,168,186,0.16)" if _DARK else "rgba(120,138,156,0.20)"
_AXIS = "rgba(150,168,186,0.34)" if _DARK else "rgba(120,138,156,0.45)"
_TEXT = "#B6C4D2" if _DARK else "#5A6B7B"
_TITLE = "#E6EDF5" if _DARK else "#0E1723"

# Paired charts share a height so the two columns of each row line up. Left
# and right at different heights leaves a ragged edge that reads as broken
# layout even when nothing is wrong.
#
#   row 1  gauge | sentiment pie          -> H_TALL
#   row 2  emotion | risk distribution    -> H_STD
#   row 3  risk donut | score histogram   -> H_STD
#   row 4  scatter (full width)           -> H_WIDE
H_TALL, H_STD, H_WIDE = 420, 380, 400

_LAYOUT = dict(
    template="none",
    margin=dict(l=48, r=32, t=64, b=44),
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
        size=13.5,
        color=_TEXT,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(
        font_size=13,
        font_family="-apple-system, system-ui, sans-serif",
        bgcolor="rgba(30,38,48,0.94)",
        bordercolor="rgba(128,145,162,0.4)",
        font=dict(color="#F0F3F6"),
    ),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_AXIS, linecolor=_AXIS,
               tickfont=dict(size=12)),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_AXIS, linecolor=_AXIS,
               tickfont=dict(size=12)),
    legend=dict(font=dict(size=12)),
)


def _finish(figure: go.Figure, height: int, **overrides) -> go.Figure:
    """Apply the shared layout, then per-chart overrides.

    `update_layout(**_LAYOUT, title=...)` raises `got multiple values for
    keyword argument` as soon as the shared layout defines a key a chart also
    sets. Merging into one dict — with nested dicts merged rather than replaced,
    so a chart overriding `xaxis.title` does not silently discard the shared
    grid colour — keeps one source of truth for styling.
    """
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _LAYOUT.items()}

    # TITLE STYLING IS APPLIED ONLY WHEN THERE IS A TITLE.
    #
    # Previously the shared layout carried a `title` dict holding font and
    # position but no `text`. Plotly then rendered the literal string
    # "undefined" above every chart that did not set its own title — which was
    # visible above the gauge. Styling now attaches to the caller's title
    # instead of existing on its own.
    if isinstance(overrides.get("title"), dict) and overrides["title"].get("text"):
        overrides["title"] = {
            "font": {"size": 17, "color": _TITLE},
            "x": 0.01, "xanchor": "left", "y": 0.97, "yanchor": "top",
            **overrides["title"],
        }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    layout["height"] = height
    # autosize=False with an explicit height stops Plotly re-measuring after
    # Streamlit has already sized the container, which is what produced a
    # scrollbar inside the chart frame.
    layout["autosize"] = False
    figure.update_layout(**layout)
    return figure


def _empty_figure(message: str, height: int = H_STD) -> go.Figure:
    """A figure that explains why it is empty, instead of a blank rectangle."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14.5, color=_TEXT),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )
    return _finish(figure, height, xaxis=dict(visible=False), yaxis=dict(visible=False))


# ==============================================================================
# Gauge — Overall Burnout %
# ==============================================================================
def burnout_gauge(
    percent: float,
    mean_percent: Optional[float] = None,
    height: int = H_TALL,
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
            # No delta. "▼−0.8" against the mean normalised score invited the
            # reading "burnout fell by 0.8", which is wrong — it is the gap
            # between two different measures, not a change over time. The mean
            # is reported as plain text beneath the chart instead.
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 46, "color": _TITLE}},
            # TWO title lines, not three.
            #
            # Plotly renders an Indicator's title INSIDE the plot area rather
            # than in the margin, so every extra <br> pushes the whole gauge
            # down. A three-line title at the previous height clipped the top
            # line off entirely. The explanatory sentence now lives as a
            # caption beneath the chart in app.py, where it does not compete
            # with the gauge for vertical space.
            title={
                "text": f"<b>Overall Burnout</b><br>"
                        f"<span style='font-size:0.75em;color:{colour}'>{label}</span>",
                "font": {"size": 17, "color": _TITLE},
            },
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%",
                         "tickfont": {"size": 11},
                         "tickcolor": _AXIS},
                "bar": {"color": colour, "thickness": 0.62},
                "borderwidth": 0,
                "bgcolor": "rgba(0,0,0,0)",
                # Translucent bands, not opaque pastels: solid light fills
                # look like white blocks on a dark page. rgba lets the page
                # surface show through in either theme.
                # Very low alpha. At 0.14-0.15 the amber and red bands mixed
                # with the dark page into a muddy brown that read as dirt
                # rather than as a scale.
                "steps": [
                    {"range": [0, 15], "color": "rgba(46,158,91,0.07)"},
                    {"range": [15, 33], "color": "rgba(217,196,65,0.07)"},
                    {"range": [33, 60], "color": "rgba(232,163,61,0.07)"},
                    {"range": [60, 100], "color": "rgba(214,69,69,0.07)"},
                ],
                "threshold": {
                    "line": {"color": RISK_COLOURS["Medium"], "width": 2},
                    "thickness": 0.75,
                    "value": 33,
                },
            },
        )
    )
    # Override the shared top margin. The Indicator's title plus its large
    # number need materially more headroom than a normal chart title, and the
    # shared t=60 was what cropped the heading.
    return _finish(figure, height, margin=dict(l=40, r=40, t=100, b=30))


# ==============================================================================
# Pie — Sentiment distribution
# ==============================================================================
def sentiment_pie(distribution: Dict[str, int], height: int = H_TALL) -> go.Figure:
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
                line=dict(color="rgba(128,145,162,0.35)", width=1.5),
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
    return _finish(figure, height, title=dict(text="Sentiment Distribution"), showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.02, x=0.5, xanchor="center"))


# ==============================================================================
# Bar — Emotion distribution
# ==============================================================================
def emotion_bar(totals: Dict[str, float], height: int = H_STD) -> go.Figure:
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
            marker=dict(color=colours, line=dict(width=0)),
            text=[f"{v:g}" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:g} word matches<extra></extra>",
        )
    )
    return _finish(figure, height, title=dict(text="Emotion Distribution (NRC lexicon word counts)"), xaxis_title=dict(text="Word matches"), yaxis_title=None, showlegend=False)


# ==============================================================================
# Bar — Risk level distribution
# ==============================================================================
def risk_distribution_bar(
    distribution: Dict[str, int], height: int = H_STD
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
    return _finish(figure, height, title=dict(text="Risk Level Distribution"), xaxis_title=None, yaxis_title=dict(text="Records"), showlegend=False, yaxis=dict(rangemode="tozero"))


def risk_donut(distribution: Dict[str, int], height: int = H_STD) -> go.Figure:
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
                line=dict(color="rgba(128,145,162,0.35)", width=1.5),
            ),
            hole=0.55,
            textinfo="label+percent",
            hovertemplate="<b>%{label} risk</b><br>%{value} records<br>"
                          "%{percent}<extra></extra>",
            sort=False,
        )
    )
    return _finish(figure, height, title=dict(text="Risk Mix"), showlegend=False)


# ==============================================================================
# Histogram — score spread
# ==============================================================================
def score_histogram(frame: pd.DataFrame, height: int = H_STD) -> go.Figure:
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
            marker=dict(color="#4C78A8", line=dict(color="rgba(128,145,162,0.3)", width=1)),
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
    return _finish(figure, height, title=dict(text="Burnout Score Distribution"), xaxis_title=dict(text="Burnout score (raw points)"), yaxis_title=dict(text="Records"), bargap=0.08, showlegend=False)


# ==============================================================================
# Scatter — score vs model confidence
# ==============================================================================
def employee_risk_scatter(frame: pd.DataFrame, height: int = H_WIDE) -> go.Figure:
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
                    line=dict(color="rgba(128,145,162,0.5)", width=1.5), opacity=0.9,
                ),
                customdata=subset[["employee_id", "clean_text"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Score: %{x:g}<br>Confidence: %{y:.0f}%<br>"
                    "%{customdata[1]}<extra></extra>"
                ),
            )
        )

    # Legend ABOVE the plot, not below.
    #
    # A horizontal legend at y=-0.25 sits in exactly the same band as the
    # x-axis title; in the rendered dashboard the two overlapped into
    # unreadable mush. Moving it to the top-right clears the axis entirely and
    # uses the empty space beside the chart title.
    return _finish(
        figure, height,
        title=dict(text="Burnout Score vs Sentiment Confidence"),
        xaxis_title=dict(text="Burnout score (raw points)"),
        yaxis_title=dict(text="Sentiment model confidence (%)"),
        margin=dict(l=64, r=32, t=84, b=66),
        legend=dict(orientation="h", yanchor="bottom", y=1.03,
                    x=1, xanchor="right"),
    )


__all__ = [
    "burnout_gauge",
    "sentiment_pie",
    "emotion_bar",
    "risk_distribution_bar",
    "risk_donut",
    "score_histogram",
    "employee_risk_scatter",
]
