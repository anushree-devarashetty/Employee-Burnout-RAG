"""
frontend/utils/theme.py — Design system for the dashboard (Person 4)
==============================================================================

WHY A THEME MODULE RATHER THAN CSS SPRINKLED THROUGH app.py

    Styling scattered across the app drifts: three different greys, four
    spacing values, and a callout that looks right in light mode and is
    unreadable in dark. Everything visual is defined here once, as tokens, and
    consumed everywhere else.

WHY THE PALETTE IS RESOLVED IN PYTHON, NOT BY `prefers-color-scheme`

    An earlier version switched palettes with a CSS media query. That is wrong
    here, and subtly so: `prefers-color-scheme` reflects the *operating system*,
    while Streamlit renders according to its OWN theme setting in
    `.streamlit/config.toml`. Someone running a light-configured app on a
    dark-mode Mac got Streamlit's light chrome wrapped around our dark panels —
    dark cards on a white page.

    So the active theme is read from Streamlit itself via
    `st.get_option("theme.base")` and the matching palette is emitted directly.
    One source of truth, no mismatch possible.

THE DARK MODE BUG THIS ALSO FIXES
    The original callout CSS set a background but no text colour, so text
    inherited Streamlit's default — near-white in dark mode, on a near-white
    panel. Every rule below sets background AND foreground together.

DESIGN DIRECTION
    A professional analytics tool, not a consumer app:

      - One accent colour, used sparingly. Green/amber/red are reserved
        exclusively for risk, so colour always carries a single meaning.
      - Hairline borders and generous whitespace rather than heavy shadows.
      - Tabular numerals so figures in a column align vertically.
      - A clear type scale — three sizes used consistently beats seven used
        approximately.

DARK MODE SPECIFICS
    Dark themes fail in predictable ways, all of which are handled here:

      - Pure black (#000) with pure white text is harsh and causes halation.
        The darkest surface here is #0F141A, text is #E6EDF5, never #FFF.
      - Elevation cannot be shown with shadows on a dark background, because
        a shadow is invisible against dark. Depth is conveyed by making each
        raised surface LIGHTER, which is how the physical metaphor actually
        works under ambient light.
      - Saturated colours vibrate against dark backgrounds. The risk colours
        are lightened and slightly desaturated for the dark palette rather
        than reused unchanged.
      - Borders need more contrast in dark mode than in light to remain
        visible at all.
==============================================================================
"""

from __future__ import annotations

ACCENT_LIGHT = "#3D6B99"
ACCENT_DARK = "#7FB2E5"

#: Semantic risk colours, tuned per theme. The dark variants are lighter and
#: less saturated: a colour that reads as confident on white vibrates against
#: a dark surface and is harder to read at small sizes.
RISK_LIGHT = {
    "Low": "#2E9E5B", "Medium": "#E8A33D", "High": "#D64545", "Unknown": "#8A8A8A",
}
RISK_DARK = {
    "Low": "#5FD08D", "Medium": "#F5C263", "High": "#F07575", "Unknown": "#9AA6B2",
}

#: Module-level aliases, rebound by `use()` so callers importing `theme.RISK`
#: always see the active palette.
RISK = dict(RISK_LIGHT)
SENTIMENT = {
    "Positive": RISK["Low"], "Neutral": "#7B8794",
    "Negative": RISK["High"], "Unknown": RISK["Unknown"],
}
ACCENT = ACCENT_LIGHT
IS_DARK = False


# ==============================================================================
# Theme resolution
# ==============================================================================
def active_is_dark() -> bool:
    """True if Streamlit is currently rendering its dark theme.

    Read from Streamlit rather than the OS, so our panels can never disagree
    with the chrome around them. Falls back to light if the option is missing,
    which is the safer default for projectors and screenshots.
    """
    try:
        import streamlit as st

        base = st.get_option("theme.base")
        return str(base).lower() == "dark"
    except Exception:
        return False


def use(dark: bool | None = None) -> bool:
    """Select the palette. Call once, before any component is rendered."""
    global RISK, SENTIMENT, ACCENT, IS_DARK

    IS_DARK = active_is_dark() if dark is None else bool(dark)
    RISK = dict(RISK_DARK if IS_DARK else RISK_LIGHT)
    ACCENT = ACCENT_DARK if IS_DARK else ACCENT_LIGHT
    SENTIMENT = {
        "Positive": RISK["Low"],
        "Neutral": "#8A97A5" if IS_DARK else "#7B8794",
        "Negative": RISK["High"],
        "Unknown": RISK["Unknown"],
    }
    return IS_DARK


# ==============================================================================
# Palettes
# ==============================================================================
_LIGHT = {
    "ink": "#0E1723", "ink_soft": "#41525F", "ink_mute": "#6B7A88",
    "surface": "#FFFFFF", "surface_2": "#F6F9FC", "surface_3": "#ECF1F6",
    "line": "#DCE4EC", "line_strong": "#C3D0DC",
    "accent": ACCENT_LIGHT, "accent_ink": "#2C5478",
    "shadow": "0 1px 2px rgba(14,23,35,.06), 0 2px 10px rgba(14,23,35,.05)",
    "mast_a": "#2C5478", "mast_b": "#3D6B99",
}

# Dark surfaces get LIGHTER as they rise. On a dark background a drop shadow is
# invisible, so elevation has to be signalled by luminance instead.
_DARK = {
    "ink": "#E6EDF5", "ink_soft": "#AFBCCA", "ink_mute": "#7D8B9A",
    "surface": "#161D26", "surface_2": "#1D2630", "surface_3": "#26313D",
    "line": "#2E3A47", "line_strong": "#3D4B5A",
    "accent": ACCENT_DARK, "accent_ink": "#A9CDEF",
    "shadow": "0 1px 2px rgba(0,0,0,.40), 0 2px 12px rgba(0,0,0,.30)",
    "mast_a": "#1B3A5C", "mast_b": "#2F5F8F",
}


def css(dark: bool | None = None) -> str:
    """The dashboard stylesheet for the active theme."""
    is_dark = use(dark)
    p = _DARK if is_dark else _LIGHT
    r = RISK

    return f"""
<style>
:root {{
  --ink:{p['ink']}; --ink-soft:{p['ink_soft']}; --ink-mute:{p['ink_mute']};
  --surface:{p['surface']}; --surface-2:{p['surface_2']}; --surface-3:{p['surface_3']};
  --line:{p['line']}; --line-strong:{p['line_strong']};
  --accent:{p['accent']}; --accent-ink:{p['accent_ink']};
  --shadow:{p['shadow']};
  --low:{r['Low']}; --medium:{r['Medium']}; --high:{r['High']};
}}

/* ==========================================================================
   SCOPE NOTE
   This stylesheet styles OUR components and nothing else.

   An earlier version overrode Streamlit's own layout internals — the flex gap
   on stVerticalBlock, padding on the chart wrapper, tight line-heights — and
   every one of those caused a visible defect: clipped headings, scrollbars
   inside charts, overlapping text. Streamlit computes widget geometry in JS at
   runtime; CSS that changes the box model underneath it will fight that
   calculation and lose.

   The rule now: colour, typography and our own components are ours to style.
   Layout, spacing between widgets, and anything Streamlit measures are not.
   ========================================================================== */

/* --------------------------------------------------------------------------
   STREAMLIT CHROME
   Hides the Deploy button, the overflow menu, the running indicator, the
   coloured top decoration and the footer — developer affordances that make a
   demo read as a dev tool.

   DO NOT hide [data-testid="stToolbar"] as a whole. An earlier version did,
   and in current Streamlit the "expand sidebar" arrow is rendered inside that
   same container: collapsing the sidebar then left no way to reopen it. Hide
   the individual controls instead, and leave the header itself alone.

   Testids have been renamed across Streamlit releases, so each control is
   listed under every name it has shipped with. Selectors that match nothing
   are harmless.
   -------------------------------------------------------------------------- */
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
.stDeployButton,
[data-testid="stMainMenu"],
#MainMenu,
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
footer {{ display: none !important; }}

/* Keep every variant of the sidebar control reachable. */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {{
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}}

/* The header must not paint a bar over the masthead once emptied. */
[data-testid="stHeader"] {{ background: transparent !important; }}

.block-container {{ padding-top:1.5rem !important; max-width:1500px; }}
#MainMenu, footer {{ visibility:hidden; }}
.stDataFrame, [data-testid="stMetricValue"], .kpi-value {{
  font-variant-numeric: tabular-nums; }}

/* ---------- masthead ---------- */
.masthead {{
  display:flex; align-items:center; gap:1rem;
  padding:1.35rem 1.5rem; margin:0 0 1.25rem;
  background:linear-gradient(135deg,{p['mast_a']} 0%,{p['mast_b']} 100%);
  border-radius:12px;
  /* overflow visible + generous line-height: the previous 1.15 with negative
     tracking clipped descenders and the cap-height of the first line. */
  overflow:visible;
}}
.masthead-mark {{
  width:48px; height:48px; flex:0 0 48px; border-radius:11px;
  background:rgba(255,255,255,.18);
  display:flex; align-items:center; justify-content:center;
  font-size:1.5rem; line-height:1; color:#FFF;
}}
.masthead h1 {{
  margin:0 !important; padding:0 !important;
  font-size:1.75rem; font-weight:670; line-height:1.35;
  color:#FFFFFF; letter-spacing:-.01em;
}}
.masthead p {{
  margin:.35rem 0 0 !important; font-size:.9rem; line-height:1.5;
  color:rgba(255,255,255,.85);
}}

/* ---------- section headings ---------- */
.section {{ margin:1.75rem 0 .75rem; }}
.section-eyebrow {{
  display:flex; align-items:center; gap:.4rem;
  font-size:.72rem; font-weight:700; letter-spacing:.11em;
  text-transform:uppercase; color:var(--accent); margin-bottom:.3rem;
  line-height:1.5;
}}
.section h2 {{
  margin:0 !important; padding:0 !important;
  font-size:1.4rem; font-weight:660; color:var(--ink); line-height:1.4;
}}
.section p.sub {{
  margin:.3rem 0 0 !important; font-size:.87rem; color:var(--ink-mute);
  line-height:1.55; max-width:90ch;
}}

/* ---------- KPI cards ---------- */
.kpi-row {{ display:flex; gap:.85rem; flex-wrap:wrap; margin-bottom:.25rem; }}
.kpi {{
  flex:1 1 200px; min-width:200px;
  background:var(--surface); color:var(--ink);
  border:1px solid var(--line); border-top:3px solid var(--accent);
  border-radius:11px; padding:.95rem 1.05rem;
}}
.kpi-label {{
  font-size:.73rem; font-weight:660; letter-spacing:.08em; line-height:1.5;
  text-transform:uppercase; color:var(--ink-mute); margin-bottom:.3rem;
}}
/* line-height 1.25, not 1.02 — the tighter value clipped the % sign and any
   descender in the value. */
.kpi-value {{ font-size:2rem; font-weight:670; color:var(--ink);
              line-height:1.25; }}
.kpi-note {{ font-size:.8rem; color:var(--ink-soft); margin-top:.25rem;
             line-height:1.5; }}

/* ---------- risk pills ---------- */
.pill {{
  display:inline-block; padding:.18rem .7rem; border-radius:999px;
  font-size:.8rem; font-weight:660; line-height:1.6;
  border:1px solid currentColor;
}}
.pill-low     {{ color:var(--low);    background:{_tint(r['Low'], is_dark)}; }}
.pill-medium  {{ color:var(--medium); background:{_tint(r['Medium'], is_dark)}; }}
.pill-high    {{ color:var(--high);   background:{_tint(r['High'], is_dark)}; }}
.pill-unknown {{ color:var(--ink-mute); background:var(--surface-3); }}

/* ---------- callouts ---------- */
.callout {{
  border-radius:9px; padding:.85rem 1.05rem; margin:.5rem 0 1rem;
  font-size:.92rem; line-height:1.65;
  background:var(--surface-2); color:var(--ink);
  border:1px solid var(--line); border-left:3px solid var(--accent);
}}
.callout strong {{ color:var(--ink); font-weight:660; }}
.callout code {{ background:var(--surface-3); color:var(--ink);
                 padding:.05rem .3rem; border-radius:4px; }}
.callout-caution {{ border-left-color:var(--medium); }}
.callout-alert   {{ border-left-color:var(--high); }}
.callout-good    {{ border-left-color:var(--low); }}

/* ---------- sidebar ---------- */
.side-group {{
  display:flex; align-items:center; gap:.42rem;
  font-size:.74rem; font-weight:700; letter-spacing:.11em; line-height:1.6;
  text-transform:uppercase; color:var(--ink-mute);
  margin:1.15rem 0 .4rem; padding-bottom:.3rem;
  border-bottom:1px solid var(--line);
}}
.status {{
  display:flex; align-items:center; gap:.5rem;
  padding:.55rem .75rem; border-radius:8px; margin-bottom:.25rem;
  background:var(--surface); color:var(--ink);
  border:1px solid var(--line);
  font-size:.9rem; font-weight:600; line-height:1.5;
}}
.status-dot {{ width:9px; height:9px; border-radius:50%; flex:0 0 9px; }}

/* Streamlit's sidebar collapse control is left ALONE.
   An attempt to replace its chevron with a CSS-drawn hamburger hid the real
   glyph and left an empty box: the control is re-rendered by Streamlit with
   its own classes, so a pseudo-element attached to a guessed selector has
   nothing stable to attach to. Its default icon works; ours did not. */

/* ---------- analysis sections ---------- */
.analysis-head {{
  font-size:.8rem; font-weight:700; letter-spacing:.09em; line-height:1.6;
  text-transform:uppercase; color:var(--accent); margin:1.1rem 0 .35rem;
}}
.analysis-body {{ font-size:.95rem; line-height:1.7; color:var(--ink); }}
.analysis-list {{ margin:.25rem 0 0; padding-left:1.15rem; }}
.analysis-list li {{ font-size:.93rem; line-height:1.65; margin-bottom:.3rem;
                     color:var(--ink); }}

/* ---------- icons ---------- */
.icon {{ vertical-align:-2px; opacity:.85; flex:0 0 auto; }}

/* ---------- tables ---------- */
/* Charts are deliberately NOT wrapped in a bordered panel. Adding padding or a
   border to the Plotly container changes the box Streamlit measured, so the
   figure overflows by exactly that amount and gains a scrollbar. */
.stDataFrame {{ border:1px solid var(--line); border-radius:9px; }}
hr {{ border-color:var(--line) !important; }}
</style>
"""


def _tint(hex_colour: str, is_dark: bool) -> str:
    """A translucent wash of a colour, for pill backgrounds.

    Dark mode gets a slightly stronger tint: a 12% wash that reads clearly on
    white almost disappears against a dark surface.
    """
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    alpha = 0.20 if is_dark else 0.13
    return f"rgba({r},{g},{b},{alpha})"


# ==============================================================================
# Icons
# ==============================================================================
#: Inline SVG rather than emoji. Emoji render differently on every OS, carry
#: colours we do not control, and look informal next to a data table. These are
#: stroked with `currentColor`, so they inherit the surrounding text colour and
#: therefore adapt to the theme automatically.
_ICONS = {
    "ask": "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
    "data": "M12 5c4.4 0 8-1.1 8-2.5S16.4 0 12 0 4 1.1 4 2.5 7.6 5 12 5z"
            "M20 9.5c0 1.4-3.6 2.5-8 2.5s-8-1.1-8-2.5"
            "M20 2.5v14c0 1.4-3.6 2.5-8 2.5s-8-1.1-8-2.5v-14",
    "report": "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"
              "M14 2v6h6M16 13H8M16 17H8M10 9H8",
    "upload": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"
              "M17 8l-5-5-5 5M12 3v12",
    "filter": "M22 3H2l8 9.46V19l4 2v-8.54z",
    "chart":  "M18 20V10M12 20V4M6 20v-6",
    "shield": "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
    "search": "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.35-4.35",
    "sparkle": "M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z",
    "download": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3",
}


def icon(name: str, size: int = 15) -> str:
    """An inline SVG icon that inherits the current text colour."""
    path = _ICONS.get(name)
    if not path:
        return ""
    return (
        f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="1.9" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'<path d="{path}"/></svg>'
    )


# ==============================================================================
# Components
# ==============================================================================
def masthead(title: str, subtitle: str) -> str:
    return (
        f'<div class="masthead"><div class="masthead-mark">◑</div>'
        f'<div><h1>{title}</h1><p>{subtitle}</p></div></div>'
    )


def section(eyebrow: str, title: str, subtitle: str = "",
            name: str = "") -> str:
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    return (
        f'<div class="section"><div class="section-eyebrow">'
        f"{icon(name, 13)}<span>{eyebrow}</span></div>"
        f"<h2>{title}</h2>{sub}</div>"
    )


def kpi_row(cards) -> str:
    """Render KPI cards from (label, value, note, colour_or_None) tuples.

    Hand-built rather than `st.metric` because the built-in offers no control
    over the accent rule or supporting note, and its delta arrow implies a
    time comparison we are not making.
    """
    html = ['<div class="kpi-row">']
    for label, value, note, colour in cards:
        style = f' style="border-top-color:{colour}"' if colour else ""
        html.append(
            f'<div class="kpi"{style}><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-note">{note}</div></div>'
        )
    html.append("</div>")
    return "".join(html)


def pill(level: str) -> str:
    return f'<span class="pill pill-{str(level).lower()}">{level}</span>'


def callout(body: str, tone: str = "", label: str = "") -> str:
    cls = f"callout callout-{tone}" if tone else "callout"
    strong = f"<strong>{label}</strong> " if label else ""
    return f'<div class="{cls}">{strong}{body}</div>'


def status(online: bool, url: str = "") -> str:
    """Backend state — just the fact, nothing else.

    Earlier versions printed the base URL and, when offline, a paragraph of
    explanation plus a copyable uvicorn command. That is troubleshooting
    documentation living permanently in the chrome: it consumed four lines of
    sidebar on every render to help with a situation that occurs rarely, and
    it made the panel look broken even when everything was fine.

    The URL is in `.env` and the fix is in the README. The sidebar states the
    fact; `title` carries the address for anyone who wants it on hover.
    """
    colour = RISK["Low"] if online else RISK["High"]
    text = "Backend online" if online else "Backend offline"
    hint = f' title="{url}"' if url else ""
    return (
        f'<div class="status"{hint}><span class="status-dot" '
        f'style="background:{colour}"></span><span>{text}</span></div>'
    )


def side_group(label: str, name: str = "") -> str:
    return f'<div class="side-group">{icon(name, 14)}<span>{label}</span></div>'


def analysis_block(heading: str, body: str = "", items=None) -> str:
    html = [f'<div class="analysis-head">{heading}</div>']
    if body:
        html.append(f'<div class="analysis-body">{body}</div>')
    if items:
        html.append('<ul class="analysis-list">')
        html.extend(f"<li>{i}</li>" for i in items)
        html.append("</ul>")
    elif items is not None:
        html.append(
            '<div class="analysis-body" style="color:var(--ink-mute)">'
            "None identified from the available evidence.</div>"
        )
    return "".join(html)


__all__ = [
    "css", "use", "active_is_dark", "icon", "masthead", "section", "kpi_row", "pill",
    "callout", "status", "side_group", "analysis_block",
    "ACCENT", "RISK", "SENTIMENT", "IS_DARK",
]
