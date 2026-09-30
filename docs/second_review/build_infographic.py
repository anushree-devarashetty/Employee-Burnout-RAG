"""SPARC Research-to-Prototype Challenge 2026 — capstone infographic (A3 portrait)."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from PIL import Image

W, H = 11.69, 16.54                      # A3 portrait, inches

NAVY  = "#16324F"
RED   = "#BC312C"
GOLD  = "#C9A227"
GOLDD = "#8A6D12"
INK   = "#16222E"
BODY  = "#2B3945"
MUTED = "#5B6B7B"
PAPER = "#F2F5F9"
LINE  = "#D3DCE6"

C_PROB, C_OBJ, C_TECH = "#B4453C", "#2E7D6B", "#2C5F96"
C_KEY, C_PROT, C_RES  = "#7A4B96", "#B4762A", "#2C5F96"
C_SOC, C_ENV, C_FUT   = "#1E7A63", "#3F7A2E", "#8B3A5E"

F = {"family": "DejaVu Sans"}
CPI = 128.0          # chars-per-inch constant: chars = width_in * CPI / fontsize

fig, ax = plt.subplots(figsize=(W, H))
ax.set_position([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
fig.patch.set_facecolor(PAPER)
ax.add_patch(Rectangle((0, 0), W, H, facecolor=PAPER, edgecolor="none", zorder=0))


# ---------------------------------------------------------------- helpers ---
def wrap(s, w_in, size):
    n = max(6, int(w_in * CPI / size))
    out = []
    for part in s.split("\n"):
        out.extend(textwrap.wrap(part, n) or [""])
    return out


def lh(size, lead=1.30):
    return size * lead / 72.0


def para(x, y, w, s, size=8.0, colour=BODY, weight="normal", ha="left",
         style="normal", lead=1.30, z=6, family=None):
    """Draw wrapped text with `y` as the TOP edge. Returns height consumed."""
    kw = dict(F) if family is None else {"family": family}
    lines = wrap(s, w, size)
    step = lh(size, lead)
    for i, ln in enumerate(lines):
        ax.text(x, y - i * step, ln, fontsize=size, color=colour,
                fontweight=weight, ha=ha, va="top", style=style, zorder=z, **kw)
    return len(lines) * step


def rbox(x, y, w, h, fc, ec=None, lw=1.0, r=0.055, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc, edgecolor=ec or fc, linewidth=lw, zorder=z))


def card(x, y, w, h, title, accent, icon=""):
    rbox(x, y, w, h, "#FFFFFF", LINE, 1.0, z=2)
    rbox(x, y + h - 0.42, w, 0.42, accent, accent, 1.0, z=3)
    ax.add_patch(Rectangle((x, y + h - 0.46), w, 0.18, facecolor=accent,
                           edgecolor="none", zorder=3))
    ax.text(x + 0.20, y + h - 0.21, f"{icon}   {title}".strip(), fontsize=10.4,
            color="#FFFFFF", fontweight="bold", va="center", zorder=5, **F)
    return y + h - 0.58          # top of the content area


def blist(x, y, w, items, size=7.6, dot=C_TECH, gap=0.065, lead=1.30):
    yy = y
    for it in items:
        ax.add_patch(Rectangle((x, yy - 0.085), 0.052, 0.052, facecolor=dot,
                               edgecolor="none", zorder=6))
        h = para(x + 0.18, yy, w - 0.18, it, size, lead=lead)
        yy -= h + gap
    return yy


def arrow(x1, y1, x2, y2, colour=MUTED, lw=1.8, ms=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, linewidth=lw, color=colour,
                                 zorder=6))


# ================================================================= HEADER ===
ax.add_patch(Rectangle((0, 14.24), W, H - 14.24, facecolor=NAVY,
                       edgecolor="none", zorder=1))
ax.add_patch(Rectangle((0, 16.40), W, 0.14, facecolor=RED, edgecolor="none",
                       zorder=2))

logo = Image.open("nmims_logo.png")
lw_in = 2.85
lh_in = lw_in * logo.height / logo.width
ax.imshow(logo, extent=(0.40, 0.40 + lw_in, 16.22 - lh_in, 16.22), zorder=4,
          aspect="auto")

ax.text(11.29, 16.16, "SPARC  ·  RESEARCH-TO-PROTOTYPE CHALLENGE 2026",
        fontsize=10.0, color=GOLD, fontweight="bold", ha="right", va="top",
        zorder=5, **F)
ax.text(11.29, 15.93, "Engineers Day Pre-Event Celebration   ·   Smart Process "
        "Automation & Research Club   ·   29 August 2026", fontsize=7.6,
        color="#B9CADB", ha="right", va="top", zorder=5, **F)
ax.text(11.29, 15.74, "CAPSTONE PROJECT   ·   A.Y. 2026-2027", fontsize=7.6,
        color="#B9CADB", fontweight="bold", ha="right", va="top", zorder=5, **F)

ax.add_patch(Rectangle((0.40, 15.42), 10.89, 0.024, facecolor=GOLD,
                       edgecolor="none", zorder=4))

ax.text(W / 2, 15.24, "Explainable Employee Burnout Detection", fontsize=25,
        color="#FFFFFF", fontweight="bold", ha="center", va="top", zorder=5, **F)
ax.text(W / 2, 14.83, "from Workplace Communication using Sentiment Analysis "
        "and Retrieval-Augmented Generation", fontsize=12.2, color="#DCE7F2",
        ha="center", va="top", zorder=5, **F)
ax.text(W / 2, 14.53, "Computer Engineering Department   ·   Mukesh Patel School "
        "of Technology Management & Engineering, NMIMS Mumbai Campus",
        fontsize=8.0, color="#9FB6CC", ha="center", va="top", zorder=5, **F)

# ------------------------------------------------------------- team strip ---
rbox(0.40, 13.46, 10.89, 0.70, "#FFFFFF", LINE, 1.0, r=0.05)
_team = [("Saumya Shah", "70322100115", "C040"),
         ("Swara Wairkar", "70322100201", "C048"),
         ("Aryaman Giri", "70322100043", "C026"),
         ("Anushree Devarashetty", "7032100175", "C038")]
ax.text(0.62, 13.81, "TEAM", fontsize=7.2, color=MUTED, fontweight="bold",
        va="center", zorder=6, **F)
cw = 7.95 / 4
for i, (nm, sap, roll) in enumerate(_team):
    cx = 1.30 + cw * (i + 0.5)
    ax.text(cx, 13.94, nm, fontsize=8.6, color=INK, fontweight="bold",
            ha="center", va="center", zorder=6, **F)
    ax.text(cx, 13.70, f"SAP {sap}  ·  {roll}", fontsize=7.0, color=MUTED,
            ha="center", va="center", zorder=6, **F)
ax.add_patch(Rectangle((9.42, 13.54), 0.014, 0.54, facecolor=LINE,
                       edgecolor="none", zorder=6))
ax.text(10.36, 13.94, "PROJECT MENTOR", fontsize=7.0, color=MUTED,
        fontweight="bold", ha="center", va="center", zorder=6, **F)
ax.text(10.36, 13.70, "Dr. Shubha Puthran", fontsize=8.8, color=RED,
        fontweight="bold", ha="center", va="center", zorder=6, **F)

LX, RX, C2 = 0.40, 6.05, 5.24
FW = 10.89

# ============================================== ROW A — PROBLEM / OBJECTIVES =
AY, AH = 10.70, 2.70
b = card(LX, AY, C2, AH, "PROBLEM STATEMENT  /  MOTIVATION", C_PROB, "!")
blist(LX + 0.22, b, C2 - 0.44, [
    "The WHO classifies burnout as an occupational phenomenon, yet organisations "
    "still detect it through quarterly self-report surveys \u2014 far too late.",
    "Communication carries the signal every single day, but commercial tools read "
    "only metadata \u2014 who mailed whom, never what was said.",
    "Machine learning returns a score with no reason. An unexplained score cannot "
    "be acted on, audited, or contested by the employee it describes.",
    "Language models explain fluently \u2014 and hallucinate, silently overruling the "
    "measured result.",
], size=7.7, dot=C_PROB, gap=0.09)
rbox(LX + 0.18, AY + 0.13, C2 - 0.36, 0.52, "#FBEEED", "#E4C3C0", 1.0, r=0.04, z=4)
para(LX + 0.32, AY + 0.56, C2 - 0.64, "THE GAP \u2014 detection and explanation are "
     "treated as ONE problem, so systems are either accurate but opaque, or "
     "fluent but unreliable.", 7.7, C_PROB, "bold")

b = card(RX, AY, C2, AH, "OBJECTIVES", C_OBJ, "\u2713")
_objs = [
    ("Reproducible preprocessing", "one clean_text contract across three corpora"),
    ("Sentiment and emotion per message", "RoBERTa classifier + 8-category NRC lexicon"),
    ("An auditable burnout score", "additive rule \u2014 every point traceable to a cause"),
    ("Explanation grounded in evidence", "vector retrieval with the source records attached"),
    ("The LLM cannot overrule the score", "risk band echoed verbatim, never regenerated"),
    ("An inspectable HR-facing interface", "evidence shown beside every answer"),
]
yy = b + 0.02
for i, (t, sub) in enumerate(_objs):
    rbox(RX + 0.20, yy - 0.215, 0.28, 0.215, C_OBJ, C_OBJ, 0, r=0.03, z=6)
    ax.text(RX + 0.34, yy - 0.108, str(i + 1), fontsize=7.3, color="#FFFFFF",
            fontweight="bold", ha="center", va="center", zorder=7, **F)
    para(RX + 0.57, yy + 0.005, C2 - 0.79, t, 8.1, INK, "bold")
    para(RX + 0.57, yy - 0.155, C2 - 0.79, sub, 7.0, MUTED)
    yy -= 0.280
rbox(RX + 0.18, AY + 0.10, C2 - 0.36, 0.30, "#EAF4F0", "#BBD9D0", 1.0, r=0.04, z=4)
ax.text(RX + C2 / 2, AY + 0.25, "ALL SIX OBJECTIVES COMPLETE", fontsize=8.6,
        color=C_OBJ, fontweight="bold", ha="center", va="center", zorder=6, **F)

# ================================================ ROW B — TECHNICAL DESIGN ===
BY, BH = 7.71, 2.85
card(LX, BY, FW, BH, "PROPOSED SOLUTION  /  TECHNICAL DESIGN", C_TECH, "\u25c6")

_stages = [
    ("1 \u00b7 DATA", C_OBJ, ["Enron corpus \u2014 517,401 emails",
                          "Burnout survey \u2014 22,750 rows",
                          "Corporate stress \u2014 2,000 rows",
                          "NRC lexicon \u2014 6,453 words",
                          "\u2192 one clean_text column"]),
    ("2 \u00b7 ML PIPELINE", C_PROT, ["RoBERTa sentiment + confidence",
                                  "NRC emotion \u2014 8 categories",
                                  "7 behavioural features",
                                  "Transparent additive scoring",
                                  "\u2192 Low / Medium / High band"]),
    ("3 \u00b7 RAG BACKEND", C_TECH, ["MiniLM 384-d embeddings",
                                  "ChromaDB, cosine + fingerprint",
                                  "Top-K retrieval with filters",
                                  "Evidence fenced as DATA",
                                  "FastAPI \u2014 7 endpoints"]),
    ("4 \u00b7 DASHBOARD", C_KEY, ["Streamlit + Plotly, 7 charts",
                               "Explainability panel",
                               "Search, filter, risk ranking",
                               "PDF export (ReportLab)",
                               "PDF / image upload + OCR"]),
]
sw, gap = 2.40, 0.42
sx0 = LX + 0.32
BOX_Y, BOX_H = BY + 1.12, 1.24
for i, (nm, col, lines) in enumerate(_stages):
    x = sx0 + i * (sw + gap)
    rbox(x, BOX_Y, sw, BOX_H, "#FFFFFF", col, 1.6, r=0.05, z=4)
    ax.add_patch(Rectangle((x, BOX_Y + BOX_H - 0.25), sw, 0.25, facecolor=col,
                           edgecolor="none", zorder=5))
    ax.text(x + sw / 2, BOX_Y + BOX_H - 0.125, nm, fontsize=8.5, color="#FFFFFF",
            fontweight="bold", ha="center", va="center", zorder=6, **F)
    yy = BOX_Y + BOX_H - 0.34
    for ln in lines:
        yy -= para(x + 0.13, yy, sw - 0.26, ln, 7.0) + 0.032
    if i < 3:
        arrow(x + sw + 0.06, BOX_Y + BOX_H / 2, x + sw + gap - 0.06,
              BOX_Y + BOX_H / 2, MUTED, 2.0)

cx0 = sx0 + sw + gap
cw_ = 2 * sw + gap
rbox(cx0, BY + 0.40, cw_, 0.54, "#FBF3D8", GOLD, 2.0, r=0.05, z=4)
ax.text(cx0 + cw_ / 2, BY + 0.78, "THE INTEGRATION CONTRACT", fontsize=9.0,
        color=GOLDD, fontweight="bold", ha="center", va="center", zorder=6, **F)
ax.text(cx0 + cw_ / 2, BY + 0.56, "outputs/burnout_scores.csv   \u00b7   20 columns   "
        "\u00b7   consumed READ-ONLY   \u00b7   risk_level is echoed, never regenerated",
        fontsize=7.2, color=INK, ha="center", va="center", zorder=6,
        family="DejaVu Sans Mono")
arrow(sx0 + sw * 1.5 + gap, BOX_Y - 0.02, sx0 + sw * 1.5 + gap, BY + 0.96, GOLDD, 2.0)
arrow(sx0 + sw * 2.5 + 2 * gap, BY + 0.96, sx0 + sw * 2.5 + 2 * gap, BOX_Y - 0.02,
      GOLDD, 2.0)
para(LX + FW / 2, BY + 0.32, FW - 0.80, "The whole system meets at ONE file. "
     "Because the risk band is echoed rather than recomputed, the language model "
     "is structurally incapable of changing it.", 7.7, MUTED, style="italic",
     ha="center")

# ============================================ ROW C — KEY FEATURES / INNOV. ==
CY, CH = 6.17, 1.40
card(LX, CY, FW, CH, "KEY FEATURES  /  INNOVATION", C_KEY, "\u2605")
_feat = [
    ("An architectural guarantee \u2014 not a prompt rule",
     "The risk band is copied from the CSV, never regenerated. No prompt can talk "
     "the model into upgrading or downgrading a person's risk."),
    ("Every answer ships with its evidence",
     "The exact records behind a conclusion sit next to it, so the reader checks "
     "the reasoning instead of trusting it. Coverage is 100%."),
    ("Runs anywhere, reads anything",
     "OpenAI, Gemini or a local Llama \u2014 switched by one line of .env. Upload a PDF "
     "or a screenshot of a chat; OCR scores and indexes it in seconds."),
]
fw = (FW - 0.40 - 2 * 0.24) / 3
for i, (t, s_) in enumerate(_feat):
    x = LX + 0.20 + i * (fw + 0.24)
    rbox(x, CY + 0.13, fw, 0.80, "#F6F1FA", "#DCCDE8", 1.0, r=0.045, z=4)
    para(x + 0.13, CY + 0.86, fw - 0.26, t, 8.0, C_KEY, "bold")
    para(x + 0.13, CY + 0.63, fw - 0.26, s_, 7.0)

# ================================== ROW D — PROTOTYPE PROGRESS / RESULTS =====
DY, DH = 4.03, 2.00
card(LX, DY, C2, DH, "PROTOTYPE  /  CURRENT PROGRESS", C_PROT, "\u25a3")
_stats = [("13,914", "lines of code"), ("45", "source files"),
          ("151", "tests passing"), ("7", "API endpoints"),
          ("17", "defects fixed"), ("100%", "objectives met")]
tw = (C2 - 0.44 - 2 * 0.13) / 3
for i, (v, k) in enumerate(_stats):
    x = LX + 0.22 + (i % 3) * (tw + 0.13)
    y = DY + 1.02 - (i // 3) * 0.50
    rbox(x, y, tw, 0.42, "#FBF4EA", "#E7D3BB", 1.0, r=0.04, z=4)
    ax.text(x + tw / 2, y + 0.27, v, fontsize=11.0, color=C_PROT,
            fontweight="bold", ha="center", va="center", zorder=6, **F)
    ax.text(x + tw / 2, y + 0.10, k, fontsize=6.7, color=MUTED, ha="center",
            va="center", zorder=6, **F)
rbox(LX + 0.20, DY + 0.13, C2 - 0.40, 0.34, "#FBF4EA", "#E7D3BB", 1.0, r=0.04, z=4)
para(LX + C2 / 2, DY + 0.42, C2 - 0.60, "WORKING END TO END \u2014 pipeline, backend, "
     "dashboard, OCR ingestion and the quantum study all run today, with zero "
     "lines of upstream code modified.", 7.1, C_PROT, "bold", ha="center")

b = card(RX, DY, C2, DH, "RESULTS  /  EVIDENCE", C_RES, "\u25a4")
_res = [
    ("Explainability coverage", "100%", "no record can exist without a reason"),
    ("Best classical model", "86.7%", "Logistic Regression on the fused features"),
    ("Quantum kernel SVM", "85.6%", "exactly reproduces the classical RBF result"),
    ("Compute overhead of quantum", "15,000\u00d7", "measured, not estimated \u2014 "
     "PennyLane and Qiskit agree to within 0.27%"),
]
yy = b + 0.04
for _i, (nm, val, sub) in enumerate(_res):
    para(RX + 0.22, yy, C2 - 1.30, nm, 8.0, INK, "bold")
    ax.text(RX + C2 - 0.22, yy, val, fontsize=9.4, color=C_RES,
            fontweight="bold", ha="right", va="top", zorder=6, **F)
    h_ = para(RX + 0.22, yy - 0.165, C2 - 0.50, sub, 7.0, MUTED)
    yy -= 0.175 + h_ + 0.07
    if _i < len(_res) - 1:
        ax.add_patch(Rectangle((RX + 0.22, yy + 0.045), C2 - 0.44, 0.008,
                               facecolor=LINE, edgecolor="none", zorder=5))

# ============================ ROW E — SOCIAL / ENVIRONMENTAL IMPACT ==========
EY, EH = 1.49, 2.40
card(LX, EY, FW, EH, "SOCIAL  /  ENVIRONMENTAL IMPACT", C_SOC, "\u25c8")
hw = (FW - 0.40 - 0.24) / 2
_panels = [
    ("SOCIAL IMPACT", C_SOC, "#EAF4F0", "#BBD9D0", [
        "Support reaches people before disengagement, not months afterwards.",
        "Anti-surveillance by design: pseudonymous IDs, aggregate insight, and "
        "evidence visible to the person described \u2014 not only to HR.",
        "Contestable \u2014 an employee can see and dispute the exact messages behind "
        "a flag. An unexplained score cannot be argued with.",
        "Never a diagnosis: the confidence note is a mandatory output section and "
        "is placed before the recommendations, not beneath them.",
        "Removes the stigma of self-reporting \u2014 nobody has to raise their hand in "
        "order to be noticed.",
    ]),
    ("ENVIRONMENTAL IMPACT", C_ENV, "#EDF4E9", "#C6DAB9", [
        "A 22M-parameter MiniLM encoder does the retrieval, not a large model \u2014 "
        "inference runs on a laptop CPU.",
        "Fingerprinted index: the vector store rebuilds only when the data "
        "actually changes, never on every restart.",
        "The contract means downstream stages never re-run the 2.1 GB pipeline \u2014 "
        "the expensive step happens exactly once.",
        "A local Ollama option removes data-centre round trips entirely.",
        "Measured, not assumed \u2014 our benchmark quantifies a 15,000\u00d7 compute "
        "overhead for quantum kernels, and we report it as a reason NOT to "
        "deploy them.",
    ]),
]
for i, (hdr, col, bg, bd, items) in enumerate(_panels):
    x = LX + 0.20 + i * (hw + 0.24)
    rbox(x, EY + 0.13, hw, 1.72, bg, bd, 1.0, r=0.045, z=4)
    ax.text(x + 0.14, EY + 1.81, hdr, fontsize=8.5, color=col,
            fontweight="bold", va="top", zorder=6, **F)
    blist(x + 0.14, EY + 1.58, hw - 0.28, items, size=7.0, dot=col, gap=0.055)

# ================================ ROW F — FUTURE SCOPE / EXPECTED OUTCOME ====
FY, FH = 0.35, 1.00
card(LX, FY, FW, FH, "FUTURE SCOPE  /  EXPECTED OUTCOME", C_FUT, "\u279c")
_fut = [
    ("VALIDATE", "Correlate risk bands with the Maslach Burnout Inventory"),
    ("MODEL TIME", "Track trajectory over weeks, not single messages"),
    ("DEPLOY", "Containerised, role-based access, GPU-backed index"),
    ("PUBLISH", "Submit the IEEE paper after empirical evaluation"),
]
aw = (FW - 0.40) / 4
for i, (t, s_) in enumerate(_fut):
    x = LX + 0.20 + i * aw
    y0, y1 = FY + 0.08, FY + 0.52
    tip, notch = 0.16, 0.16
    pts = [(x, y0), (x + aw - 0.09 - tip, y0), (x + aw - 0.09, (y0 + y1) / 2),
           (x + aw - 0.09 - tip, y1), (x, y1)]
    if i > 0:
        pts.append((x + notch, (y0 + y1) / 2))
    ax.add_patch(Polygon(pts, closed=True, facecolor="#F7EDF2",
                         edgecolor="#DCC0CE", linewidth=1.0, zorder=4))
    ax.text(x + 0.24, y1 - 0.055, f"{i + 1}.  {t}", fontsize=7.8, color=C_FUT,
            fontweight="bold", va="top", zorder=6, **F)
    para(x + 0.24, y1 - 0.21, aw - 0.44, s_, 6.5, BODY, lead=1.22)

# ================================================================= FOOTER ===
ax.add_patch(Rectangle((0, 0), W, 0.30, facecolor=NAVY, edgecolor="none", zorder=3))
ax.text(0.40, 0.15, "Under the guidance of Dr. Shubha Puthran   \u00b7   Computer "
        "Engineering Department, MPSTME", fontsize=7.2, color="#B9CADB",
        va="center", zorder=6, **F)
ax.text(11.29, 0.15, "SPARC Research-to-Prototype Challenge 2026", fontsize=7.2,
        color=GOLD, fontweight="bold", ha="right", va="center", zorder=6, **F)

fig.savefig("SPARC_Infographic.png", dpi=200, facecolor=PAPER)
fig.savefig("SPARC_Infographic.pdf", facecolor=PAPER)
plt.close(fig)
print("SPARC_Infographic.png / .pdf")
