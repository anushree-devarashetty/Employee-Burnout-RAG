"""Architecture and timeline diagrams for the 2nd review deck."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

INK   = "#12212F"
MUTED = "#5B6B7B"
P1    = "#2E7D6B"   # data
P2    = "#B4762A"   # ML
P3    = "#2C5F96"   # RAG backend
P4    = "#7A4B96"   # dashboard
P5    = "#8B3A5E"   # quantum
GOLD  = "#C9A227"   # the contract
LINE  = "#C3CEDA"

FONT = {"family": "DejaVu Sans"}


def box(ax, x, y, w, h, title, lines, colour, fs=8.2, tfs=9.4, alpha=0.10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.05",
        linewidth=1.5, edgecolor=colour, facecolor=colour, alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.05",
        linewidth=1.5, edgecolor=colour, facecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=colour, zorder=4, **FONT)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.46 - i * 0.215, ln, ha="center", va="top",
                fontsize=fs, color=INK, zorder=4, **FONT)


def arrow(ax, x1, y1, x2, y2, colour=MUTED, style="-|>", lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=colour, linestyle=ls, zorder=5,
        connectionstyle=f"arc3,rad={rad}"))


def band(ax, y, h, label, colour, x0=0.15, x1=15.85):
    ax.add_patch(Rectangle((x0, y), x1 - x0, h, facecolor=colour, alpha=0.035,
                           edgecolor="none", zorder=0))
    ax.text(x0 + 0.10, y + h - 0.13, label, fontsize=8.6, fontweight="bold",
            color=colour, alpha=0.95, va="top", zorder=1, **FONT)


# ==========================================================================
# 1. SYSTEM ARCHITECTURE
# ==========================================================================
fig, ax = plt.subplots(figsize=(16, 11.85))
ax.set_xlim(0, 16); ax.set_ylim(0, 11.85); ax.axis("off")

def band2(ax, ybot, ytop, label, colour, x0=0.15, x1=15.85):
    ax.add_patch(Rectangle((x0, ybot), x1 - x0, ytop - ybot, facecolor=colour,
                           alpha=0.035, edgecolor="none", zorder=0))

def blabel(ax, x, y, text, colour):
    ax.text(x, y, text, fontsize=9.0, fontweight="bold", color=colour,
            va="bottom", ha="left", zorder=1, **FONT)

ax.text(8, 11.56, "Explainable Employee Burnout Detection \u2014 System Architecture",
        ha="center", fontsize=17, fontweight="bold", color=INK, **FONT)
ax.text(8, 11.24,
        "Four sequential stages joined by ONE narrow contract, plus two independent extensions",
        ha="center", fontsize=10.2, color=MUTED, style="italic", **FONT)

# ---------- Stage 1 : data ----------
band2(ax, 9.72, 11.06, "", P1)
blabel(ax, 0.30, 10.84, "STAGE 1  \u00b7  DATA SOURCES   (Person 1)", P1)
box(ax, 0.45, 9.86, 3.5, 0.92, "Enron Email Corpus",
    ["517,401 emails \u00b7 2.1 GB", "file \u00b7 message \u00b7 clean_text"], P1)
box(ax, 4.25, 9.86, 3.5, 0.92, "Employee Burnout Analysis",
    ["22,750 survey responses", "Burn Rate = ground truth"], P1)
box(ax, 8.05, 9.86, 3.5, 0.92, "Indian Corporate Stress",
    ["2,000 rows \u00d7 30 columns", "Stress_Level \u00b7 Burnout_Symptoms"], P1)
box(ax, 11.85, 9.86, 3.7, 0.92, "NRC Emotion Lexicon",
    ["6,453 associated words", "8 emotion categories"], P1)
for x in (2.20, 6.00, 9.80):
    arrow(ax, x, 9.84, x, 9.54, P1)
ax.text(8, 9.38, "P R E P R O C E S S I N G      \u2192      single   clean_text   column",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color=P1, **FONT)
arrow(ax, 8, 9.22, 8, 8.92, P1, lw=2.0)

# ---------- Stage 2 : ML ----------
band2(ax, 7.48, 9.06, "", P2)
blabel(ax, 0.30, 8.83, "STAGE 2  \u00b7  MACHINE LEARNING PIPELINE   (Person 2)", P2)
box(ax, 0.45, 7.62, 3.5, 1.15, "1 \u00b7 Sentiment",
    ["RoBERTa (cardiffnlp)", "Positive / Neutral / Negative", "+ confidence 0\u20131"], P2)
box(ax, 4.25, 7.62, 3.5, 1.15, "2 \u00b7 Emotion",
    ["NRC lexicon lookup", "8 categories, word counts", "inspectable, not inferred"], P2)
box(ax, 8.05, 7.62, 3.5, 1.15, "3 \u00b7 Behaviour",
    ["7 features: length, caps,", "punctuation, sentences", "HOW they write, not what"], P2)
box(ax, 11.85, 7.62, 3.7, 1.15, "4 \u00b7 Burnout Score",
    ["TRANSPARENT additive rule", "Neg +2 \u00b7 anger/fear/sadness +1", "\u22642 Low \u00b7 \u22645 Medium \u00b7 >5 High"], P2)
for x in (3.95, 7.75, 11.55):
    arrow(ax, x, 8.20, x + 0.30, 8.20, P2)
arrow(ax, 13.70, 7.60, 12.05, 6.92, P2, lw=2.0, rad=-0.20)

# ---------- THE CONTRACT ----------
for lw_, fc, al in ((2.6, GOLD, 0.13), (2.6, "none", 1.0)):
    ax.add_patch(FancyBboxPatch((4.05, 6.50), 7.9, 0.72,
        boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=lw_,
        edgecolor=GOLD, facecolor=fc, alpha=al, zorder=3))
ax.text(8.0, 7.03, "THE INTEGRATION CONTRACT", ha="center", va="top",
        fontsize=10.4, fontweight="bold", color="#8A6D12", zorder=5, **FONT)
ax.text(8.0, 6.76,
        "outputs/burnout_scores.csv    \u00b7    20 columns    \u00b7    consumed READ-ONLY",
        ha="center", va="top", fontsize=9.4, color=INK, zorder=5,
        family="DejaVu Sans Mono")
for x, c in ((6.4, P3), (9.6, P4)):
    arrow(ax, x, 6.48, x, 6.12, c, lw=2.0)
arrow(ax, 11.95, 6.62, 14.35, 5.72, P5, lw=1.6, ls=(0, (4, 3)), rad=-0.15)

# ---------- Stage 3 : RAG backend ----------
band2(ax, 3.74, 6.34, "", P3, x1=14.05)
blabel(ax, 0.30, 6.08, "STAGE 3  \u00b7  RAG BACKEND   (Person 3)   \u2014   FastAPI, 7 endpoints", P3)
for i, (t, l) in enumerate([
        ("Embeddings", ["all-MiniLM-L6-v2", "384-dim, L2-normalised"]),
        ("Vector Store", ["ChromaDB, persistent", "cosine + fingerprint"]),
        ("Retriever", ["Top-K + metadata filters", "employee ID beats similarity"]),
        ("Grounded Prompt", ["evidence fenced as DATA", "no diagnosis, no re-scoring"]),
        ("LLM Client", ["OpenAI \u00b7 Gemini \u00b7 Ollama", "switch via .env only"])]):
    box(ax, 0.45 + i * 2.75, 5.02, 2.55, 1.00, t, l, P3, fs=7.9, tfs=9.0)
for i in range(4):
    arrow(ax, 3.00 + i * 2.75, 5.52, 3.20 + i * 2.75, 5.52, P3, lw=1.4)
box(ax, 0.45, 3.88, 6.30, 1.00, "Document Ingestion  \u00b7  POST /ingest",
    ["PDF (pdfplumber) + image OCR (Tesseract) \u2192 4-strategy splitter",
     "scored by Person 2's OWN functions, lifted via ast \u2014 file untouched"],
    P3, fs=7.9, tfs=9.0)
box(ax, 7.05, 3.88, 6.95, 1.00, "Structured Output  \u00b7  8 fixed keys",
    ["summary \u00b7 emotional analysis \u00b7 causes \u00b7 risk level \u00b7 interventions",
     "HR recommendations \u00b7 wellness \u00b7 MANDATORY confidence note"],
    P3, fs=7.9, tfs=9.0)
arrow(ax, 12.72, 5.00, 12.72, 4.90, P3, lw=1.4)
arrow(ax, 4.40, 4.90, 4.40, 5.00, P3, lw=1.4)

# ---------- Stage 4 : dashboard ----------
band2(ax, 2.30, 3.70, "", P4, x1=14.05)
blabel(ax, 0.30, 3.50, "STAGE 4  \u00b7  DASHBOARD   (Person 4)   \u2014   Streamlit", P4)
box(ax, 0.45, 2.44, 4.4, 1.00, "Interactive Charts",
    ["7 Plotly views \u00b7 gauge, pie, bars,", "histogram, scatter \u00b7 search & filter"], P4, fs=7.9, tfs=9.0)
box(ax, 5.10, 2.44, 4.4, 1.00, "Explainability Panel",
    ["every answer displayed WITH the", "records it was grounded in"], P4, fs=7.9, tfs=9.0)
box(ax, 9.75, 2.44, 4.25, 1.00, "ReportLab PDF Export",
    ["confidence note placed BEFORE", "the recommendations"], P4, fs=7.9, tfs=9.0)
arrow(ax, 10.5, 3.86, 10.5, 3.48, P4, lw=1.6)

# ---------- Quantum column ----------
for fc, al in ((P5, 0.09), ("none", 1.0)):
    ax.add_patch(FancyBboxPatch((14.20, 2.44), 1.45, 3.58,
        boxstyle="round,pad=0.012,rounding_size=0.05", linewidth=1.5,
        edgecolor=P5, facecolor=fc, alpha=al, zorder=2))
for i, t in enumerate([
        "QUANTUM ML", "COMPARATIVE", "STUDY", "", "PennyLane", "and Qiskit", "",
        "Variational", "classifier", "", "Quantum", "kernel SVM", "",
        "vs 4 classical", "baselines", "", "6 features \u2192 6 qubits"]):
    ax.text(14.92, 5.80 - i * 0.192, t, ha="center", va="top",
            fontsize=8.0 if i > 2 else 8.8,
            fontweight="bold" if i < 3 else "normal",
            color=P5 if i < 3 else INK, zorder=4, **FONT)

# ---------- key principle ----------
for fc, al in ((INK, 0.055), ("none", 1.0)):
    ax.add_patch(FancyBboxPatch((0.45, 1.10), 15.1, 0.92,
        boxstyle="round,pad=0.015,rounding_size=0.05", linewidth=1.6,
        edgecolor=INK, facecolor=fc, alpha=al, zorder=2))
ax.text(8.0, 1.86, "THE ARCHITECTURAL CONTRIBUTION", ha="center", va="top",
        fontsize=9.8, fontweight="bold", color=INK, **FONT)
ax.text(8.0, 1.59,
        "Prediction and explanation are treated as TWO problems with different correctness criteria \u2014 "
        "the ML owns the numbers, the language model owns the words.",
        ha="center", va="top", fontsize=9.3, color=INK, **FONT)
ax.text(8.0, 1.34,
        "risk_level is echoed verbatim and can never be overruled by the more fluent component   \u00b7   "
        "every layer degrades rather than fails",
        ha="center", va="top", fontsize=8.7, color=MUTED, style="italic", **FONT)

fig.savefig("system_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.18)
plt.close(fig)
print("system_architecture.png")


# ==========================================================================
# 2. TIMELINE
# ==========================================================================
fig, ax = plt.subplots(figsize=(16, 8.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 8.6); ax.axis("off")

ax.text(8, 8.34, "Project Timeline — Topic Approval to Second Review",
        ha="center", fontsize=16.5, fontweight="bold", color=INK, **FONT)
ax.text(8, 8.04, "Work completed to date, by workstream",
        ha="center", fontsize=10, color=MUTED, style="italic", **FONT)

WEEKS = ["Jul\nW1", "Jul\nW2", "Jul\nW3", "Jul\nW4",
         "Aug\nW1", "Aug\nW2", "Aug\nW3", "Aug\nW4"]
X0, X1 = 3.30, 15.30
step = (X1 - X0) / len(WEEKS)

for i, w in enumerate(WEEKS):
    ax.text(X0 + step * (i + 0.5), 7.62, w, ha="center", va="center",
            fontsize=8.6, color=MUTED, **FONT)
    ax.plot([X0 + step * i] * 2, [0.95, 7.42], color=LINE, lw=0.8,
            ls=(0, (2, 3)), zorder=0)
ax.plot([X1, X1], [0.95, 7.42], color=LINE, lw=0.8, ls=(0, (2, 3)), zorder=0)

TASKS = [
    ("Topic approval presentation",            0, 1, P1, "done"),
    ("Dataset collection — 3 corpora",          0, 2, P1, "done"),
    ("Preprocessing → clean_text contract",     1, 3, P1, "done"),
    ("Literature review — 14 references",       1, 4, P1, "done"),
    ("RoBERTa sentiment analysis",              2, 4, P2, "done"),
    ("NRC emotion + behavioural features",      3, 5, P2, "done"),
    ("Feature fusion + burnout scoring",        4, 5, P2, "done"),
    ("RAG backend — 14 modules, FastAPI",       4, 7, P3, "done"),
    ("Embeddings + ChromaDB vector store",      4, 6, P3, "done"),
    ("Multi-provider LLM client",               5, 6, P3, "done"),
    ("PDF / image ingestion with OCR",          6, 7, P3, "done"),
    ("Streamlit dashboard + 7 charts",          5, 7, P4, "done"),
    ("ReportLab PDF export",                    6, 7, P4, "done"),
    ("Quantum ML comparative study",            6, 8, P5, "done"),
    ("151 automated tests + verification",      5, 8, INK, "done"),
    ("Interim report (IEEE format)",            6, 8, INK, "done"),
]

y = 7.22
for label, a, b, colour, _ in TASKS:
    xa, xb = X0 + step * a, X0 + step * b
    ax.add_patch(FancyBboxPatch(
        (xa + 0.03, y - 0.135), (xb - xa) - 0.06, 0.27,
        boxstyle="round,pad=0.004,rounding_size=0.03",
        linewidth=1.1, edgecolor=colour, facecolor=colour, alpha=0.30, zorder=3))
    ax.text(3.16, y, label, ha="right", va="center", fontsize=8.5,
            color=INK, **FONT)
    ax.text(xb - 0.13, y, "✓", ha="right", va="center", fontsize=8.6,
            fontweight="bold", color=colour, zorder=5, **FONT)
    y -= 0.385

# milestones
ax.plot([X0, X1], [1.02, 1.02], color=MUTED, lw=1.6, zorder=4)
for x, txt, col in ((X0, "TOPIC\nAPPROVED", P1), (X1, "SECOND\nREVIEW", "#B03434")):
    ax.plot([x], [1.02], marker="D", markersize=9, color=col, zorder=6)
    ax.text(x, 0.86, txt, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=col, **FONT)
ax.text((X0 + X1) / 2, 0.86, "8 WEEKS  \u00b7  16 WORKSTREAMS  \u00b7  ALL COMPLETE",
        ha="center", va="top", fontsize=9.0, fontweight="bold", color=INK, **FONT)

legend = [("Person 1 \u2014 Data", P1), ("Person 2 \u2014 ML", P2),
          ("Person 3 \u2014 RAG backend", P3), ("Person 4 \u2014 Dashboard", P4),
          ("Quantum study", P5), ("Team-wide", INK)]
for i, (lab, col) in enumerate(legend):
    x = 0.55 + i * 2.62
    ax.add_patch(Rectangle((x, 0.16), 0.22, 0.14, facecolor=col, alpha=0.55,
                           edgecolor=col, lw=1.0))
    ax.text(x + 0.30, 0.23, lab, va="center", fontsize=8.2, color=INK, **FONT)

fig.savefig("timeline.png", dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.16)
plt.close(fig)
print("timeline.png")
