"""Second Review deck, built on the approved Topic Approval template."""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SRC = "/sessions/sweet-zealous-brown/mnt/uploads/Topic Approval Presentation.pptx"
OUT = "Second_Review_Presentation.pptx"

NAVY  = RGBColor(0x1B, 0x3A, 0x5C)
INK   = RGBColor(0x1A, 0x1A, 0x1A)
ACC   = RGBColor(0x8A, 0x6D, 0x12)
GREY  = RGBColor(0x55, 0x5F, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BANDF = RGBColor(0xE8, 0xEE, 0xF5)

SERIF = "Times New Roman"

T_L, T_T, T_W, T_H = Inches(0.50), Inches(1.54), Inches(9.00), Inches(0.56)
B_L, B_T, B_W, B_H = Inches(0.62), Inches(2.12), Inches(8.86), Inches(4.80)

prs = Presentation(SRC)
LAYOUT_TITLE = prs.slide_layouts[0]
LAYOUT_BODY = prs.slide_layouts[6]

# ---- strip every slide except the title slide -------------------------------
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:]:
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)


def _clear(slide, keep_title=False):
    for sh in list(slide.shapes):
        if keep_title and sh == slide.shapes.title:
            continue
        sh._element.getparent().remove(sh._element)


def new_slide(title, rubric=None):
    s = prs.slides.add_slide(LAYOUT_BODY)
    _clear(s)
    tb = s.shapes.add_textbox(T_L, T_T, T_W, T_H)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.size = Pt(22); r.font.bold = True
    r.font.name = SERIF; r.font.color.rgb = NAVY
    # rule under the title
    from pptx.enum.shapes import MSO_SHAPE
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, T_L, Inches(2.02), T_W, Pt(2.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACC
    ln.line.fill.background(); ln.shadow.inherit = False
    if rubric:
        tag = s.shapes.add_textbox(Inches(0.52), Inches(7.03), Inches(5.20), Inches(0.28))
        tp = tag.text_frame.paragraphs[0]
        tp.alignment = PP_ALIGN.LEFT
        rr = tp.add_run(); rr.text = rubric
        rr.font.size = Pt(10.5); rr.font.bold = True
        rr.font.name = SERIF; rr.font.color.rgb = ACC
    return s


def bullets(slide, items, size=16.5, top=None, height=None, lead=Pt(7)):
    tb = slide.shapes.add_textbox(B_L, top or B_T, B_W, height or B_H)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        lvl, txt = it if isinstance(it, tuple) else (0, it)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = min(lvl, 4)
        p.space_after = lead
        p.space_before = Pt(2) if lvl == 0 else Pt(0)
        bold = txt.startswith("**")
        txt = txt.replace("**", "")
        if lvl == 0:
            mark = "▪  "
        elif lvl == 1:
            mark = "     –  "
        else:
            mark = "          ·  "
        r = p.add_run()
        r.text = mark + txt
        r.font.size = Pt(size - lvl * 1.3)
        r.font.name = SERIF
        r.font.bold = bold
        r.font.color.rgb = INK if lvl == 0 else GREY
    return tb


def table(slide, headers, rows, widths, top=Inches(2.16), size=12.5, hsize=12.5,
          row_h=Inches(0.34)):
    total = sum(widths)
    left = Inches(0.62) if total > 8.4 else Inches((10 - total) / 2)
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   left, top, Inches(total),
                                   row_h * (len(rows) + 1))
    tbl = shape.table
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = ""
        p = c.text_frame.paragraphs[0]
        r = p.add_run(); r.text = h
        r.font.size = Pt(hsize); r.font.bold = True
        r.font.name = SERIF; r.font.color.rgb = WHITE
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
        c.margin_left = c.margin_right = Inches(0.06)
        c.margin_top = c.margin_bottom = Inches(0.03)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            p.word_wrap = True
            bold = str(val).startswith("**")
            r = p.add_run(); r.text = str(val).replace("**", "")
            r.font.size = Pt(size); r.font.name = SERIF
            r.font.bold = bold
            r.font.color.rgb = INK
            c.fill.solid()
            c.fill.fore_color.rgb = BANDF if i % 2 else WHITE
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.025)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shape


def note(slide, text, top=Inches(6.50), size=12, colour=ACC, italic=True):
    tb = slide.shapes.add_textbox(Inches(0.62), top, Inches(8.86), Inches(0.55))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.italic = italic; r.font.bold = True
    r.font.name = SERIF; r.font.color.rgb = colour
    return tb


def picture(slide, path, top=Inches(2.12), max_h=5.30, max_w=9.00):
    from PIL import Image
    w, h = Image.open(path).size
    ar = w / h
    pw, ph = max_w, max_w / ar
    if ph > max_h:
        ph, pw = max_h, max_h * ar
    slide.shapes.add_picture(path, Inches((10 - pw) / 2), top,
                             Inches(pw), Inches(ph))


def speaker(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# =============================================================================
# 1 · TITLE
# =============================================================================
s1 = prs.slides[0]
shapes = list(s1.shapes)
for sh in shapes[1:]:
    sh._element.getparent().remove(sh._element)
shapes[0].left, shapes[0].top = Inches(0.62), Inches(1.56)
shapes[0].width, shapes[0].height = Inches(8.76), Inches(2.34)
tf = shapes[0].text_frame
tf.clear()
for i, (txt, sz, bold) in enumerate([
        ("Capstone Project", 22, True),
        ("Second Review Presentation", 22, True),
        ("A.Y. 2026-2027", 18, False),
        ("", 8, False),
        ("Project Title:", 18, True),
        ("Explainable Employee Burnout Detection from Workplace "
         "Communication using Sentiment Analysis and Retrieval-Augmented "
         "Generation", 19, True)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt
    r.font.size = Pt(sz); r.font.bold = bold
    r.font.name = SERIF; r.font.color.rgb = NAVY if bold else INK

# presenter block, rebuilt as an aligned borderless table
tb = s1.shapes.add_textbox(Inches(0.62), Inches(3.96), Inches(8.76), Inches(0.32))
p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "Presented by"
r.font.size = Pt(15); r.font.bold = True
r.font.name = SERIF; r.font.color.rgb = NAVY

_team = [("Saumya Shah", "70322100115", "C040"),
         ("Swara Wairkar", "70322100201", "C048"),
         ("Aryaman Giri", "70322100043", "C026"),
         ("Anushree Devarashetty", "7032100175", "C038")]
_tb = s1.shapes.add_table(3, 4, Inches(0.55), Inches(4.34),
                          Inches(8.90), Inches(0.90)).table
for j in range(4):
    _tb.columns[j].width = Inches(2.225)
for i in range(3):
    for j, member in enumerate(_team):
        c = _tb.cell(i, j)
        c.text = ""
        pp = c.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
        rr = pp.add_run(); rr.text = member[i]
        rr.font.size = Pt(12.5 if i == 0 else 11.5)
        rr.font.bold = (i == 0)
        rr.font.name = SERIF
        rr.font.color.rgb = INK if i == 0 else GREY
        c.fill.background()
        c.margin_left = c.margin_right = Inches(0.02)
        c.margin_top = c.margin_bottom = Inches(0.01)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        for ln_tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
            from pptx.oxml.ns import qn
            e = c._tc.get_or_add_tcPr().makeelement(qn(ln_tag), {})
            e.append(e.makeelement(qn("a:noFill"), {}))
            c._tc.get_or_add_tcPr().append(e)

tb = s1.shapes.add_textbox(Inches(0.62), Inches(5.42), Inches(8.76), Inches(0.80))
tf2 = tb.text_frame; tf2.word_wrap = True
for i, (txt, sz, bold, col) in enumerate([
        ("Under the guidance of : Dr. Shubha Puthran", 16, True, NAVY),
        ("Computer Engineering Department, MPSTME, Mumbai Campus", 12, False,
         GREY)]):
    pp = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
    pp.alignment = PP_ALIGN.CENTER
    rr = pp.add_run(); rr.text = txt
    rr.font.size = Pt(sz); rr.font.bold = bold
    rr.font.name = SERIF; rr.font.color.rgb = col

speaker(s1, "Good morning. We are presenting the second review of our capstone "
            "project on explainable employee burnout detection.")

# =============================================================================
# 2 · RUBRIC MAP
# =============================================================================
s = new_slide("Agenda — Mapped to the Review Rubric")
table(s, ["#", "Evaluation criterion", "Addressed on slides", "Evidence we present"],
      [["A", "**Literature Review / Market Survey**", "6 – 9",
        "14 peer-reviewed references; tool comparison"],
       ["B", "**Clarity of problem statement**", "4 – 5",
        "Gap statement + 6 measurable objectives"],
       ["C", "**Design / System level representation**", "10 – 13",
        "Full architecture; the integration contract"],
       ["D", "**Implementation / Results**", "14 – 19",
        "13,900 lines of code; working system; benchmark"],
       ["E", "**Interim Report**", "20 – 21",
        "IEEE-format draft complete; timeline on track"]],
      [0.42, 3.10, 1.55, 3.79], top=Inches(2.20), row_h=Inches(0.62), size=12)
note(s, "Every slide in this deck is tagged in the bottom-left corner with the "
        "rubric criterion it answers.", top=Inches(6.05))
speaker(s, "Rather than make the panel hunt for evidence, we have mapped the deck "
           "directly onto the five evaluation criteria. Each slide carries a tag "
           "in the top right showing which criterion it answers.")

# =============================================================================
# 3 · PROGRESS SINCE TOPIC APPROVAL
# =============================================================================
s = new_slide("Progress Since Topic Approval")
table(s, ["Committed at topic approval", "Status", "Delivered"],
      [["Acquire and clean workplace communication data", "**Complete**",
        "3 corpora, 542,151 records, unified schema"],
       ["Sentiment and emotion analysis pipeline", "**Complete**",
        "RoBERTa + NRC lexicon, 20-column output"],
       ["Transparent burnout scoring", "**Complete**",
        "Additive rule, 3 risk bands, per-record reason"],
       ["Retrieval-Augmented Generation backend", "**Complete**",
        "14 modules, FastAPI, 7 endpoints"],
       ["Interactive dashboard", "**Complete**",
        "Streamlit, 7 charts, PDF export"],
       ["— (added after approval)", "**Bonus**",
        "Document/image ingestion with OCR"],
       ["— (added after approval)", "**Bonus**",
        "Quantum ML comparative study — PennyLane + Qiskit"]],
      [3.35, 1.05, 4.46], top=Inches(2.16), row_h=Inches(0.50), size=11.5)
note(s, "All approved deliverables are complete. Two additional workstreams were "
        "undertaken beyond the approved scope.", top=Inches(6.26))
speaker(s, "Everything we committed to at topic approval is finished, and we have "
           "added two workstreams beyond the approved scope: document ingestion "
           "with OCR, and a quantum machine-learning comparative study.")

# =============================================================================
# 4 · PROBLEM STATEMENT  (B)
# =============================================================================
s = new_slide("Problem Statement", "RUBRIC B · PROBLEM STATEMENT")
bullets(s, [
    "**Burnout is an occupational phenomenon recognised by the WHO, yet it is "
    "detected late.**",
    (1, "Organisations rely on periodic self-report surveys — low frequency, "
        "low response rates, and vulnerable to under-reporting."),
    (1, "By the time a survey registers distress, the disengagement has usually "
        "already happened."),
    "**Workplace communication carries the signal continuously, but is not used.**",
    (1, "Email and chat are generated daily and reflect emotional state as it "
        "changes, not months later."),
    "**Existing machine-learning approaches produce a number, not a reason.**",
    (1, "A score of 7 tells an HR partner nothing about what to do, who to talk "
        "to, or why the employee was flagged."),
    (1, "An unexplained score is not actionable, and cannot be audited or "
        "contested by the person it describes."),
    "**Language models can explain fluently — but they also hallucinate.**",
    (1, "Left unconstrained, a language model will invent evidence and silently "
        "overrule the measured result."),
], size=15.5)
note(s, "The gap: detection and explanation are treated as one problem, so systems "
        "are either accurate but opaque, or fluent but unreliable.",
     top=Inches(6.54), size=12.5)
speaker(s, "Burnout is recognised by the WHO as an occupational phenomenon, but "
           "organisations detect it late because they rely on periodic surveys. "
           "Workplace communication carries the signal continuously. The problem "
           "is that existing ML approaches give you a number with no reason, and "
           "language models give you a fluent reason that may be invented.")

# =============================================================================
# 5 · AIM AND OBJECTIVES  (B)
# =============================================================================
s = new_slide("Aim and Objectives", "RUBRIC B · PROBLEM STATEMENT")
bullets(s, [
    "**Aim** — to build a system that detects burnout risk from workplace "
    "communication and explains every decision using only the evidence that "
    "produced it.",
], size=16, height=Inches(0.85))
table(s, ["#", "Objective", "Measurable outcome", "Status"],
      [["1", "Build a reproducible preprocessing pipeline",
        "Single clean_text contract across 3 corpora", "**Done**"],
       ["2", "Classify sentiment and emotion per message",
        "RoBERTa + 8-category NRC lexicon", "**Done**"],
       ["3", "Produce an auditable burnout score",
        "Additive rule; every point traceable", "**Done**"],
       ["4", "Ground explanation in retrieved evidence",
        "Vector retrieval + fenced-evidence prompt", "**Done**"],
       ["5", "Prevent the LLM overruling the measurement",
        "risk_level echoed verbatim, never regenerated", "**Done**"],
       ["6", "Deliver an inspectable HR-facing interface",
        "Dashboard + PDF, evidence shown with answer", "**Done**"]],
      [0.34, 3.35, 3.85, 0.78], top=Inches(3.02), row_h=Inches(0.46), size=11.5)
speaker(s, "Our aim is one sentence: detect burnout risk from communication, and "
           "explain every decision using only the evidence that produced it. We "
           "broke that into six objectives, each with a measurable outcome. All "
           "six are complete.")

# =============================================================================
# 6 · LITERATURE REVIEW I  (A)
# =============================================================================
s = new_slide("Literature Review I — Burnout and Language Models",
              "RUBRIC A · LITERATURE REVIEW")
table(s, ["Ref", "Work", "Contribution", "What we take from it"],
      [["[1]", "WHO (ICD-11)", "Burnout classified as an occupational phenomenon",
        "Defines responsible-use boundary: not a diagnosis"],
       ["[2]", "Maslach & Jackson", "Maslach Burnout Inventory — three-part schema",
        "Exhaustion / cynicism / efficacy framing"],
       ["[3]", "Xames et al.", "ML for burnout prediction from HR and survey data",
        "Confirms the field's reliance on self-report"],
       ["[4]", "Vaswani et al.", "Transformer architecture",
        "Basis of the sentiment encoder"],
       ["[5]", "Devlin et al.", "BERT — bidirectional pre-training",
        "Contextual representation of workplace text"],
       ["[6]", "Liu et al.", "RoBERTa — refined pre-training",
        "Backbone of our classifier"],
       ["[7]", "Barbieri et al.", "TweetEval — sentiment fine-tuning",
        "The exact checkpoint we deploy"],
       ["[8]", "Mohammad & Turney", "NRC Word–Emotion Association Lexicon",
        "Inspectable emotion counts, not activations"]],
      [0.45, 1.65, 3.15, 3.60], top=Inches(2.16), row_h=Inches(0.53), size=10.5,
      hsize=11)
speaker(s, "Our literature review covers three threads. The first is burnout "
           "measurement itself — the WHO classification and the Maslach "
           "inventory. The key finding is that the ML literature overwhelmingly "
           "uses self-report and HR data rather than natural language. The second "
           "thread is the transformer line that gives us our sentiment classifier.")

# =============================================================================
# 7 · LITERATURE REVIEW II  (A)
# =============================================================================
s = new_slide("Literature Review II — Retrieval-Augmented Generation",
              "RUBRIC A · LITERATURE REVIEW")
table(s, ["Ref", "Work", "Contribution", "What we take from it"],
      [["[9]", "Lewis et al.", "RAG — parametric + non-parametric knowledge",
        "Core architecture; provenance for every claim"],
       ["[10]", "Karpukhin et al.", "Dense Passage Retrieval",
        "Learned dual encoders beat lexical search"],
       ["[11]", "Robertson & Zaragoza", "BM25 probabilistic relevance",
        "The sparse baseline we improve on"],
       ["[12]", "Malkov & Yashunin", "HNSW graphs for ANN search",
        "Index structure underlying our vector store"],
       ["[13]", "Johnson et al.", "GPU billion-scale similarity search",
        "Scaling path beyond the current corpus"],
       ["[14]", "Reimers & Gurevych", "Sentence-BERT",
        "all-MiniLM-L6-v2 encoder; cosine comparison"]],
      [0.48, 1.75, 3.20, 3.42], top=Inches(2.16), row_h=Inches(0.55), size=10.5,
      hsize=11)
note(s, "Sentence-BERT reduces exhaustive pairwise comparison from hours to "
        "seconds — the property that makes per-employee retrieval feasible "
        "at interactive speed.", top=Inches(6.15), size=11.5)
speaker(s, "The third thread is retrieval-augmented generation. Lewis and "
           "colleagues showed that combining model weights with an external index "
           "improves both accuracy and provenance. Sentence-BERT is what makes it "
           "fast enough to be interactive — it turns hours of pairwise "
           "comparison into seconds.")

# =============================================================================
# 8 · MARKET SURVEY  (A)
# =============================================================================
s = new_slide("Market Survey — Existing Approaches",
              "RUBRIC A · MARKET SURVEY")
table(s, ["Approach", "Example", "Signal", "Frequency", "Explains?", "Auditable?"],
      [["Engagement surveys", "Gallup Q12, Culture Amp", "Self-report",
        "Quarterly", "Partly", "Yes"],
       ["Pulse survey tools", "Officevibe, Peakon", "Self-report",
        "Weekly", "Partly", "Yes"],
       ["HR analytics / attrition ML", "IBM, Workday", "Structured HR",
        "Monthly", "No", "Partly"],
       ["Workplace analytics", "Viva Insights", "Metadata only",
        "Continuous", "No", "Partly"],
       ["Generic LLM assistant", "Prompted chatbot", "Free text",
        "On demand", "Fluently", "**No**"],
       ["**This project**", "**Burnout RAG**", "**Message content**",
        "**Continuous**", "**Yes**", "**Yes**"]],
      [1.75, 1.72, 1.35, 1.18, 1.02, 1.02], top=Inches(2.16),
      row_h=Inches(0.56), size=10.5, hsize=10.5)
note(s, "No surveyed tool reads message content continuously AND shows the "
        "evidence behind its conclusion. That intersection is our contribution.",
     top=Inches(6.24), size=12)
speaker(s, "We surveyed the commercial landscape. Survey tools are auditable but "
           "infrequent. Workplace analytics is continuous but only reads metadata "
           "— who mailed whom, not what they said. A generic chatbot explains "
           "fluently but cannot be audited. Nothing occupies the intersection: "
           "continuous, content-aware, and evidence-backed.")

# =============================================================================
# 9 · RESEARCH GAP  (A)
# =============================================================================
s = new_slide("Research Gap and Positioning", "RUBRIC A · LITERATURE REVIEW")
bullets(s, [
    "**Gap identified**",
    (1, "To the best of our knowledge, prior work treats burnout detection and "
        "burnout explanation as a single modelling problem."),
    (1, "That forces a choice: an accurate model that cannot explain itself, or a "
        "fluent model that cannot be trusted."),
    "**Our positioning — the contribution is architectural, not a new model**",
    (1, "We treat them as two problems with different correctness criteria, "
        "joined by a deliberately narrow contract."),
    (1, "Correctness for the ML stage means the score is reproducible and every "
        "point is traceable to a rule."),
    (1, "Correctness for the language stage means every sentence is supported by "
        "a retrieved record that is shown to the user."),
    (1, "The contract prevents the more fluent component from overruling the more "
        "accountable one."),
], size=15.5)
note(s, "Novelty claim: separation of prediction and explanation under an "
        "enforced contract — not a larger model or a new dataset.",
     top=Inches(6.52), size=12.5)
speaker(s, "This gives us our gap. Prior work treats detection and explanation as "
           "one problem, which forces a choice between accuracy and "
           "explainability. Our contribution is architectural: we separate them, "
           "and we enforce the separation with a contract. That is the novelty we "
           "are claiming — not a bigger model.")

# =============================================================================
# 10 · SYSTEM ARCHITECTURE  (C)
# =============================================================================
s = new_slide("System Architecture", "RUBRIC C · SYSTEM DESIGN")
picture(s, "system_architecture.png", top=Inches(2.14), max_h=4.76, max_w=9.30)
speaker(s, "This is the complete system. Four sequential stages, colour-coded by "
           "owner, joined by a single contract in the centre. Stage one is data. "
           "Stage two is the machine-learning pipeline. The gold box is the "
           "contract. Stage three is the RAG backend. Stage four is the "
           "dashboard. On the right is the quantum study, which reads the same "
           "contract independently.")

# =============================================================================
# 11 · THE INTEGRATION CONTRACT  (C)
# =============================================================================
s = new_slide("Key Design Decision — The Integration Contract",
              "RUBRIC C · SYSTEM DESIGN")
bullets(s, [
    "**The entire system is joined at exactly one point: outputs/burnout_scores.csv "
    "— 20 columns, consumed strictly read-only.**",
], size=15.5, height=Inches(0.76))
table(s, ["Property of the contract", "Why it was designed this way"],
      [["One file, one direction",
        "Stages 3 and 4 can be developed, tested and demonstrated without "
        "re-running the 2.1 GB pipeline"],
       ["risk_level is echoed, never recomputed",
        "The language model is structurally incapable of upgrading or "
        "downgrading a person's risk band"],
       ["Evidence is fenced as data in the prompt",
        "Retrieved records cannot be read as instructions — blocks "
        "prompt-injection through message content"],
       ["Read-only enforcement, verified",
        "Person 1 and Person 2 source files were confirmed byte-identical after "
        "integration (git diff + MD5)"],
       ["Scoring functions reused, not rewritten",
        "Uploaded documents are scored by the original authors' own functions, "
        "lifted at runtime via Python's ast module"]],
      [2.95, 5.85], top=Inches(2.96), row_h=Inches(0.65), size=11.5)
speaker(s, "The single most important design decision is the contract. The whole "
           "system meets at one file. Because risk level is echoed rather than "
           "regenerated, the language model is structurally incapable of changing "
           "someone's risk band — that is not a prompt instruction, it is an "
           "architectural guarantee. We verified the read-only property with git "
           "diff and MD5 checksums.")

# =============================================================================
# 12 · COMPONENT DESIGN  (C)
# =============================================================================
s = new_slide("Component-Level Design — RAG Backend",
              "RUBRIC C · SYSTEM DESIGN")
table(s, ["Module", "Responsibility", "Key design choice"],
      [["embeddings.py", "Encode text to vectors",
        "all-MiniLM-L6-v2, 384-dim, L2-normalised so cosine = dot product"],
       ["vectorstore.py", "Persist and search vectors",
        "ChromaDB with a content fingerprint — rebuilds only when data changes"],
       ["retriever.py", "Select evidence",
        "Explicit employee ID overrides similarity; filters relax before failing"],
       ["burnout_prompt.py", "Constrain the model",
        "Evidence fenced as data; 8 mandatory output sections"],
       ["llm_client.py", "Talk to a provider",
        "Abstract base + registry: OpenAI, Gemini, Ollama — switched by .env only"],
       ["rag_pipeline.py", "Orchestrate",
        "Four-strategy JSON parsing; falls back to a rule-based analysis"],
       ["app.py", "Expose the API",
        "7 FastAPI endpoints, Pydantic-validated request and response"]],
      [1.60, 2.05, 5.15], top=Inches(2.16), row_h=Inches(0.60), size=10.5,
      hsize=11)
speaker(s, "Zooming into stage three. Seven modules, each with one "
           "responsibility. Two choices worth calling out: embeddings are "
           "L2-normalised so cosine similarity becomes a dot product, which is "
           "cheaper; and the LLM client is a registry behind an abstract base, so "
           "switching from OpenAI to a local Llama model is a one-line change in "
           "the .env file with no code modification.")

# =============================================================================
# 13 · DESIGN FOR FAILURE  (C)
# =============================================================================
s = new_slide("Design for Failure and Responsible Use",
              "RUBRIC C · SYSTEM DESIGN")
table(s, ["Failure or risk", "System response"],
      [["Language model unreachable or slow",
        "Rule-based analysis is returned with degraded = true and HTTP 200 — "
        "the dashboard stays usable"],
       ["Model returns malformed JSON",
        "Four parsing strategies in sequence; a structured fallback if all four "
        "fail"],
       ["No evidence matches the query",
        "Filters are relaxed step by step; if still empty the system says so "
        "rather than inventing an answer"],
       ["Message content contains injected instructions",
        "Retrieved text is fenced as data; the prompt forbids treating it as "
        "instruction"],
       ["Output mistaken for a clinical diagnosis",
        "A confidence note is a mandatory output section and is placed BEFORE the "
        "recommendations in the PDF"],
       ["Employee privacy",
        "Pseudonymous employee IDs; evidence shown is limited to the records that "
        "produced the conclusion"]],
      [2.95, 5.85], top=Inches(2.16), row_h=Inches(0.68), size=11.5)
speaker(s, "We designed for failure explicitly. Every layer degrades rather than "
           "failing. If the language model is unreachable the system returns a "
           "rule-based analysis flagged as degraded, and the dashboard keeps "
           "working. The confidence note is a mandatory output section, and in the "
           "PDF it is placed before the recommendations, not buried at the bottom.")

# =============================================================================
# 14 · IMPLEMENTATION STATUS  (D)
# =============================================================================
s = new_slide("Implementation Status", "RUBRIC D · IMPLEMENTATION")
table(s, ["Component", "Owner", "Files", "Lines", "Status"],
      [["Data acquisition and preprocessing", "Person 1", "7", "399",
        "**Complete**"],
       ["ML pipeline — sentiment, emotion, behaviour, scoring", "Person 2",
        "included above", "—", "**Complete**"],
       ["RAG backend — FastAPI, 7 endpoints", "Person 3", "17", "7,162",
        "**Complete**"],
       ["Dashboard — Streamlit, 7 charts, PDF", "Person 4", "7", "2,824",
        "**Complete**"],
       ["Quantum ML comparative study", "Team", "7", "2,103", "**Complete**"],
       ["Automated test suite", "Team", "7", "1,426", "**151 passing**"],
       ["**Total**", "", "**45**", "**13,914**", ""]],
      [3.55, 1.15, 1.35, 1.05, 1.30], top=Inches(2.16), row_h=Inches(0.52),
      size=11.5)
note(s, "Zero lines of Person 1 and Person 2 source were modified during "
        "integration — verified byte-for-byte.", top=Inches(6.40))
speaker(s, "Just under fourteen thousand lines of code across forty-five files, "
           "with a hundred and fifty-one automated tests passing. The line worth "
           "emphasising is the last one: we integrated two new subsystems without "
           "modifying a single line of the earlier stages.")

# =============================================================================
# 15 · API SURFACE  (D)
# =============================================================================
s = new_slide("Backend Implementation — API Surface",
              "RUBRIC D · IMPLEMENTATION")
table(s, ["Method", "Endpoint", "Purpose"],
      [["GET", "/", "Service identity and version"],
       ["GET", "/health", "Provider reachability, model name, record count, "
                          "masked credentials"],
       ["GET", "/statistics", "Aggregate risk, sentiment and emotion counts for "
                              "the dashboard"],
       ["POST", "/query", "Natural-language question → grounded answer + the "
                          "records used"],
       ["POST", "/report", "Full 8-section burnout analysis for one employee or "
                           "the organisation"],
       ["POST", "/refresh", "Rebuild the vector index after the ML pipeline "
                            "re-runs"],
       ["POST", "/ingest", "Upload a PDF or a screenshot — OCR, split, "
                           "score, index"]],
      [0.85, 1.55, 6.30], top=Inches(2.16), row_h=Inches(0.55), size=11.5)
note(s, "Provider-agnostic: OpenAI, Google Gemini and local Llama via Ollama are "
        "all supported by changing one line in .env.", top=Inches(6.65), size=12)
speaker(s, "Seven endpoints. Two are worth noting. Health reports whether the "
           "provider is actually reachable and masks the credentials, so it is "
           "safe to expose. Ingest accepts a PDF or even a screenshot of a chat, "
           "runs OCR, splits it into messages, and scores them using the original "
           "pipeline's own functions.")

# =============================================================================
# 16 · DASHBOARD  (D)
# =============================================================================
s = new_slide("Dashboard Implementation", "RUBRIC D · IMPLEMENTATION")
bullets(s, [
    "**Explainable Employee Burnout Detection using AI + RAG** — Streamlit "
    "and Plotly, light and dark themes.",
    (1, "Overall burnout gauge, risk distribution, sentiment breakdown, emotion "
        "profile, score histogram, burnout-versus-sentiment scatter, and a "
        "ranked high-risk table."),
    (1, "Free-text search and risk filtering across every record."),
    "**Explainability panel — the feature that answers the problem statement**",
    (1, "Every generated answer is displayed together with the exact records it "
        "was grounded in, so a reader can check the reasoning rather than trust "
        "it."),
    "**Document upload**",
    (1, "Drop in a PDF or a screenshot of a conversation; the record appears in "
        "the charts and becomes searchable within seconds."),
    "**PDF export via ReportLab**",
    (1, "All 8 sections, with the confidence note placed before the "
        "recommendations."),
], size=15)
speaker(s, "The dashboard is the HR-facing surface. Seven charts, search and "
           "filtering, and light and dark themes. The panel that matters most is "
           "the explainability panel: every answer is shown together with the "
           "records it came from, so the reader checks the reasoning instead of "
           "trusting it. We will demonstrate this live.")

# =============================================================================
# 17 · RESULTS I  (D)
# =============================================================================
s = new_slide("Results I — Pipeline Output and Verification",
              "RUBRIC D · RESULTS")
table(s, ["What we measured", "Result"],
      [["Records processed end to end",
        "3 corpora unified — 517,401 emails, 22,750 survey rows, 2,000 "
        "stress records"],
       ["Burnout score output",
        "20-column table: score, risk band, sentiment, confidence, 8 emotions, "
        "7 behaviour features, reason"],
       ["Explainability coverage",
        "100% — every record carries a human-readable reason for its score"],
       ["Automated tests",
        "151 passing across data loading, schemas, retrieval, vector store, API "
        "and ingestion"],
       ["Integration safety",
        "Person 1 and Person 2 files byte-identical before and after (git diff "
        "+ MD5)"],
       ["Defects found by our own tests, then fixed",
        "17 — including a silent vector-store failure and a "
        "misattributed dominant emotion"]],
      [3.15, 5.65], top=Inches(2.16), row_h=Inches(0.68), size=11.5)
speaker(s, "On results. The pipeline runs end to end across all three corpora and "
           "produces a twenty-column table in which every single record carries a "
           "human-readable reason. A hundred and fifty-one automated tests pass. "
           "And importantly, those tests found seventeen real defects in our own "
           "code before the review — we will show the two most serious.")

# =============================================================================
# 18 · RESULTS II  (D)
# =============================================================================
s = new_slide("Results II — Defects Found and Fixed by Our Own Tests",
              "RUBRIC D · RESULTS")
table(s, ["Defect", "Why it mattered", "Fix"],
      [["ChromaDB returned collection objects on 1.x and plain strings on 0.6.x",
        "A bare exception handler reported “0 records” after a "
        "successful write — a silent data-loss failure",
        "Normalise both return shapes; remove the silent handler and log"],
       ["Dominant emotion computed with idxmax on all-zero rows",
        "Six of ten records were labelled “anger” when no emotion word "
        "was present at all",
        "Return “neutral” when every emotion count is zero"],
       ["Missing behaviour columns silently created as zero",
        "Fabricated values would have reached the vector store and been cited as "
        "evidence",
        "Emotions zero-filled (absent = 0 is true); behaviour left absent"],
       ["Message minimum length of 12 characters",
        "Dropped “not ok” and “I’m done” — the "
        "strongest burnout signals in the corpus",
        "Per-strategy thresholds; chat messages accepted from 3 characters"]],
      [2.55, 3.35, 2.90], top=Inches(2.16), row_h=Inches(0.90), size=10.5,
      hsize=11)
note(s, "We report these deliberately: a system that is trusted with wellbeing "
        "data has to be able to show how it fails, not only how it works.",
     top=Inches(6.72), size=11.5)
speaker(s, "We are reporting our own defects deliberately. The first one is the "
           "most instructive: a version difference in the vector database "
           "combined with a bare exception handler meant the system reported zero "
           "records after a completely successful write. The second is the one we "
           "care about most ethically — six of ten records were being "
           "labelled angry when no emotion word was present at all.")

# =============================================================================
# 19 · RESULTS III — QUANTUM  (D)
# =============================================================================
s = new_slide("Results III — Quantum ML Comparative Study",
              "RUBRIC D · RESULTS")
table(s, ["Model", "Family", "Framework", "Accuracy", "Runtime"],
      [["Logistic Regression", "Classical", "scikit-learn", "**0.8667**", "< 0.01 s"],
       ["SVM (RBF kernel)", "Classical", "scikit-learn", "0.8587", "< 0.01 s"],
       ["Random Forest", "Classical", "scikit-learn", "0.8533", "0.20 s"],
       ["SVM-RBF (same 400 rows)", "Classical", "scikit-learn", "0.8560",
        "0.008 s"],
       ["Quantum kernel SVM", "Quantum", "PennyLane", "**0.8560**", "0.67 s"],
       ["QSVC (quantum kernel)", "Quantum", "Qiskit", "0.8533", "123.6 s"],
       ["VQC (variational)", "Quantum", "PennyLane", "0.7067", "—"],
       ["VQC (variational)", "Quantum", "Qiskit", "0.7227", "292.2 s"]],
      [2.35, 1.15, 1.45, 1.15, 1.25], top=Inches(2.12), row_h=Inches(0.42),
      size=11)
bullets(s, [
    "**The quantum kernel reproduces the classical RBF result to within 0.0000 on "
    "the same 400 rows — at roughly 84× (PennyLane) and 15,000× "
    "(Qiskit) the compute cost.**",
    (1, "Cross-framework agreement of 0.27% between two independent "
        "implementations validates that the circuits are correct."),
], size=13, top=Inches(6.00), height=Inches(1.15))
speaker(s, "Our faculty mentor asked us to investigate quantum machine learning, "
           "so we implemented it twice — in PennyLane and in Qiskit — "
           "and benchmarked both against four classical baselines. The honest "
           "result: the quantum kernel exactly reproduces the classical RBF "
           "accuracy at up to fifteen thousand times the compute cost. We report "
           "this as a negative result, and the agreement between the two "
           "independent frameworks tells us the implementation is correct rather "
           "than broken.")

# =============================================================================
# 20 · INTERIM REPORT  (E)
# =============================================================================
s = new_slide("Interim Report Status", "RUBRIC E · INTERIM REPORT")
table(s, ["Section", "Content", "Status"],
      [["I. Introduction", "Motivation, contributions, scope", "**Complete**"],
       ["II. Related Work", "4 subsections, 14 references, positioning",
        "**Complete**"],
       ["III. System Architecture", "Four stages and the integration contract",
        "**Complete**"],
       ["IV. ML Pipeline", "Preprocessing, sentiment, emotion, behaviour, fusion, "
                           "scoring", "**Complete**"],
       ["V. RAG Backend", "Embedding, retrieval, prompting, generation",
        "**Complete**"],
       ["VI. Dashboard", "Interface design and explainability panel",
        "**Complete**"],
       ["VII. Verification", "5 subsections — attribution, sparsity, "
                             "classification quality", "**Complete**"],
       ["VIII. Defects", "Every defect found, its disposition and its owner",
        "**Complete**"],
       ["IX. Future Work", "Validation study, temporal modelling, deployment",
        "**Complete**"]],
      [1.95, 5.20, 1.20], top=Inches(2.12), row_h=Inches(0.44), size=10.5,
      hsize=11)
note(s, "Drafted in IEEE conference format with 11 figures. Under review by our "
        "guide, Dr. Shubha Puthran.", top=Inches(6.66), size=11.5)
speaker(s, "The interim report is drafted in full IEEE conference format — "
           "nine sections, fourteen references and eleven figures. Two sections "
           "are unusual for a student report and we included them deliberately: a "
           "verification section, and a defects section that records every problem "
           "we found and how it was resolved.")

# =============================================================================
# 21 · TIMELINE  (E)
# =============================================================================
s = new_slide("Project Timeline and Progress", "RUBRIC E · INTERIM REPORT")
picture(s, "timeline.png", top=Inches(2.20), max_h=4.50, max_w=9.20)
note(s, "16 workstreams across 8 weeks, all complete and on schedule.",
     top=Inches(6.78), size=12)
speaker(s, "This is our timeline from topic approval to today. Sixteen "
           "workstreams across eight weeks, colour-coded by owner, all complete. "
           "The overlap between rows is deliberate — because stages three "
           "and four consume a file rather than calling code, they could be built "
           "in parallel with the pipeline still being tuned.")

# =============================================================================
# 22 · LIMITATIONS  (E)
# =============================================================================
s = new_slide("Limitations and Ethical Considerations",
              "RUBRIC E · INTERIM REPORT")
table(s, ["Limitation", "How we are handling it"],
      [["No clinically labelled ground truth for burnout",
        "We report risk bands, never a diagnosis; a validation study against the "
        "Maslach inventory is planned"],
       ["Behavioural features measured on cleaned text lose punctuation and casing",
        "Documented in the report as a defect owned by the preprocessing stage; "
        "the fix belongs upstream"],
       ["Emotion counts are sparse on short messages",
        "Reported honestly as “neutral” rather than forcing a dominant "
        "emotion"],
       ["Surveillance risk if deployed carelessly",
        "Designed for aggregate wellbeing insight, pseudonymous IDs, and evidence "
        "visible to the person described"],
       ["Enron corpus is a single organisation from one era",
        "Two additional corpora included; generalisation is stated as a threat to "
        "validity"]],
      [3.30, 5.50], top=Inches(2.16), row_h=Inches(0.76), size=11.5)
speaker(s, "We want to be explicit about limitations. We have no clinically "
           "labelled ground truth, so we report risk bands and never a diagnosis. "
           "And there is a real surveillance risk in a system like this, which is "
           "why the evidence is visible to the person being described rather than "
           "only to HR.")

# =============================================================================
# 23 · FUTURE WORK
# =============================================================================
s = new_slide("Future Work — Towards the Final Review")
bullets(s, [
    "**Validation** — correlate produced risk bands against Maslach Burnout "
    "Inventory responses on a consenting sample.",
    "**Temporal modelling** — track an individual's trajectory over weeks "
    "rather than scoring messages independently; trend is a stronger signal than "
    "any single message.",
    "**Retrieval evaluation** — measure recall@k and answer faithfulness "
    "against a hand-annotated evidence set.",
    "**Deployment** — containerise the backend and dashboard; role-based "
    "access so managers see aggregates and only the individual sees their own "
    "evidence.",
    "**Scaling the index** — move from ChromaDB to a GPU-backed index for "
    "the full 517,401-message corpus.",
    "**Paper submission** — complete the empirical evaluation and submit to "
    "the target venue.",
], size=15.5)
speaker(s, "Between now and the final review, the priority is validation — "
           "correlating our risk bands against a recognised instrument on a "
           "consenting sample. Second is temporal modelling: a trajectory over "
           "weeks is a much stronger signal than any single message. Third is "
           "deployment with role-based access.")

# =============================================================================
# 24 · CONCLUSION
# =============================================================================
s = new_slide("Conclusion")
bullets(s, [
    "**We set out to detect burnout risk from workplace communication and explain "
    "every decision using only the evidence that produced it.**",
    (1, "All six objectives from topic approval are complete."),
    (1, "The system is running end to end: 3 corpora, an ML pipeline, a RAG "
        "backend with 7 endpoints, and an interactive dashboard."),
    (1, "13,914 lines of code, 151 automated tests, 17 self-reported defects "
        "found and fixed."),
    (1, "Two extensions beyond the approved scope: document and image ingestion, "
        "and a two-framework quantum ML study."),
    (1, "The interim report is complete in IEEE format with 14 references."),
    "**The contribution is architectural: prediction and explanation are two "
    "problems, and the contract between them is what makes the output "
    "trustworthy.**",
], size=15.5)
speaker(s, "To conclude. Every objective from topic approval is complete, the "
           "system runs end to end, and the interim report is written. If the "
           "panel takes one thing from this review, we would like it to be the "
           "contract: prediction and explanation are two different problems, and "
           "keeping them separate is what makes the output something an HR "
           "partner can actually defend.")

# =============================================================================
# 25 · REFERENCES
# =============================================================================
s = new_slide("References")
bullets(s, [
    "[1] World Health Organization, “Burn-out an occupational phenomenon,” "
    "ICD-11, 2019.",
    "[2] C. Maslach and S. E. Jackson, “The measurement of experienced "
    "burnout,” J. Occup. Behav., 1981.",
    "[3] M. D. Xames et al., “Machine learning approaches for employee "
    "burnout prediction.”",
    "[4] A. Vaswani et al., “Attention is all you need,” NeurIPS, 2017.",
    "[5] J. Devlin et al., “BERT: Pre-training of deep bidirectional "
    "transformers,” NAACL, 2019.",
    "[6] Y. Liu et al., “RoBERTa: A robustly optimized BERT pretraining "
    "approach,” 2019.",
    "[7] F. Barbieri et al., “TweetEval: Unified benchmark for tweet "
    "classification,” EMNLP Findings, 2020.",
    "[8] S. Mohammad and P. Turney, “Crowdsourcing a word–emotion "
    "association lexicon,” 2013.",
    "[9] P. Lewis et al., “Retrieval-augmented generation for "
    "knowledge-intensive NLP tasks,” NeurIPS, 2020.",
    "[10] V. Karpukhin et al., “Dense passage retrieval for open-domain "
    "question answering,” EMNLP, 2020.",
    "[11] S. Robertson and H. Zaragoza, “The probabilistic relevance "
    "framework: BM25 and beyond,” 2009.",
    "[12] Y. Malkov and D. Yashunin, “Efficient and robust approximate "
    "nearest neighbor search using HNSW,” 2020.",
    "[13] J. Johnson et al., “Billion-scale similarity search with GPUs,” "
    "IEEE Trans. Big Data, 2021.",
    "[14] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings "
    "using siamese BERT-networks,” EMNLP, 2019.",
], size=11.8, lead=Pt(3))
speaker(s, "Our references.")

# =============================================================================
# 26 · THANK YOU
# =============================================================================
s = prs.slides.add_slide(LAYOUT_BODY)
_clear(s)
tb = s.shapes.add_textbox(Inches(0.6), Inches(2.70), Inches(8.8), Inches(2.2))
tf = tb.text_frame; tf.word_wrap = True
for i, (txt, sz, bold, col) in enumerate([
        ("Thank You", 40, True, NAVY),
        ("", 10, False, INK),
        ("Questions and discussion", 20, False, GREY),
        ("", 10, False, INK),
        ("Saumya Shah  ·  Swara Wairkar  ·  Aryaman Giri  ·  "
         "Anushree Devarashetty", 14, False, INK),
        ("Under the guidance of Dr. Shubha Puthran", 14, True, INK),
        ("Computer Engineering Department, MPSTME, Mumbai Campus", 11, False,
         GREY)]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = txt
    r.font.size = Pt(sz); r.font.bold = bold
    r.font.name = SERIF; r.font.color.rgb = col
speaker(s, "Thank you. We are happy to take questions, and we can demonstrate the "
           "running system now if the panel would like.")

# ---- slide numbers ----------------------------------------------------------
for i, sl in enumerate(prs.slides, start=1):
    if i == 1:
        continue
    tb = sl.shapes.add_textbox(Inches(9.00), Inches(7.03), Inches(0.62),
                               Inches(0.30))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(i)
    r.font.size = Pt(10); r.font.name = SERIF; r.font.color.rgb = GREY

prs.save(OUT)
print(f"saved {OUT}: {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
