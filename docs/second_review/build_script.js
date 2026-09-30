const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  PageBreak, Header, Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const NAVY = "1B3A5C", ACC = "8A6D12", GREY = "555F6B", INK = "1A1A1A";
const BAND = "E8EEF5";
const FONT = "Times New Roman";

const P = (text, o = {}) => new Paragraph({
  alignment: o.align, spacing: { before: o.before ?? 0, after: o.after ?? 100 },
  indent: o.indent, border: o.border,
  children: [new TextRun({
    text, font: FONT, size: (o.size ?? 11) * 2, bold: o.bold, italics: o.italics,
    color: o.color ?? INK,
  })],
});

const RICH = (runs, o = {}) => new Paragraph({
  alignment: o.align, spacing: { before: o.before ?? 0, after: o.after ?? 100 },
  indent: o.indent,
  children: runs.map(r => new TextRun({
    text: r[0], font: FONT, size: (r[2] ?? o.size ?? 11) * 2,
    bold: r[1] === "b" || r[1] === "bi", italics: r[1] === "i" || r[1] === "bi",
    color: r[3] ?? INK,
  })),
});

const H1 = t => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 340, after: 150 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: ACC, space: 4 } },
  children: [new TextRun({ text: t, font: FONT, size: 30, bold: true, color: NAVY })],
});

const H2 = t => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 100 },
  children: [new TextRun({ text: t, font: FONT, size: 24, bold: true, color: NAVY })],
});

// speaker header bar
const SPK = (name, role, slides, mins) => new Table({
  width: { size: convertInchesToTwip(6.5), type: WidthType.DXA },
  columnWidths: [convertInchesToTwip(6.5)],
  borders: {
    top: { style: BorderStyle.SINGLE, size: 2, color: NAVY },
    bottom: { style: BorderStyle.SINGLE, size: 2, color: NAVY },
    left: { style: BorderStyle.SINGLE, size: 2, color: NAVY },
    right: { style: BorderStyle.SINGLE, size: 2, color: NAVY },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
  },
  rows: [new TableRow({
    children: [new TableCell({
      width: { size: convertInchesToTwip(6.5), type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: NAVY },
      margins: { top: 90, bottom: 90, left: 140, right: 140 },
      children: [
        RICH([[name, "b", 14, "FFFFFF"]], { after: 30 }),
        RICH([[`${role}   ·   Slides ${slides}   ·   approx. ${mins}`, "", 10, "D8E2EE"]], { after: 0 }),
      ],
    })],
  })],
});

// slide cue block
const CUE = (slide, title) => new Table({
  width: { size: convertInchesToTwip(6.5), type: WidthType.DXA },
  columnWidths: [convertInchesToTwip(0.75), convertInchesToTwip(5.75)],
  borders: {
    top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
  },
  rows: [new TableRow({
    children: [
      new TableCell({
        width: { size: convertInchesToTwip(0.75), type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: ACC },
        margins: { top: 50, bottom: 50, left: 60, right: 60 },
        children: [RICH([[`SLIDE ${slide}`, "b", 8.5, "FFFFFF"]], { align: AlignmentType.CENTER, after: 0 })],
      }),
      new TableCell({
        width: { size: convertInchesToTwip(5.75), type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: BAND },
        margins: { top: 50, bottom: 50, left: 120, right: 60 },
        children: [RICH([[title, "b", 10.5, NAVY]], { after: 0 })],
      }),
    ],
  })],
});

const SAY = t => RICH([[t, "", 11]], { indent: { left: 200 }, after: 110 });
const STAGE = t => RICH([["▸  " + t, "i", 9.5, ACC]], { indent: { left: 200 }, after: 130 });

const tbl = (headers, rows, widths) => {
  const W = widths.map(convertInchesToTwip);
  const total = W.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: W,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "AAB6C4" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "AAB6C4" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "AAB6C4" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "AAB6C4" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "CFD8E2" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "CFD8E2" },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => new TableCell({
          width: { size: W[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: NAVY },
          margins: { top: 60, bottom: 60, left: 90, right: 90 },
          children: [RICH([[h, "b", 9.5, "FFFFFF"]], { after: 0 })],
        })),
      }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => new TableCell({
          width: { size: W[i], type: WidthType.DXA },
          shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "FFFFFF" : BAND },
          margins: { top: 55, bottom: 55, left: 90, right: 90 },
          children: [RICH([[String(c).replace(/\*\*/g, ""), String(c).startsWith("**") ? "b" : "", 9.5]], { after: 0 })],
        })),
      })),
    ],
  });
};

const kids = [];

// ============================== COVER ==============================
kids.push(
  P("MPSTME  ·  Computer Engineering  ·  A.Y. 2026-2027", { align: AlignmentType.CENTER, size: 10, color: GREY, after: 260 }),
  P("Capstone Project — Second Review", { align: AlignmentType.CENTER, size: 13, bold: true, color: GREY, after: 90 }),
  P("Presentation Script for Four Speakers", { align: AlignmentType.CENTER, size: 22, bold: true, color: NAVY, after: 160 }),
  P("Explainable Employee Burnout Detection from Workplace Communication using Sentiment Analysis and Retrieval-Augmented Generation",
    { align: AlignmentType.CENTER, size: 12, italics: true, color: INK, after: 320 }),
  tbl(["Speaker", "Section", "Slides", "Time"], [
    ["Saumya Shah", "Opening, problem statement, literature review, market survey", "1 – 9", "4 min"],
    ["Swara Wairkar", "System design, the integration contract, component design", "10 – 13", "4 min"],
    ["Aryaman Giri", "Implementation, results, defects, quantum study", "14 – 19", "5 min"],
    ["Anushree Devarashetty", "Interim report, timeline, limitations, future work, close", "20 – 26", "4 min"],
  ], [1.55, 3.05, 0.85, 0.65]),
  P("", { after: 200 }),
  RICH([["Total speaking time: ", "b", 10.5], ["approximately 17 minutes, leaving room for questions in a 25-minute slot.", "", 10.5]], { after: 90 }),
  RICH([["Adjust the name-to-section mapping if your work split differs — the content of each section is what matters, not who delivers it.", "i", 10, GREY]], { after: 90 }),
);

// ============================== HOW TO USE ==============================
kids.push(H1("How to use this script"));
kids.push(
  P("Every slide has a cue block, the words to say, and a stage direction in gold telling you what to do with your hands, the clicker or the laptop. The wording is written to be spoken, not read — it is short sentences, no sub-clauses, and no jargon that you would not use out loud.", { size: 10.5 }),
  P("Three rules the panel will notice:", { size: 10.5, bold: true, after: 60 }),
  P("1.  Do not read the slide. The slide is the evidence; you are the argument. If you find yourself saying a bullet word for word, stop and say the point instead.", { size: 10.5, indent: { left: 200 }, after: 50 }),
  P("2.  Hand over by name. Every speaker ends with a sentence that names the next speaker and what they will cover. It looks rehearsed because it is.", { size: 10.5, indent: { left: 200 }, after: 50 }),
  P("3.  Own the negatives. Slides 18 and 19 report our own defects and a negative quantum result. Deliver them at normal speed and without apology — reporting them is the strongest thing in the deck.", { size: 10.5, indent: { left: 200 }, after: 50 }),
  P("The same text is also in the PowerPoint speaker notes, so you can present from Presenter View if you prefer.", { size: 10.5, italics: true, color: GREY, before: 120 }),
);

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ============================== SPEAKER 1 ==============================
kids.push(H1("Speaker 1 — Saumya Shah"));
kids.push(SPK("Saumya Shah", "Opening · Problem · Literature Review · Market Survey", "1 – 9", "4 minutes"));
kids.push(P("", { after: 120 }));

kids.push(CUE(1, "Title"));
kids.push(SAY("Good morning. I am Saumya, and with me are Swara, Aryaman and Anushree. We are presenting the second review of our capstone project, Explainable Employee Burnout Detection, guided by Dr. Shubha Puthran."));
kids.push(STAGE("Do not linger. One breath, then advance."));

kids.push(CUE(2, "Agenda — Mapped to the Review Rubric"));
kids.push(SAY("Before we begin, one note on how we have organised this. Rather than ask the panel to hunt for evidence, we have mapped the deck directly onto the five evaluation criteria for this review. Each slide carries a tag in the bottom-left corner showing which criterion it answers. Literature review is slides six to nine. Problem statement is four and five. Design is ten to thirteen. Implementation and results is fourteen to nineteen. The interim report is twenty and twenty-one."));
kids.push(STAGE("This slide buys goodwill. Say it calmly and do not rush it."));

kids.push(CUE(3, "Progress Since Topic Approval"));
kids.push(SAY("This is where we stand against what we committed to at topic approval. Every one of the five approved deliverables is complete — the data pipeline, the sentiment and emotion analysis, the burnout scoring, the retrieval backend and the dashboard. We also took on two workstreams that were not in the approved scope: document and image ingestion with OCR, and a quantum machine-learning comparative study that our faculty mentor suggested."));

kids.push(CUE(4, "Problem Statement"));
kids.push(SAY("Now the problem itself. Burnout is recognised by the World Health Organization as an occupational phenomenon, but organisations detect it late, because they rely on periodic self-report surveys. By the time a quarterly survey registers distress, the disengagement has usually already happened."));
kids.push(SAY("Meanwhile, workplace communication carries the same signal continuously. People write email and chat every day, and how they write changes as their emotional state changes."));
kids.push(SAY("So why has this not been solved? Two reasons. Machine-learning approaches give you a number with no reason — a score of seven tells an HR partner nothing about what to do or why the person was flagged, and it cannot be contested by the employee it describes. Language models give you a fluent reason, but left unconstrained they will invent evidence and quietly overrule the measured result."));
kids.push(STAGE("Pause after 'invent evidence'. That sentence is the hinge of the whole presentation."));

kids.push(CUE(5, "Aim and Objectives"));
kids.push(SAY("Our aim fits in one sentence: detect burnout risk from workplace communication, and explain every decision using only the evidence that produced it. We broke that into six objectives, each with a measurable outcome rather than a vague intention. As you can see in the right-hand column, all six are complete."));

kids.push(CUE(6, "Literature Review I — Burnout and Language Models"));
kids.push(SAY("Our literature review runs along three threads. The first is burnout measurement itself — the WHO classification, which defines the boundary of responsible use and is why we never issue a diagnosis, and the Maslach Burnout Inventory, which gives the three-part schema the field still uses. The important finding here is negative: the machine-learning literature on burnout works almost entirely from self-report and structured HR data, not from natural language."));
kids.push(SAY("The second thread is the transformer line — the original architecture, BERT, and RoBERTa — ending at the specific TweetEval checkpoint we deploy. And the NRC lexicon, which we chose deliberately because its emotion counts can be inspected by a human, in a way that a neural activation cannot."));

kids.push(CUE(7, "Literature Review II — Retrieval-Augmented Generation"));
kids.push(SAY("The third thread is retrieval-augmented generation. Lewis and colleagues showed that combining what a model knows in its weights with knowledge retrieved at inference time improves both accuracy and provenance on knowledge-intensive tasks. Dense passage retrieval showed that learned encoders beat lexical search like BM25. And Sentence-BERT is what makes the whole thing practical — it reduces exhaustive pairwise comparison from hours to seconds, which is the only reason per-employee retrieval can happen at interactive speed."));

kids.push(CUE(8, "Market Survey — Existing Approaches"));
kids.push(SAY("We also surveyed what is actually on the market. Engagement and pulse survey tools are auditable, but they are infrequent and depend on people self-reporting. HR analytics platforms read structured data and do not explain. Workplace analytics tools like Viva Insights are continuous, but they read metadata only — who mailed whom and at what hour, never what was said. And a generic chatbot explains fluently but cannot be audited at all."));
kids.push(SAY("Look at the bottom row. Nothing we surveyed occupies the intersection: continuous, content-aware, and evidence-backed. That intersection is where we sit."));

kids.push(CUE(9, "Research Gap and Positioning"));
kids.push(SAY("Which gives us our gap. To the best of our knowledge, prior work treats burnout detection and burnout explanation as a single modelling problem. That forces a choice between an accurate model that cannot explain itself and a fluent model that cannot be trusted."));
kids.push(SAY("Our contribution is architectural, not a bigger model and not a new dataset. We treat them as two problems with different definitions of correctness, and we join them with a deliberately narrow contract that stops the more fluent component overruling the more accountable one."));
kids.push(SAY("Swara will now take you through that design."));
kids.push(STAGE("Hand the clicker over physically. Do not walk off while Swara is starting."));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ============================== SPEAKER 2 ==============================
kids.push(H1("Speaker 2 — Swara Wairkar"));
kids.push(SPK("Swara Wairkar", "System Design · The Integration Contract · Component Design", "10 – 13", "4 minutes"));
kids.push(P("", { after: 120 }));

kids.push(CUE(10, "System Architecture"));
kids.push(SAY("Thank you, Saumya. This is the complete system on one slide. Four sequential stages, colour-coded by owner, and a single point in the middle where they all meet."));
kids.push(SAY("Stage one, in green, is data — three corpora, half a million Enron emails, a burnout survey dataset, an Indian corporate stress dataset, and the NRC lexicon, all reduced to a single clean-text column."));
kids.push(SAY("Stage two, in orange, is the machine-learning pipeline. Four steps: RoBERTa gives sentiment, the NRC lexicon gives eight emotion counts, seven behavioural features capture how someone writes rather than what they write, and then a transparent additive rule turns all of that into a score and one of three risk bands."));
kids.push(SAY("The gold box in the centre is the integration contract, and I will spend the next slide on it."));
kids.push(SAY("Stage three, in blue, is the retrieval backend. Stage four, in purple, is the dashboard. And on the right, in maroon, is the quantum study, which reads the same contract completely independently."));
kids.push(STAGE("Point at the gold box when you say 'contract'. Trace the flow top to bottom with your hand — do not use the mouse pointer."));

kids.push(CUE(11, "Key Design Decision — The Integration Contract"));
kids.push(SAY("This is the single most important decision in the project. The entire system is joined at exactly one point — one CSV file, twenty columns, consumed strictly read-only."));
kids.push(SAY("Four consequences. First, because it is one file in one direction, stages three and four could be built and demonstrated without ever re-running a two-gigabyte pipeline. Second, and this is the one that answers the hallucination problem from slide four: the risk level is echoed, never recomputed. The language model is structurally incapable of upgrading or downgrading a person's risk band. That is not a polite instruction in a prompt that a model can ignore — it is an architectural guarantee."));
kids.push(SAY("Third, the retrieved evidence is fenced as data inside the prompt, so message content cannot be read as an instruction. That blocks prompt injection through the corpus itself. And fourth, we verified the read-only property rather than assuming it. After integration we confirmed with git diff and MD5 checksums that the earlier stages' source files were byte-identical."));
kids.push(STAGE("Slow down on 'structurally incapable'. If the panel remembers one line from the review, this is the one."));

kids.push(CUE(12, "Component-Level Design — RAG Backend"));
kids.push(SAY("Zooming into stage three. Seven modules, each with exactly one responsibility. Two choices worth calling out."));
kids.push(SAY("The embeddings are L2-normalised, which makes cosine similarity equivalent to a dot product — cheaper at query time, and it means the distance the vector store reports converts directly into a similarity we can show the user."));
kids.push(SAY("And the LLM client is a registry behind an abstract base class. Switching from OpenAI to Google Gemini to a local Llama model running on our own machine is a one-line change in the environment file. No code changes at all. That matters for a wellbeing system, because an organisation may not be willing to send employee messages to a third-party API."));

kids.push(CUE(13, "Design for Failure and Responsible Use"));
kids.push(SAY("We designed for failure explicitly, because a system that people are asked to trust has to behave predictably when it breaks."));
kids.push(SAY("Every layer degrades rather than fails. If the language model is unreachable, the system returns a rule-based analysis flagged as degraded and the dashboard keeps working. If the model returns malformed JSON, there are four parsing strategies in sequence and a structured fallback. If no evidence matches a query, the filters relax step by step, and if it is still empty the system says so instead of inventing an answer."));
kids.push(SAY("On responsible use — the confidence note is a mandatory output section, and in the exported PDF it is placed before the recommendations, not buried at the bottom where nobody reads it. Employee identifiers are pseudonymous, and the evidence shown is limited to the records that actually produced the conclusion."));
kids.push(SAY("Aryaman will now show you what we have actually built."));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ============================== SPEAKER 3 ==============================
kids.push(H1("Speaker 3 — Aryaman Giri"));
kids.push(SPK("Aryaman Giri", "Implementation · Results · Defects · Quantum Study", "14 – 19", "5 minutes"));
kids.push(P("", { after: 120 }));

kids.push(CUE(14, "Implementation Status"));
kids.push(SAY("Thank you, Swara. Here is what exists. Just under fourteen thousand lines of code across forty-five files, with a hundred and fifty-one automated tests passing. Every component is complete — the pipeline, the backend, the dashboard, the quantum study and the test suite."));
kids.push(SAY("The line I would draw your attention to is the one at the bottom. Zero lines of the earlier stages' source code were modified during integration, and we verified that byte for byte. Two new subsystems were added to a working codebase without touching it."));

kids.push(CUE(15, "Backend Implementation — API Surface"));
kids.push(SAY("Seven endpoints. Two are worth a sentence each."));
kids.push(SAY("Health does not just return 'OK' — it reports whether the language provider is actually reachable, which model is loaded, how many records are indexed, and it masks the credentials so the endpoint is safe to expose."));
kids.push(SAY("Ingest is the extension we added after topic approval. You can upload a PDF, or even a screenshot of a chat conversation. It runs OCR, splits it into individual messages using one of four strategies depending on the format, and scores them — and it scores them by calling the original pipeline's own functions, which we lift at runtime rather than copying, so the two can never drift apart."));

kids.push(CUE(16, "Dashboard Implementation"));
kids.push(SAY("The dashboard is the HR-facing surface. Seven charts, free-text search, risk filtering, light and dark themes."));
kids.push(SAY("But the panel that matters is the explainability panel, because it is the one that answers the problem statement Saumya opened with. Every generated answer is displayed together with the exact records it was grounded in. The reader checks the reasoning instead of trusting it. If the system says an employee is at elevated risk, the messages that led to that are right there on the screen, and so is the rule that added each point."));
kids.push(STAGE("If the panel wants a live demo, offer it here — but keep going unless they ask. Have the dashboard already running in a second window."));

kids.push(CUE(17, "Results I — Pipeline Output and Verification"));
kids.push(SAY("On results. The pipeline runs end to end across all three corpora and produces a twenty-column table in which every single record carries a human-readable reason for its score. Explainability coverage is a hundred percent by construction — a record cannot exist without a reason."));
kids.push(SAY("A hundred and fifty-one automated tests pass, covering data loading, schemas, retrieval, the vector store, the API and ingestion. And those tests found seventeen real defects in our own code before this review. I want to show you two of them."));

kids.push(CUE(18, "Results II — Defects Found and Fixed by Our Own Tests"));
kids.push(SAY("We are reporting our own defects deliberately, and I would ask you to read the second column rather than the first."));
kids.push(SAY("The first defect is the most instructive engineering lesson we had. Two versions of the vector database return different types from the same function. Combined with a bare exception handler, that meant the system reported zero records after a completely successful write. It looked like an empty database. It was silent data loss, and only a test caught it."));
kids.push(SAY("The second is the one we care about most ethically. The dominant emotion was computed with a function that returns the first column when everything is tied at zero. On short messages with no emotion words at all, that meant six of ten records were being labelled 'anger'. In a wellbeing system, telling an HR partner that an employee is angry when there is no evidence of anger is not a cosmetic bug — it is the kind of error that damages a person. It now returns 'neutral' when every count is zero."));
kids.push(STAGE("Deliver this at normal speed. Do not apologise and do not speed up. Reporting these is the strongest thing in the deck."));

kids.push(CUE(19, "Results III — Quantum ML Comparative Study"));
kids.push(SAY("Finally, the quantum study. Our faculty mentor asked us to investigate whether quantum machine learning had anything to offer here, so we implemented it twice — once in PennyLane and once in Qiskit — and benchmarked both against four classical baselines on the same data."));
kids.push(SAY("The honest answer is no, and the table shows why. The best classical model reaches eighty-six point seven percent. The quantum kernel reaches eighty-five point six — which is not approximately the classical RBF result, it is exactly the classical RBF result on the same four hundred rows, to four decimal places. It reproduces the classical answer at roughly eighty-four times the compute cost in PennyLane, and about fifteen thousand times in Qiskit. The variational classifier does noticeably worse than any classical baseline."));
kids.push(SAY("We report this as a negative result, and it is a useful one. The agreement between two completely independent framework implementations — within nought point two seven percent — tells us the circuits are correct rather than broken. So the conclusion is not 'quantum does not work', it is 'quantum reproduces the classical result at enormous cost on a problem of this size', which is exactly what the current theory predicts for six qubits on a simulator."));
kids.push(SAY("Anushree will close with the report, the timeline and where we go next."));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ============================== SPEAKER 4 ==============================
kids.push(H1("Speaker 4 — Anushree Devarashetty"));
kids.push(SPK("Anushree Devarashetty", "Interim Report · Timeline · Limitations · Future Work · Close", "20 – 26", "4 minutes"));
kids.push(P("", { after: 120 }));

kids.push(CUE(20, "Interim Report Status"));
kids.push(SAY("Thank you, Aryaman. The interim report is drafted in full IEEE conference format — nine sections, fourteen references and eleven figures — and it is currently with Dr. Puthran for review."));
kids.push(SAY("Two of these sections are unusual for a student interim report, and we included them on purpose. Section seven is verification: five subsections that test our own claims, including whether the highest-scoring record actually deserves to be highest and whether the emotion signal is dense enough to rely on. Section eight is a defects register, recording every problem we found, who owns it and how it was resolved — including the ones Aryaman just showed you."));

kids.push(CUE(21, "Project Timeline and Progress"));
kids.push(SAY("This is our timeline from topic approval to today. Sixteen workstreams over eight weeks, colour-coded by owner, all complete and on schedule."));
kids.push(SAY("The overlap between the rows is deliberate rather than accidental. Because stages three and four consume a file rather than calling code, they could be built in parallel while the pipeline was still being tuned. That is the practical dividend of the contract Swara described — it bought us parallel development across four people."));

kids.push(CUE(22, "Limitations and Ethical Considerations"));
kids.push(SAY("We want to be explicit about what this system cannot do. We have no clinically labelled ground truth for burnout, so we report risk bands and never a diagnosis, and validating against a recognised instrument is our next priority."));
kids.push(SAY("Some of our behavioural features are measured on cleaned text that has lost its punctuation and casing, which weakens them. We have documented that in the report as a defect that belongs to the preprocessing stage rather than papering over it."));
kids.push(SAY("And there is a real surveillance risk in a system like this. Our answer is in the design: it is built for aggregate wellbeing insight, identifiers are pseudonymous, and the evidence behind a conclusion is visible to the person being described — not only to the organisation."));
kids.push(STAGE("Say the surveillance point looking at the panel, not the screen. It is the question they are most likely to ask, so answer it before they do."));

kids.push(CUE(23, "Future Work — Towards the Final Review"));
kids.push(SAY("Between now and the final review, three priorities. First, validation — correlating our risk bands against Maslach Burnout Inventory responses on a consenting sample, which is what turns this from a working system into an evaluated one."));
kids.push(SAY("Second, temporal modelling. At the moment we score messages independently. A trajectory over several weeks is a far stronger signal than any single message, and it is also fairer to the individual — one bad Tuesday should not flag anyone."));
kids.push(SAY("Third, deployment with role-based access, so that managers see aggregates and only the individual sees their own evidence. Alongside that, scaling the index to the full corpus, and submitting the paper once the empirical evaluation is complete."));

kids.push(CUE(24, "Conclusion"));
kids.push(SAY("To conclude. Every objective from topic approval is complete. The system runs end to end — three corpora, an ML pipeline, a retrieval backend with seven endpoints, and an interactive dashboard. Fourteen thousand lines of code, a hundred and fifty-one tests, seventeen self-reported defects found and fixed, two extensions beyond the approved scope, and the interim report written."));
kids.push(SAY("If the panel takes one idea away from this review, we would like it to be the contract. Prediction and explanation are two different problems with two different definitions of correctness, and keeping them separate is what makes the output something an HR partner can actually stand behind when someone asks why they were flagged."));
kids.push(STAGE("Land the last sentence and stop. Do not add 'so, yeah'. Silence is fine."));

kids.push(CUE(25, "References"));
kids.push(SAY("Our references — fourteen peer-reviewed sources across burnout measurement, transformer language models and retrieval-augmented generation."));
kids.push(STAGE("Do not read these aloud. Advance within two seconds."));

kids.push(CUE(26, "Thank You"));
kids.push(SAY("Thank you. We are happy to take questions, and the system is running if the panel would like a live demonstration."));

kids.push(new Paragraph({ children: [new PageBreak()] }));

// ============================== Q&A ==============================
kids.push(H1("Anticipated questions and prepared answers"));
kids.push(P("Answer in one or two sentences. If you do not know, say so and say what you would do to find out — that scores better than guessing.", { size: 10.5, italics: true, color: GREY, after: 160 }));

const QA = [
  ["Aryaman", "How do you know the burnout scores are correct?",
   "We do not claim they are clinically correct — we have no labelled ground truth, which is why we report risk bands and never a diagnosis. What we can guarantee is that every score is reproducible and every point is traceable to a stated rule. Validating against the Maslach inventory on a consenting sample is our next priority."],
  ["Swara", "What stops the language model from making things up?",
   "Three things. The risk level is echoed from the CSV rather than regenerated, so it structurally cannot be changed. The evidence is fenced as data in the prompt, so message text cannot act as an instruction. And every answer is shown together with the records it came from, so a fabrication is visible to the reader rather than hidden."],
  ["Anushree", "Is this not employee surveillance?",
   "It could be, if it were deployed badly, which is why the design pushes the other way. Identifiers are pseudonymous, the intended use is aggregate wellbeing insight, and the evidence behind a conclusion is visible to the person described, not just to HR. We treat that as a design constraint rather than a policy afterthought."],
  ["Saumya", "Why RAG rather than just fine-tuning a model on burnout data?",
   "Fine-tuning would bury the reasoning in weights, which is exactly the problem we set out to solve. Retrieval keeps the evidence outside the model where it can be shown and checked, and it means the system updates when the data updates, with no retraining."],
  ["Aryaman", "The quantum result is negative. Was it worth doing?",
   "Yes, for two reasons. It answers the question our mentor asked with measured evidence rather than an opinion. And the agreement between two independent framework implementations to within nought point two seven percent tells us the circuits are correct — a negative result you can trust is more useful than a positive one you cannot."],
  ["Swara", "Why did you build your own contract instead of calling the pipeline directly?",
   "Calling directly would have coupled four people's work to one another's runtime, and re-running a two-gigabyte pipeline for every test. A file boundary let all four workstreams proceed in parallel, and it made the read-only guarantee verifiable rather than merely intended."],
  ["Aryaman", "Can it handle a language other than English?",
   "Not today. The sentiment checkpoint and the NRC lexicon are both English. The architecture does not assume English anywhere — swapping in a multilingual encoder and lexicon would be a configuration change — but we would not claim it works until we had tested it."],
  ["Anushree", "How large is the dataset you actually process?",
   "Half a million Enron emails, twenty-two thousand survey responses and two thousand stress records. For interactive demonstration we index a working subset, because the full corpus is two point one gigabytes; scaling the index to the whole corpus is in our future work."],
  ["Saumya", "What is genuinely novel here, as opposed to engineering?",
   "The separation itself. Prior work treats detection and explanation as one modelling problem. We treat them as two, with different correctness criteria, joined by a contract that prevents the fluent component overruling the accountable one. That is an architectural claim, and it is the one we are putting in the paper."],
  ["Aryaman", "Which of your defects worried you most?",
   "The emotion mislabelling. Six of ten records were being labelled angry with no emotion word present. A silent database failure costs you an afternoon; telling an organisation that someone is angry when they are not is the kind of error that harms a person."],
];

kids.push(tbl(["Lead", "Question", "Answer"],
  QA.map(q => ["**" + q[0], q[1], q[2]]), [0.75, 1.75, 4.0]));

kids.push(P("", { after: 200 }));
kids.push(H2("If the demo is requested"));
kids.push(P("Have all three running before you walk in: the backend on port 8000, the dashboard on port 8501, and the model provider. Open the dashboard in a clean browser window with no bookmarks bar. Show three things in this order and nothing else — the gauge and risk distribution, one natural-language question with the retrieved evidence panel open beneath the answer, and one PDF export. Two minutes, then stop and return to the deck.", { size: 10.5 }));

kids.push(H2("If something breaks live"));
kids.push(P("Say what the system is designed to do and let the design cover you: “If the provider is unreachable the backend returns a rule-based analysis flagged as degraded, which is what you are seeing — the dashboard stays usable.” Then move on. A graceful failure demonstrated live is better evidence than a demo that never wobbles.", { size: 10.5 }));

// ============================== BUILD ==============================
const doc = new Document({
  creator: "Capstone Team",
  title: "Second Review — Presentation Script",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1100, bottom: 1000, left: 1150, right: 1150 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Second Review — Presentation Script          ", font: FONT, size: 16, color: GREY }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GREY }),
          ],
        })],
      }),
    },
    children: kids,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("Second_Review_Script.docx", b);
  console.log("Second_Review_Script.docx", b.length, "bytes");
});
