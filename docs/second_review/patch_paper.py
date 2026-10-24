"""Add the quantum study to the capstone paper, in place, preserving styling.

Edits word/document.xml directly rather than regenerating the document, so the
IEEE layout, author table, existing tables and figures are untouched.
"""
import re
from pathlib import Path

DOC = Path("word/document.xml")
x = DOC.read_text(encoding="utf8")


# ------------------------------------------------------------------ helpers
def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_PID = [0x900]


def pid() -> str:
    _PID[0] += 1
    return f"{_PID[0]:08X}"


def P(body: str) -> str:
    return (f'<w:p w:rsidR="00000000" w:rsidDel="00000000" w:rsidP="00000000" '
            f'w:rsidRDefault="00000000" w:rsidRPr="00000000" '
            f'w14:paraId="{pid()}">{body}</w:p>')


def heading(text: str) -> str:
    """Centred small-caps section heading, matching existing sections."""
    return P('<w:pPr><w:spacing w:after="60" w:before="120" w:lineRule="auto"/>'
             '<w:jc w:val="center"/><w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:smallCaps w:val="1"/><w:rtl w:val="0"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
             '<w:r><w:rPr><w:rtl w:val="0"/></w:rPr></w:r>')


def subheading(text: str) -> str:
    """Italic left-aligned subsection heading."""
    return P('<w:pPr><w:spacing w:after="40" w:before="80" w:lineRule="auto"/>'
             '<w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:i w:val="1"/><w:iCs w:val="1"/><w:rtl w:val="0"/>'
             f'</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
             '<w:r><w:rPr><w:rtl w:val="0"/></w:rPr></w:r>')


def body(text: str, indent: bool = True) -> str:
    """Justified body paragraph with first-line indent."""
    ind = '<w:ind w:firstLine="200"/>' if indent else ""
    return P('<w:pPr><w:spacing w:after="10" w:before="120" w:lineRule="auto"/>'
             f'{ind}<w:jc w:val="both"/><w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:rtl w:val="0"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def equation(text: str) -> str:
    """Centred display expression."""
    return P('<w:pPr><w:spacing w:after="80" w:before="80" w:lineRule="auto"/>'
             '<w:jc w:val="center"/><w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:i w:val="1"/><w:rtl w:val="0"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def caption(label: str, title: str) -> str:
    out = P('<w:pPr><w:spacing w:after="0" w:before="160" w:lineRule="auto"/>'
            '<w:jc w:val="center"/><w:rPr/></w:pPr>'
            '<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/>'
            f'<w:rtl w:val="0"/></w:rPr><w:t xml:space="preserve">{esc(label)}'
            '</w:t></w:r>')
    out += P('<w:pPr><w:spacing w:after="60" w:before="0" w:lineRule="auto"/>'
             '<w:jc w:val="center"/><w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:smallCaps w:val="1"/><w:sz w:val="16"/>'
             f'<w:szCs w:val="16"/><w:rtl w:val="0"/></w:rPr>'
             f'<w:t xml:space="preserve">{esc(title)}</w:t></w:r>')
    return out


def cell(text: str, width: int, header: bool, align: str = "center") -> str:
    shade = '<w:shd w:fill="dce6f1" w:val="clear"/>' if header else ""
    bold = '<w:b w:val="1"/><w:bCs w:val="1"/>' if header else ""
    # Schema order inside <w:tcPr> is fixed: tcW, then shd, then tcMar.
    # Emitting shd before tcW fails XSD validation.
    return (f'<w:tc><w:tcPr><w:tcW w:w="{width}.0" w:type="dxa"/>{shade}'
            '<w:tcMar><w:top w:w="40.0" w:type="dxa"/>'
            '<w:left w:w="70.0" w:type="dxa"/>'
            '<w:bottom w:w="40.0" w:type="dxa"/>'
            '<w:right w:w="70.0" w:type="dxa"/></w:tcMar></w:tcPr>'
            + P(f'<w:pPr><w:jc w:val="{align}"/><w:rPr/></w:pPr>'
                f'<w:r><w:rPr>{bold}<w:sz w:val="14"/><w:szCs w:val="14"/>'
                f'<w:rtl w:val="0"/></w:rPr>'
                f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')
            + '</w:tc>')


def table(headers, rows, widths) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = ('<w:tbl><w:tblPr><w:tblStyle w:val="Table1"/>'
           f'<w:tblW w:w="{sum(widths)}.0" w:type="dxa"/>'
           '<w:jc w:val="left"/><w:tblLayout w:type="fixed"/>'
           f'<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>')
    out += ('<w:tr><w:trPr><w:cantSplit w:val="0"/>'
            '<w:tblHeader w:val="1"/></w:trPr>'
            + "".join(cell(h, w, True) for h, w in zip(headers, widths))
            + '</w:tr>')
    for r in rows:
        out += ('<w:tr><w:trPr><w:cantSplit w:val="0"/>'
                '<w:tblHeader w:val="0"/></w:trPr>'
                + "".join(cell(str(c), w, False,
                               "left" if i == 0 else "center")
                          for i, (c, w) in enumerate(zip(r, widths)))
                + '</w:tr>')
    return out + '</w:tbl>'


def reference(num: str, text: str) -> str:
    return P('<w:pPr><w:spacing w:after="10" w:lineRule="auto"/>'
             '<w:ind w:left="260" w:hanging="260"/><w:jc w:val="both"/>'
             '<w:rPr/></w:pPr>'
             '<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/>'
             f'<w:rtl w:val="0"/></w:rPr><w:t xml:space="preserve">{esc(num)}'
             f'</w:t><w:tab/><w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def replace_text(old: str, new: str, count: int = 1) -> None:
    """Swap the contents of a <w:t> node."""
    global x
    needle = f'<w:t xml:space="preserve">{esc(old)}</w:t>'
    assert needle in x, f"text not found: {old[:60]}"
    x = x.replace(needle, f'<w:t xml:space="preserve">{esc(new)}</w:t>', count)


# ============================================================ THE NEW SECTION
SEC = []
SEC.append(heading("IX.  Quantum Machine Learning Comparative Study"))

SEC.append(subheading("A.  Motivation and Scope"))
SEC.append(body(
    "The study addresses a question separate from the detection pipeline: "
    "whether quantum machine learning confers any advantage on the structured "
    "burnout data. It is deliberately isolated from the rest of the system. It "
    "consumes the Employee Burnout Analysis dataset directly and touches "
    "neither the communication pipeline, nor the integration contract, nor the "
    "retrieval subsystem. The service and the interface operate with no "
    "quantum dependency installed; the module writes its results to a separate "
    "artifact which downstream components read only when present. A negative "
    "outcome therefore constrains nothing else in the architecture."))

SEC.append(subheading("B.  Encoding and Circuit Design"))
SEC.append(body(
    "Six attributes are selected — designation, resource allocation, "
    "mental fatigue score, gender, company type and remote-working "
    "availability — and mapped to six qubits, one attribute per qubit. No "
    "dimensionality reduction is applied. The choice is methodological rather "
    "than incidental: projecting features onto principal components to fit a "
    "qubit budget discards variance and confounds the comparison, because the "
    "classical and quantum models would then no longer receive the same "
    "inputs. At six attributes and six qubits they receive identical inputs, "
    "and the kernel function remains the only variable."))
SEC.append(body(
    "Values are scaled to the interval [0, π] and encoded as single-qubit "
    "rotations. The variational ansatz applies two layers of rotations per "
    "qubit with controlled-NOT gates entangling neighbouring qubits in a ring "
    "whose wrap-around connection closes the loop. Estimated two-qubit depth "
    "is five. The circuit is declared once in a framework-neutral "
    "specification from which both implementations are constructed."))
SEC.append(body(
    "One asymmetry between the two variational implementations must be "
    "recorded. The PennyLane ansatz applies RY and RZ rotations per qubit per "
    "layer, giving 24 trainable parameters, whereas Qiskit\u2019s "
    "RealAmplitudes construction applies RY rotations only, giving 18. The "
    "two variational classifiers are therefore closely related but not "
    "identical, and the difference between their accuracies should be read "
    "with that in mind. The kernel comparison is unaffected, since the "
    "fidelity kernel has no trainable parameters at all \u2014 a further "
    "reason for treating the kernel rather than the variational classifier as "
    "the controlled element of the study."))

SEC.append(subheading("C.  Models and Protocol"))
SEC.append(body(
    "Four classical baselines — a majority-class predictor, logistic "
    "regression, a support vector machine with radial basis function kernel, "
    "and a random forest — are evaluated against two quantum models: a "
    "variational quantum classifier and a quantum kernel support vector "
    "machine. The quantum kernel is the state fidelity between encoded "
    "samples, and is supplied as a precomputed Gram matrix to the same "
    "support vector classifier used for the classical radial basis function "
    "model [17]. Optimiser, regularisation constant, random seed and data "
    "split are held identical across both. The kernel matrix is the sole "
    "quantity that differs, which makes the comparison controlled rather than "
    "merely parallel, and which is why the support vector machine rather than "
    "the variational classifier is the appropriate vehicle: its training "
    "objective is convex, so the solution is a deterministic function of the "
    "Gram matrix alone, with no initialisation or local minima to confound "
    "the result."))
SEC.append(body(
    "Each quantum model is implemented twice, in PennyLane [21] and in Qiskit "
    "[22]. Agreement between two independent implementations is what "
    "distinguishes a correct circuit from a defective one; a single framework "
    "returns a number with nothing to check it against."))

SEC.append(subheading("D.  Results"))
SEC.append(body(
    "Table VII reports the outcome. The variational classifier underperforms "
    "every classical baseline. The quantum kernel reproduces the accuracy of "
    "the classical radial basis function kernel on the same 400 rows to four "
    "decimal places, at 0.8560, and the two framework implementations agree "
    "to within 0.27 per cent, which indicates that the circuits are correctly "
    "constructed rather than defective."))
SEC.append(body(
    "The computational cost is not comparable. Fitting and evaluating the "
    "classical kernel on those 400 rows takes 0.004 seconds; the PennyLane "
    "quantum kernel takes 1.81 seconds and the Qiskit quantum kernel 267.81 "
    "seconds. The corresponding ratios are of the order of 10\u00b2 and "
    "10\u2074 respectively, and are quoted as orders of magnitude rather than "
    "precise factors because the classical denominator is small enough to be "
    "dominated by timing noise. The complete benchmark occupies 528.3 "
    "seconds, of which the four classical models account for less than 0.2 "
    "seconds in total."))
SEC.append(caption("TABLE VII", "Classical and Quantum Model Comparison"))
SEC.append(table(
    ["Model", "Framework", "Accuracy", "F1", "Time (s)"],
    [["Majority baseline", "sklearn", "0.5813", "0.4274", "0.00"],
     ["Logistic Regression", "sklearn", "0.8667", "0.8667", "0.03"],
     ["SVM (RBF)", "sklearn", "0.8587", "0.8586", "0.01"],
     ["Random Forest", "sklearn", "0.8533", "0.8534", "0.18"],
     ["SVM-RBF (400 rows)", "sklearn", "0.8560", "0.8551", "0.004"],
     ["VQC", "PennyLane", "0.7067", "0.7006", "15.04"],
     ["Quantum kernel", "PennyLane", "0.8560", "0.8557", "1.81"],
     ["VQC", "Qiskit", "0.7227", "0.7084", "239.50"],
     ["Quantum kernel", "Qiskit", "0.8533", "0.8514", "267.81"]],
    # Column tables in this template are 5040 dxa wide. A wider table
    # overflows the column and is clipped.
    [1560, 1080, 880, 780, 740]))

SEC.append(subheading("E.  Analytical Explanation of the Null Result"))
SEC.append(body(
    "The equality between the quantum and classical kernels is not "
    "coincidental, and does not require an empirical explanation. The angle "
    "feature map applies one rotation per qubit and no entangling operation, "
    "so the encoded state is separable. The fidelity consequently factorises "
    "across qubits as"))
SEC.append(equation("K(x, y) = ∏ cos²((xᵢ − yᵢ) / 2),"))
SEC.append(body(
    "which is computable classically in time linear in the number of "
    "attributes. The map therefore admits no quantum advantage by "
    "construction. The expression is furthermore a smooth, monotonically "
    "decreasing function of the elementwise difference, which places it in "
    "the same family of distance-based similarity measures as the radial "
    "basis function kernel. That the two yield identical accuracy is the "
    "predicted outcome rather than a surprising one. The factorisation was "
    "verified numerically against the simulated circuit to a maximum absolute "
    "deviation of 4.4 × 10⁻¹⁶.", indent=False))
SEC.append(body(
    "The result is reported because it locates the limitation correctly. A "
    "quantum advantage requires a feature map inducing a similarity structure "
    "that is simultaneously informative for the data and hard to compute "
    "classically. The second-order map of Havlíček et al. [16], "
    "which entangles rotations on products of attributes, satisfies the "
    "second condition; it is implemented in the present codebase but has not "
    "yet been evaluated at scale. A further consideration is that the "
    "capacity of a kernel machine is governed by the margin relative to the "
    "data radius rather than by the dimension of the feature space, so the "
    "exponential dimension of the quantum Hilbert space confers no advantage "
    "in itself."))

SEC.append(subheading("F.  Execution on a National Quantum Platform"))
SEC.append(body(
    "All figures above derive from local state-vector simulation, in which "
    "fidelity is computed analytically. Physical devices and remote "
    "simulators instead estimate it from a finite number of measurement "
    "shots. To quantify that difference the kernel was ported to Qniverse, "
    "the quantum computing platform of the Centre for Development of Advanced "
    "Computing under the Ministry of Electronics and Information Technology "
    "[20]."))
SEC.append(body(
    "The port is not a translation of the classifier. The platform exposes "
    "gate-level circuit construction and a job submission interface, so the "
    "variational classifier — which would require one remote job per "
    "parameter per gradient step — was not ported. The kernel was, "
    "because each entry reduces to a single circuit in which the sample is "
    "encoded, the inverse encoding of a second sample is applied, and the "
    "probability of the all-zero measurement outcome is the fidelity "
    "estimate."))
SEC.append(body(
    "The cost model governs the experimental design. One circuit constitutes "
    "one job, and a Gram matrix over n training rows requires n(n − 1)/2 "
    "jobs together with a further product of the test and training counts. "
    "Three measures reduce that cost without altering the result. The "
    "diagonal is never submitted, since a state has unit fidelity with "
    "itself. Only the upper triangle of the symmetric block is submitted, the "
    "kernel being symmetric. Every entry is cached by content hash, so a "
    "resumed or repeated run never resubmits a completed pair. Finite "
    "sampling perturbs the Gram matrix away from positive semi-definiteness; "
    "the smallest eigenvalues are therefore clipped to zero before fitting, "
    "and the magnitude of that correction is recorded as an indicator of "
    "shot adequacy."))

SEC.append(subheading("G.  Threats to Validity"))
SEC.append(body(
    "Three limitations qualify the comparison. First, the classical support "
    "vector machine receives a kernel bandwidth selected by a library "
    "heuristic, whereas the quantum kernel bandwidth is fixed implicitly by "
    "the encoding range and was not tuned; the comparison is therefore "
    "between a tuned classical model and an untuned quantum one, and "
    "bandwidth is known to dominate quantum kernel performance [19]. Second, "
    "results derive from a single train-test split without repetition, so the "
    "difference between the closest models is not established as "
    "statistically significant. Third, the reported cost ratios measure "
    "classical simulation of quantum circuits rather than execution on "
    "quantum hardware, where the binding constraints are shot count and queue "
    "latency instead."))

NEW = "".join(SEC)

# ----------------------------------------------------- insert before Future Work
anchor = '<w:t xml:space="preserve">VII.  Future Work</w:t>'
assert anchor in x, "Future Work heading not found"
p_start = x.rfind("<w:p ", 0, x.find(anchor))
x = x[:p_start] + NEW + x[p_start:]

# ------------------------------------------------------------- fix numbering
replace_text("VII.  Future Work", "X.  Future Work")

# ------------------------------------------------- abstract, contributions etc.
replace_text(
    "reporting of flaws, four of which remain unfixed upstream.",
    "reporting of flaws, four of which remain unfixed upstream; and")
CONTRIB_ANCHOR = ('<w:t xml:space="preserve">'
                  + esc("reporting of flaws, four of which remain unfixed "
                        "upstream; and") + '</w:t>')
ci = x.find(CONTRIB_ANCHOR)
cp_end = x.find("</w:p>", ci) + 6
bullet = x[x.rfind("<w:p ", 0, ci):cp_end]
new_bullet = re.sub(
    r'<w:t xml:space="preserve">[^<]*</w:t>',
    '<w:t xml:space="preserve">' + esc(
        "a controlled two-framework comparison of quantum and classical "
        "models, reported as a null result with an analytical explanation.")
    + '</w:t>', bullet, count=1)
new_bullet = re.sub(r'w14:paraId="[0-9A-Fa-f]+"', f'w14:paraId="{pid()}"',
                    new_bullet)
x = x[:cp_end] + new_bullet + x[cp_end:]

# ------------------------------------------------------------------ references
REFS = [
    ("[16]", "V. Havlíček, A. D. Córcoles, K. Temme, A. W. "
             "Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "
             "“Supervised learning with quantum-enhanced feature "
             "spaces,” Nature, vol. 567, pp. 209–212, 2019."),
    ("[17]", "M. Schuld and N. Killoran, “Quantum machine learning in "
             "feature Hilbert spaces,” Physical Review Letters, vol. 122, "
             "no. 4, 040504, 2019."),
    ("[18]", "J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. "
             "Neven, “Barren plateaus in quantum neural network training "
             "landscapes,” Nature Communications, vol. 9, 4812, 2018."),
    ("[19]", "R. Shaydulin and S. M. Wild, “Importance of kernel "
             "bandwidth in quantum machine learning,” Physical Review A, "
             "vol. 106, no. 4, 042407, 2022."),
    ("[20]", "Qniverse, Quantum Technology Group, Centre for Development of "
             "Advanced Computing (C-DAC), Bengaluru, Ministry of Electronics "
             "and Information Technology, Government of India, 2026."),
    ("[21]", "V. Bergholm et al., “PennyLane: Automatic differentiation "
             "of hybrid quantum-classical computations,” arXiv:1811.04968, "
             "2018."),
    ("[22]", "Qiskit contributors, “Qiskit: An open-source framework for "
             "quantum computing,” 2024."),
]
last = '<w:t xml:space="preserve">[15]</w:t>'
assert last in x, "reference [15] not found"
li = x.find(last)
lp_end = x.find("</w:p>", li) + 6
x = x[:lp_end] + "".join(reference(n, t) for n, t in REFS) + x[lp_end:]

# ------------------------------------------- abstract / setup / future / conclusion
def append_to(tail_fragment: str, extra: str) -> None:
    """Append a sentence to the <w:t> node whose text ends with `tail_fragment`."""
    global x
    pat = re.compile(r'<w:t xml:space="preserve">([^<]*'
                     + re.escape(esc(tail_fragment)) + r')</w:t>')
    m = pat.search(x)
    assert m, f"anchor not found: {tail_fragment[:50]}"
    x = (x[:m.start()]
         + f'<w:t xml:space="preserve">{m.group(1)}{esc(extra)}</w:t>'
         + x[m.end():])




# Experimental setup: the quantum configuration.
append_to(
    "since otherwise the dashboard aborts mid-generation.",
    " The quantum study is configured separately. Six attributes are encoded "
    "on six qubits with two variational layers and ring entanglement, giving "
    "24 trainable parameters, and no dimensionality reduction is applied. "
    "Quantum results are obtained on state-vector simulators under PennyLane "
    "0.45.1 and Qiskit 2.5.2 with qiskit-machine-learning 0.9.1 on Python "
    "3.14, over 1,500 sampled rows split 1,125 train and 375 test, with the "
    "quantum kernel capped at 400 training rows, and are additionally "
    "executed with finite-shot sampling on the Qniverse platform [20].")

# Future work: quantum directions.
append_to(
    "compared within a single formula.",
    " Within the quantum study three steps follow from Section IX-G. The "
    "second-order entangling feature map should be evaluated at scale, since "
    "the map used here is provably classically simulable. The kernel bandwidth "
    "should be swept rather than fixed implicitly by the encoding range, so "
    "that a tuned quantum model is compared against a tuned classical one "
    "[19]. Repetition across random seeds with confidence intervals would "
    "establish whether the observed differences are significant, which a "
    "single split cannot.")

# Conclusion: one closing sentence on the quantum result.
append_to(
    "is the substantive contribution.",
    " The accompanying quantum study reaches a null result and explains it: "
    "the encoding adopted yields a kernel that factorises into a classically "
    "computable product, so its agreement with the classical kernel was "
    "determined by construction rather than discovered by experiment. "
    "Reporting that derivation, rather than the accuracy figure alone, is "
    "what makes the comparison informative.")



ABSTRACT_OLD = 'Burnout is an occupational condition classified by WHO. It arises due to chronic and uncontrolled workplace stress. It is not a medical condition. AI-driven analytics are now increasingly being used to track workforce wellbeing. Such framing dictates how any system should be able to detect and report on burnout. The currently existing detection method is based on infrequent self-reporting inventories. It is prone to response biases. It yields a numerical score without any explanation. Neural solutions increase coverage. However, at the cost of one opacity for another.This paper introduces an explainable burnout detection system. It disentangles detection and explanation process. A four-stage pipeline calculates workplace communication score using a transformer sentiment classifier, a word-emotion lexicon, and behavioral features that are combined using a rule-based formula. Each point can be traced to a named signal. The Retrieval-Augmented Generation system then explains the scores in natural language. It references concrete records for each answer. Users get access to the evidence rather than unsubstantiated claims of the generator. The two systems communicate through a read-only file. It guarantees that the scoring model remains authoritative. While the language model is restricted to explanation only. The system was verified using 86 automated tests. It gracefully fails if the language model is unavailable. Using evaluation data, it yielded reliable risk stratification. Six resolved and four open integration issues were discovered.'
ABSTRACT_NEW = 'Burnout is classified by the World Health Organization as an occupational phenomenon rather than a medical condition, a distinction that constrains what an automated system may legitimately claim. Detection in practice rests on periodic self-report inventories, administered infrequently, vulnerable to response bias, and silent on why a score was produced. Neural classifiers widen coverage but substitute one form of opacity for another. This paper presents a burnout detection system that separates measurement from explanation and enforces that separation architecturally. A four-stage pipeline scores workplace communication using a transformer sentiment classifier, a word-emotion lexicon and surface behavioural features, combined by a transparent additive rule in which every point is attributable to a named signal by construction. A retrieval-augmented language model then explains those scores and returns the specific records supporting each statement, so that a reader can check the reasoning rather than trust it. The two stages communicate through a single read-only file, which keeps the scoring stage authoritative and leaves the language model no means of altering a risk band. The implementation is verified by 151 automated tests and degrades to a rule-based analysis with explicit disclosure when generation is unavailable; of ten integration defects identified, six were repaired and four are documented as open. A quantum machine learning study conducted in two independent frameworks is reported alongside: the quantum kernel reproduces the classical result exactly at four orders of magnitude greater cost, and an analytical derivation shows why the chosen encoding determined that outcome in advance.'
INTRO1_OLD = 'Workplace communication now leaves a continuous digital trace, It is recognized now in ICD-11 entry QD85: burnout is classified as occupational, not medical, encompassing exhaustion, cynicism towards the job, and inefficacy [1]. Inference engines have to maintain this distinction when building their models.'
INTRO1_NEW = "The International Classification of Diseases, Eleventh Revision (ICD-11), is the World Health Organization's global standard for coding diseases and health-related conditions. Its entry QD85 defines burnout as an occupational phenomenon resulting from chronic workplace stress that has not been successfully managed, characterised by exhaustion, increased mental distance from or cynicism towards one's job, and reduced professional efficacy [1]. The entry places burnout among factors influencing health status rather than among diseases, and that boundary binds any automated system built on it: such a system may report occupational risk, but it may not issue a diagnosis. Workplace communication, meanwhile, now leaves a continuous digital trace in which these signals are expressed daily, far more often than any survey can sample."
INTRO2_OLD = 'Currently, the go-to solution for professionals is Maslach Burnout Inventory (MBI) that is reliable but periodic, self-reporting and unexplained no reasons for scoring are provided. Machine learning can increase comprehensiveness, achieving R² 0.9567 with ensembles by weighting workload, fatigue, and tenure [3], but the models are explainable only on population level, not for individuals the manager has to speak to.'
INTRO2_NEW = 'The instrument used in practice is the Maslach Burnout Inventory, which is reliable but administered periodically, dependent on self-report, and silent on why a particular score was produced [2]. Machine learning widens coverage, with ensemble models reaching an R² of 0.9567 by weighting workload, fatigue and tenure [3], but their explanations hold at the level of the population rather than of the individual a manager must actually speak to.'
INTRO3_OLD = 'In this paper we separate prediction and explanation into two tasks by using an explicit agreement: a transparent, rule-based formula calculates fully accountable score while retrieval-augmented language model provides explanations without any possibility of altering it, referencing the documents supporting each statement.'
INTRO3_NEW = 'This paper separates prediction from explanation and binds the two with an explicit contract: a transparent rule-based formula produces a fully accountable score, and a retrieval-augmented language model explains that score without any means of altering it, citing the records that support each statement.'
FIG11_OLD = 'Fig. 11.  Flowchart for system architecture.'
FIG11_NEW = 'Fig. 11.  End-to-end data flow: the four-stage scoring pipeline, the read-only contract at which it terminates, and the retrieval-augmented subsystem that consumes the resulting scores to produce explanations.'


# ====================== REVISIONS REQUESTED BY THE PROJECT GUIDE ============
# 1. Abstract rewritten: the original was 28 clipped sentences, several of
#    which carried no content ("Such framing dictates how any system should be
#    able to detect and report on burnout"), one was a fragment beginning
#    "While", and two ran together on a missing space.
# 2. Introduction opening repaired: it was a comma splice in which "It" had no
#    referent, and it cited ICD-11 and entry QD85 without defining either.
# 3. Figure 11 caption made specific about what the flowchart shows.
def replace_whole(old: str, new: str) -> None:
    global x
    needle = f'<w:t xml:space="preserve">{esc(old)}</w:t>'
    assert needle in x, f"replace_whole missed: {old[:70]}"
    x = x.replace(needle, f'<w:t xml:space="preserve">{esc(new)}</w:t>', 1)


replace_whole(ABSTRACT_OLD, ABSTRACT_NEW)
replace_whole(INTRO1_OLD, INTRO1_NEW)
replace_whole(INTRO2_OLD, INTRO2_NEW)
replace_whole(INTRO3_OLD, INTRO3_NEW)
replace_whole(FIG11_OLD, FIG11_NEW)
replace_whole("Contributions Include", "Contributions include")


# ============ REFLECT THE CURRENT CODEBASE: 151 TESTS, INGESTION ===========
# Table V reported 86 tests across five modules. `pytest --collect-only` now
# reports 151 across six, the additional module being document ingestion.
# That component was never described in the paper, so a table row for it
# would reference something the reader has not been told about. Section V-F
# is added first; the table is rebuilt second.
ING = []
ING.append(subheading("F.  Document Ingestion"))
ING.append(body(
    "The subsystem accepts documents in addition to the committed corpus. A "
    "PDF is parsed for its text layer, and any page whose recovered text "
    "falls below a threshold is re-read by optical character recognition, "
    "because a scanned or photographed page yields an empty string silently "
    "rather than an error. Screenshots of conversations are read by the same "
    "path. Extracted text is divided into messages by whichever of four "
    "strategies first produces a plausible result — email headers, chat "
    "timestamps, speaker labels, or paragraph breaks — and the minimum "
    "message length is set per strategy, since a two-word reply carrying a "
    "strong signal is valid in a chat transcript and is not valid in an "
    "email body."))
ING.append(body(
    "Uploaded messages are scored by the same functions that produced the "
    "committed corpus. Those functions are lifted from the scoring module at "
    "run time by parsing its abstract syntax tree and evaluating only its "
    "function definitions, which avoids executing the module's top-level code "
    "and guarantees that uploaded and corpus records cannot diverge in how "
    "they are scored. Uploaded records are never written into the contract "
    "file."))

_vi = '<w:t xml:space="preserve">VI.  Experimental Setup</w:t>'
assert _vi in x, "section VI heading not found"
_p = x.rfind("<w:p ", 0, x.find(_vi))
x = x[:_p] + "".join(ING) + x[_p:]

replace_whole(
    "The automated suite comprises 86 tests distributed as shown in Table V, "
    "exercising the schema contract, the data layer, the vector store, the "
    "retrieval and generation pipeline and the service interface, including "
    "the degradation path.",
    "The automated suite comprises 151 tests distributed as shown in Table V, "
    "exercising the schema contract, the data layer, the vector store, the "
    "retrieval and generation pipeline, the document ingestion path and the "
    "service interface, including the degradation path.")

_cap = '<w:t xml:space="preserve">Automated Test Distribution</w:t>'
assert _cap in x, "Table V caption not found"
_ts = x.find("<w:tbl>", x.find(_cap))
_te = x.find("</w:tbl>", _ts) + 8
x = x[:_ts] + table(
    ["Module under test", "Tests", "Share"],
    [("Retrieval and generation", "37", "24.5%"),
     ("Response schemas", "33", "21.9%"),
     ("Service interface", "24", "15.9%"),
     ("Document ingestion", "22", "14.6%"),
     ("Data layer", "20", "13.2%"),
     ("Vector store", "15", "9.9%"),
     ("Total", "151", "100%")],
    [2620, 1180, 1240]) + x[_te:]


# =================== CUT TO SIX PAGES: FIGURE REDUCTION ====================
# The paper carried eleven figures. Ten were exploratory plots of the input
# corpora, supporting one sentence of analysis each, while occupying roughly
# a page and a half of a two-column layout. At six pages that is the least
# load-bearing content in the paper, so the corpus-characterisation figures
# are removed and only the three that carry argument are retained:
#   - burn rate distribution      (the target variable)
#   - fatigue versus burn rate    (the strongest single predictor)
#   - the architecture flowchart  (the contribution itself)
# Figures are renumbered 1-3 and the sentences introducing the removed plots
# are deleted rather than left dangling.

def _paragraphs():
    return [(m.start(), m.end())
            for m in re.finditer(r'<w:p [^>]*>.*?</w:p>', x, re.S)]


def drop_figure(caption_prefix: str) -> None:
    """Remove the caption paragraph and the image paragraph above it."""
    global x
    paras = _paragraphs()
    for i, (a, b) in enumerate(paras):
        seg = x[a:b]
        body_text = "".join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', seg))
        if body_text.strip().startswith(caption_prefix):
            prev_a, prev_b = paras[i - 1]
            assert "<wp:extent" in x[prev_a:prev_b], \
                f"paragraph above {caption_prefix} is not an image"
            x = x[:prev_a] + x[b:]
            return
    raise AssertionError(f"figure not found: {caption_prefix}")


def drop_paragraph(starts_with: str) -> None:
    global x
    for a, b in _paragraphs():
        body_text = "".join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', x[a:b]))
        if body_text.strip().startswith(starts_with):
            x = x[:a] + x[b:]
            return
    raise AssertionError(f"paragraph not found: {starts_with}")


for _fig in ("Fig. 9.", "Fig. 10.",
             "Fig. 1.", "Fig. 2.", "Fig. 3.",
             "Fig. 4.", "Fig. 5.", "Fig. 6."):
    drop_figure(_fig)

drop_paragraph("Fig. 1–Fig. 6 characterise")

# Renumber the survivors and repair the sentence that referenced the removed
# distribution plots.
replace_whole("Fig. 7.  Distribution of burn rate in the Employee Burnout "
              "Analysis dataset.",
              "Fig. 1.  Distribution of burn rate in the Employee Burnout "
              "Analysis dataset.")
replace_whole("Fig. 8.  Mental fatigue score versus burn rate in the Employee "
              "Burnout Analysis dataset.",
              "Fig. 2.  Mental fatigue score versus burn rate in the Employee "
              "Burnout Analysis dataset.")
for _old, _new in ((FIG11_NEW, FIG11_NEW.replace("Fig. 11.", "Fig. 3.")),):
    replace_whole(_old, _new)

# The sentence pointed at four plots; two survive.
x = x.replace(esc("Fig. 7–Fig. 10 summarise their key distributions."),
              esc("Fig. 1 and Fig. 2 summarise the distribution of the "
                  "target variable and its strongest single correlate."), 1)

# ---- prose trims to reach six pages -------------------------------------
# Related Work carried a sentence of scene-setting before each citation;
# Future Work listed eight directions where the paper only defends five.
for _old, _new in [
    ("The Maslach Burnout Inventory was the work that established the "
     "three-part schema that still underlies today’s categorisation",
     "The Maslach Burnout Inventory established the three-part schema "
     "underlying current categorisation"),
]:
    if esc(_old) in x:
        x = x.replace(esc(_old), esc(_new), 1)



# ======================= CUT TO SIX PAGES: PART 2 =========================
# Figures scaled down, and the longest paragraphs tightened. Nothing is
# removed here: every claim, citation and number survives. Several of these
# paragraphs also had defects of their own ("he system prompt", "conjuction",
# "NLU" used to mean the vector store), which are corrected in passing.

# ---- scale the surviving figures ----------------------------------------
def scale_figures(factor: float) -> None:
    global x
    def _ext(m):
        cx, cy = int(m.group(1)), int(m.group(2))
        return f'<wp:extent cx="{int(cx*factor)}" cy="{int(cy*factor)}"'
    x = re.sub(r'<wp:extent cx="(\d+)" cy="(\d+)"', _ext, x)
    def _xfrm(m):
        cx, cy = int(m.group(1)), int(m.group(2))
        return f'<a:ext cx="{int(cx*factor)}" cy="{int(cy*factor)}"'
    x = re.sub(r'<a:ext cx="(\d+)" cy="(\d+)"', _xfrm, x)


# Fig. 1 and Fig. 2 are simple distribution plots and survive heavy scaling.
# Fig. 3 is the architecture diagram and carries labelled text; scaling it to
# 0.72 rendered that text unreadable in print, which is a worse outcome than
# an extra page, so it is reduced only slightly.
def scale_nth_figure(index: int, factor: float) -> None:
    global x
    spans = [m.span() for m in re.finditer(r'<wp:extent cx="(\d+)" cy="(\d+)"', x)]
    if index >= len(spans):
        return
    a, b = spans[index]
    m = re.match(r'<wp:extent cx="(\d+)" cy="(\d+)"', x[a:b])
    cx, cy = int(m.group(1)), int(m.group(2))
    x = (x[:a] + f'<wp:extent cx="{int(cx*factor)}" cy="{int(cy*factor)}"' + x[b:])
    # the matching a:ext inside the drawing must move with it
    spans2 = [mm.span() for mm in re.finditer(r'<a:ext cx="(\d+)" cy="(\d+)"', x)]
    if index < len(spans2):
        a2, b2 = spans2[index]
        m2 = re.match(r'<a:ext cx="(\d+)" cy="(\d+)"', x[a2:b2])
        cx2, cy2 = int(m2.group(1)), int(m2.group(2))
        x = (x[:a2] + f'<a:ext cx="{int(cx2*factor)}" cy="{int(cy2*factor)}"'
             + x[b2:])


for _i, _f in ((0, 0.64), (1, 0.64), (2, 0.86)):
    scale_nth_figure(_i, _f)

# ---- tighten the longest paragraphs -------------------------------------
TIGHTEN = [
("All results are obtained on the committed ten-record corpus. Encoding runs on CPU by default, with optional acceleration on Metal or CUDA selected by configuration and silently falling back to CPU with a logged warning when the requested device is unavailable, so that a device setting committed by one collaborator cannot break the service for another. Generation defaults to a local model, with temperature 0.2, a maximum of 1600 output tokens and a 120-second timeout. Retrieval depth defaults to five with a ceiling of twenty-five. The interface timeout of 180 seconds is required to exceed the generation timeout, since otherwise the dashboard aborts mid-generation.",
 "All results are obtained on the committed ten-record corpus. Encoding runs on CPU by default, with optional Metal or CUDA acceleration that falls back to CPU with a logged warning when unavailable, so one collaborator's device setting cannot break the service for another. Generation defaults to a local model at temperature 0.2, 1600 output tokens and a 120-second timeout; retrieval depth defaults to five, capped at twenty-five; the 180-second interface timeout must exceed the generation timeout, or the dashboard aborts mid-generation."),

("Encoding uses all-MiniLM-L6-v2, a six-layer sentence transformer producing 384-dimensional vectors from a model of approximately eighty megabytes [14]. Three properties of the implementation are consequential. Vectors are L2-normalised at encode time, which reduces cosine similarity to a dot product and, in combination with a cosine-configured index, makes the returned distance exactly one minus similarity, yielding a bounded relevance score in the unit interval that can be shown to a non-technical reader rather than an unbounded distance that means nothing on a dashboard. The model is loaded lazily behind a double-checked lock, because a service that loads the encoder at import time would download eighty megabytes during test collection and would load two copies if two first requests arrived concurrently on a thread pool. The class implements the embed_documents and embed_query interface without importing the orchestration framework, so that a framework version change cannot break encoding.",
 "Encoding uses all-MiniLM-L6-v2, a six-layer sentence transformer producing 384-dimensional vectors from an eighty-megabyte model [14]. Three implementation properties are consequential. Vectors are L2-normalised at encode time, reducing cosine similarity to a dot product and, with a cosine-configured index, making the returned distance exactly one minus similarity: a bounded relevance score in the unit interval, rather than an unbounded distance that means nothing on a dashboard. The model loads lazily behind a double-checked lock, since loading at import time would fetch eighty megabytes during test collection and load two copies under concurrent first requests. The class implements the embedding interface without importing the orchestration framework, so a framework version change cannot break encoding."),

("Retrieval takes the question, optionally applies a metadata filter, and produces the k closest items along with the similarity. There are three peculiarities to mention. A concrete identification of the employee in the question is recognized by means of a regular expression that handles variations of separators and extra spaces, and transforms the employee id into the canonical form with zeroes, so it would match the synthesized ids. Several filter conditions are combined with the conjunction explicitly, since otherwise the store will treat it as a multihash lookup with multiple keys in it, and apply just one of those filters; a simple condition is emitted without conjunctions, since in some versions of NLU, a one-item conjuction is not recognized. Depth asked for is limited between one and twenty-five; it ensures that asking for any larger depth does not lead to context window overflow on the provider side.",
 "Retrieval takes the question, optionally applies a metadata filter, and returns the k closest records with their similarity. Three details matter. An employee identifier named in the question is recognised by a regular expression tolerant of separator and spacing variants and normalised to the zero-padded canonical form, so that it matches the synthesised identifiers. Multiple filter conditions are combined under an explicit conjunction, because the store otherwise treats them as a multi-key lookup and applies only one; a single condition is emitted without the conjunction, which some versions reject. Depth is clamped to between one and twenty-five, so that an excessive request cannot overflow the provider's context window."),

("he system prompt fulfills four separate roles that correspond to different types of failure modes. It insists that every sentence must be based on the given records and not invent any employees, quotes, dates, events, job titles or metrics, since an invented particularity of a burnout report can send a manager to discuss something that did not happen. It prohibits deriving risk level again, insisting that the label is repeated verbatim from the pipeline output, since a classifier that disagreed silently would create disagreement between the dashboard and the model. It prohibits diagnostic and clinical terms, in accord with the occupational instead of medical nature of the phenomenon [1]. And it says that any text inside of a record is data and not an instruction so that a message with instructions is interpreted as information on the author rather than a request for the model.",
 "The system prompt serves four roles, each answering a distinct failure mode. It requires every sentence to rest on the supplied records and forbids inventing employees, quotes, dates, events, job titles or metrics, since an invented detail could send a manager to discuss something that never happened. It forbids re-deriving the risk level, requiring the label to be echoed verbatim, since a silently disagreeing classifier would put the dashboard and the model at odds. It forbids diagnostic and clinical vocabulary, consistent with the occupational rather than medical framing [1]. And it declares record text to be data rather than instruction, so that a message containing commands is read as evidence about its author."),

("Retrieval-Augmented Generation combines parametric knowledge held in model weights with non-parametric knowledge retrieved from an external index at inference time, and was shown to improve performance and provenance on knowledge-intensive tasks [9]. Dense passage retrieval established that learned dual encoders outperform sparse lexical baselines such as the probabilistic relevance framework underlying BM25 [10], [11]. Efficient similarity search over the resulting vectors is supported by hierarchical navigable small world graphs [12] and by GPU-accelerated billion-scale indexing [13]. Sentence-BERT provides the sentence-level encoder family used here, deriving semantically meaningful fixed-length embeddings from siamese and triplet networks that can be compared by cosine similarity, reducing the cost of exhaustive pairwise comparison from hours to seconds [14].",
 "Retrieval-Augmented Generation combines parametric knowledge in model weights with non-parametric knowledge retrieved at inference time, improving both performance and provenance on knowledge-intensive tasks [9]. Dense passage retrieval established that learned dual encoders outperform sparse lexical baselines such as BM25 [10], [11], and efficient search over the resulting vectors is supported by hierarchical navigable small world graphs [12] and GPU-accelerated billion-scale indexing [13]. Sentence-BERT supplies the encoder family used here, deriving fixed-length embeddings comparable by cosine similarity and reducing exhaustive pairwise comparison from hours to seconds [14]."),

("The immediate priority is repair of the four open upstream defects, followed by regeneration of the scored outputs from the full corpus [15]. Beyond that, five directions follow directly from the limitations. Confidence-weighted scoring would multiply the sentiment contribution by its retained probability, which the present analysis indicates would materially reduce the influence of the misclassification identified in Section VIII-D. Temporal aggregation over a per-person window would allow the chronicity central to the construct to be modelled, and requires the identity synthesis to move from message to author. Validation against the standard inventory on a consenting cohort would convert face validity into measured construct validity [2]. Retrieval quality warrants formal evaluation, with a hybrid of dense and sparse retrieval [10], [11] compared against the present dense-only configuration, and with faithfulness of the generated analysis to the retrieved evidence measured rather than assumed. Finally, the fusion stage should admit the structured organisational attributes that ensemble studies identify as dominant predictors [3], which would allow communication-derived and operational signals to be compared within a single formula. Within the quantum study three steps follow from Section IX-G. The second-order entangling feature map should be evaluated at scale, since the map used here is provably classically simulable. The kernel bandwidth should be swept rather than fixed implicitly by the encoding range, so that a tuned quantum model is compared against a tuned classical one [19]. Repetition across random seeds with confidence intervals would establish whether the observed differences are significant, which a single split cannot.",
 "The immediate priority is repair of the four open upstream defects and regeneration of the scored outputs from the full corpus [15]. Five directions follow from the limitations: confidence-weighted scoring, multiplying the sentiment contribution by its retained probability, which would damp the misclassification of Section VII-D; temporal aggregation over a per-person window, which would model the chronicity central to the construct and requires identity synthesis to move from message to author; validation against the standard inventory on a consenting cohort, converting face validity into measured construct validity [2]; formal retrieval evaluation, comparing hybrid dense-sparse retrieval [10], [11] against the present dense-only configuration and measuring faithfulness rather than assuming it; and admission of the structured organisational attributes that ensemble studies identify as dominant predictors [3] into the fusion stage. Within the quantum study, the entangling feature map should be evaluated at scale, since the map used here is provably classically simulable; the kernel bandwidth should be swept rather than fixed by the encoding range [19]; and results should be repeated across seeds with confidence intervals."),

("Evaluation on the committed ten-record corpus is reported in full and is candidly insufficient to support generalisation. What the evaluation does establish is that the attribution property holds, that the degradation path functions, and that the architecture surfaces defects rather than absorbing them: six integration defects were identified and repaired, and four upstream defects were documented and tolerated without fabricating the measurements they suppressed. In a system whose output concerns identified individuals, the discipline of preserving the distinction between absent and zero, and of never allowing a fluent generator to overrule an accountable measurement, is the substantive contribution.",
 "Evaluation on the committed ten-record corpus is reported in full and is candidly insufficient to support generalisation. It does establish that the attribution property holds, that the degradation path functions, and that the architecture surfaces defects rather than absorbing them: six integration defects were identified and repaired, and four upstream defects documented and tolerated without fabricating the measurements they suppressed. In a system whose output concerns identified individuals, preserving the distinction between absent and zero, and never letting a fluent generator overrule an accountable measurement, is the substantive contribution."),
]


TIGHTEN += [["The Maslach Burnout Inventory established the three-part schema underlying current categorisation [2], while the WHO classification provided a further categorisation of the issue as an occupational phenomenon, thus fixing the framework within which responsible use is limited [1]. Research on machine learning algorithms for burnout detection reveals an industry focused on self-reporting and HR data over natural language processing. The work carried out by the group of Xames is an example of the same in its success and its limitation [3].", "The Maslach Burnout Inventory established the three-part schema that still underlies current categorisation [2], and the WHO classification fixed burnout as an occupational phenomenon, bounding what responsible use permits [1]. Machine learning research on burnout detection is dominated by self-report and HR data rather than natural language; the work of Xames et al. exemplifies both the success and the limitation of that approach [3]."], ["Transformer language models provide the contemporary basis for sentiment classification. The architecture originates with the Transformer [4], was adapted to bidirectional pre-training by BERT [5], and was refined by RoBERTa through modified pre-training objectives and larger corpora [6]. The classifier used in this work is a RoBERTa checkpoint further pre-trained on a large Twitter corpus and fine-tuned for sentiment as part of the TweetEval benchmark [7]. Complementary to distributional sentiment, the NRC Word–Emotion Association Lexicon supplies crowd-sourced associations between English terms and eight basic emotions together with two polarities [8], offering counts that are directly inspectable in a way that neural activations are not.", "Transformer language models provide the contemporary basis for sentiment classification: the architecture originates with the Transformer [4], was adapted to bidirectional pre-training by BERT [5] and refined by RoBERTa through modified objectives and larger corpora [6]. The classifier used here is a RoBERTa checkpoint further pre-trained on Twitter text and fine-tuned for sentiment within TweetEval [7]. Complementing distributional sentiment, the NRC Word–Emotion Association Lexicon supplies crowd-sourced associations between English terms and eight basic emotions with two polarities [8], giving counts that are directly inspectable in a way neural activations are not."], ["Two structured datasets were reviewed as candidate benchmarks for the construct-validity work proposed in Section X: the Employee Burnout Analysis dataset (approximately 22,750 structured employee records covering mental fatigue score, burn rate, resource allocation, work-from-home status and designation) and the Indian Corporate Stress dataset (employee survey data covering stress level, work-life balance, job satisfaction, workload and burnout symptoms). Neither dataset is consumed by the present pipeline; both are shown here to characterise the kind of labelled ground truth against which the rule-based score could eventually be validated.", "Two structured datasets were reviewed as candidate benchmarks for the construct-validity work proposed in Section X: the Employee Burnout Analysis dataset (approximately 22,750 records covering mental fatigue, burn rate, resource allocation, work-from-home status and designation) and the Indian Corporate Stress dataset (stress level, work-life balance, job satisfaction, workload and symptoms). Neither is consumed by the present pipeline; both characterise the labelled ground truth against which the rule-based score could eventually be validated."], ["The system is organised as four stages with a single integration contract between the second and the third. Stage one performs data collection and preprocessing, emitting a corpus in which each row carries a normalised clean_text field. Stage two performs sentiment analysis, emotion analysis, behavioural feature extraction, feature fusion and burnout scoring, emitting outputs/burnout_scores.csv. Stage three is the retrieval-augmented backend. Stage four is the analyst-facing dashboard.", "The system is organised as four stages with a single integration contract between the second and the third. Stage one performs collection and preprocessing, emitting a corpus in which each row carries a normalised clean_text field. Stage two performs sentiment analysis, emotion analysis, behavioural feature extraction, fusion and burnout scoring, emitting outputs/burnout_scores.csv. Stage three is the retrieval-augmented backend; stage four the analyst-facing dashboard."], ["The contract is deliberately narrow. Stages three and four consume outputs/burnout_scores.csv read-only and never re-execute the scoring pipeline. Row reading is delegated to the loader supplied by stage two rather than reimplemented, so that the validation contract requiring the presence of clean_text is enforced in exactly one place. All aggregation used by both the backend and the dashboard is computed in a single module, which eliminates by construction the failure in which a dashboard displays one overall burnout figure while the language model reasons over another.", "The contract is deliberately narrow. Stages three and four consume outputs/burnout_scores.csv read-only and never re-execute the scoring pipeline. Row reading is delegated to the loader supplied by stage two rather than reimplemented, so the validation contract requiring clean_text is enforced in exactly one place. All aggregation used by the backend and the dashboard is computed in a single module, eliminating by construction the failure in which a dashboard shows one overall burnout figure while the language model reasons over another."], ["One point worth noting about this is that the identity of an employee is entirely absent from the upstream data itself. The scored data is one row per message without any employee identity. Since the specification for the dashboard requires ranking employees for high risks and searches against each employee's profile, the identity is generated deterministically based on the row number as EMP_0001, EMP_0002 etc., padded with zeros so that string sort order in dashboard tables is correctly reflected as numerical sort order. The important aspect is deterministic nature; otherwise, the record id generation will depend on the ordering and therefore could not survive across restarts. If in future any actual employee_id column comes up as part of the data then it will automatically use that.", "Employee identity is entirely absent from the upstream data: the scored file is one row per message with no identifier. Since the dashboard must rank employees by risk and search per employee, identifiers are generated deterministically from the row number as EMP_0001, EMP_0002 and so on, zero-padded so that string sort order in dashboard tables matches numerical order. Determinism is the essential property; an identifier derived from ordering alone would not survive a restart. Should an actual employee_id column later appear, it is used in preference."], ["The committed corpus is datasets/sample.csv, a ten-row development set containing a single clean_text column of short workplace utterances. The sentiment script is written against datasets/processed/enron_clean.csv and truncates its input to the first one thousand messages by an explicit development-mode statement; that file is not present in the repository, and the committed outputs therefore derive from the ten-row sample. This is stated plainly because the distinction between the intended corpus and the evaluated corpus materially bounds every result reported in Section VIII.", "The committed corpus is datasets/sample.csv, a ten-row development set containing a single clean_text column of short workplace utterances. The sentiment script is written against datasets/processed/enron_clean.csv and truncates to the first thousand messages by an explicit development-mode statement; that file is absent from the repository, so the committed outputs derive from the ten-row sample. This is stated plainly because the distinction between the intended and the evaluated corpus materially bounds every result in Section VII."], ["Each message is tokenised with truncation at 512 tokens and classified by a RoBERTa sequence classification head into Negative, Neutral or Positive [6], [7]. The softmax of the output logits is taken, the arg max index selects the label, and the corresponding probability is retained to four decimal places as a per-record confidence. Retaining the confidence rather than the label alone is what later permits a low-confidence classification to be identified as such in the evidence presented to the reader.", "Each message is tokenised with truncation at 512 tokens and classified by a RoBERTa sequence classification head into Negative, Neutral or Positive [6], [7]. The softmax of the logits is taken, the arg max selects the label, and the corresponding probability is retained to four decimal places as a per-record confidence. Retaining confidence rather than the label alone is what later allows a low-confidence classification to be flagged as such in the evidence shown to the reader."], ["The NRC Word–Emotion Association Lexicon is loaded from its word-level distribution, in which each line associates a term with an emotion and a binary indicator [8]. The file contains 141,540 such lines. Only associations with an indicator of one are retained, producing a mapping from term to emotion list. Each message is lower-cased and split on whitespace, and every token found in the lexicon increments the counter for each of its associated emotions across anger, anticipation, disgust, fear, joy, sadness, surprise and trust.", "The NRC Word–Emotion Association Lexicon is loaded from its word-level distribution, in which each of 141,540 lines associates a term with an emotion and a binary indicator [8]. Only associations with an indicator of one are retained, producing a term-to-emotion mapping. Each message is lower-cased and split on whitespace, and every token found in the lexicon increments the counter for each associated emotion across anger, anticipation, disgust, fear, joy, sadness, surprise and trust."], ["The semantics of this counter deserve emphasis, because they are frequently misread. The value is a word-match count and not an intensity. A count of two means that two matching tokens appeared in the message. It does not mean that the emotion was twice as strong. This distinction is carried explicitly into the system prompt of the explanation layer, so that the language model cannot silently reinterpret a frequency as a magnitude.", "The semantics of this counter deserve emphasis because they are frequently misread. The value is a word-match count, not an intensity: a count of two means two matching tokens appeared, not that the emotion was twice as strong. The distinction is carried explicitly into the explanation layer's system prompt, so the language model cannot silently reinterpret a frequency as a magnitude."], ["Three of these features are computed from a column named message. That column does not exist in the upstream emotion output, whose text field is clean_text. The script therefore raises a key error if executed against the current data. The consequence propagates: sentence_count is never written, and it is absent from the committed scored file. This defect is analysed in Section IX and is deliberately not repaired by the downstream layers.", "Three of these features are computed from a column named message, which does not exist in the upstream emotion output, whose text field is clean_text. The script therefore raises a key error against current data. The consequence propagates: sentence_count is never written and is absent from the committed scored file. This defect is analysed in Section VIII and deliberately not repaired downstream."], ["The fusion stage is presently a pass-through that copies the behavioural feature frame to outputs/employee_features.csv. It is retained as an explicit stage rather than elided, because it is the designated join point at which structured organisational attributes such as tenure, workload allocation or leave utilisation would be merged. The features identified as dominant in ensemble burnout prediction [3] are precisely the class of attribute this stage exists to admit.", "The fusion stage is presently a pass-through copying the behavioural feature frame to outputs/employee_features.csv. It is retained as an explicit stage rather than elided because it is the designated join point at which structured organisational attributes such as tenure, workload allocation or leave utilisation would be merged — precisely the class of attribute that ensemble studies identify as dominant [3]."], ["Risk stratification applies fixed thresholds to S, given in Table II. A configuration value of twelve is used as the nominal ceiling when rendering the score as a percentage for display, and is never used to classify risk; the risk label is always read verbatim from the pipeline output. This separation matters because a display normalisation constant that silently participated in classification would make the visual and the label disagree whenever the constant was retuned.", "Risk stratification applies fixed thresholds to S, given in Table II. A configuration value of twelve is the nominal ceiling when rendering the score as a percentage for display and is never used to classify; the risk label is always read verbatim from the pipeline output. The separation matters because a display constant that silently participated in classification would make the visual and the label disagree whenever it was retuned."], ["In parallel with the score, a textual explanation is assembled by appending a human-readable reason for each triggering condition, joined by commas, with a fallback string when no condition fires. This artefact serves two purposes. It is the per-record justification shown alongside the score, and it is the substrate from which a degraded response is constructed when the language model is unreachable, as described in Section VI. The explanation layer is therefore not decorative; it is load-bearing for availability.", "In parallel with the score, a textual explanation is assembled by appending a human-readable reason for each triggering condition, with a fallback string when none fires. It serves two purposes: the per-record justification shown alongside the score, and the substrate from which a degraded response is built when the language model is unreachable, as described in Section VI. The explanation layer is therefore load-bearing for availability, not decorative."], ["Each scored row is rendered into a natural-language document string before embedding, rather than being embedded as raw comma-separated values. Embedding a serialised sentence places the record in the same representational space as the analyst question, which is itself a natural-language sentence; embedding a delimiter-separated field list does not.", "Each scored row is rendered into a natural-language document string before embedding rather than as raw comma-separated values. A serialised sentence places the record in the same representational space as the analyst's question, which is itself a sentence; a delimiter-separated field list does not."], ["Relevant records will be presented as delimited evidence fragments carrying identifier, risk level, score, sentiment, dominant emotion, the rule-based reasoning and the similarity score. The prompt tells the model that emotion scores represent matching frequencies rather than intensity values and the recommendations must be specific.", "Relevant records are presented as delimited evidence fragments carrying identifier, risk level, score, sentiment, dominant emotion, the rule-based reasoning and the similarity. The prompt states that emotion scores are matching frequencies rather than intensities, and that recommendations must be specific."], ["The generation client supports a local runtime and two hosted providers behind a single configuration switch, with no code change required to move between them. Credentials are validated at import time for the selected provider only, so that a collaborator running a local model is never asked for a hosted key, and a missing credential surfaces as a startup error naming the exact variable to correct rather than as a server error during a later request. Secrets are masked to their final four characters in all diagnostic output, because the health endpoint is routinely captured in screenshots.", "The generation client supports a local runtime and two hosted providers behind a single configuration switch, with no code change to move between them. Credentials are validated at import time for the selected provider only, so a collaborator running a local model is never asked for a hosted key, and a missing credential surfaces as a startup error naming the exact variable rather than a server error during a later request. Secrets are masked to their final four characters in all diagnostic output, since the health endpoint is routinely captured in screenshots."], ["Complete per-record results are given in Table IV. Aggregate outcomes are six Negative and four Positive classifications, nine Low and one Medium risk record, no High risk record, a maximum score of three, and an overall burnout figure of ten percent. Sentiment confidence exhibits a pronounced asymmetry: positive classifications carry a mean confidence of 0.971, whereas negative classifications carry a mean of 0.720, a separation of approximately 0.25 that is consistent across every record and is discussed below.", "Complete per-record results are given in Table IV. Aggregate outcomes are six Negative and four Positive classifications, nine Low and one Medium risk record, no High risk record, a maximum score of three and an overall burnout figure of ten percent. Sentiment confidence is pronouncedly asymmetric: positive classifications carry a mean of 0.971 against 0.720 for negative, a separation of roughly 0.25 consistent across every record and discussed below."], ["EMP_0006 is the only record classified Medium. Its score of three decomposes exactly as two points for Negative sentiment plus one point for a single sadness token, with no behavioural condition satisfied. This decomposition is not an approximation recovered by a post-hoc attribution method; it is the arithmetic by which the score was produced, and it is recoverable from the stored feature values alone. That property is the central claim of the design.", "EMP_0006 is the only record classified Medium. Its score of three decomposes exactly as two points for Negative sentiment plus one for a single sadness token, with no behavioural condition satisfied. This is not an approximation recovered by post-hoc attribution; it is the arithmetic by which the score was produced, recoverable from the stored feature values alone. That property is the central claim of the design."], ["The eight emotion counters across the entire corpus total nine matches: joy three, trust two, anticipation two, sadness one and surprise one, with anger, fear and disgust at zero throughout. Six of the ten records have an entirely zero emotion vector. This sparsity is a direct consequence of message length, which ranges from four to seven tokens; a lexicon-matching approach over utterances of this length has almost no surface to act upon. The finding is important because it means the emotion term in (1) contributes to only four of ten scores, and the formula is, on this corpus, dominated by the sentiment term.", "The eight emotion counters across the corpus total nine matches: joy three, trust two, anticipation two, sadness one, surprise one, with anger, fear and disgust at zero throughout. Six of ten records have an entirely zero emotion vector. The sparsity follows directly from message length, which ranges from four to seven tokens; lexicon matching over utterances this short has almost no surface to act upon. The finding matters because the emotion term in (1) then contributes to only four of ten scores, leaving the formula dominated by sentiment on this corpus."], ["No behavioural condition fires on any record. Word counts range from four to seven against a threshold of 120, and uppercase ratios range from 0.026 to 0.050 against a threshold of 0.25. The behavioural component of the formula is therefore entirely untested by this corpus. The thresholds were evidently calibrated for full-length email correspondence and cannot be validated on utterances of this length.", "No behavioural condition fires on any record. Word counts range from four to seven against a threshold of 120, and uppercase ratios from 0.026 to 0.050 against 0.25. The behavioural component is therefore entirely untested by this corpus; its thresholds were evidently calibrated for full-length email and cannot be validated on utterances of this length."], ["An account of defects is included because their disposition illustrates the architectural principle more clearly than any successful path does. Six defects were found and repaired during integration; four remain open in the upstream pipeline and are tolerated rather than repaired. Table VI summarises the repaired set.", "Defects are reported because their disposition illustrates the architectural principle more clearly than any successful path. Six were found and repaired during integration; four remain open in the upstream pipeline and are tolerated rather than repaired. Table VI summarises the repaired set."], ["This paper presented a burnout detection system organised around a separation between measurement and explanation. Measurement is performed by a transparent integer formula over transformer sentiment, lexicon emotion counts and surface behavioural features, in which every point of every score is attributable to a named signal by construction rather than by post-hoc approximation. Explanation is performed by a retrieval-augmented language model that is architecturally prevented from altering the measurement, that returns the evidence supporting each claim, and that degrades to a rule-based answer with explicit disclosure when unavailable.", "This paper presented a burnout detection system organised around a separation between measurement and explanation. Measurement is performed by a transparent integer formula over transformer sentiment, lexicon emotion counts and surface behavioural features, in which every point is attributable to a named signal by construction rather than by post-hoc approximation. Explanation is performed by a retrieval-augmented language model architecturally prevented from altering the measurement, which returns the evidence supporting each claim and degrades to a rule-based answer with explicit disclosure when unavailable."]]


TIGHTEN += [('The intended full-scale corpus is the Enron email collection, introduced as a benchmark for email classification research and remaining the principal publicly available body of authentic organisational correspondence [15].', 'The intended full-scale corpus is the Enron email collection, the principal publicly available body of authentic organisational correspondence and a standing benchmark for email classification [15].'), ('The automated suite comprises 151 tests distributed as shown in Table V, exercising the schema contract, the data layer, the vector store, the retrieval and generation pipeline, the document ingestion path and the HTTP surface.', 'The automated suite comprises 151 tests distributed as shown in Table V, exercising the schema contract, data layer, vector store, retrieval and generation pipeline, ingestion path and HTTP surface.'), ('Each quantum model is implemented twice, in PennyLane [21] and in Qiskit [22]. Agreement between two independent implementations is what distinguishes a correct circuit from a defective one; a single implementation cannot establish this.', 'Each quantum model is implemented twice, in PennyLane [21] and Qiskit [22]. Agreement between two independent implementations is what distinguishes a correct circuit from a defective one; a single implementation cannot.'), ('The equality between the quantum and classical kernels is not coincidental, and does not require an empirical explanation. The angle feature map applies one rotation per qubit and no entangling operation, so the encoded state is separable across qubits.', 'The equality between the quantum and classical kernels is not coincidental and needs no empirical explanation. The angle feature map applies one rotation per qubit and no entangling operation, so the encoded state is separable across qubits.')]

_saved = 0
for _o, _n in TIGHTEN:
    if esc(_o) in x:
        _saved += len(_o) - len(_n)
        x = x.replace(esc(_o), esc(_n), 1)
    else:
        print(f"  [tighten] not found: {_o[:55]}...")
print(f"  tightened prose by {_saved} characters (~{_saved//6} words)")



# ---- drop optional DOI suffixes from references (IEEE permits omission) ---
_before = len(x)
x = re.sub(r',? ?doi: ?10\.[^<\s]*?(?=[<\s])', '', x)
x = re.sub(r'\. doi: ?10\.[^<]*?\.', '.', x)
print(f"  stripped DOIs: {_before - len(x)} characters")

DOC.write_text(x, encoding="utf8")
print(f"document.xml written: {len(x):,} chars")
print(f"new section paragraphs: {len(SEC)}")
print(f"references added: {len(REFS)}")
