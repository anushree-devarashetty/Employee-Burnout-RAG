"""
tests/test_ingest.py — Document ingestion
==============================================================================
Covers extraction, message splitting, and the reuse of Person 2's scoring.

The RoBERTa sentiment head is stubbed (it is a ~500MB download and a network
dependency). Everything else runs for real: PDF parsing, the NRC lexicon,
behavioural features, and Person 2's actual scoring functions.
==============================================================================
"""

from __future__ import annotations

import hashlib

import pytest

from backend.ingest.extractors import (
    ExtractionError,
    extract,
    extract_from_text,
    ocr_available,
)
from backend.ingest.message_splitter import MIN_CHARS_BY_STRATEGY, split_messages


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------
@pytest.fixture
def email_pdf_bytes() -> bytes:
    """A small PDF shaped like an exported email thread."""
    from io import BytesIO

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    style = getSampleStyleSheet()["Normal"]
    story = []
    for sender, subject, body in [
        ("priya@corp.com", "Sprint",
         "I am completely exhausted after working late every night."),
        ("anushree@corp.com", "Sprint",
         "Thanks for flagging this, I enjoy collaborating with the team."),
    ]:
        story += [
            Paragraph(f"From: {sender}", style),
            Paragraph("To: team@corp.com", style),
            Paragraph("Sent: Mon, 14 Jul 2026 22:14", style),
            Paragraph(f"Subject: {subject}", style),
            Spacer(1, 6),
            Paragraph(body, style),
            Spacer(1, 18),
        ]

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4).build(story)
    return buffer.getvalue()


# ------------------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------------------
def test_pdf_with_text_layer_does_not_use_ocr(email_pdf_bytes):
    result = extract(email_pdf_bytes, "emails.pdf")
    assert result.method == "pdf-text"
    assert result.ocr_used is False
    assert "exhausted after working late" in result.text


def test_unsupported_extension_is_rejected_with_a_helpful_message():
    with pytest.raises(ExtractionError) as exc:
        extract(b"data", "malware.exe")
    assert "unsupported" in str(exc.value).lower()
    assert ".pdf" in str(exc.value)


def test_empty_file_is_rejected():
    with pytest.raises(ExtractionError):
        extract(b"", "empty.pdf")


def test_oversized_file_is_rejected_before_parsing():
    from backend.ingest.extractors import MAX_UPLOAD_BYTES

    with pytest.raises(ExtractionError, match="limit"):
        extract(b"x" * (MAX_UPLOAD_BYTES + 1), "huge.pdf")


def test_windows_encoded_text_is_decoded():
    """Outlook exports are frequently cp1252, not UTF-8."""
    result = extract_from_text("Deadline – it's tough".encode("cp1252"), "a.txt")
    assert "tough" in result.text


def test_ligatures_are_normalised():
    """PDF extraction emits "ﬁ" as one glyph, which breaks lexicon lookups
    for exactly the words this project cares about."""
    result = extract_from_text("This is diﬃcult and ineﬃcient".encode(), "a.txt")
    assert "difficult" in result.text
    assert "inefficient" in result.text


def test_behavioural_punctuation_is_preserved():
    """Cleaning must not destroy features the burnout score depends on."""
    result = extract_from_text(b"THIS IS URGENT!!! Why?? ", "a.txt")
    assert result.text.count("!") == 3
    assert result.text.count("?") == 2
    assert "THIS IS URGENT" in result.text


def test_ocr_availability_reports_a_reason_when_missing():
    available, reason = ocr_available()
    assert available or reason


# ------------------------------------------------------------------------------
# Splitting
# ------------------------------------------------------------------------------
def test_email_split(email_pdf_bytes):
    text = extract(email_pdf_bytes, "emails.pdf").text
    result = split_messages(text, "emails.pdf")

    assert result.strategy == "email"
    assert result.count == 2
    assert result.messages[0].sender.startswith("priya")
    assert result.messages[0].subject == "Sprint"
    assert "From:" not in result.messages[0].text


def test_chat_split_with_timestamps():
    result = split_messages(
        "[09:14] Priya: running late\n"
        "[09:40] Anushree: no problem at all\n"
        "[09:41] Priya: I am completely exhausted",
        "chat.txt",
    )
    assert result.strategy == "chat"
    assert result.count == 3
    assert result.messages[2].sender == "Priya"
    assert result.messages[2].timestamp == "09:41"


def test_whatsapp_export_format():
    result = split_messages(
        "12/07/2026, 22:14 - Priya: I don't feel motivated anymore\n"
        "12/07/2026, 22:16 - Anushree: what is going on?",
        "wa.txt",
    )
    assert result.count == 2
    assert result.messages[0].sender == "Priya"


def test_short_messages_are_kept_in_chat_mode():
    """The most important splitter behaviour for this project.

    "not ok" (6 chars) and "I'm done" (8) are among the strongest burnout
    signals a person sends. A single global minimum length would discard them.
    """
    result = split_messages(
        "[09:00] A: not ok\n[09:01] B: what happened\n[09:02] A: I'm done", "s"
    )
    assert result.count == 3
    assert "not ok" in [m.text for m in result.messages]
    assert "I'm done" in [m.text for m in result.messages]


def test_paragraph_mode_keeps_a_higher_floor():
    """Without structural evidence, short blocks are page furniture."""
    assert MIN_CHARS_BY_STRATEGY["paragraph"] > MIN_CHARS_BY_STRATEGY["chat"]

    text = "Page 1 of 4\n\nHeading\n\n" + ("A real paragraph about deadlines. " * 4)
    assert split_messages(text, "n").count == 1


def test_duplicate_messages_are_removed():
    """Email threads repeat earlier messages inside each reply; counting them
    repeatedly would inflate a team's apparent burnout rate."""
    result = split_messages(
        "[09:00] A: the workload is overwhelming\n"
        "[09:01] B: that sounds hard\n"
        "[09:02] A: the workload is overwhelming",
        "d",
    )
    assert result.count == 2


def test_signature_boilerplate_is_stripped():
    result = split_messages(
        "From: a@b.com\nSubject: x\n\nI am exhausted.\nSent from my iPhone\n\n"
        "From: c@d.com\nSubject: y\n\nUnderstood, let us talk tomorrow.",
        "e",
    )
    assert all("Sent from my iPhone" not in m.text for m in result.messages)


def test_empty_document_returns_no_messages():
    assert split_messages("", "x").count == 0


def test_unsplittable_text_warns_about_dilution():
    result = split_messages("short", "x")
    assert result.strategy == "whole-document"
    assert any("diluted" in w for w in result.warnings)


# ------------------------------------------------------------------------------
# Reuse of Person 2's scoring
# ------------------------------------------------------------------------------
def test_person2_scoring_extracted_without_running_their_script():
    """The central integration guarantee of this feature.

    `models/burnout/burnout_score.py` executes top-level code that OVERWRITES
    `outputs/burnout_scores.csv`. Extracting its functions must not trigger
    that. This test hashes their file before and after to prove it.
    """
    from backend import REPO_ROOT
    from backend.ingest.pipeline_runner import load_person2_scoring

    csv_path = REPO_ROOT / "outputs" / "burnout_scores.csv"
    before = hashlib.md5(csv_path.read_bytes()).hexdigest()

    functions = load_person2_scoring()

    after = hashlib.md5(csv_path.read_bytes()).hexdigest()
    assert before == after, "Person 2's CSV was modified — a plain import ran"
    assert set(functions) == {"calculate_score", "risk_level", "explanation"}
    assert all(callable(f) for f in functions.values())


def test_uploaded_text_scores_identically_to_the_dataset():
    """A message from an upload must score the same as the same message in
    the CSV, or the dashboard would be comparing incomparable numbers."""
    import pandas as pd

    from backend.ingest.pipeline_runner import (
        behaviour_features,
        count_emotions,
        load_person2_scoring,
    )

    text = "I am completely exhausted after working late."
    row = {"clean_text": text, "sentiment": "Negative", "confidence": 0.80}
    row.update(count_emotions(text))
    row.update(behaviour_features(text))

    functions = load_person2_scoring()
    series = pd.Series(row)

    # EMP_0006 in the committed CSV: same text, score 3, Medium.
    assert functions["calculate_score"](series) == 3
    assert functions["risk_level"](3) == "Medium"


def test_nrc_lexicon_matches_multiple_emotions():
    from backend.ingest.pipeline_runner import count_emotions

    counts = count_emotions("I am furious and terrified about this awful deadline")
    assert counts["anger"] > 0
    assert counts["fear"] > 0


def test_behaviour_features_include_sentence_count():
    """Uploads get the feature Person 2's own CSV is missing.

    Their `behavior_features.py` crashes before writing `sentence_count`
    (docs/PERSON2_FIXES.md), so the committed CSV lacks it and their scoring
    function cannot use it. Uploaded messages have it, so `calculate_score`
    runs with its full intended feature set.
    """
    from backend.ingest.pipeline_runner import behaviour_features

    features = behaviour_features("First one. Second one. Third one.")
    assert features["sentence_count"] == 3
    assert features["word_count"] == 6


def test_uppercase_ratio_is_computed_not_destroyed():
    from backend.ingest.pipeline_runner import behaviour_features

    assert behaviour_features("SHOUTING")["uppercase_ratio"] == 1.0
    assert behaviour_features("quiet")["uppercase_ratio"] == 0.0
    assert behaviour_features("")["uppercase_ratio"] == 0
