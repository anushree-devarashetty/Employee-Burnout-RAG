"""
backend/ingest/message_splitter.py — Raw text into individual messages
==============================================================================

WHY THIS EXISTS AND WHY IT MATTERS MORE THAN IT LOOKS

    Person 2's pipeline scores ONE MESSAGE per row. Burnout indicators are
    per-utterance: "I am completely exhausted" scores 3 on its own, but the
    same sentence buried in a 2,000-word transcript alongside forty neutral
    lines is diluted into invisibility — the emotion counts stay the same
    while `word_count` explodes, and the risk signal disappears.

    So splitting correctly is not preprocessing housekeeping. It determines
    whether the analysis detects anything at all.

FOUR STRATEGIES, TRIED IN ORDER OF SPECIFICITY

    1. Email    "From: ... Subject: ..."          Outlook/Gmail PDF exports
    2. Chat     "[10:30] Priya: message"          Slack, Teams, WhatsApp, OCR
    3. Speaker  "PRIYA: message"                  meeting transcripts
    4. Paragraph  blank-line separated            everything else

    The first strategy that produces a plausible result wins. "Plausible"
    means more than one message and a sensible average length — a pattern that
    matches once and swallows the whole document has not really matched.

WHAT IS DELIBERATELY DISCARDED
    Sender names, timestamps and email headers are stripped from the scored
    text but retained as metadata. They are not communication *content*: a
    timestamp contributes nothing to sentiment, while "Sent from my iPhone"
    would inflate `word_count` on every single message and shift behavioural
    features across the board.
==============================================================================
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

#: Minimum message length, per strategy. A single global threshold is wrong,
#: because the strategies differ in how much evidence they have that a fragment
#: is a real message.
#:
#:   chat / speaker — a timestamp and a sender name already PROVE this is a
#:       message someone sent. The bar is therefore very low. This matters
#:       specifically for burnout detection: "not ok" (6), "I'm done" (8) and
#:       "I can't" (7) are among the strongest signals a person can send, and a
#:       12-character floor would discard all three.
#:
#:   email — headers prove it is a message, but PDF extraction leaves stray
#:       fragments between header blocks, so a slightly higher bar is warranted.
#:
#:   paragraph — no structural evidence at all. Short blocks here are usually
#:       page numbers, headings or OCR noise rather than communication.
MIN_CHARS_BY_STRATEGY = {
    "chat": 3,
    "speaker": 3,
    "email": 10,
    "paragraph": 25,
    "whole-document": 1,
}

#: Fallback for callers that do not specify a strategy.
MIN_MESSAGE_CHARS = 3

#: Longer than this and it is almost certainly an unsplit document. Truncated
#: rather than dropped, because its opening usually carries the real signal.
MAX_MESSAGE_CHARS = 4000


@dataclass
class Message:
    """One extracted message, ready for scoring."""

    text: str
    sender: Optional[str] = None
    timestamp: Optional[str] = None
    subject: Optional[str] = None
    index: int = 0
    source: str = ""

    def metadata(self) -> Dict[str, str]:
        return {
            key: value
            for key, value in (
                ("sender", self.sender),
                ("timestamp", self.timestamp),
                ("subject", self.subject),
                ("source", self.source),
            )
            if value
        }


@dataclass
class SplitResult:
    """Outcome of splitting one document."""

    messages: List[Message] = field(default_factory=list)
    strategy: str = "none"
    warnings: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.messages)


# ==============================================================================
# Patterns
# ==============================================================================

# Email header block. Requires From: plus at least one other header, so a
# message merely *containing* the word "From:" is not misread as a boundary.
_EMAIL_HEADER = re.compile(
    r"^\s*From:\s*(?P<sender>.+?)\s*$"
    r"(?P<rest>(?:\n^\s*(?:To|Cc|Bcc|Sent|Date|Subject):.*$)+)",
    re.MULTILINE,
)
_SUBJECT = re.compile(r"^\s*Subject:\s*(?P<subject>.+?)\s*$", re.MULTILINE)
_DATE = re.compile(r"^\s*(?:Sent|Date):\s*(?P<date>.+?)\s*$", re.MULTILINE)

# Chat lines. Covers the common export and screenshot shapes:
#   [10:30] Priya: text
#   10:30 AM  Priya: text
#   12/07/2026, 10:30 - Priya: text          (WhatsApp)
#   Priya (10:30 AM): text
_CHAT_LINE = re.compile(
    r"^\s*"
    r"(?:\[?(?P<ts1>\d{1,2}[:/.]\d{2}(?:[:.]\d{2})?(?:\s*[APap]\.?[Mm]\.?)?)\]?\s*[-–—]?\s*)?"
    r"(?:(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),?\s*"
    r"(?P<ts2>\d{1,2}:\d{2}(?:\s*[APap]\.?[Mm]\.?)?)?\s*[-–—]\s*)?"
    # {0,30} not {1,30}: anonymised transcripts commonly use single-letter
    # speakers ("A:", "B:"), and requiring two characters silently drops the
    # entire conversation back to the paragraph fallback.
    r"(?P<sender>[A-Z][\w .'\-]{0,30}?)"
    r"(?:\s*\((?P<ts3>[^)]{1,20})\))?"
    r"\s*:\s"
    r"(?P<text>.+)$",
    re.MULTILINE,
)

# Meeting-transcript speaker labels: "PRIYA:" or "Priya Giri:" at line start.
_SPEAKER_LINE = re.compile(
    r"^\s*(?P<sender>[A-Z][A-Za-z .'\-]{0,30})\s*:\s*(?P<text>.+)$", re.MULTILINE
)

#: Boilerplate that would otherwise inflate behavioural features on every row.
_NOISE_PATTERNS = [
    re.compile(r"^\s*(?:Sent|Get) from my \w+.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:This (?:e-?mail|message) (?:and any attachments )?is confidential"
               r"|CONFIDENTIALITY NOTICE)[\s\S]{0,600}$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*(?:Unsubscribe|View this email in your browser).*$",
               re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*<?https?://\S+>?\s*$", re.MULTILINE),
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.MULTILINE | re.IGNORECASE),
    # WhatsApp system lines and OCR-visible UI chrome.
    re.compile(r"^\s*(?:Messages and calls are end-to-end encrypted"
               r"|<Media omitted>|This message was deleted"
               r"|You deleted this message).*$", re.MULTILINE | re.IGNORECASE),
]

#: Quoted-reply prefix; kept out of the scored text so a thread is not counted
#: repeatedly, once per reply level.
_QUOTE_PREFIX = re.compile(r"^\s*>+\s?", re.MULTILINE)


def _clean_body(text: str) -> str:
    """Strip boilerplate and quoted replies from a message body."""
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    text = _QUOTE_PREFIX.sub("", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _acceptable(messages: List[Message], total_chars: int) -> bool:
    """Decide whether a strategy actually worked.

    A pattern that matches once and swallows the document has not matched.
    Requires at least two messages, and that they collectively account for a
    meaningful share of the source text — otherwise a single stray "Note:" line
    could beat a correct paragraph split.
    """
    if len(messages) < 2:
        return False
    captured = sum(len(m.text) for m in messages)
    return captured >= 0.30 * max(total_chars, 1)


# ==============================================================================
# Strategies
# ==============================================================================
def _split_emails(text: str) -> List[Message]:
    """Split on email header blocks."""
    matches = list(_EMAIL_HEADER.finditer(text))
    if not matches:
        return []

    messages: List[Message] = []
    for position, match in enumerate(matches):
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        header_block = match.group(0)
        body = _clean_body(text[start:end])

        if len(body) < MIN_CHARS_BY_STRATEGY["email"]:
            continue

        subject = _SUBJECT.search(header_block)
        date = _DATE.search(header_block)

        messages.append(
            Message(
                text=body[:MAX_MESSAGE_CHARS],
                sender=match.group("sender").strip()[:80] or None,
                subject=subject.group("subject").strip()[:120] if subject else None,
                timestamp=date.group("date").strip()[:60] if date else None,
            )
        )
    return messages


def _split_chat(text: str) -> List[Message]:
    """Split on timestamped chat lines.

    Continuation lines — a multi-line message where only the first line carries
    the "Name:" prefix — are appended to the previous message rather than being
    dropped or promoted to messages of their own.
    """
    messages: List[Message] = []
    last_end = 0
    pending: Optional[Message] = None

    for match in _CHAT_LINE.finditer(text):
        # Text between the previous match and this one belongs to the message
        # before it.
        if pending is not None and match.start() > last_end:
            trailing = text[last_end : match.start()].strip()
            if trailing:
                pending.text = f"{pending.text} {trailing}".strip()

        if pending is not None:
            messages.append(pending)

        timestamp = match.group("ts1") or match.group("ts2") or match.group("ts3")
        date = match.group("date")
        pending = Message(
            text=_clean_body(match.group("text")),
            sender=(match.group("sender") or "").strip()[:80] or None,
            timestamp=" ".join(filter(None, (date, timestamp)))[:60] or None,
        )
        last_end = match.end()

    if pending is not None:
        trailing = text[last_end:].strip()
        if trailing:
            pending.text = f"{pending.text} {trailing}".strip()
        messages.append(pending)

    minimum = MIN_CHARS_BY_STRATEGY["chat"]
    return [
        Message(
            text=m.text[:MAX_MESSAGE_CHARS], sender=m.sender, timestamp=m.timestamp
        )
        for m in messages
        if len(m.text) >= minimum
    ]


def _split_speakers(text: str) -> List[Message]:
    """Split a meeting transcript on speaker labels."""
    messages: List[Message] = []
    for match in _SPEAKER_LINE.finditer(text):
        body = _clean_body(match.group("text"))
        if len(body) < MIN_CHARS_BY_STRATEGY["speaker"]:
            continue
        messages.append(
            Message(
                text=body[:MAX_MESSAGE_CHARS],
                sender=match.group("sender").strip()[:80] or None,
            )
        )
    return messages


def _split_paragraphs(text: str) -> List[Message]:
    """Fallback: blank-line separated blocks.

    Blocks longer than the cap are split on sentence boundaries rather than
    truncated, so a wall of text still yields analysable units instead of one
    diluted mega-message.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    messages: List[Message] = []

    minimum = MIN_CHARS_BY_STRATEGY["paragraph"]
    for block in blocks:
        block = _clean_body(block)
        if len(block) < minimum:
            continue

        if len(block) <= MAX_MESSAGE_CHARS:
            messages.append(Message(text=block))
            continue

        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            if len(current) + len(sentence) > MAX_MESSAGE_CHARS and current:
                messages.append(Message(text=current.strip()))
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if len(current) >= minimum:
            messages.append(Message(text=current.strip()))

    return messages


# ==============================================================================
# Entry point
# ==============================================================================
def split_messages(text: str, source: str = "") -> SplitResult:
    """Split extracted document text into individual messages.

    Args:
        text: Cleaned text from `extractors.extract`.
        source: Original filename, attached to each message as metadata.

    Returns:
        A :class:`SplitResult` naming the strategy that succeeded.
    """
    if not text or not text.strip():
        return SplitResult(warnings=["The document contained no readable text."])

    total_chars = len(text)
    warnings: List[str] = []

    for name, splitter in (
        ("email", _split_emails),
        ("chat", _split_chat),
        ("speaker", _split_speakers),
    ):
        try:
            candidate = splitter(text)
        except Exception as exc:  # a bad regex path must not lose the document
            logger.warning("Splitter '%s' failed: %s", name, exc)
            continue

        if _acceptable(candidate, total_chars):
            logger.info("Split %d message(s) using the '%s' strategy.",
                        len(candidate), name)
            return SplitResult(
                messages=_finalise(candidate, source), strategy=name, warnings=warnings
            )

    paragraphs = _split_paragraphs(text)
    if paragraphs:
        logger.info("Split %d message(s) using the 'paragraph' fallback.",
                    len(paragraphs))
        warnings.append(
            "No email or chat structure was detected, so the document was split "
            "on paragraph breaks. If this is a conversation, results improve "
            "when sender names and timestamps are present."
        )
        return SplitResult(
            messages=_finalise(paragraphs, source),
            strategy="paragraph",
            warnings=warnings,
        )

    # Nothing split cleanly: treat the whole document as one message rather
    # than returning nothing.
    warnings.append(
        "The document could not be split into separate messages and was "
        "analysed as a single block. Burnout signals are diluted in long "
        "text, so treat the score as indicative only."
    )
    return SplitResult(
        messages=_finalise([Message(text=text[:MAX_MESSAGE_CHARS])], source),
        strategy="whole-document",
        warnings=warnings,
    )


def _finalise(messages: List[Message], source: str) -> List[Message]:
    """Number the messages and drop exact duplicates.

    Duplicates are common and damaging: an email thread repeats earlier
    messages inside each reply, and counting the same complaint five times
    would inflate the apparent burnout rate of a whole team.
    """
    seen: set = set()
    result: List[Message] = []

    for message in messages:
        fingerprint = re.sub(r"\s+", " ", message.text.lower()).strip()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        message.index = len(result)
        message.source = source
        result.append(message)

    return result


__all__ = [
    "Message",
    "SplitResult",
    "split_messages",
    "MIN_MESSAGE_CHARS",
    "MIN_CHARS_BY_STRATEGY",
]
