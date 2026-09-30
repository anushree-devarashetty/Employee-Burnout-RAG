"""
backend/ingest/pipeline_runner.py — Score uploaded messages
==============================================================================

Runs Person 2's four analysis stages over newly uploaded text, producing rows
with the exact same 20-column schema as `outputs/burnout_scores.csv`.

    text -> RoBERTa sentiment -> NRC emotions -> behaviour -> burnout score

WHY THE SCHEMA MUST MATCH EXACTLY
    Uploaded messages are shown on the same dashboard, in the same charts, and
    retrieved by the same RAG pipeline as the committed dataset. If they were
    scored even slightly differently, every comparison the dashboard invites a
    user to make would be invalid.

THE INTERESTING PROBLEM: REUSING PERSON 2'S SCORING WITHOUT RUNNING THEIR SCRIPT

    Person 2's burnout logic lives in three functions inside
    `models/burnout/burnout_score.py`: `calculate_score`, `risk_level` and
    `explanation`. Reimplementing them here would mean two copies of the
    scoring rule, guaranteed to drift the first time anyone tuned a weight.

    But that file is a script, not a module. Below the function definitions it
    executes top-level code that reads `outputs/employee_features.csv` and
    OVERWRITES `outputs/burnout_scores.csv`. A plain `import` would destroy
    Person 2's committed output every time the backend started.

    Solution: parse the file with `ast`, keep only the `FunctionDef` and import
    nodes, discard every other top-level statement, and execute the remainder
    in an isolated namespace. Person 2's genuine scoring rule is reused
    verbatim, their file is not modified, and nothing is read or written.

    Verified against their file: `calculate_score`, `risk_level` and
    `explanation` are all module-level `FunctionDef` nodes, so all three
    extract cleanly.

WHAT IS MIRRORED RATHER THAN REUSED, AND WHY
    Emotion counting and behavioural features were written by Person 2 as
    inline top-level pandas expressions, not as functions, so there is nothing
    to extract. Those two stages are reimplemented here, matching their source
    line for line, with the original referenced in each docstring. If Person 2
    ever moves that code into functions, this module should switch to
    extracting them too.

    One deliberate difference: `sentence_count` IS computed here. Person 2's
    `behavior_features.py` fails to produce it because of the `df["message"]`
    bug (see docs/PERSON2_FIXES.md), which means their committed CSV lacks the
    column and their own scoring function cannot use it. Uploaded messages have
    it, so `calculate_score` runs with its full intended feature set.
==============================================================================
"""

from __future__ import annotations

import ast
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from backend import REPO_ROOT
from backend.config import Settings, get_settings
from backend.data_loader import BEHAVIOR_COLUMNS, EMOTION_COLUMNS
from backend.ingest.message_splitter import Message

logger = logging.getLogger(__name__)

#: Person 2's model, named exactly as in models/sentiment/roberta_sentiment.py.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

#: Label order matching that script's `labels` list.
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

_LEXICON_PATH = REPO_ROOT / "datasets" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
_SCORING_PATH = REPO_ROOT / "models" / "burnout" / "burnout_score.py"


class PipelineError(RuntimeError):
    """Raised when uploaded messages cannot be scored."""


# ==============================================================================
# Person 2's scoring functions, extracted without side effects
# ==============================================================================
_scoring_lock = threading.Lock()
_scoring_functions: Optional[Dict[str, Callable]] = None


def load_person2_scoring() -> Dict[str, Callable]:
    """Extract Person 2's scoring functions without executing their script.

    Returns:
        ``{"calculate_score": fn, "risk_level": fn, "explanation": fn}``

    Raises:
        PipelineError: if the file is missing or its functions have been
            renamed — in which case the message names the file to look at,
            rather than failing with an opaque KeyError later.
    """
    global _scoring_functions
    if _scoring_functions is not None:
        return _scoring_functions

    with _scoring_lock:
        if _scoring_functions is not None:
            return _scoring_functions

        if not _SCORING_PATH.is_file():
            raise PipelineError(
                f"Person 2's scoring module was not found at {_SCORING_PATH}. "
                f"Uploaded messages cannot be scored consistently with the "
                f"main dataset without it."
            )

        try:
            tree = ast.parse(_SCORING_PATH.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise PipelineError(
                f"Could not parse {_SCORING_PATH}: {exc}"
            ) from exc

        # Keep ONLY function definitions and imports. Every other top-level
        # statement in that file performs file I/O and must not run.
        safe_body = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
        ]
        module = ast.Module(body=safe_body, type_ignores=[])

        namespace: Dict[str, Any] = {}
        try:
            exec(compile(module, str(_SCORING_PATH), "exec"), namespace)
        except Exception as exc:
            raise PipelineError(
                f"Could not load scoring functions from {_SCORING_PATH}: {exc}"
            ) from exc

        required = ("calculate_score", "risk_level", "explanation")
        missing = [name for name in required if not callable(namespace.get(name))]
        if missing:
            raise PipelineError(
                f"{_SCORING_PATH} no longer defines: {', '.join(missing)}.\n"
                f"Uploaded messages are scored with Person 2's own functions, "
                f"so this module must be updated if they are renamed."
            )

        _scoring_functions = {name: namespace[name] for name in required}
        logger.info(
            "Loaded Person 2's scoring functions (%s) without executing their script.",
            ", ".join(required),
        )
        return _scoring_functions


# ==============================================================================
# Stage 1 — sentiment (Person 2's model)
# ==============================================================================
class SentimentAnalyser:
    """RoBERTa sentiment, mirroring models/sentiment/roberta_sentiment.py.

    Lazily loaded: the model is ~500 MB and must not be pulled in at import
    time, nor at all unless somebody actually uploads a document.
    """

    def __init__(self, model_name: str = SENTIMENT_MODEL) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise PipelineError(
                    "transformers is not installed, so uploaded documents "
                    "cannot be scored for sentiment.\n"
                    "  Fix: pip install -r backend/requirements.txt"
                ) from exc

            logger.info(
                "Loading sentiment model '%s' (first run downloads ~500MB)...",
                self.model_name,
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self._model.eval()
            except Exception as exc:
                raise PipelineError(
                    f"Could not load '{self.model_name}': {exc}\n"
                    f"  On first use this is downloaded from HuggingFace — "
                    f"check your internet connection."
                ) from exc

            logger.info("Sentiment model ready.")

    def predict(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        """Classify texts, returning label and confidence per input.

        Truncated at 512 tokens exactly as Person 2's script does, so a long
        message is scored on the same span either way.
        """
        if not texts:
            return []

        self._load()

        import torch
        from scipy.special import softmax

        results: List[Dict[str, Any]] = []
        with torch.no_grad():
            for text in texts:
                encoded = self._tokenizer(
                    str(text), return_tensors="pt", truncation=True, max_length=512
                )
                scores = softmax(self._model(**encoded).logits.detach().numpy()[0])
                index = int(scores.argmax())
                results.append(
                    {
                        "sentiment": SENTIMENT_LABELS[index],
                        "confidence": round(float(scores[index]), 4),
                    }
                )
        return results


_analyser: Optional[SentimentAnalyser] = None
_analyser_lock = threading.Lock()


def get_sentiment_analyser() -> SentimentAnalyser:
    """Process-wide sentiment analyser."""
    global _analyser
    if _analyser is None:
        with _analyser_lock:
            if _analyser is None:
                _analyser = SentimentAnalyser()
    return _analyser


# ==============================================================================
# Stage 2 — NRC emotions
# ==============================================================================
_lexicon: Optional[Dict[str, List[str]]] = None
_lexicon_lock = threading.Lock()


def load_nrc_lexicon() -> Dict[str, List[str]]:
    """Load the NRC lexicon, mirroring models/emotion/emotion_analysis.py.

    Same parse: tab-separated ``word<TAB>emotion<TAB>association``, keeping
    only rows where association is "1".
    """
    global _lexicon
    if _lexicon is not None:
        return _lexicon

    with _lexicon_lock:
        if _lexicon is not None:
            return _lexicon

        if not _LEXICON_PATH.is_file():
            raise PipelineError(
                f"The NRC Emotion Lexicon was not found at {_LEXICON_PATH}."
            )

        lexicon: Dict[str, List[str]] = {}
        with open(_LEXICON_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                word, emotion, association = parts
                if association == "1":
                    lexicon.setdefault(word, []).append(emotion)

        _lexicon = lexicon
        logger.info("NRC lexicon loaded: %d words.", len(lexicon))
        return _lexicon


def count_emotions(text: str) -> Dict[str, int]:
    """Count NRC emotion word matches.

    Mirrors emotion_analysis.py: lowercase, split on whitespace, count matches
    across the eight categories. Punctuation is not stripped there either, so
    it is not stripped here — matching their behaviour matters more than
    marginally better matching.
    """
    lexicon = load_nrc_lexicon()
    scores = {emotion: 0 for emotion in EMOTION_COLUMNS}

    for word in str(text).lower().split():
        for emotion in lexicon.get(word, ()):
            if emotion in scores:
                scores[emotion] += 1

    return scores


# ==============================================================================
# Stage 3 — behavioural features
# ==============================================================================
def behaviour_features(text: str) -> Dict[str, float]:
    """Compute the seven behavioural features.

    Mirrors models/features/behavior_features.py, with `clean_text` as the
    source column throughout — which is what that script intended. It currently
    reads a non-existent `message` column (docs/PERSON2_FIXES.md), so this is
    the corrected form of their logic, not a different one.
    """
    text = str(text)
    words = text.split()
    letters = [c for c in text if c.isalpha()]
    uppercase = sum(1 for c in letters if c.isupper())

    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_length": round(
            sum(len(word) for word in words) / max(len(words), 1), 2
        ),
        "sentence_count": len([s for s in text.split(".") if s.strip()]),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": round(uppercase / len(letters), 3) if letters else 0,
    }


# ==============================================================================
# Full pipeline
# ==============================================================================
@dataclass
class ScoringResult:
    """Scored rows plus a summary of the run."""

    frame: pd.DataFrame
    message_count: int
    warnings: List[str]

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


def score_messages(
    messages: Sequence[Message],
    settings: Optional[Settings] = None,
    id_prefix: str = "DOC",
) -> ScoringResult:
    """Run the full analysis pipeline over extracted messages.

    Args:
        messages: Output of `message_splitter.split_messages`.
        settings: Configuration; defaults to the process-wide instance.
        id_prefix: Identifier prefix. ``DOC`` by default so uploaded records
            are visually distinct from the dataset's ``EMP_`` rows.

    Returns:
        A :class:`ScoringResult` whose frame matches the 20-column schema of
        `outputs/burnout_scores.csv`, plus `sender`, `timestamp` and `source`.
    """
    settings = settings or get_settings()
    warnings: List[str] = []

    if not messages:
        return ScoringResult(pd.DataFrame(), 0, ["No messages to analyse."])

    texts = [message.text for message in messages]

    # --- stage 1: sentiment ---------------------------------------------------
    logger.info("Scoring %d message(s) for sentiment...", len(texts))
    sentiments = get_sentiment_analyser().predict(texts)

    # --- stages 2 and 3 -------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    for position, (message, sentiment) in enumerate(zip(messages, sentiments)):
        row: Dict[str, Any] = {
            "clean_text": message.text,
            "sentiment": sentiment["sentiment"],
            "confidence": sentiment["confidence"],
        }
        row.update(count_emotions(message.text))
        row.update(behaviour_features(message.text))
        row["employee_id"] = f"{id_prefix}_{position + 1:04d}"
        row["sender"] = message.sender or ""
        row["timestamp"] = message.timestamp or ""
        row["subject"] = message.subject or ""
        row["source"] = message.source or ""
        rows.append(row)

    frame = pd.DataFrame(rows)

    # --- stage 4: Person 2's own scoring functions ---------------------------
    scoring = load_person2_scoring()
    try:
        frame["burnout_score"] = frame.apply(scoring["calculate_score"], axis=1)
        frame["risk_level"] = frame["burnout_score"].apply(scoring["risk_level"])
        frame["explanation"] = frame.apply(scoring["explanation"], axis=1)
    except Exception as exc:
        raise PipelineError(
            f"Person 2's scoring functions failed on the uploaded messages: "
            f"{exc}\n"
            f"This usually means their expected columns have changed. See "
            f"{_SCORING_PATH}."
        ) from exc

    # --- derived display fields, matching backend/data_loader.py -------------
    frame["burnout_percent"] = (
        frame["burnout_score"] / float(settings.burnout_score_max) * 100.0
    ).clip(0, 100).round(1)

    emotions = frame[list(EMOTION_COLUMNS)]
    max_emotion = emotions.max(axis=1)
    frame["dominant_emotion"] = emotions.idxmax(axis=1).where(
        max_emotion > 0, other="neutral"
    )
    frame["dominant_emotion_score"] = max_emotion.astype(float)

    frame["document_text"] = frame.apply(_document_text, axis=1)

    at_risk = int((frame["risk_level"].isin(["Medium", "High"])).sum())
    if at_risk:
        warnings.append(
            f"{at_risk} of {len(frame)} uploaded message(s) scored at Medium or "
            f"High risk."
        )

    logger.info(
        "Scored %d message(s): %s",
        len(frame),
        frame["risk_level"].value_counts().to_dict(),
    )
    return ScoringResult(frame=frame, message_count=len(frame), warnings=warnings)


def _document_text(row: pd.Series) -> str:
    """Render a scored row as the text embedded into the vector store.

    Same shape as `backend/data_loader.build_document_text`, with sender and
    timestamp added, since an uploaded conversation has a participant and a
    time that the anonymous dataset rows do not.
    """
    parts: List[str] = []

    who = str(row.get("sender", "")).strip()
    parts.append(
        f"Message from {who} (record {row['employee_id']})."
        if who
        else f"Message {row['employee_id']}."
    )

    when = str(row.get("timestamp", "")).strip()
    if when:
        parts.append(f"Sent {when}.")

    subject = str(row.get("subject", "")).strip()
    if subject:
        parts.append(f"Subject: {subject}.")

    parts.append(f'Message: "{str(row["clean_text"]).strip()}"')
    parts.append(
        f"Sentiment: {row['sentiment']} "
        f"(model confidence {float(row['confidence']):.2f})."
    )

    active = [
        f"{emotion} ({int(row[emotion])})"
        for emotion in EMOTION_COLUMNS
        if float(row.get(emotion, 0) or 0) > 0
    ]
    parts.append(
        f"Detected emotions: {', '.join(active)}."
        if active
        else "Detected emotions: none identified by the NRC lexicon."
    )

    behaviour = [
        f"{column.replace('_', ' ')} {row[column]:g}"
        for column in BEHAVIOR_COLUMNS
        if column in row.index and pd.notna(row.get(column))
    ]
    if behaviour:
        parts.append(f"Communication behaviour: {', '.join(behaviour)}.")

    parts.append(
        f"Burnout score: {row['burnout_score']} "
        f"({row['burnout_percent']:.0f} out of 100)."
    )
    parts.append(f"Risk level: {row['risk_level']}.")

    explanation = str(row.get("explanation", "")).strip()
    if explanation:
        parts.append(f"Model explanation: {explanation}.")

    return " ".join(parts)


__all__ = [
    "score_messages",
    "ScoringResult",
    "PipelineError",
    "load_person2_scoring",
    "load_nrc_lexicon",
    "count_emotions",
    "behaviour_features",
    "get_sentiment_analyser",
    "SENTIMENT_MODEL",
]
