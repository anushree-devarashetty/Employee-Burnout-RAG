"""
backend/data_loader.py — The single gateway to Person 2's output
==============================================================================

WHY THIS MODULE EXISTS

    Person 3 (RAG backend) and Person 4 (dashboard) both need the same numbers.
    If each wrote its own `pd.read_csv` and its own "overall burnout %", they
    would drift apart the first time anyone changed a threshold, and the
    dashboard would confidently display one figure while the LLM reasoned over
    another. Every read of `outputs/burnout_scores.csv` in this project goes
    through this file, so that class of bug cannot occur.

REUSE, NOT REIMPLEMENTATION
    Row reading is delegated to Person 2's own `models/utils/loader.py`:

        from models.utils.loader import load_data

    That function already validates the presence of `clean_text` and raises a
    clear error if it is missing. Re-implementing it here would duplicate
    Person 2's contract and let the two copies diverge. This module adds only
    what Person 2's function does not provide: identity, derived display
    fields, and aggregation.

WHAT IS ADDED ON TOP OF PERSON 2'S CSV
    employee_id        Stable synthetic identity (see below)
    burnout_percent    burnout_score mapped onto 0-100 for display only
    dominant_emotion   argmax across the eight NRC emotion columns
    document_text      Natural-language rendering used for embedding

WHAT IS NEVER RECOMPUTED
    sentiment, confidence, the eight emotion counts, the behavioural features,
    burnout_score, risk_level and explanation are read verbatim. This module
    has no opinion about them. If a number looks wrong, the fix belongs in
    `models/burnout/burnout_score.py`, not here.

THE MISSING employee_id
    Person 2's CSV has no employee identifier — each row is one message. The
    dashboard specification requires "Top High Risk Employees" and "Search
    Employee", so identity has to come from somewhere.

    Chosen approach: derive it deterministically from row position, producing
    EMP_0001, EMP_0002, ... Deterministic matters because the vector store,
    the API and the dashboard must agree on which record is which across
    restarts; a random or hash-based ID would break that.

    If Person 2 ever adds a real `employee_id` column, it is detected and used
    automatically — no code change here, and no coordination needed.

A KNOWN DEFECT IN PERSON 2's CODE THAT THIS MODULE TOLERATES
    `models/features/behavior_features.py` reads `df["message"]`, a column that
    does not exist upstream, so `sentence_count` was never written to the CSV.
    `models/burnout/burnout_score.py` then reads `row["sentence_count"]`.
    Both scripts therefore raise KeyError if re-run today, and the committed
    CSVs predate the bug.

    That is Person 2's territory to fix. This module simply treats every
    behavioural column as optional, so the backend works with the CSV exactly
    as committed, and will keep working unchanged once Person 2 repairs the
    pipeline and `sentence_count` reappears.
==============================================================================
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------
# Script-mode self-bootstrap (see backend/config.py for the full rationale)
# ------------------------------------------------------------------------------
if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import pandas as pd

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ==============================================================================
# Schema — mirrors outputs/burnout_scores.csv exactly as Person 2 produces it
# ==============================================================================

#: Columns without which the backend genuinely cannot function.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "clean_text",
    "sentiment",
    "burnout_score",
    "risk_level",
)

#: The eight NRC emotion categories, in the order emotion_analysis.py writes.
EMOTION_COLUMNS: Tuple[str, ...] = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)

#: Behavioural features from behavior_features.py.
#: ALL are optional. `sentence_count` is absent from the committed CSV because
#: of the upstream defect documented in this module's docstring, and the
#: backend must not crash over it.
BEHAVIOR_COLUMNS: Tuple[str, ...] = (
    "word_count",
    "char_count",
    "avg_word_length",
    "sentence_count",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
)

#: Risk levels exactly as burnout_score.py emits them (<=2 Low, <=5 Medium, else High).
RISK_LEVELS: Tuple[str, ...] = ("Low", "Medium", "High")

#: Which levels count towards the headline "Overall Burnout %" gauge.
#: Chosen over normalising the raw point score because it is scale-independent:
#: if Person 2 retunes the scoring weights, this figure stays meaningful.
AT_RISK_LEVELS: Tuple[str, ...] = ("Medium", "High")

#: Sentiment labels from roberta_sentiment.py (cardiffnlp/twitter-roberta-base).
SENTIMENT_LABELS: Tuple[str, ...] = ("Positive", "Neutral", "Negative")


# ==============================================================================
# Container
# ==============================================================================
@dataclass(frozen=True)
class BurnoutDataset:
    """An immutable, enriched snapshot of Person 2's output."""

    frame: pd.DataFrame
    source_path: Path
    source_mtime: float
    loaded_at: datetime
    missing_optional_columns: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def record_count(self) -> int:
        return int(len(self.frame))

    def describe(self) -> Dict[str, Any]:
        """Provenance block for ``GET /health`` and PDF report footers."""
        return {
            "source_path": str(self.source_path),
            "record_count": self.record_count,
            "columns": list(self.frame.columns),
            "missing_optional_columns": list(self.missing_optional_columns),
            "source_modified": datetime.fromtimestamp(
                self.source_mtime, tz=timezone.utc
            ).isoformat(),
            "loaded_at": self.loaded_at.isoformat(),
            "owner": "Person 2 — consumed read-only",
        }


# ==============================================================================
# Person 2 reuse
# ==============================================================================
def _read_with_person2_loader(csv_path: Path) -> pd.DataFrame:
    """Read the CSV using Person 2's ``models/utils/loader.load_data``.

    Falls back to a direct read only if that module cannot be imported (for
    example if someone copies `backend/` into another project). The fallback
    reproduces Person 2's `clean_text` check so behaviour stays identical
    either way.
    """
    try:
        from models.utils.loader import load_data  # Person 2's module

        logger.debug("Reading via Person 2's models.utils.loader.load_data")
        return load_data(str(csv_path))

    except ImportError:
        logger.warning(
            "models.utils.loader unavailable — falling back to a direct read. "
            "This is expected only outside the full repository."
        )
        frame = pd.read_csv(csv_path)
        if "clean_text" not in frame.columns:
            # Mirrors the exact contract of Person 2's loader.
            raise ValueError("Dataset must contain 'clean_text' column.")
        return frame


# ==============================================================================
# Enrichment
# ==============================================================================
def _attach_employee_ids(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Ensure an ``employee_id`` column exists.

    Uses a real column if Person 2 has added one; otherwise generates stable,
    zero-padded IDs from row position. Zero padding keeps IDs sorting correctly
    as strings in dashboard tables (EMP_0002 before EMP_0010).
    """
    if "employee_id" in frame.columns:
        logger.info("Using the real 'employee_id' column present in the CSV.")
        frame["employee_id"] = frame["employee_id"].astype(str)
        return frame

    width = max(4, len(str(len(frame))))
    frame["employee_id"] = [
        f"{prefix}_{i + 1:0{width}d}" for i in range(len(frame))
    ]
    logger.info(
        "No 'employee_id' column found — generated %d synthetic IDs (%s_...).",
        len(frame),
        prefix,
    )
    return frame


def _coerce_numeric(
    frame: pd.DataFrame,
    columns: Tuple[str, ...],
    create_missing_as_zero: bool = False,
) -> List[str]:
    """Force listed columns to numeric, filling unparseable values with 0.

    CSV round-trips turn integers into strings and empty cells into NaN. Every
    downstream consumer (argmax, aggregation, Chroma metadata) assumes real
    numbers, so normalise once here rather than defensively everywhere else.

    ``create_missing_as_zero`` encodes a distinction that matters for an
    explainability project, where a fabricated number is worse than a gap:

      EMOTION columns  -> True.  The NRC lexicon assigns a count of 0 to any
          category it does not detect, so an absent column and a zero column
          carry the same meaning. Materialising it as 0 keeps all eight
          categories present for the dashboard's emotion chart.

      BEHAVIOUR columns -> False. `sentence_count` is absent because Person 2's
          `behavior_features.py` crashed on a non-existent `message` column,
          NOT because the messages contain zero sentences. Writing 0 would
          state a falsehood, and that falsehood would be embedded into the
          vector store and shown to the LLM as evidence. Absent stays absent,
          and every consumer here already guards with `if col in row.index`.

    Returns:
        Names of the requested columns that were absent from the CSV.
    """
    missing: List[str] = []
    for column in columns:
        if column not in frame.columns:
            missing.append(column)
            if create_missing_as_zero:
                frame[column] = 0.0
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return missing


def _add_dominant_emotion(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``dominant_emotion``: the highest-scoring NRC category per row.

    Rows where every emotion count is zero are labelled ``"neutral"`` rather
    than silently reporting ``"anger"`` — which is what a naive ``idxmax``
    returns for an all-zero row, since it takes the first column on a tie.
    That mislabelling would be actively harmful in a wellbeing dashboard.
    """
    present = [c for c in EMOTION_COLUMNS if c in frame.columns]
    if not present:
        frame["dominant_emotion"] = "unknown"
        frame["dominant_emotion_score"] = 0
        return frame

    emotions = frame[present]
    max_score = emotions.max(axis=1)

    frame["dominant_emotion"] = emotions.idxmax(axis=1).where(
        max_score > 0, other="neutral"
    )
    frame["dominant_emotion_score"] = max_score.astype(float)
    return frame


def _add_burnout_percent(frame: pd.DataFrame, score_max: int) -> pd.DataFrame:
    """Map Person 2's integer point score onto 0-100 for display only.

    Never used to classify risk. `risk_level` always comes straight from
    Person 2's column, so the label a user sees and the label the ML pipeline
    produced can never disagree.
    """
    scores = pd.to_numeric(frame["burnout_score"], errors="coerce").fillna(0)
    frame["burnout_percent"] = (scores / float(score_max) * 100.0).clip(0, 100).round(1)
    return frame


def _normalise_risk_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Title-case risk labels so 'high', 'HIGH' and 'High' group together."""
    frame["risk_level"] = (
        frame["risk_level"].astype(str).str.strip().str.title()
    )
    unexpected = set(frame["risk_level"].unique()) - set(RISK_LEVELS)
    if unexpected:
        logger.warning(
            "Unexpected risk_level value(s) in the CSV: %s. Expected %s.",
            sorted(unexpected),
            list(RISK_LEVELS),
        )
    return frame


def build_document_text(row: pd.Series) -> str:
    """Render one record as the natural-language text that gets embedded.

    Design note: a raw ``clean_text`` embedding would only ever match on
    message wording, so the question "who is at high risk?" would retrieve
    nothing useful — the words "high risk" appear in no employee's message.
    Folding the structured signals into the embedded text lets semantic search
    reach the analysis as well as the prose.

    Kept human-readable rather than key=value, because this exact string is
    also shown to the LLM as evidence and appears in API responses, where a
    reviewer needs to be able to read it.
    """
    parts: List[str] = [
        f"Employee {row['employee_id']}.",
        f"Message: \"{str(row['clean_text']).strip()}\"",
        f"Sentiment: {row['sentiment']}",
    ]

    if "confidence" in row.index and pd.notna(row.get("confidence")):
        parts.append(f"(model confidence {float(row['confidence']):.2f}).")

    active = [
        f"{emotion} ({int(row[emotion])})"
        for emotion in EMOTION_COLUMNS
        if emotion in row.index and float(row.get(emotion, 0) or 0) > 0
    ]
    parts.append(
        f"Detected emotions: {', '.join(active)}."
        if active
        else "Detected emotions: none identified by the NRC lexicon."
    )

    behaviour = [
        f"{col.replace('_', ' ')} {row[col]:g}"
        for col in BEHAVIOR_COLUMNS
        if col in row.index and pd.notna(row.get(col))
    ]
    if behaviour:
        parts.append(f"Communication behaviour: {', '.join(behaviour)}.")

    parts.append(
        f"Burnout score: {row['burnout_score']} "
        f"({row['burnout_percent']:.0f} out of 100)."
    )
    parts.append(f"Risk level: {row['risk_level']}.")

    if "explanation" in row.index and pd.notna(row.get("explanation")):
        parts.append(f"Model explanation: {row['explanation']}.")

    return " ".join(parts)


# ==============================================================================
# Loading (cached on file mtime)
# ==============================================================================
_CACHE: Dict[str, BurnoutDataset] = {}
_CACHE_LOCK = threading.Lock()


def load_burnout_dataset(
    settings: Optional[Settings] = None,
    force_reload: bool = False,
) -> BurnoutDataset:
    """Load, enrich and cache Person 2's burnout output.

    The cache key includes the file's modification time, so re-running Person
    2's pipeline is picked up automatically on the next call without a restart
    and without a manual refresh.

    Thread-safe: FastAPI serves requests from a thread pool and Streamlit reruns
    scripts on its own threads, so concurrent first-loads are a real scenario.

    Args:
        settings: Configuration. Defaults to the process-wide instance.
        force_reload: Bypass the cache. Used by ``POST /refresh``.

    Returns:
        An enriched :class:`BurnoutDataset`.

    Raises:
        FileNotFoundError: if the CSV is missing (message includes the exact
            commands to regenerate it).
        ValueError: if required columns are absent.
    """
    settings = settings or get_settings()
    csv_path = settings.ensure_data_available()
    mtime = csv_path.stat().st_mtime
    cache_key = f"{csv_path}:{mtime}"

    with _CACHE_LOCK:
        if not force_reload and cache_key in _CACHE:
            logger.debug("Serving burnout dataset from cache.")
            return _CACHE[cache_key]

        logger.info("Loading burnout data from %s", csv_path)
        frame = _read_with_person2_loader(csv_path)

        missing_required = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing_required:
            raise ValueError(
                f"{csv_path} is missing required column(s): "
                f"{', '.join(missing_required)}.\n"
                f"Expected at minimum: {', '.join(REQUIRED_COLUMNS)}.\n"
                f"Found: {', '.join(frame.columns)}.\n"
                f"Re-run Person 2's pipeline "
                f"(python models/burnout/burnout_score.py) to regenerate it."
            )

        frame = frame.copy()
        frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
        frame["sentiment"] = frame["sentiment"].fillna("Unknown").astype(str).str.title()
        if "explanation" in frame.columns:
            frame["explanation"] = frame["explanation"].fillna("").astype(str)

        _coerce_numeric(frame, ("burnout_score", "confidence"))
        # Emotions: absent == not detected == 0. Safe to materialise.
        missing_emotions = _coerce_numeric(
            frame, EMOTION_COLUMNS, create_missing_as_zero=True
        )
        # Behaviour: absent == not measured. Never fabricated.
        missing_behaviour = _coerce_numeric(
            frame, BEHAVIOR_COLUMNS, create_missing_as_zero=False
        )

        frame = _normalise_risk_level(frame)
        frame = _attach_employee_ids(frame, settings.employee_id_prefix)
        frame = _add_burnout_percent(frame, settings.burnout_score_max)
        frame = _add_dominant_emotion(frame)
        frame["document_text"] = frame.apply(build_document_text, axis=1)

        missing_optional = tuple(missing_emotions + missing_behaviour)
        if missing_emotions:
            logger.info(
                "Emotion column(s) absent from the CSV, materialised as 0 "
                "(the NRC lexicon detecting nothing IS zero): %s",
                ", ".join(missing_emotions),
            )
        if missing_behaviour:
            logger.warning(
                "Behaviour column(s) absent from the CSV and left ABSENT, not "
                "zero-filled — a fabricated measurement would be embedded into "
                "the vector store and shown to the LLM as evidence: %s. "
                "Cause: models/features/behavior_features.py reads a "
                "non-existent 'message' column. Owner: Person 2.",
                ", ".join(missing_behaviour),
            )

        dataset = BurnoutDataset(
            frame=frame,
            source_path=csv_path,
            source_mtime=mtime,
            loaded_at=datetime.now(timezone.utc),
            missing_optional_columns=missing_optional,
        )

        _CACHE.clear()          # only ever keep the newest snapshot
        _CACHE[cache_key] = dataset

        logger.info(
            "Loaded %d records | risk mix: %s",
            dataset.record_count,
            frame["risk_level"].value_counts().to_dict(),
        )
        return dataset


def load_burnout_dataframe(
    settings: Optional[Settings] = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper returning just the enriched DataFrame."""
    return load_burnout_dataset(settings=settings, force_reload=force_reload).frame


def clear_cache() -> None:
    """Drop the cached dataset. Used by ``POST /refresh`` and by tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
    logger.info("Burnout dataset cache cleared.")


# ==============================================================================
# Vector-store support
# ==============================================================================
def row_to_metadata(row: pd.Series) -> Dict[str, Any]:
    """Build a ChromaDB metadata dict for one record.

    Chroma accepts only str, int, float and bool — no None, no lists, no
    nested dicts — and rejects the whole batch otherwise. Every value is
    therefore explicitly coerced here rather than passed through, and this is
    the only place that conversion happens.

    These fields are what `retriever.py` filters on, so anything the API needs
    to filter by must be present in this dict.
    """
    metadata: Dict[str, Any] = {
        "employee_id": str(row["employee_id"]),
        "sentiment": str(row["sentiment"]),
        "risk_level": str(row["risk_level"]),
        "burnout_score": float(row["burnout_score"]),
        "burnout_percent": float(row["burnout_percent"]),
        "dominant_emotion": str(row.get("dominant_emotion", "unknown")),
        "clean_text": str(row["clean_text"])[:1000],
        "explanation": str(row.get("explanation", ""))[:1000],
        "confidence": float(row.get("confidence", 0.0) or 0.0),
    }
    for column in EMOTION_COLUMNS:
        metadata[column] = float(row.get(column, 0) or 0)
    for column in BEHAVIOR_COLUMNS:
        if column in row.index:
            metadata[column] = float(row.get(column, 0) or 0)
    return metadata


# ==============================================================================
# Aggregation — the numbers Person 4's dashboard renders
# ==============================================================================
def compute_statistics(frame: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate the metrics shown on the dashboard and in the PDF report.

    Lives in the backend, not the frontend, so the gauge in Streamlit, the
    figures in the PDF and any value quoted by the LLM all originate from one
    calculation.

    ``overall_burnout_percent`` is the share of records at Medium or High risk
    — deliberately not the mean point score, which would silently change
    meaning if Person 2 retuned the scoring weights.
    """
    total = int(len(frame))
    if total == 0:
        return {
            "total_records": 0,
            "overall_burnout_percent": 0.0,
            "mean_burnout_percent": 0.0,
            "mean_burnout_score": 0.0,
            "at_risk_count": 0,
            "risk_distribution": {level: 0 for level in RISK_LEVELS},
            "sentiment_distribution": {},
            "emotion_totals": {emotion: 0 for emotion in EMOTION_COLUMNS},
            "dominant_emotion": "neutral",
            "top_high_risk": [],
        }

    risk_counts = frame["risk_level"].value_counts().to_dict()
    risk_distribution = {level: int(risk_counts.get(level, 0)) for level in RISK_LEVELS}
    for level, count in risk_counts.items():          # keep anything unexpected visible
        risk_distribution.setdefault(str(level), int(count))

    at_risk = int(sum(risk_distribution.get(level, 0) for level in AT_RISK_LEVELS))

    emotion_totals = {
        emotion: float(frame[emotion].sum())
        for emotion in EMOTION_COLUMNS
        if emotion in frame.columns
    }
    dominant = (
        max(emotion_totals, key=emotion_totals.get)
        if emotion_totals and max(emotion_totals.values()) > 0
        else "neutral"
    )

    ranked = frame.sort_values(
        by=["burnout_score", "employee_id"], ascending=[False, True]
    )

    return {
        "total_records": total,
        "overall_burnout_percent": round(at_risk / total * 100.0, 1),
        "mean_burnout_percent": round(float(frame["burnout_percent"].mean()), 1),
        "mean_burnout_score": round(float(frame["burnout_score"].mean()), 2),
        "at_risk_count": at_risk,
        "risk_distribution": risk_distribution,
        "sentiment_distribution": {
            str(k): int(v) for k, v in frame["sentiment"].value_counts().items()
        },
        "emotion_totals": emotion_totals,
        "dominant_emotion": dominant,
        "top_high_risk": ranked.head(10)[
            [
                "employee_id",
                "risk_level",
                "burnout_score",
                "burnout_percent",
                "sentiment",
                "dominant_emotion",
                "clean_text",
                "explanation",
            ]
        ].to_dict(orient="records"),
    }


# ==============================================================================
# Lookup helpers used by the API and the dashboard
# ==============================================================================
def get_employee_record(
    employee_id: str, frame: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """Return one record by employee ID, or ``None``. Case-insensitive."""
    frame = load_burnout_dataframe() if frame is None else frame
    matches = frame[
        frame["employee_id"].str.lower() == str(employee_id).strip().lower()
    ]
    return None if matches.empty else matches.iloc[0].to_dict()


def search_records(
    query: str,
    frame: Optional[pd.DataFrame] = None,
    risk_levels: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Plain substring search over employee ID and message text.

    Intentionally NOT semantic. This backs the dashboard's "Search Employee"
    box, where a user typing ``EMP_0007`` expects that exact record, not the
    five most conceptually similar ones. Semantic search is the retriever's
    job and is reached through the RAG endpoints instead.
    """
    frame = load_burnout_dataframe() if frame is None else frame
    result = frame

    if risk_levels:
        wanted = {level.strip().title() for level in risk_levels}
        result = result[result["risk_level"].isin(wanted)]

    term = (query or "").strip().lower()
    if term:
        result = result[
            result["employee_id"].str.lower().str.contains(term, regex=False)
            | result["clean_text"].str.lower().str.contains(term, regex=False)
            | result["explanation"].str.lower().str.contains(term, regex=False)
        ]

    return result.head(limit).to_dict(orient="records")


__all__ = [
    "BurnoutDataset",
    "REQUIRED_COLUMNS",
    "EMOTION_COLUMNS",
    "BEHAVIOR_COLUMNS",
    "RISK_LEVELS",
    "AT_RISK_LEVELS",
    "SENTIMENT_LABELS",
    "load_burnout_dataset",
    "load_burnout_dataframe",
    "clear_cache",
    "build_document_text",
    "row_to_metadata",
    "compute_statistics",
    "get_employee_record",
    "search_records",
]


# ==============================================================================
# Manual inspection:  python backend/data_loader.py
# ==============================================================================
if __name__ == "__main__":
    import json

    dataset = load_burnout_dataset()

    print("=" * 78)
    print("PROVENANCE")
    print("=" * 78)
    print(json.dumps(dataset.describe(), indent=2))

    print("\n" + "=" * 78)
    print("STATISTICS")
    print("=" * 78)
    stats = compute_statistics(dataset.frame)
    print(json.dumps({k: v for k, v in stats.items() if k != "top_high_risk"}, indent=2))

    print("\n" + "=" * 78)
    print("TOP HIGH RISK")
    print("=" * 78)
    for record in stats["top_high_risk"][:5]:
        print(
            f"  {record['employee_id']}  {record['risk_level']:<6} "
            f"score={record['burnout_score']:<5g} ({record['burnout_percent']:g}%)  "
            f"{record['clean_text'][:48]}"
        )

    print("\n" + "=" * 78)
    print("EMBEDDED DOCUMENT TEXT (first record)")
    print("=" * 78)
    print(dataset.frame.iloc[0]["document_text"])
