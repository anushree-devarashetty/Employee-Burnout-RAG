"""
backend/rag/retriever.py — Top-K record retrieval (Person 3, stage 3)
==============================================================================

ROLE
    Turns a natural-language question into the K most relevant employee
    records. Owns query encoding, metadata filtering, the top_k ceiling, and
    conversion of raw Chroma hits into `RetrievedRecord` objects.

    It does not build prompts and does not call an LLM.

TWO BEHAVIOURS THAT ARE NOT OBVIOUS

1.  AN EMPTY RESULT SET IS RECOVERED FROM, NOT RETURNED
        A filtered search can legitimately match nothing — asking about High
        risk when the dataset contains only Low and Medium is exactly your
        current data. Returning zero records would make the LLM answer from
        no evidence at all, which is the worst possible outcome for an
        explainability system.

        So when a filtered search comes back empty, the filter is dropped and
        the search retried, with a note recorded on the response. The user is
        told the filter matched nothing rather than being handed a confident
        answer grounded in nothing.

2.  A DETECTED EMPLOYEE ID SHORT-CIRCUITS SEMANTIC SEARCH
        "What is going on with EMP_0006?" should return EMP_0006 — not the
        five records whose wording is most similar to that sentence. Embedding
        models are poor at identifier matching, so an explicit ID in the
        question is honoured directly, then topped up with semantically
        similar records for context.
==============================================================================
"""

from __future__ import annotations

import logging
import threading
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from backend.config import Settings, get_settings
from backend.rag.embeddings import EmbeddingModel, get_embedding_model
from backend.rag.vectorstore import BurnoutVectorStore, get_vector_store
from backend.schemas import RetrievedRecord, RiskLevel

logger = logging.getLogger(__name__)


class RetrieverError(RuntimeError):
    """Raised when retrieval cannot be completed."""


@dataclass
class RetrievalResult:
    """Records retrieved for one question, plus how they were obtained."""

    records: List[RetrievedRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    filter_applied: Optional[Dict[str, Any]] = None
    filter_relaxed: bool = False
    requested_k: int = 0

    @property
    def count(self) -> int:
        return len(self.records)

    def as_context_blocks(self) -> List[str]:
        """The evidence strings handed to the prompt builder."""
        return [
            record.document_text or record.clean_text for record in self.records
        ]


# ==============================================================================
# Helpers
# ==============================================================================
def extract_employee_id(question: str, prefix: str = "EMP") -> Optional[str]:
    """Find an explicit employee ID in the question text.

    Matches ``EMP_0006``, ``emp-6``, ``EMP 12`` and bare ``EMP0006``, then
    normalises to the canonical zero-padded form so it matches the IDs
    `data_loader` generates.
    """
    if not question:
        return None

    pattern = rf"\b{re.escape(prefix)}[\s_\-]?0*(\d{{1,6}})\b"
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if not match:
        return None
    return f"{prefix}_{int(match.group(1)):04d}"


def _build_where_clause(
    risk_levels: Optional[Sequence[RiskLevel]] = None,
    employee_id: Optional[str] = None,
    sentiment: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble a ChromaDB metadata filter.

    Chroma requires an explicit ``$and`` when combining conditions; a plain
    multi-key dict is silently interpreted as a single condition, which would
    apply only one of the filters. Single conditions are emitted bare because
    some Chroma versions reject a one-element ``$and``.
    """
    conditions: List[Dict[str, Any]] = []

    if employee_id:
        conditions.append({"employee_id": {"$eq": str(employee_id)}})

    if risk_levels:
        values = [
            level.value if isinstance(level, RiskLevel) else str(level)
            for level in risk_levels
            if str(level) != RiskLevel.UNKNOWN.value
        ]
        if values:
            conditions.append(
                {"risk_level": {"$eq": values[0]} if len(values) == 1
                 else {"$in": values}}
            )

    if sentiment:
        conditions.append({"sentiment": {"$eq": str(sentiment).title()}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _hit_to_record(hit: Dict[str, Any]) -> RetrievedRecord:
    """Convert one raw Chroma hit into a validated `RetrievedRecord`."""
    metadata = hit.get("metadata") or {}
    return RetrievedRecord(
        employee_id=str(metadata.get("employee_id", hit.get("id", "unknown"))),
        clean_text=str(metadata.get("clean_text", "")),
        sentiment=str(metadata.get("sentiment", "Unknown")),
        risk_level=metadata.get("risk_level", "Unknown"),
        burnout_score=float(metadata.get("burnout_score", 0.0) or 0.0),
        burnout_percent=float(metadata.get("burnout_percent", 0.0) or 0.0),
        dominant_emotion=str(metadata.get("dominant_emotion", "unknown")),
        explanation=str(metadata.get("explanation", "")),
        similarity=round(float(hit.get("similarity", 0.0)), 4),
        document_text=hit.get("document"),
    )


# ==============================================================================
# Retriever
# ==============================================================================
class BurnoutRetriever:
    """Semantic retrieval of employee burnout records."""

    def __init__(
        self,
        store: BurnoutVectorStore,
        embedder: EmbeddingModel,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings

    # --------------------------------------------------------------------------
    def _clamp_k(self, top_k: Optional[int]) -> int:
        """Apply the configured default and the hard ceiling.

        The ceiling is what stops an API caller requesting 10,000 records and
        overflowing the LLM's context window — which fails as an opaque
        provider-side error rather than anything actionable.
        """
        requested = top_k or self.settings.retriever_top_k
        clamped = max(1, min(int(requested), self.settings.retriever_max_k))
        if clamped != requested:
            logger.info(
                "top_k %s clamped to %d (RETRIEVER_MAX_K).", requested, clamped
            )
        return clamped

    # --------------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        risk_levels: Optional[Sequence[RiskLevel]] = None,
        employee_id: Optional[str] = None,
        auto_detect_employee: bool = True,
    ) -> RetrievalResult:
        """Retrieve the most relevant records for a question.

        Args:
            question: The user's natural-language question.
            top_k: Records to return. Defaults to RETRIEVER_TOP_K.
            risk_levels: Restrict to these bands.
            employee_id: Restrict to one employee.
            auto_detect_employee: Honour an ID mentioned in the question text.

        Returns:
            A :class:`RetrievalResult`, possibly with notes explaining any
            filter relaxation.
        """
        k = self._clamp_k(top_k)
        notes: List[str] = []

        if self.store.count() == 0:
            raise RetrieverError(
                "The vector store is empty. Build it first:\n"
                "  POST /refresh   (or click 'Refresh Database' in the dashboard)\n"
                "  or run: python backend/rag/vectorstore.py"
            )

        # --- explicit ID beats semantic similarity ---------------------------
        detected = employee_id
        if not detected and auto_detect_employee:
            detected = extract_employee_id(question, self.settings.employee_id_prefix)
            if detected:
                notes.append(
                    f"Detected employee {detected} in the question; that record "
                    f"is included directly rather than by similarity."
                )

        records: List[RetrievedRecord] = []
        if detected:
            exact = self.store.get_by_id(detected)
            if exact:
                records.append(_hit_to_record(exact))
            else:
                notes.append(
                    f"No record exists for {detected}; falling back to semantic "
                    f"search across all records."
                )
                detected = None

        # --- semantic search --------------------------------------------------
        query_vector = self.embedder.embed_query(question)
        where = _build_where_clause(risk_levels=risk_levels)
        filter_relaxed = False

        remaining = max(0, k - len(records))
        hits: List[Dict[str, Any]] = []
        if remaining:
            hits = self.store.query(query_vector, top_k=remaining + len(records),
                                    where=where)

            if not hits and where is not None:
                # Recover rather than answer from nothing — see module docstring.
                logger.info("Filtered search returned nothing; relaxing the filter.")
                filter_relaxed = True
                requested = ", ".join(
                    lvl.value if isinstance(lvl, RiskLevel) else str(lvl)
                    for lvl in (risk_levels or [])
                )
                notes.append(
                    f"No records matched the requested risk level(s): {requested}. "
                    f"Showing the most relevant records from the full dataset "
                    f"instead — the answer below is NOT filtered as requested."
                )
                hits = self.store.query(query_vector, top_k=remaining)

        seen = {record.employee_id for record in records}
        for hit in hits:
            record = _hit_to_record(hit)
            if record.employee_id in seen:
                continue
            records.append(record)
            seen.add(record.employee_id)
            if len(records) >= k:
                break

        if not records:
            notes.append("No records were retrieved for this question.")

        logger.info(
            "Retrieved %d/%d records (filter=%s, relaxed=%s)",
            len(records), k, where, filter_relaxed,
        )

        return RetrievalResult(
            records=records,
            notes=notes,
            filter_applied=where,
            filter_relaxed=filter_relaxed,
            requested_k=k,
        )

    # --------------------------------------------------------------------------
    def retrieve_by_scope(
        self,
        scope: str,
        max_records: int,
        employee_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Select records for a report, by scope rather than by question.

        Reports summarise a *population*, so ranking by burnout score is the
        correct selection here — semantic similarity has no meaning without a
        question. Reads straight from the dataframe, bypassing the vector
        store entirely.
        """
        from backend.data_loader import load_burnout_dataframe

        frame = load_burnout_dataframe()
        notes: List[str] = []

        if scope == "employee":
            if not employee_id:
                raise RetrieverError("scope='employee' requires an employee_id.")
            frame = frame[
                frame["employee_id"].str.lower() == employee_id.strip().lower()
            ]
            if frame.empty:
                raise RetrieverError(f"No record found for employee '{employee_id}'.")

        elif scope == "at_risk":
            frame = frame[frame["risk_level"].isin(["Medium", "High"])]
            if frame.empty:
                notes.append(
                    "No records are at Medium or High risk; reporting on all "
                    "records instead."
                )
                frame = load_burnout_dataframe()

        elif scope == "high_only":
            frame = frame[frame["risk_level"] == "High"]
            if frame.empty:
                notes.append(
                    "No records are at High risk; reporting on Medium and High "
                    "instead."
                )
                frame = load_burnout_dataframe()
                frame = frame[frame["risk_level"].isin(["Medium", "High"])]
                if frame.empty:
                    notes.append("No at-risk records at all; reporting on everything.")
                    frame = load_burnout_dataframe()

        frame = frame.sort_values(
            by=["burnout_score", "employee_id"], ascending=[False, True]
        ).head(max_records)

        records = [
            RetrievedRecord(
                employee_id=str(row["employee_id"]),
                clean_text=str(row["clean_text"]),
                sentiment=str(row["sentiment"]),
                risk_level=row["risk_level"],
                burnout_score=float(row["burnout_score"]),
                burnout_percent=float(row["burnout_percent"]),
                dominant_emotion=str(row.get("dominant_emotion", "unknown")),
                explanation=str(row.get("explanation", "")),
                similarity=1.0,  # selected deterministically, not by similarity
                document_text=str(row["document_text"]),
            )
            for _, row in frame.iterrows()
        ]

        return RetrievalResult(
            records=records, notes=notes, requested_k=max_records
        )


# ==============================================================================
# Singleton accessor
# ==============================================================================
_INSTANCE: Optional[BurnoutRetriever] = None
_INSTANCE_LOCK = threading.Lock()


def get_retriever(settings: Optional[Settings] = None) -> BurnoutRetriever:
    """Return the process-wide :class:`BurnoutRetriever`.

    Manual singleton rather than ``@lru_cache``: Pydantic v2 ``Settings`` is
    unhashable and lru_cache would raise on an explicit settings argument.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            resolved = settings or get_settings()
            _INSTANCE = BurnoutRetriever(
                store=get_vector_store(resolved),
                embedder=get_embedding_model(resolved),
                settings=resolved,
            )
    return _INSTANCE


def reset_retriever() -> None:
    """Drop the cached instance. Used by tests."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


__all__ = [
    "BurnoutRetriever",
    "RetrievalResult",
    "RetrieverError",
    "get_retriever",
    "extract_employee_id",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    from backend.rag.vectorstore import build_index

    build_index(force=False)
    result = get_retriever().retrieve("Who is showing signs of burnout?", top_k=3)
    for note in result.notes:
        print(f"NOTE: {note}")
    for record in result.records:
        print(
            f"  {record.similarity:.4f}  {record.employee_id}  "
            f"{record.risk_level.value:<7} {record.clean_text[:50]}"
        )
