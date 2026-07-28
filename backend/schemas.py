"""
backend/schemas.py — API request/response contracts (Person 3)
==============================================================================

WHAT THIS FILE IS FOR

    Every byte crossing the FastAPI boundary is described here as a Pydantic
    model. That buys three things at once:

      1. Validation. Person 4's dashboard cannot send a malformed request
         without getting a precise 422 naming the offending field.
      2. Documentation. FastAPI generates /docs directly from these classes,
         so the interactive API reference can never drift from the code.
      3. A parsing target. `rag_pipeline.py` validates raw LLM output against
         `BurnoutAnalysis`, which is what turns a text completion into the
         guaranteed structured JSON the specification requires.

THE EIGHT-KEY CONTRACT
    `BurnoutAnalysis` is the heart of this file and implements the specified
    output exactly:

        burnout_summary              str
        emotional_analysis           str
        possible_causes              list[str]
        risk_level                   Low | Medium | High
        suggested_interventions      list[str]
        hr_recommendations           list[str]
        mental_wellness_suggestions  list[str]
        confidence_note              str

DEFENSIVE COERCION, AND WHY IT IS NOT OPTIONAL
    LLMs are asked for JSON; they do not reliably return it in the requested
    shape. A model told to produce a list of causes will, perhaps one time in
    twenty, return a newline-separated string or a numbered paragraph instead.
    Rejecting that response would surface to an HR user as a hard 500 error on
    a perfectly good answer.

    The validators below therefore accept the near-misses and normalise them:
    a string becomes a one-element list, bullet and numeric prefixes are
    stripped, `"HIGH RISK"` becomes `High`. This is deliberate tolerance at the
    boundary — the internal representation stays strict, so nothing downstream
    ever has to ask what shape a field is in.

    The one thing NOT tolerated is a missing `confidence_note`. See below.
==============================================================================
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# Enumerations
# ==============================================================================
class RiskLevel(str, Enum):
    """Risk bands exactly as produced by Person 2's ``burnout_score.py``.

    Thresholds live in Person 2's code (score <= 2 Low, <= 5 Medium, else
    High) and are deliberately NOT duplicated here — this enum names the
    values, it does not decide them.
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"


class ReportScope(str, Enum):
    """Which records a generated report should cover."""

    ALL = "all"
    AT_RISK = "at_risk"        # Medium + High
    HIGH_ONLY = "high_only"
    EMPLOYEE = "employee"      # a single employee_id


class ServiceStatus(str, Enum):
    """Overall backend health."""

    OK = "ok"
    DEGRADED = "degraded"      # serving, but something is wrong (e.g. LLM down)
    ERROR = "error"


# ==============================================================================
# Shared coercion helpers
# ==============================================================================
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*•–—]|\d+[.)])\s*")


def _to_string_list(value: Any) -> List[str]:
    """Normalise an LLM's idea of "a list" into an actual ``list[str]``.

    Handles, in order of how often they occur in practice:
        ["a", "b"]                      already correct
        "a\\nb\\nc"                      newline-separated string
        "- a\\n- b"  /  "1. a\\n2. b"    bulleted or numbered string
        "single sentence"               one-element list
        [{"cause": "workload"}]         list of single-key dicts
        None / ""                       empty list

    Empty strings are dropped so a trailing newline does not become a blank
    bullet in the dashboard or the PDF.
    """
    if value is None:
        return []

    if isinstance(value, str):
        items = [line for line in value.splitlines() if line.strip()]
        if len(items) <= 1:
            items = [value] if value.strip() else []
    elif isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            if isinstance(item, dict):
                # e.g. {"cause": "..."} or {"text": "..."} — take the values.
                items.extend(str(v) for v in item.values())
            else:
                items.append(str(item))
    elif isinstance(value, dict):
        items = [str(v) for v in value.values()]
    else:
        items = [str(value)]

    cleaned = [_BULLET_PREFIX.sub("", str(item)).strip() for item in items]
    return [item for item in cleaned if item]


def _to_risk_level(value: Any) -> Any:
    """Coerce loose risk wording onto the :class:`RiskLevel` enum.

    Accepts ``"high"``, ``"HIGH RISK"``, ``"Risk: Medium"``, ``"low risk"``.
    Anything unrecognisable becomes ``Unknown`` rather than raising — a model
    that phrases the band oddly should not discard an otherwise sound analysis,
    and ``Unknown`` is visibly wrong in the UI, which is the desired behaviour.
    """
    if isinstance(value, RiskLevel):
        return value
    if not isinstance(value, str):
        return RiskLevel.UNKNOWN

    text = value.strip().lower()
    # Order matters: check High before Low, so "not low, high" resolves to High.
    if "high" in text or "severe" in text or "critical" in text:
        return RiskLevel.HIGH
    if "medium" in text or "moderate" in text:
        return RiskLevel.MEDIUM
    if "low" in text or "minimal" in text:
        return RiskLevel.LOW
    return RiskLevel.UNKNOWN


# ==============================================================================
# Retrieval
# ==============================================================================
class RetrievedRecord(BaseModel):
    """One employee record returned by the retriever as evidence.

    Included in API responses so the dashboard can display *what the answer was
    based on*. For an explainability project this is not a debugging nicety —
    an HR user acting on a burnout recommendation must be able to see the
    underlying messages, and `similarity` tells them how relevant each one
    actually was.
    """

    model_config = ConfigDict(populate_by_name=True)

    employee_id: str = Field(..., description="Stable record identifier.")
    clean_text: str = Field(..., description="The employee's message text.")
    sentiment: str = Field(..., description="RoBERTa label from Person 2.")
    risk_level: RiskLevel = Field(..., description="Risk band from Person 2.")
    burnout_score: float = Field(..., description="Raw integer point score.")
    burnout_percent: float = Field(..., ge=0, le=100, description="Score as 0-100.")
    dominant_emotion: str = Field(..., description="Highest NRC emotion, or 'neutral'.")
    explanation: str = Field(default="", description="Person 2's rule-based reason.")
    similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity to the question. 1.0 = identical.",
    )
    document_text: Optional[str] = Field(
        default=None,
        description="The exact text embedded and shown to the LLM as evidence.",
    )

    @field_validator("risk_level", mode="before")
    @classmethod
    def _coerce_risk(cls, v: Any) -> Any:
        return _to_risk_level(v)


# ==============================================================================
# The eight-key analysis contract
# ==============================================================================
class BurnoutAnalysis(BaseModel):
    """Structured burnout analysis produced by the LLM.

    This is the specified output format. `rag_pipeline.py` validates raw model
    output against this class, so any response reaching the dashboard is
    guaranteed to carry all eight fields with correct types.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "burnout_summary": (
                    "One employee shows moderate burnout risk driven by "
                    "exhaustion after sustained late working."
                ),
                "emotional_analysis": (
                    "Sadness is the dominant NRC signal, paired with negative "
                    "RoBERTa sentiment at 0.80 confidence. Joy and trust are "
                    "absent, which matters more than the sadness count alone."
                ),
                "possible_causes": [
                    "Sustained overtime without recovery periods",
                    "Workload exceeding capacity for the current sprint",
                ],
                "risk_level": "Medium",
                "suggested_interventions": [
                    "Hold a workload review 1:1 within the week",
                    "Redistribute at least one deliverable this sprint",
                ],
                "hr_recommendations": [
                    "Audit overtime patterns across the team, not just this individual",
                    "Confirm the employee's leave balance is being used",
                ],
                "mental_wellness_suggestions": [
                    "Share the Employee Assistance Programme details",
                    "Encourage protected non-working hours in the evenings",
                ],
                "confidence_note": (
                    "Based on 1 message from a 10-record sample. This is a "
                    "signal for a human conversation, not a clinical finding."
                ),
            }
        }
    )

    burnout_summary: str = Field(
        ...,
        min_length=1,
        description="Plain-language overview of the burnout situation.",
    )
    emotional_analysis: str = Field(
        ...,
        min_length=1,
        description="What the sentiment and NRC emotion signals indicate.",
    )
    possible_causes: List[str] = Field(
        default_factory=list,
        description="Likely contributing factors, grounded in the retrieved records.",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.UNKNOWN,
        description="Risk band. Echoed from Person 2's data, not re-derived.",
    )
    suggested_interventions: List[str] = Field(
        default_factory=list,
        description="Concrete manager-level actions.",
    )
    hr_recommendations: List[str] = Field(
        default_factory=list,
        description="Policy and process actions for HR.",
    )
    mental_wellness_suggestions: List[str] = Field(
        default_factory=list,
        description="Supportive, non-clinical wellbeing steps.",
    )
    confidence_note: str = Field(
        ...,
        min_length=1,
        description=(
            "Honest statement of evidence limits. REQUIRED — see the validator."
        ),
    )

    # --------------------------------------------------------------------------
    @field_validator(
        "possible_causes",
        "suggested_interventions",
        "hr_recommendations",
        "mental_wellness_suggestions",
        mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v: Any) -> List[str]:
        return _to_string_list(v)

    @field_validator("risk_level", mode="before")
    @classmethod
    def _coerce_risk(cls, v: Any) -> Any:
        return _to_risk_level(v)

    @field_validator("burnout_summary", "emotional_analysis", mode="before")
    @classmethod
    def _coerce_prose(cls, v: Any) -> str:
        """Flatten a list back into prose if the model over-structures a field."""
        if isinstance(v, (list, tuple)):
            return " ".join(str(item).strip() for item in v if str(item).strip())
        return "" if v is None else str(v).strip()

    @field_validator("confidence_note", mode="before")
    @classmethod
    def _require_confidence_note(cls, v: Any) -> str:
        """Substitute a default rather than let this field go missing.

        Every other field may legitimately be sparse. This one may not.

        The system infers a sensitive human state from a handful of short text
        samples using a lexicon and a general-purpose sentiment model. The
        specification requires that limitation to be shown to the HR user, and
        an omission here would silently present a probabilistic inference as a
        finding of fact — precisely the black-box behaviour this project exists
        to avoid. If the model drops the field, we supply the caveat ourselves
        instead of rendering the analysis without one.
        """
        if isinstance(v, (list, tuple)):
            v = " ".join(str(item).strip() for item in v if str(item).strip())
        text = "" if v is None else str(v).strip()
        if not text:
            return (
                "The language model did not supply a confidence statement. "
                "Treat this analysis as an indicative signal derived from a "
                "limited sample of workplace text, not a clinical or diagnostic "
                "conclusion, and confirm through direct conversation."
            )
        return text


# ==============================================================================
# GET /
# ==============================================================================
class RootResponse(BaseModel):
    """Service banner returned by ``GET /``."""

    service: str = Field(default="Employee Burnout RAG API")
    description: str = Field(
        default=(
            "Explainable Employee Burnout Detection using Machine Learning, "
            "Sentiment Analysis and Retrieval-Augmented Generation."
        )
    )
    version: str
    status: ServiceStatus = Field(default=ServiceStatus.OK)
    docs_url: str = Field(default="/docs")
    endpoints: Dict[str, str] = Field(
        default_factory=lambda: {
            "GET /": "This service banner",
            "GET /health": "Health check, configuration and data provenance",
            "GET /statistics": "Aggregate metrics for the dashboard",
            "POST /query": "Ask a natural-language question (full RAG pipeline)",
            "POST /report": "Generate a structured organisational report",
            "POST /refresh": "Rebuild the vector store from Person 2's CSV",
        }
    )


# ==============================================================================
# GET /health
# ==============================================================================
class HealthResponse(BaseModel):
    """Health, configuration and data provenance for ``GET /health``.

    Credentials are masked by ``Settings.safe_summary()`` before they reach
    this model — /health is routinely screenshotted for demos and reports.
    """

    status: ServiceStatus
    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    configuration: Dict[str, Any] = Field(
        ..., description="Redacted settings snapshot."
    )
    data: Dict[str, Any] = Field(
        ..., description="Provenance of Person 2's CSV."
    )
    vector_store: Dict[str, Any] = Field(
        ..., description="ChromaDB collection state."
    )
    llm_reachable: Optional[bool] = Field(
        default=None,
        description="Whether the configured LLM answered. None = not probed.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal problems, e.g. columns absent from the CSV.",
    )


# ==============================================================================
# GET /statistics
# ==============================================================================
class StatisticsResponse(BaseModel):
    """Aggregate metrics backing every chart on Person 4's dashboard."""

    total_records: int
    overall_burnout_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Share of records at Medium or High risk. Drives the gauge. "
            "Scale-independent, so it stays meaningful if Person 2 retunes "
            "the point weights."
        ),
    )
    mean_burnout_percent: float = Field(
        ..., ge=0, le=100, description="Mean normalised score, reported alongside."
    )
    mean_burnout_score: float = Field(..., description="Mean raw point score.")
    at_risk_count: int = Field(..., ge=0)
    risk_distribution: Dict[str, int]
    sentiment_distribution: Dict[str, int]
    emotion_totals: Dict[str, float]
    dominant_emotion: str
    top_high_risk: List[Dict[str, Any]]
    source: Dict[str, Any] = Field(
        default_factory=dict, description="CSV provenance."
    )


# ==============================================================================
# POST /query
# ==============================================================================
class QueryRequest(BaseModel):
    """A natural-language question for the RAG pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Which employees show the strongest signs of burnout, and why?",
                "top_k": 5,
                "risk_levels": ["Medium", "High"],
                "include_sources": True,
            }
        }
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The HR user's question.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Records to retrieve. Defaults to RETRIEVER_TOP_K and is clamped "
            "to RETRIEVER_MAX_K by the pipeline, so a caller cannot overflow "
            "the LLM context window."
        ),
    )
    risk_levels: Optional[List[RiskLevel]] = Field(
        default=None,
        description="Restrict retrieval to these bands. None = no filter.",
    )
    employee_id: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a single employee.",
    )
    include_sources: bool = Field(
        default=True,
        description="Return the retrieved records that grounded the answer.",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        """Reject whitespace-only questions.

        `min_length` alone passes `"   "`, which would send an empty question
        to the LLM and return a confidently meaningless answer.
        """
        text = v.strip()
        if len(text) < 3:
            raise ValueError(
                "question must contain at least 3 non-whitespace characters."
            )
        return text

    @field_validator("employee_id")
    @classmethod
    def _tidy_employee_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        text = v.strip()
        return text or None

    @field_validator("risk_levels", mode="before")
    @classmethod
    def _coerce_risk_levels(cls, v: Any) -> Any:
        """Accept ``"High"``, ``["high","medium"]`` or a comma-separated string."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [part for part in re.split(r"[,\s]+", v) if part]
        return [_to_risk_level(item) for item in v]


class QueryResponse(BaseModel):
    """Full result of ``POST /query``."""

    question: str
    answer: BurnoutAnalysis
    sources: List[RetrievedRecord] = Field(
        default_factory=list,
        description="Records retrieved as evidence. Empty if include_sources=False.",
    )
    retrieved_count: int = Field(..., ge=0)
    llm_provider: str
    llm_model: str
    elapsed_seconds: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    degraded: bool = Field(
        default=False,
        description=(
            "True when the LLM was unreachable and the response was assembled "
            "from Person 2's rule-based explanations instead. The dashboard "
            "shows a banner so the user is never misled about the source."
        ),
    )
    notes: List[str] = Field(
        default_factory=list, description="Non-fatal warnings for this request."
    )


# ==============================================================================
# POST /report
# ==============================================================================
class ReportRequest(BaseModel):
    """Request for a structured organisational burnout report."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scope": "at_risk",
                "title": "Q3 Burnout Review — Engineering",
                "max_records": 25,
                "include_sources": True,
            }
        }
    )

    scope: ReportScope = Field(
        default=ReportScope.ALL, description="Which records to cover."
    )
    employee_id: Optional[str] = Field(
        default=None, description="Required when scope='employee'."
    )
    title: str = Field(
        default="Employee Burnout Analysis Report",
        max_length=200,
    )
    focus: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional steer, e.g. 'focus on workload distribution'.",
    )
    max_records: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Cap on records summarised, to bound LLM context size.",
    )
    include_sources: bool = Field(default=True)

    @field_validator("employee_id")
    @classmethod
    def _tidy_employee_id(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v and v.strip() else None


class ReportResponse(BaseModel):
    """Result of ``POST /report``.

    Returns structured JSON rather than a binary PDF on purpose: Person 4's
    `pdf_report.py` renders it with ReportLab client-side, which keeps document
    styling out of the API and lets the same payload feed the on-screen panel
    and the downloadable file without a second request.
    """

    title: str
    scope: ReportScope
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    analysis: BurnoutAnalysis
    statistics: StatisticsResponse
    sources: List[RetrievedRecord] = Field(default_factory=list)
    record_count: int = Field(..., ge=0)
    llm_provider: str
    llm_model: str
    elapsed_seconds: float = Field(..., ge=0)
    degraded: bool = Field(default=False)
    notes: List[str] = Field(default_factory=list)


# ==============================================================================
# POST /refresh
# ==============================================================================
class RefreshRequest(BaseModel):
    """Request to rebuild the vector store from Person 2's CSV."""

    force: bool = Field(
        default=True,
        description=(
            "True wipes and re-embeds every record. False re-embeds only when "
            "the CSV's fingerprint has changed since the last build."
        ),
    )


class RefreshResponse(BaseModel):
    """Result of ``POST /refresh`` — backs the dashboard's Refresh button."""

    status: ServiceStatus
    message: str
    records_indexed: int = Field(..., ge=0)
    rebuilt: bool = Field(
        ..., description="False when the index was already current."
    )
    collection: str
    persist_dir: str
    elapsed_seconds: float = Field(..., ge=0)
    source: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Errors
# ==============================================================================
class ErrorResponse(BaseModel):
    """Uniform error body for every non-2xx response.

    A single shape means Person 4's `api_client.py` has exactly one error path
    to handle, rather than guessing at FastAPI defaults versus custom bodies.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "DataUnavailable",
                "detail": "Person 2's burnout output was not found at outputs/burnout_scores.csv",
                "hint": "Run: python models/burnout/burnout_score.py",
            }
        }
    )

    error: str = Field(..., description="Machine-readable error class.")
    detail: str = Field(..., description="Human-readable explanation.")
    hint: Optional[str] = Field(
        default=None, description="Concrete next action, where one exists."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "RiskLevel",
    "ReportScope",
    "ServiceStatus",
    "RetrievedRecord",
    "BurnoutAnalysis",
    "RootResponse",
    "HealthResponse",
    "StatisticsResponse",
    "QueryRequest",
    "QueryResponse",
    "ReportRequest",
    "ReportResponse",
    "RefreshRequest",
    "RefreshResponse",
    "ErrorResponse",
]
