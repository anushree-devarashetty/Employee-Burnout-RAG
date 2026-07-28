"""
backend/rag/rag_pipeline.py — Pipeline orchestration (Person 3, stage 5)
==============================================================================

    User Question
         |
         v
    Embedding      (embeddings.py)
         |
         v
    Retriever      (retriever.py -> vectorstore.py)
         |
         v
    Prompt         (prompts/burnout_prompt.py)
         |
         v
    LLM            (llm_client.py — OpenAI | Gemini | Ollama)
         |
         v
    Structured JSON (schemas.py::BurnoutAnalysis)

This is the only module that knows about all five stages. Every other module
owns exactly one, which is what keeps them independently testable.

THE HARD PART IS THE LAST STEP
    Getting text out of an LLM is easy. Getting *reliably parseable* JSON out
    of one is not. Even with JSON mode enabled, models wrap output in markdown
    fences, prepend "Here is the analysis:", or emit trailing commas.

    `parse_llm_json` therefore tries four strategies in increasing order of
    desperation, and only then gives up. Combined with the tolerant validators
    in `schemas.py`, this is what makes "structured JSON" an actual guarantee
    rather than an aspiration.

DEGRADATION IS A FEATURE
    If the LLM is unreachable — overwhelmingly the most likely failure, since
    the default provider is a local Ollama server the user must remember to
    start — the pipeline returns Person 2's rule-based findings with
    `degraded=True` instead of raising. An HR user gets a usable, clearly
    labelled answer rather than a stack trace.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from pydantic import ValidationError

from backend.config import Settings, get_settings
from backend.data_loader import compute_statistics, load_burnout_dataframe
from backend.prompts.burnout_prompt import (
    SYSTEM_PROMPT,
    build_fallback_analysis,
    build_query_prompt,
    build_report_prompt,
)
from backend.rag.llm_client import (
    BaseLLMClient,
    LLMError,
    LLMUnavailableError,
    get_llm_client,
)
from backend.rag.retriever import BurnoutRetriever, RetrieverError, get_retriever
from backend.schemas import (
    BurnoutAnalysis,
    QueryRequest,
    QueryResponse,
    ReportRequest,
    ReportResponse,
    ReportScope,
    RetrievedRecord,
    StatisticsResponse,
)

logger = logging.getLogger(__name__)

#: Matches ```json ... ``` and bare ``` ... ``` fences.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot produce a result at all."""


# ==============================================================================
# JSON extraction
# ==============================================================================
def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Four strategies, in order:

      1. Parse the whole string. Works when JSON mode behaved.
      2. Strip a markdown fence and parse the contents. Models add ```json
         fences even when told not to.
      3. Slice from the first ``{`` to the last ``}``. Handles prose wrapped
         around the object ("Here is the analysis: {...} Hope that helps").
      4. Repair common malformations — trailing commas before ``}`` or ``]``,
         and Python's True/False/None — then retry the slice.

    Raises:
        PipelineError: if no strategy yields a JSON object.
    """
    if not raw or not raw.strip():
        raise PipelineError("The language model returned an empty response.")

    text = raw.strip()

    # 1. straight parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. markdown fence
    fence = _FENCE.search(text)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            if isinstance(parsed, dict):
                logger.debug("Recovered JSON from a markdown fence.")
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. widest brace span
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                logger.debug("Recovered JSON by brace slicing.")
                return parsed
        except json.JSONDecodeError:
            # 4. repair and retry
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)   # trailing commas
            repaired = re.sub(r"\bTrue\b", "true", repaired)
            repaired = re.sub(r"\bFalse\b", "false", repaired)
            repaired = re.sub(r"\bNone\b", "null", repaired)
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    logger.info("Recovered JSON after repairing malformations.")
                    return parsed
            except json.JSONDecodeError:
                pass

    raise PipelineError(
        f"Could not extract JSON from the model's response. "
        f"First 400 characters: {text[:400]}"
    )


def coerce_to_analysis(payload: Dict[str, Any]) -> BurnoutAnalysis:
    """Validate a parsed payload into a :class:`BurnoutAnalysis`.

    If a model nests everything under a wrapper key (``{"analysis": {...}}``,
    ``{"result": {...}}``), the wrapper is unwrapped before validation.
    """
    if not any(key in payload for key in ("burnout_summary", "risk_level")):
        for wrapper in ("analysis", "result", "response", "output", "data"):
            inner = payload.get(wrapper)
            if isinstance(inner, dict):
                logger.debug("Unwrapped nested '%s' key.", wrapper)
                payload = inner
                break

    try:
        return BurnoutAnalysis(**payload)
    except ValidationError as exc:
        # Almost always a missing required prose field. Supply a neutral
        # placeholder rather than discarding an otherwise complete analysis.
        logger.warning("LLM output failed validation; repairing. %s", exc)
        repaired = dict(payload)
        repaired.setdefault(
            "burnout_summary",
            "The language model did not provide a summary for this analysis.",
        )
        repaired.setdefault(
            "emotional_analysis",
            "The language model did not provide an emotional analysis.",
        )
        repaired.setdefault("confidence_note", "")   # validator substitutes text
        try:
            return BurnoutAnalysis(**repaired)
        except ValidationError as inner_exc:
            raise PipelineError(
                f"The model's response could not be validated: {inner_exc}"
            ) from inner_exc


# ==============================================================================
# Pipeline
# ==============================================================================
class RAGPipeline:
    """Wires retrieval, prompting and generation into one workflow."""

    def __init__(
        self,
        retriever: BurnoutRetriever,
        llm: BaseLLMClient,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # --------------------------------------------------------------------------
    def _statistics(self) -> Dict[str, Any]:
        """Dataset-wide aggregates, used as orienting context in prompts."""
        return compute_statistics(load_burnout_dataframe())

    def _generate(
        self,
        prompt: str,
        records: List[RetrievedRecord],
        stats: Dict[str, Any],
    ) -> Tuple[BurnoutAnalysis, bool, List[str]]:
        """Call the LLM and validate, degrading gracefully on failure.

        Returns:
            ``(analysis, degraded, notes)``.
        """
        notes: List[str] = []

        try:
            raw = self.llm.complete(
                user_prompt=prompt, system_prompt=SYSTEM_PROMPT, as_json=True
            )
            return coerce_to_analysis(parse_llm_json(raw)), False, notes

        except LLMUnavailableError as exc:
            reason = (
                f"The {self.llm.provider_name} language model could not be "
                f"reached."
            )
            logger.warning("LLM unavailable; degrading. %s", exc)
            notes.append(f"{reason} Showing rule-based findings instead.")
            notes.append(str(exc).split("\n")[0])

        except (LLMError, PipelineError) as exc:
            reason = (
                f"The {self.llm.provider_name} language model returned an "
                f"unusable response."
            )
            logger.warning("LLM response unusable; degrading. %s", exc)
            notes.append(f"{reason} Showing rule-based findings instead.")

        fallback = build_fallback_analysis(records, stats, reason=reason)
        return BurnoutAnalysis(**fallback), True, notes

    # --------------------------------------------------------------------------
    def query(self, request: QueryRequest) -> QueryResponse:
        """Run the full RAG workflow for ``POST /query``."""
        started = time.perf_counter()

        try:
            retrieval = self.retriever.retrieve(
                question=request.question,
                top_k=request.top_k,
                risk_levels=request.risk_levels,
                employee_id=request.employee_id,
            )
        except RetrieverError as exc:
            raise PipelineError(str(exc)) from exc

        stats = self._statistics()
        prompt = build_query_prompt(
            question=request.question,
            records=retrieval.records,
            statistics=stats,
            notes=retrieval.notes,
        )

        analysis, degraded, llm_notes = self._generate(
            prompt, retrieval.records, stats
        )

        return QueryResponse(
            question=request.question,
            answer=analysis,
            sources=retrieval.records if request.include_sources else [],
            retrieved_count=retrieval.count,
            llm_provider=self.llm.provider_name,
            llm_model=self.llm.model_name,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            degraded=degraded,
            notes=retrieval.notes + llm_notes,
        )

    # --------------------------------------------------------------------------
    def report(self, request: ReportRequest) -> ReportResponse:
        """Generate a structured report for ``POST /report``."""
        started = time.perf_counter()

        scope = (
            request.scope.value
            if isinstance(request.scope, ReportScope)
            else str(request.scope)
        )

        try:
            retrieval = self.retriever.retrieve_by_scope(
                scope=scope,
                max_records=request.max_records,
                employee_id=request.employee_id,
            )
        except RetrieverError as exc:
            raise PipelineError(str(exc)) from exc

        stats = self._statistics()
        prompt = build_report_prompt(
            records=retrieval.records,
            statistics=stats,
            scope=scope,
            title=request.title,
            focus=request.focus,
            notes=retrieval.notes,
        )

        analysis, degraded, llm_notes = self._generate(
            prompt, retrieval.records, stats
        )

        frame = load_burnout_dataframe()
        statistics = StatisticsResponse(
            **stats,
            source={
                "path": str(self.settings.burnout_csv),
                "records": int(len(frame)),
                "owner": "Person 2 — consumed read-only",
            },
        )

        return ReportResponse(
            title=request.title,
            scope=request.scope,
            analysis=analysis,
            statistics=statistics,
            sources=retrieval.records if request.include_sources else [],
            record_count=retrieval.count,
            llm_provider=self.llm.provider_name,
            llm_model=self.llm.model_name,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            degraded=degraded,
            notes=retrieval.notes + llm_notes,
        )


# ==============================================================================
# Singleton accessor
# ==============================================================================
_INSTANCE: Optional[RAGPipeline] = None
_INSTANCE_LOCK = threading.Lock()


def get_pipeline(settings: Optional[Settings] = None) -> RAGPipeline:
    """Return the process-wide :class:`RAGPipeline`.

    Manual singleton rather than ``@lru_cache``: Pydantic v2 ``Settings`` is
    unhashable and lru_cache would raise on an explicit settings argument.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            resolved = settings or get_settings()
            _INSTANCE = RAGPipeline(
                retriever=get_retriever(resolved),
                llm=get_llm_client(resolved),
                settings=resolved,
            )
    return _INSTANCE


def reset_pipeline() -> None:
    """Drop the cached pipeline. Used by tests."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


__all__ = [
    "RAGPipeline",
    "PipelineError",
    "get_pipeline",
    "parse_llm_json",
    "coerce_to_analysis",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    from backend.rag.vectorstore import build_index

    build_index(force=False)
    response = get_pipeline().query(
        QueryRequest(question="Which employees show signs of burnout, and why?")
    )
    print(json.dumps(json.loads(response.model_dump_json()), indent=2)[:3000])
