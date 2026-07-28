"""
backend/app.py — FastAPI application (Person 3)
==============================================================================

ENDPOINTS
    GET  /            Service banner
    GET  /health      Health, configuration and data provenance
    GET  /statistics  Aggregate metrics for Person 4's dashboard
    POST /query       Ask a question (full RAG pipeline)
    POST /report      Generate a structured report
    POST /refresh     Rebuild the vector store from Person 2's CSV

    /statistics and /refresh are additions beyond the four specified endpoints.
    /statistics exists so the dashboard's charts do not each re-read and
    re-aggregate the CSV; /refresh backs the specified "Refresh Database"
    sidebar button, which needs a server-side action to invoke.

RUN
    From the REPOSITORY ROOT (not from inside backend/):
        uvicorn backend.app:app --reload --port 8000

    Interactive docs: http://localhost:8000/docs

STARTUP BEHAVIOUR
    The vector store is built on startup only if missing or stale. On a warm
    start the fingerprint matches and nothing is re-embedded.

    Startup deliberately does NOT abort when indexing fails. A missing CSV or a
    stopped Ollama server should leave the API up and answering /health with a
    precise explanation — an HR user seeing "connection refused" learns
    nothing, whereas a 503 naming the missing file is actionable. Configuration
    errors are the exception: those still refuse to boot, because a server
    running with invalid settings is worse than one that will not start.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import __version__
from backend.config import get_settings
from backend.data_loader import (
    clear_cache,
    compute_statistics,
    load_burnout_dataset,
)
from backend.rag.embeddings import EmbeddingError, get_embedding_model
from backend.rag.llm_client import get_llm_client
from backend.rag.rag_pipeline import PipelineError, get_pipeline
from backend.rag.retriever import RetrieverError
from backend.rag.vectorstore import VectorStoreError, get_vector_store
from backend.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RefreshRequest,
    RefreshResponse,
    ReportRequest,
    ReportResponse,
    RootResponse,
    ServiceStatus,
    StatisticsResponse,
)

settings = get_settings()
settings.configure_logging()
logger = logging.getLogger("backend.app")

#: Populated during startup and reported by /health.
_STARTUP: Dict[str, Any] = {"indexed": False, "error": None, "records": 0}


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the index on startup; never let a data problem block the boot."""
    logger.info("=" * 70)
    logger.info("Employee Burnout RAG API v%s starting", __version__)
    logger.info("Provider: %s / %s", settings.llm_provider.value,
                settings.active_model_name())
    logger.info("Data:     %s", settings.burnout_csv)
    logger.info("=" * 70)

    try:
        dataset = load_burnout_dataset()
        result = get_vector_store().build(dataset=dataset, force=False)
        _STARTUP.update(
            indexed=True, records=result.records_indexed, error=None
        )
        logger.info(
            "Vector store ready: %d records (%s)",
            result.records_indexed,
            "rebuilt" if result.rebuilt else "already current",
        )
    except (FileNotFoundError, ValueError, VectorStoreError, EmbeddingError) as exc:
        # Stay up and explain via /health — see the module docstring.
        _STARTUP.update(indexed=False, error=str(exc))
        logger.error("Startup indexing failed. API is up but /query will 503.")
        logger.error("%s", exc)

    yield
    logger.info("Employee Burnout RAG API shutting down.")


app = FastAPI(
    title="Employee Burnout RAG API",
    description=(
        "Explainable Employee Burnout Detection using Machine Learning, "
        "Sentiment Analysis and Retrieval-Augmented Generation.\n\n"
        "Consumes `outputs/burnout_scores.csv` produced by the ML pipeline and "
        "serves retrieval-augmented explanations over it. The ML pipeline is "
        "never re-run by this service."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Error handling
# ==============================================================================
def _error(status_code: int, error: str, detail: str, hint: str | None = None):
    """Build a uniform `ErrorResponse` body.

    One shape for every failure means Person 4's `api_client.py` has a single
    error path instead of guessing between FastAPI defaults and custom bodies.
    """
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error, detail=detail, hint=hint
        ).model_dump(mode="json"),
    )


@app.exception_handler(FileNotFoundError)
async def _handle_missing_data(request: Request, exc: FileNotFoundError):
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "DataUnavailable",
        str(exc),
        "Run the ML pipeline: python models/burnout/burnout_score.py",
    )


@app.exception_handler(VectorStoreError)
async def _handle_vector_store(request: Request, exc: VectorStoreError):
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "VectorStoreError",
        str(exc),
        "Rebuild the index with POST /refresh",
    )


@app.exception_handler(EmbeddingError)
async def _handle_embedding(request: Request, exc: EmbeddingError):
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "EmbeddingError",
        str(exc),
        "pip install -r backend/requirements.txt",
    )


@app.exception_handler(RetrieverError)
async def _handle_retriever(request: Request, exc: RetrieverError):
    return _error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "RetrievalUnavailable",
        str(exc),
        "Build the vector store with POST /refresh",
    )


@app.exception_handler(PipelineError)
async def _handle_pipeline(request: Request, exc: PipelineError):
    return _error(
        status.HTTP_502_BAD_GATEWAY,
        "PipelineError",
        str(exc),
        "Check the LLM provider settings in .env, then see GET /health",
    )


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "InternalServerError",
        f"{type(exc).__name__}: {exc}",
        "See the backend logs for the full traceback.",
    )


# ==============================================================================
# GET /
# ==============================================================================
@app.get("/", response_model=RootResponse, tags=["meta"])
async def root() -> RootResponse:
    """Service banner and endpoint index."""
    return RootResponse(
        version=__version__,
        status=ServiceStatus.OK if _STARTUP["indexed"] else ServiceStatus.DEGRADED,
    )


# ==============================================================================
# GET /health
# ==============================================================================
@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(probe_llm: bool = False) -> HealthResponse:
    """Health, configuration and data provenance.

    Args:
        probe_llm: Actually contact the LLM provider. Off by default so a
            monitoring poll never blocks on a cold Ollama model load.
    """
    warnings: list[str] = []
    service_status = ServiceStatus.OK

    data_info: Dict[str, Any] = {}
    try:
        dataset = load_burnout_dataset()
        data_info = dataset.describe()
        if dataset.missing_optional_columns:
            warnings.append(
                "Column(s) absent from the CSV: "
                f"{', '.join(dataset.missing_optional_columns)}. "
                "Behaviour columns are left absent rather than zero-filled, so "
                "no fabricated measurement reaches the model."
            )
    except Exception as exc:
        data_info = {"error": str(exc)}
        warnings.append(f"Burnout data unavailable: {exc}")
        service_status = ServiceStatus.ERROR

    try:
        store_info = get_vector_store().info()
        if store_info.get("record_count", 0) == 0:
            warnings.append(
                "The vector store is empty. Build it with POST /refresh."
            )
            service_status = ServiceStatus.DEGRADED
    except Exception as exc:
        store_info = {"error": str(exc)}
        warnings.append(f"Vector store unavailable: {exc}")
        service_status = ServiceStatus.DEGRADED

    if _STARTUP["error"]:
        warnings.append(f"Startup indexing failed: {_STARTUP['error']}")
        service_status = ServiceStatus.DEGRADED

    llm_reachable = None
    if probe_llm:
        try:
            llm_reachable = get_llm_client().health_check()
            if not llm_reachable:
                warnings.append(
                    f"The {settings.llm_provider.value} provider did not "
                    f"respond. /query will still work but returns degraded, "
                    f"rule-based answers."
                )
                service_status = ServiceStatus.DEGRADED
        except Exception as exc:
            llm_reachable = False
            warnings.append(f"LLM health check failed: {exc}")

    configuration = settings.safe_summary()
    configuration["embeddings"] = get_embedding_model().info()

    return HealthResponse(
        status=service_status,
        version=__version__,
        configuration=configuration,
        data=data_info,
        vector_store=store_info,
        llm_reachable=llm_reachable,
        warnings=warnings,
    )


# ==============================================================================
# GET /statistics
# ==============================================================================
@app.get("/statistics", response_model=StatisticsResponse, tags=["data"])
async def statistics() -> StatisticsResponse:
    """Aggregate metrics backing every chart on the dashboard."""
    dataset = load_burnout_dataset()
    stats = compute_statistics(dataset.frame)
    return StatisticsResponse(**stats, source=dataset.describe())


# ==============================================================================
# POST /query
# ==============================================================================
@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question through the full RAG pipeline.

    Returns 200 with `degraded=true` when the LLM is unreachable, rather than
    failing: the caller still receives the rule-based findings, clearly
    labelled as such.
    """
    return get_pipeline().query(request)


# ==============================================================================
# POST /report
# ==============================================================================
@app.post("/report", response_model=ReportResponse, tags=["rag"])
async def report(request: ReportRequest) -> ReportResponse:
    """Generate a structured burnout report.

    Returns JSON, not a PDF. Person 4's `pdf_report.py` renders it with
    ReportLab, which keeps document styling out of the API and lets one
    response feed both the on-screen panel and the download.
    """
    if request.scope.value == "employee" and not request.employee_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="scope='employee' requires an employee_id.",
        )
    return get_pipeline().report(request)


# ==============================================================================
# POST /refresh
# ==============================================================================
@app.post("/refresh", response_model=RefreshResponse, tags=["data"])
async def refresh(request: RefreshRequest | None = None) -> RefreshResponse:
    """Rebuild the vector store from Person 2's CSV.

    Backs the dashboard's "Refresh Database" button. Call this after re-running
    the ML pipeline. The dataframe cache is cleared first so the rebuild reads
    the file from disk rather than a stale in-memory copy.
    """
    started = time.perf_counter()
    request = request or RefreshRequest()

    clear_cache()
    dataset = load_burnout_dataset(force_reload=True)
    result = get_vector_store().build(dataset=dataset, force=request.force)

    _STARTUP.update(indexed=True, records=result.records_indexed, error=None)

    return RefreshResponse(
        status=ServiceStatus.OK,
        message=result.reason,
        records_indexed=result.records_indexed,
        rebuilt=result.rebuilt,
        collection=result.collection,
        persist_dir=result.persist_dir,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        source=dataset.describe(),
    )


# ==============================================================================
# python backend/app.py  (equivalent to the uvicorn command)
# ==============================================================================
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
