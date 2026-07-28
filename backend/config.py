"""
backend/config.py — Typed, validated configuration for Person 3's RAG backend
==============================================================================

DESIGN GOALS

1.  ONE SOURCE OF TRUTH
    Every tunable value in the backend comes from `<repo_root>/.env`. No module
    calls `os.getenv` directly. If a setting is not defined here, it does not
    exist.

2.  PROVIDER SWITCHING WITHOUT CODE CHANGES
    Setting `LLM_PROVIDER=openai|gemini|ollama` in .env is the only action
    required to change LLM backend. This module validates that the credentials
    for the *selected* provider are present and refuses to start otherwise.

3.  FAIL LOUDLY AT STARTUP, NOT HALFWAY THROUGH A REQUEST
    A missing API key must surface the moment the server boots with a message
    naming the exact variable to fix — never as a 500 error twenty minutes
    later when an HR user clicks "Ask Question". Pydantic validators here run
    at import time for that reason.

4.  PATHS THAT WORK FROM ANY WORKING DIRECTORY
    `.env` stores repo-relative paths (`outputs/burnout_scores.csv`) because
    that is what a human wants to read and edit. This module resolves them
    against REPO_ROOT into absolute paths, so uvicorn, Streamlit and pytest all
    agree on where files are regardless of where they were launched.

WHY pydantic-settings
    Already pinned in the project's root requirements.txt (pydantic-settings
    2.14.2, pydantic 2.12.5), so this adds no new dependency to the team's
    environment. It gives type coercion, range validation and a single
    declarative place for defaults.
==============================================================================
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# ------------------------------------------------------------------------------
# Script-mode self-bootstrap  (must run before `from backend import ...`)
# ------------------------------------------------------------------------------
# When imported normally (`from backend.config import settings`), Python has
# already executed backend/__init__.py, so the repo root is on sys.path.
#
# But when this file is run DIRECTLY for inspection:
#
#     python backend/config.py
#
# Python puts `backend/` on sys.path — not the repository root — so
# `import backend` fails with ModuleNotFoundError. The bootstrap we need lives
# inside the very package we cannot yet import.
#
# `__package__` is None or "" only in that script case, so this block costs
# nothing during normal imports and makes every module in this package
# directly runnable on Windows, macOS and Linux.
if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Importing the package (not a submodule) runs the sys.path bootstrap in
# backend/__init__.py and hands us the resolved repository root.
from backend import BACKEND_DIR, REPO_ROOT

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================
class LLMProvider(str, Enum):
    """Supported LLM providers.

    Inherits from ``str`` so the members compare equal to plain strings and
    serialise cleanly into JSON API responses without extra encoders.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


class EmbeddingDevice(str, Enum):
    """Torch device used by sentence-transformers."""

    CPU = "cpu"
    CUDA = "cuda"      # NVIDIA GPU
    MPS = "mps"        # Apple Silicon GPU


# Placeholder values shipped in .env.example. Treated as "not configured" so a
# user who copies the template but forgets to paste a real key gets a clear
# error instead of a confusing 401 from the provider.
_PLACEHOLDER_SECRETS = {
    "",
    "replace-me",
    "sk-replace-me",
    "your-api-key",
    "your-api-key-here",
    "changeme",
    "none",
}


# ==============================================================================
# Settings
# ==============================================================================
class Settings(BaseSettings):
    """All backend configuration, loaded and validated from ``.env``."""

    model_config = SettingsConfigDict(
        # Absolute path so the file is found no matter the working directory.
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env also carries Person 4's frontend keys (API_BASE_URL, etc.).
        # "ignore" lets one shared .env serve both halves of the project
        # instead of forcing two files that inevitably drift apart.
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # 1. LLM provider selection
    # --------------------------------------------------------------------------
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Which LLM backend to use. The only switch needed.",
    )

    # --------------------------------------------------------------------------
    # 2. Ollama
    # --------------------------------------------------------------------------
    ollama_model: str = Field(default="llama3.1")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --------------------------------------------------------------------------
    # 3. OpenAI (and any OpenAI-compatible server)
    # --------------------------------------------------------------------------
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_base_url: str = Field(
        default="",
        description="Blank for real OpenAI. Set for Groq / Together / vLLM.",
    )

    # --------------------------------------------------------------------------
    # 4. Gemini
    # --------------------------------------------------------------------------
    google_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.0-flash")

    # --------------------------------------------------------------------------
    # 5. Shared generation settings
    # --------------------------------------------------------------------------
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1600, ge=64, le=32768)
    llm_timeout: int = Field(default=120, ge=5, le=1800)

    # --------------------------------------------------------------------------
    # 6. Data source — Person 2's output, consumed read-only
    # --------------------------------------------------------------------------
    burnout_csv_path: str = Field(default="outputs/burnout_scores.csv")
    employee_id_prefix: str = Field(default="EMP")
    burnout_score_max: int = Field(
        default=12,
        ge=1,
        description=(
            "Ceiling of Person 2's integer point scale, used ONLY to render a "
            "0-100 display percentage. Never used to classify risk — risk_level "
            "is always taken verbatim from Person 2's column."
        ),
    )

    # --------------------------------------------------------------------------
    # 7. Embeddings
    # --------------------------------------------------------------------------
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: EmbeddingDevice = Field(default=EmbeddingDevice.CPU)
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)

    # --------------------------------------------------------------------------
    # 8. ChromaDB
    # --------------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="backend/chroma_db")
    chroma_collection: str = Field(default="burnout_records")

    # --------------------------------------------------------------------------
    # 9. Retrieval
    # --------------------------------------------------------------------------
    retriever_top_k: int = Field(default=5, ge=1)
    retriever_max_k: int = Field(default=25, ge=1, le=200)

    # --------------------------------------------------------------------------
    # 10. FastAPI server
    # --------------------------------------------------------------------------
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000, ge=1, le=65535)
    # Kept as a plain string, NOT a List[str]. pydantic-settings tries to
    # JSON-decode env values for complex types, so `CORS_ORIGINS=*` would raise
    # a JSONDecodeError. Parsed on demand by `cors_origins_list` below.
    cors_origins: str = Field(default="*")
    log_level: str = Field(default="INFO")

    # --------------------------------------------------------------------------
    # 11. Frontend (read here so /health can report the expected pairing)
    # --------------------------------------------------------------------------
    api_base_url: str = Field(default="http://localhost:8000")
    api_timeout: int = Field(default=180, ge=5, le=1800)
    frontend_port: int = Field(default=8501, ge=1, le=65535)

    # ==========================================================================
    # Field validators
    # ==========================================================================
    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: Any) -> Any:
        """Accept ``OpenAI``, ``ollama``, `` GEMINI `` and similar spellings.

        People edit .env by hand; casing and stray whitespace should not be a
        failure mode.
        """
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("embedding_device", mode="before")
    @classmethod
    def _normalise_device(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, v: Any) -> Any:
        """Uppercase and reject anything the logging module cannot use."""
        if not isinstance(v, str):
            return v
        level = v.strip().upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in valid:
            raise ValueError(
                f"LOG_LEVEL={v!r} is not valid. "
                f"Use one of: {', '.join(sorted(valid))}"
            )
        return level

    @field_validator(
        "ollama_base_url",
        "openai_base_url",
        "api_base_url",
        mode="before",
    )
    @classmethod
    def _strip_trailing_slash(cls, v: Any) -> Any:
        """Normalise URLs so joining paths never produces a double slash."""
        if isinstance(v, str):
            return v.strip().rstrip("/")
        return v

    @field_validator("employee_id_prefix")
    @classmethod
    def _validate_prefix(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("EMPLOYEE_ID_PREFIX must not be empty.")
        return v

    @field_validator("retriever_max_k")
    @classmethod
    def _max_k_not_below_top_k(cls, v: int, info: ValidationInfo) -> int:
        """``RETRIEVER_MAX_K`` must not be lower than ``RETRIEVER_TOP_K``.

        Otherwise the default retrieval depth would itself exceed the ceiling
        meant to protect the LLM context window — a silent contradiction.
        """
        top_k = info.data.get("retriever_top_k")
        if top_k is not None and v < top_k:
            raise ValueError(
                f"RETRIEVER_MAX_K ({v}) must be >= RETRIEVER_TOP_K ({top_k})."
            )
        return v

    # ==========================================================================
    # Cross-field validation
    # ==========================================================================
    @model_validator(mode="after")
    def _validate_selected_provider_credentials(self) -> "Settings":
        """Verify credentials exist for the *selected* provider only.

        Deliberately does not require keys for providers that are not in use —
        a teammate running local Ollama should never be asked for an OpenAI key.
        """
        provider = self.llm_provider

        if provider is LLMProvider.OPENAI:
            key_missing = self.openai_api_key.strip().lower() in _PLACEHOLDER_SECRETS
            # An OpenAI-compatible local server (vLLM, LM Studio, Ollama's
            # OpenAI shim) needs no real key, so only demand one when talking
            # to a remote endpoint.
            if key_missing and not self.openai_base_url:
                raise ValueError(
                    "LLM_PROVIDER=openai but OPENAI_API_KEY is missing or is "
                    "still the placeholder from .env.example.\n"
                    "  Fix: put a real key in .env  ->  OPENAI_API_KEY=sk-...\n"
                    "  Get one at https://platform.openai.com/api-keys\n"
                    "  Or, to use a local OpenAI-compatible server instead, set "
                    "OPENAI_BASE_URL (e.g. http://localhost:8000/v1).\n"
                    "  Or switch provider entirely:  LLM_PROVIDER=ollama"
                )

        elif provider is LLMProvider.GEMINI:
            if self.google_api_key.strip().lower() in _PLACEHOLDER_SECRETS:
                raise ValueError(
                    "LLM_PROVIDER=gemini but GOOGLE_API_KEY is missing or is "
                    "still the placeholder from .env.example.\n"
                    "  Fix: put a real key in .env  ->  GOOGLE_API_KEY=...\n"
                    "  Get one at https://aistudio.google.com/app/apikey\n"
                    "  Or switch provider entirely:  LLM_PROVIDER=ollama"
                )

        elif provider is LLMProvider.OLLAMA:
            # No credential required. Reachability is checked lazily by
            # llm_client.py, because Ollama may legitimately still be starting
            # when the API server boots.
            if not self.ollama_base_url:
                raise ValueError(
                    "LLM_PROVIDER=ollama but OLLAMA_BASE_URL is empty.\n"
                    "  Fix: OLLAMA_BASE_URL=http://localhost:11434"
                )

        return self

    # ==========================================================================
    # Derived, absolute paths
    # ==========================================================================
    @property
    def repo_root(self) -> Path:
        """Absolute path to the repository root."""
        return REPO_ROOT

    @property
    def backend_dir(self) -> Path:
        """Absolute path to the ``backend/`` package directory."""
        return BACKEND_DIR

    def _resolve(self, raw: str) -> Path:
        """Turn a possibly-relative .env path into an absolute one.

        Relative values are resolved against REPO_ROOT rather than the current
        working directory, which is what makes the same .env work for uvicorn
        (launched at repo root), Streamlit (launched at repo root but executing
        ``frontend/app.py``) and pytest alike.
        """
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def burnout_csv(self) -> Path:
        """Absolute path to Person 2's ``outputs/burnout_scores.csv``."""
        return self._resolve(self.burnout_csv_path)

    @property
    def chroma_dir(self) -> Path:
        """Absolute path to the persistent ChromaDB directory."""
        return self._resolve(self.chroma_persist_dir)

    @property
    def cors_origins_list(self) -> List[str]:
        """``CORS_ORIGINS`` split into a list for FastAPI's middleware."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    # ==========================================================================
    # Runtime checks and reporting
    # ==========================================================================
    def ensure_data_available(self) -> Path:
        """Confirm Person 2's CSV exists, with an actionable error if not.

        Called at application startup and again before rebuilding the vector
        store. Kept out of the validators on purpose: configuration can be
        perfectly valid while the pipeline simply has not been run yet, and
        those are different failures deserving different messages.

        Returns:
            Absolute path to the CSV.

        Raises:
            FileNotFoundError: with instructions for regenerating the file.
        """
        csv_path = self.burnout_csv
        if csv_path.is_file():
            return csv_path

        raise FileNotFoundError(
            f"Person 2's burnout output was not found at:\n"
            f"    {csv_path}\n\n"
            f"The RAG backend consumes this file and never regenerates it.\n"
            f"To produce it, run Person 2's pipeline from the repository root:\n"
            f"    python models/sentiment/roberta_sentiment.py\n"
            f"    python models/emotion/emotion_analysis.py\n"
            f"    python models/features/behavior_features.py\n"
            f"    python models/fusion/fusion.py\n"
            f"    python models/burnout/burnout_score.py\n\n"
            f"Or point BURNOUT_CSV_PATH in .env at an existing file."
        )

    def active_model_name(self) -> str:
        """Model identifier for the currently selected provider."""
        return {
            LLMProvider.OLLAMA: self.ollama_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.GEMINI: self.gemini_model,
        }[self.llm_provider]

    @staticmethod
    def _mask(secret: str) -> str:
        """Render a credential safe for logs and HTTP responses.

        Shows only the last four characters. Never returns the full value —
        /health is frequently screenshotted for reports and demos.
        """
        secret = (secret or "").strip()
        if secret.lower() in _PLACEHOLDER_SECRETS:
            return "not set"
        if len(secret) <= 4:
            return "****"
        return f"****{secret[-4:]}"

    def safe_summary(self) -> Dict[str, Any]:
        """Redacted configuration snapshot for ``GET /health`` and logs."""
        return {
            "llm_provider": self.llm_provider.value,
            "llm_model": self.active_model_name(),
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "llm_timeout_seconds": self.llm_timeout,
            "credential": {
                LLMProvider.OPENAI: self._mask(self.openai_api_key),
                LLMProvider.GEMINI: self._mask(self.google_api_key),
                LLMProvider.OLLAMA: "not required (local)",
            }[self.llm_provider],
            "openai_base_url": self.openai_base_url or "default (api.openai.com)",
            "ollama_base_url": self.ollama_base_url,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device.value,
            "vector_store": {
                "backend": "chromadb",
                "collection": self.chroma_collection,
                "persist_dir": str(self.chroma_dir),
                "exists": self.chroma_dir.is_dir(),
            },
            "retriever_top_k": self.retriever_top_k,
            "retriever_max_k": self.retriever_max_k,
            "data_source": {
                "path": str(self.burnout_csv),
                "exists": self.burnout_csv.is_file(),
                "owner": "Person 2 (read-only)",
            },
            "burnout_score_max": self.burnout_score_max,
            "employee_id_prefix": self.employee_id_prefix,
        }

    def configure_logging(self) -> None:
        """Apply ``LOG_LEVEL`` to the root logger.

        ``force=True`` overrides the handler uvicorn installs first, so the
        configured level actually takes effect instead of being silently
        ignored — a common and time-wasting surprise.
        """
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )


# ==============================================================================
# Singleton accessor
# ==============================================================================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    Cached because reading and validating .env on every request would be
    wasteful, and because a single instance guarantees every module sees
    identical configuration.

    Raises:
        pydantic.ValidationError: if .env is invalid. Allowed to propagate —
            the server must not start with broken configuration.
    """
    settings = Settings()
    settings.configure_logging()
    logger.debug("Configuration loaded: %s", settings.safe_summary())
    return settings


# Module-level convenience handle. Importing this module validates .env, so a
# configuration error surfaces at import time with a precise message.
settings = get_settings()


__all__ = [
    "Settings",
    "LLMProvider",
    "EmbeddingDevice",
    "get_settings",
    "settings",
]


# ==============================================================================
# Manual inspection:  python backend/config.py
# ==============================================================================
if __name__ == "__main__":
    import json

    print("=" * 78)
    print("Employee Burnout RAG — backend configuration")
    print("=" * 78)
    print(json.dumps(get_settings().safe_summary(), indent=2))
    print("=" * 78)

    try:
        path = get_settings().ensure_data_available()
        print(f"OK  Person 2's data found: {path}")
    except FileNotFoundError as exc:
        print(f"MISSING DATA\n{exc}")
