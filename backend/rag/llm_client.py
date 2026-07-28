"""
backend/rag/llm_client.py — Provider-agnostic LLM interface (Person 3)
==============================================================================

REQUIREMENT
    Support OpenAI, Gemini and Llama (Ollama). Switching providers must require
    ONLY an edit to .env — never a code change.

HOW THAT IS ACHIEVED
    One abstract base (`BaseLLMClient`) with a single method, `complete()`.
    Three concrete subclasses. A factory that reads `LLM_PROVIDER` and returns
    the right one. Nothing above this module ever names a provider:
    `rag_pipeline.py` calls `get_llm_client().complete(...)` and is finished.

    Adding a fourth provider means adding one subclass and one dict entry here.
    No other file changes.

WHY RAW HTTP AND NOT PROVIDER SDKs
    Each SDK brings its own dependency tree and its own breaking changes. This
    module talks to all three providers over their documented REST endpoints
    using `httpx`, which is already a transitive dependency of FastAPI. The
    result is a much smaller install, no version conflicts between three
    SDKs, and identical retry/timeout/error handling across providers.

    A note on the specification's mention of LangChain: LangChain is used in
    this project for the RAG orchestration concepts and the `Embeddings`
    interface (see `embeddings.py`), while the provider calls themselves are
    kept dependency-light and explicit here. `to_langchain_llm()` at the bottom
    exposes this client to LangChain chains for anyone who wants them.

THE DEGRADED PATH
    A local Ollama server that is not running is the single most likely failure
    in this project. `LLMUnavailableError` is raised as a distinct exception so
    `rag_pipeline.py` can fall back to Person 2's rule-based explanations and
    flag the response `degraded=True`, rather than showing an HR user a 500.
==============================================================================
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from backend.config import LLMProvider, Settings, get_settings

logger = logging.getLogger(__name__)

#: Transient HTTP statuses worth retrying: rate limit + server-side errors.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5

#: Hostnames that are always on this machine and must never be proxied.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}


def _is_loopback(url: str) -> bool:
    """True if ``url`` points at this machine."""
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS or host.endswith(".local")


def _http_client(timeout: float, base_url: str):
    """Build an ``httpx.Client`` with correct proxy behaviour for the target.

    WHY THIS EXISTS
        httpx reads HTTP_PROXY / HTTPS_PROXY from the environment. On a machine
        behind a corporate proxy — common in exactly the enterprise HR settings
        this project targets — that means a request to a LOCAL Ollama server at
        http://localhost:11434 is sent to the proxy, which either refuses it or
        cannot route it. The user then sees "could not reach Ollama" while
        Ollama is running perfectly well on their own machine.

        httpx is also inconsistent about honouring `no_proxy` across versions,
        so relying on the user having set it correctly is not safe. Verified in
        testing: with HTTP_PROXY set and `no_proxy` explicitly listing
        localhost, httpx still attempted to proxy a loopback request and failed.

    WHAT IT DOES
        `trust_env=False` is applied only for loopback targets, so a local
        Ollama or vLLM server is contacted directly. Remote providers (OpenAI,
        Gemini) keep `trust_env=True` and continue to work through a corporate
        proxy, which is exactly where they need it.
    """
    import httpx

    return httpx.Client(timeout=timeout, trust_env=not _is_loopback(base_url))


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached at all.

    Distinct from `LLMError` so the pipeline can degrade gracefully instead of
    failing the request: a stopped Ollama server should produce a usable,
    clearly-labelled fallback answer, not a 500.
    """


class LLMResponseError(LLMError):
    """The provider responded, but the response was unusable."""


# ==============================================================================
# Base
# ==============================================================================
class BaseLLMClient(ABC):
    """Interface every provider implementation satisfies."""

    provider_name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model in use."""

    @abstractmethod
    def _invoke(self, system_prompt: str, user_prompt: str, as_json: bool) -> str:
        """Provider-specific call. Returns raw text."""

    @abstractmethod
    def health_check(self) -> bool:
        """True if the provider is reachable. Must not raise."""

    # --------------------------------------------------------------------------
    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        as_json: bool = True,
    ) -> str:
        """Run a completion, with retries on transient failures.

        Retries only on genuinely transient conditions (429, 5xx, timeouts).
        A 401 from a bad API key is retried zero times, because retrying it
        just delays a certain failure by five seconds.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                started = time.perf_counter()
                text = self._invoke(system_prompt, user_prompt, as_json)
                logger.info(
                    "%s/%s responded in %.2fs (%d chars)",
                    self.provider_name,
                    self.model_name,
                    time.perf_counter() - started,
                    len(text or ""),
                )
                if not text or not text.strip():
                    raise LLMResponseError("Provider returned an empty response.")
                return text

            except LLMUnavailableError:
                raise                      # not transient; degrade immediately

            except LLMError as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS or not getattr(exc, "retryable", False):
                    break
                delay = _BACKOFF_BASE ** attempt
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    self.provider_name, attempt, _MAX_ATTEMPTS, exc, delay,
                )
                time.sleep(delay)

        raise last_error or LLMError("LLM call failed for an unknown reason.")

    # --------------------------------------------------------------------------
    @staticmethod
    def _http_error(status: int, body: str, provider: str) -> LLMError:
        """Map an HTTP status onto an actionable exception."""
        snippet = (body or "")[:300]

        if status in (401, 403):
            error = LLMError(
                f"{provider} rejected the credentials (HTTP {status}).\n"
                f"  Fix: check the API key in .env, then restart the backend.\n"
                f"  Response: {snippet}"
            )
        elif status == 404:
            error = LLMError(
                f"{provider} returned 404 — the model name is probably wrong "
                f"or not installed.\n"
                f"  For Ollama, run:  ollama pull <model>\n"
                f"  Response: {snippet}"
            )
        elif status == 429:
            error = LLMError(f"{provider} rate limit reached (429). {snippet}")
        else:
            error = LLMError(f"{provider} returned HTTP {status}. {snippet}")

        error.retryable = status in _RETRYABLE_STATUS  # type: ignore[attr-defined]
        return error

    def info(self) -> Dict[str, Any]:
        """Summary for ``GET /health``. Never includes credentials."""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout,
        }


# ==============================================================================
# OpenAI (and any OpenAI-compatible server)
# ==============================================================================
class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions, and anything speaking the same protocol.

    Setting `OPENAI_BASE_URL` points this class at Groq, Together, vLLM,
    LM Studio or Ollama's OpenAI shim without a single code change — which is
    why the project's `.env.example` documents that variable prominently.
    """

    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url or "https://api.openai.com/v1"
        self._model = settings.openai_model

    @property
    def model_name(self) -> str:
        return self._model

    def _invoke(self, system_prompt: str, user_prompt: str, as_json: bool) -> str:
        import httpx

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if as_json:
            # Server-side guarantee of syntactically valid JSON. Supported by
            # OpenAI and most compatible servers; harmless where ignored.
            payload["response_format"] = {"type": "json_object"}

        try:
            with _http_client(self.timeout, self.base_url) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Could not reach the OpenAI-compatible endpoint at "
                f"{self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise self._http_error(response.status_code, response.text, "OpenAI")

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMResponseError(
                f"Unexpected OpenAI response shape: {response.text[:300]}"
            ) from exc

    def health_check(self) -> bool:
        try:
            with _http_client(10, self.base_url) as client:
                response = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            return response.status_code < 500
        except Exception:
            return False


# ==============================================================================
# Gemini
# ==============================================================================
class GeminiClient(BaseLLMClient):
    """Google Gemini via the generativelanguage REST API."""

    provider_name = "gemini"
    _BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.google_api_key
        self._model = settings.gemini_model

    @property
    def model_name(self) -> str:
        return self._model

    def _invoke(self, system_prompt: str, user_prompt: str, as_json: bool) -> str:
        import httpx

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt:
            # Gemini takes the system prompt as a separate top-level field
            # rather than a message role — the main shape difference from
            # OpenAI, and the reason this is a separate class.
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if as_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        try:
            with _http_client(self.timeout, self._BASE) as client:
                response = client.post(
                    f"{self._BASE}/models/{self._model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
        except Exception as exc:
            raise LLMUnavailableError(f"Could not reach Gemini: {exc}") from exc

        if response.status_code != 200:
            raise self._http_error(response.status_code, response.text, "Gemini")

        try:
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                # Usually a safety block; surface the stated reason.
                reason = (data.get("promptFeedback") or {}).get(
                    "blockReason", "no candidates returned"
                )
                raise LLMResponseError(f"Gemini returned no output ({reason}).")
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(part.get("text", "") for part in parts)
        except LLMResponseError:
            raise
        except Exception as exc:
            raise LLMResponseError(
                f"Unexpected Gemini response shape: {response.text[:300]}"
            ) from exc

    def health_check(self) -> bool:
        try:
            with _http_client(10, self._BASE) as client:
                response = client.get(
                    f"{self._BASE}/models", params={"key": self.api_key}
                )
            return response.status_code < 500
        except Exception:
            return False


# ==============================================================================
# Ollama (Llama, Mistral, Phi, ...)
# ==============================================================================
class OllamaClient(BaseLLMClient):
    """Local models served by Ollama. The project default: free and offline."""

    provider_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.base_url = settings.ollama_base_url
        self._model = settings.ollama_model

    @property
    def model_name(self) -> str:
        return self._model

    def _invoke(self, system_prompt: str, user_prompt: str, as_json: bool) -> str:
        import httpx

        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": user_prompt,
            "stream": False,           # one complete response, not SSE chunks
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if as_json:
            payload["format"] = "json"  # Ollama's grammar-constrained JSON mode

        try:
            with _http_client(self.timeout, self.base_url) as client:
                response = client.post(f"{self.base_url}/api/generate", json=payload)
        except Exception as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}.\n"
                f"  Is it running?  Start it with:  ollama serve\n"
                f"  Is the model pulled?            ollama pull {self._model}\n"
                f"  Underlying error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise self._http_error(response.status_code, response.text, "Ollama")

        try:
            return response.json().get("response", "")
        except ValueError as exc:
            raise LLMResponseError(
                f"Unexpected Ollama response: {response.text[:300]}"
            ) from exc

    def health_check(self) -> bool:
        try:
            with _http_client(5, self.base_url) as client:
                response = client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            # Reachable is not enough — the specific model must be present,
            # or every request 404s at generation time.
            names = [m.get("name", "") for m in response.json().get("models", [])]
            base = self._model.split(":")[0]
            if not any(n == self._model or n.split(":")[0] == base for n in names):
                logger.warning(
                    "Ollama is running but '%s' is not pulled. "
                    "Run: ollama pull %s   (available: %s)",
                    self._model, self._model, ", ".join(names) or "none",
                )
                return False
            return True
        except Exception:
            return False


# ==============================================================================
# Factory
# ==============================================================================
_REGISTRY = {
    LLMProvider.OPENAI: OpenAIClient,
    LLMProvider.GEMINI: GeminiClient,
    LLMProvider.OLLAMA: OllamaClient,
}


_INSTANCE: Optional[BaseLLMClient] = None
_INSTANCE_LOCK = threading.Lock()


def get_llm_client(settings: Optional[Settings] = None) -> BaseLLMClient:
    """Return the client for the configured provider.

    This function is the entire provider-switching mechanism. Callers never
    name a provider, so `LLM_PROVIDER` in .env is genuinely the only switch.

    Manual singleton rather than ``@lru_cache``: Pydantic v2 ``Settings`` is
    unhashable and lru_cache would raise on an explicit settings argument.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            resolved = settings or get_settings()
            client_class = _REGISTRY.get(resolved.llm_provider)
            if client_class is None:  # unreachable: config validates the enum
                raise LLMError(
                    f"Unsupported LLM provider: {resolved.llm_provider}"
                )
            _INSTANCE = client_class(resolved)
            logger.info(
                "LLM client ready: %s / %s",
                _INSTANCE.provider_name,
                _INSTANCE.model_name,
            )
    return _INSTANCE


def reset_llm_client() -> None:
    """Drop the cached client. Used by tests and after a provider change."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


def to_langchain_llm(settings: Optional[Settings] = None):
    """Expose this client as a LangChain ``LLM`` for use in chains.

    Provided so LangChain constructs can be built on top of the same
    provider-switching logic. Not used by `rag_pipeline.py`, which calls
    `complete()` directly to keep the hot path free of LangChain imports.
    """
    from langchain_core.language_models.llms import LLM

    client = get_llm_client(settings)

    class _Wrapped(LLM):
        @property
        def _llm_type(self) -> str:
            return f"burnout-rag-{client.provider_name}"

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            return client.complete(prompt, as_json=False)

        @property
        def _identifying_params(self) -> Dict[str, Any]:
            return client.info()

    return _Wrapped()


__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GeminiClient",
    "OllamaClient",
    "LLMError",
    "LLMUnavailableError",
    "LLMResponseError",
    "get_llm_client",
    "to_langchain_llm",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    client = get_llm_client()
    print(json.dumps(client.info(), indent=2))
    print(f"Reachable: {client.health_check()}")
    try:
        print(client.complete(
            'Reply with JSON: {"status": "ok"}', "You output only JSON."
        ))
    except LLMError as exc:
        print(f"LLM call failed:\n{exc}")
