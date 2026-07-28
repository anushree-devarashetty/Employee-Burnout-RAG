"""
frontend/utils/api_client.py — HTTP client for Person 3's backend (Person 4)
==============================================================================

Every call the dashboard makes to the FastAPI service goes through this module.

DESIGN

1.  NEVER RAISES INTO THE UI
        Streamlit reruns the whole script on each interaction, so an unhandled
        exception replaces the entire dashboard with a traceback — charts,
        tables and all. Every method here returns an `APIResult` carrying
        either data or a readable error, so a backend problem degrades one
        panel instead of blanking the page.

2.  ERRORS ARE TRANSLATED, NOT FORWARDED
        "ConnectionError: [Errno 61]" tells an HR user nothing. Each failure is
        mapped to what actually went wrong and what to do about it — usually
        "the backend is not running, start it with this command".

3.  TIMEOUTS ARE PER-ENDPOINT
        /health must answer in seconds. /query runs a full RAG pipeline against
        a local model on CPU and can legitimately take a minute. A single
        global timeout is either too short for /query or too long for /health.
==============================================================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import requests

logger = logging.getLogger(__name__)

#: Fast endpoints: metadata only, no model inference.
_FAST_TIMEOUT = 15


@dataclass
class APIResult:
    """Outcome of one API call. Truthy when successful."""

    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    hint: Optional[str] = None
    status_code: Optional[int] = None

    def __bool__(self) -> bool:
        return self.ok

    def get(self, key: str, default: Any = None) -> Any:
        """Read a key from the payload, or ``default`` if the call failed."""
        return (self.data or {}).get(key, default)


class BurnoutAPIClient:
    """Client for the Employee Burnout RAG API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("API_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = int(timeout or os.getenv("API_TIMEOUT", "180"))
        self._session = requests.Session()

    # --------------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> APIResult:
        """Perform one HTTP call, converting every failure into an `APIResult`."""
        url = f"{self.base_url}{path}"
        effective_timeout = timeout or self.timeout

        try:
            response = self._session.request(
                method, url, json=payload, params=params,
                timeout=effective_timeout,
            )
        except requests.exceptions.ConnectionError:
            return APIResult(
                ok=False,
                error=f"Cannot reach the backend at {self.base_url}.",
                hint=(
                    "Start it from the repository root:\n"
                    "    uvicorn backend.app:app --reload --port 8000"
                ),
            )
        except requests.exceptions.Timeout:
            return APIResult(
                ok=False,
                error=(
                    f"The backend did not respond within {effective_timeout}s."
                ),
                hint=(
                    "A local Ollama model on CPU can be slow on its first call. "
                    "Raise API_TIMEOUT in .env, or switch to a smaller model."
                ),
            )
        except requests.exceptions.RequestException as exc:
            return APIResult(ok=False, error=f"Request failed: {exc}")

        if response.status_code >= 400:
            return self._translate_error(response)

        try:
            return APIResult(
                ok=True, data=response.json(), status_code=response.status_code
            )
        except ValueError:
            return APIResult(
                ok=False,
                error="The backend returned a response that was not JSON.",
                status_code=response.status_code,
            )

    @staticmethod
    def _translate_error(response: requests.Response) -> APIResult:
        """Turn an error response into readable text.

        Handles the backend's uniform `ErrorResponse` shape and FastAPI's own
        422 validation format, which nests field errors in a list.
        """
        try:
            body = response.json()
        except ValueError:
            return APIResult(
                ok=False,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )

        detail = body.get("detail")
        if isinstance(detail, list):        # FastAPI 422
            messages = [
                f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: "
                f"{item.get('msg', '')}"
                for item in detail
            ]
            return APIResult(
                ok=False,
                error="Invalid request: " + "; ".join(messages),
                status_code=response.status_code,
            )

        return APIResult(
            ok=False,
            error=body.get("detail") or body.get("error") or response.text[:200],
            hint=body.get("hint"),
            status_code=response.status_code,
        )

    # --------------------------------------------------------------------------
    # Endpoints
    # --------------------------------------------------------------------------
    def root(self) -> APIResult:
        """``GET /`` — service banner."""
        return self._request("GET", "/", timeout=_FAST_TIMEOUT)

    def health(self, probe_llm: bool = False) -> APIResult:
        """``GET /health`` — health, configuration and provenance."""
        return self._request(
            "GET", "/health",
            params={"probe_llm": str(probe_llm).lower()},
            # Probing the LLM may trigger a cold model load, so allow longer.
            timeout=45 if probe_llm else _FAST_TIMEOUT,
        )

    def statistics(self) -> APIResult:
        """``GET /statistics`` — aggregate metrics for the charts."""
        return self._request("GET", "/statistics", timeout=_FAST_TIMEOUT)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        risk_levels: Optional[List[str]] = None,
        employee_id: Optional[str] = None,
        include_sources: bool = True,
    ) -> APIResult:
        """``POST /query`` — ask a question through the RAG pipeline."""
        payload: Dict[str, Any] = {
            "question": question,
            "include_sources": include_sources,
        }
        if top_k:
            payload["top_k"] = int(top_k)
        if risk_levels:
            payload["risk_levels"] = list(risk_levels)
        if employee_id:
            payload["employee_id"] = employee_id
        return self._request("POST", "/query", payload=payload)

    def report(
        self,
        scope: str = "all",
        title: str = "Employee Burnout Analysis Report",
        employee_id: Optional[str] = None,
        focus: Optional[str] = None,
        max_records: int = 25,
        include_sources: bool = True,
    ) -> APIResult:
        """``POST /report`` — generate a structured report."""
        payload: Dict[str, Any] = {
            "scope": scope,
            "title": title,
            "max_records": int(max_records),
            "include_sources": include_sources,
        }
        if employee_id:
            payload["employee_id"] = employee_id
        if focus:
            payload["focus"] = focus
        return self._request("POST", "/report", payload=payload)

    def refresh(self, force: bool = True) -> APIResult:
        """``POST /refresh`` — rebuild the vector store."""
        return self._request(
            "POST", "/refresh", payload={"force": force},
            # Re-embedding every record can take a while on a large dataset.
            timeout=max(self.timeout, 300),
        )

    # --------------------------------------------------------------------------
    def is_online(self) -> bool:
        """Quick reachability check for the sidebar status indicator."""
        return bool(self._request("GET", "/", timeout=5))


_client: Optional[BurnoutAPIClient] = None


def get_client(
    base_url: Optional[str] = None, timeout: Optional[int] = None
) -> BurnoutAPIClient:
    """Return a shared client, reusing its HTTP connection pool.

    A module-level singleton rather than `st.cache_resource`, so this module
    stays importable and testable without a Streamlit runtime.
    """
    global _client
    if _client is None or base_url is not None:
        _client = BurnoutAPIClient(base_url=base_url, timeout=timeout)
    return _client


__all__ = ["BurnoutAPIClient", "APIResult", "get_client"]
