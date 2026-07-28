"""
tests/test_api.py — endpoint tests against the real FastAPI app
==============================================================================
Uses `TestClient`, so the routes, lifespan, validators and exception handlers
are all genuinely exercised. Marked `slow` because startup builds a real
ChromaDB index.
==============================================================================
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def client(isolated_chroma):
    """A `TestClient` with the lifespan run, against a throwaway index."""
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------------------
# GET /
# ------------------------------------------------------------------------------
def test_root_lists_every_endpoint(client):
    body = client.get("/").json()
    assert body["service"] == "Employee Burnout RAG API"
    for path in ("GET /", "GET /health", "POST /query", "POST /report",
                 "POST /refresh"):
        assert path in body["endpoints"]


# ------------------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------------------
def test_health_reports_data_and_index_state(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert body["data"]["record_count"] == 10
    assert body["vector_store"]["record_count"] == 10
    assert body["vector_store"]["distance_metric"] == "cosine"


def test_health_never_leaks_a_credential(client, monkeypatch):
    """/health is routinely screenshotted for demos and reports."""
    body = client.get("/health").json()
    credential = body["configuration"]["credential"]
    assert credential.startswith("****") or credential in (
        "not set", "not required (local)"
    )


def test_health_does_not_probe_the_llm_by_default(client):
    """A monitoring poll must not block on a cold model load."""
    assert client.get("/health").json()["llm_reachable"] is None


# ------------------------------------------------------------------------------
# GET /statistics
# ------------------------------------------------------------------------------
def test_statistics_matches_the_loader(client, statistics):
    body = client.get("/statistics").json()
    assert body["total_records"] == statistics["total_records"]
    assert body["overall_burnout_percent"] == statistics["overall_burnout_percent"]
    assert body["risk_distribution"] == statistics["risk_distribution"]


# ------------------------------------------------------------------------------
# POST /query
# ------------------------------------------------------------------------------
def test_query_returns_200_and_degrades_when_the_llm_is_down(client):
    """The most important behaviour in the service.

    A stopped Ollama server must not produce a 500 for an HR user. It must
    produce a usable answer, clearly labelled as not model-generated.
    """
    response = client.post("/query", json={"question": "Who is at risk?", "top_k": 3})
    assert response.status_code == 200

    body = response.json()
    assert body["degraded"] is True
    assert body["retrieved_count"] > 0
    assert body["answer"]["confidence_note"].startswith("DEGRADED RESPONSE")


def test_query_always_returns_all_eight_keys(client):
    answer = client.post("/query", json={"question": "Who is at risk?"}).json()["answer"]
    assert set(answer) == {
        "burnout_summary", "emotional_analysis", "possible_causes", "risk_level",
        "suggested_interventions", "hr_recommendations",
        "mental_wellness_suggestions", "confidence_note",
    }


def test_query_returns_the_evidence_it_used(client):
    body = client.post(
        "/query", json={"question": "Who is at risk?", "include_sources": True}
    ).json()
    assert body["sources"]
    for source in body["sources"]:
        assert source["employee_id"].startswith("EMP_")
        assert 0.0 <= source["similarity"] <= 1.0


def test_query_can_omit_sources(client):
    body = client.post(
        "/query", json={"question": "Who is at risk?", "include_sources": False}
    ).json()
    assert body["sources"] == []


def test_query_honours_an_employee_id_in_the_question(client):
    body = client.post(
        "/query", json={"question": "What is going on with EMP_0006?", "top_k": 3}
    ).json()
    assert body["sources"][0]["employee_id"] == "EMP_0006"
    assert any("Detected employee" in note for note in body["notes"])


def test_query_says_so_when_a_risk_filter_matches_nothing(client):
    """Answering from an empty context is the worst outcome for explainability."""
    body = client.post(
        "/query", json={"question": "Who is at risk?", "risk_levels": ["High"]}
    ).json()
    assert body["retrieved_count"] > 0
    assert any("NOT filtered as requested" in note for note in body["notes"])


@pytest.mark.parametrize("payload", [{}, {"question": "   "}, {"question": "hi"},
                                     {"question": "valid question", "top_k": 0}])
def test_query_rejects_invalid_requests(client, payload):
    assert client.post("/query", json=payload).status_code == 422


# ------------------------------------------------------------------------------
# POST /report
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("scope", ["all", "at_risk", "high_only"])
def test_report_scopes(client, scope):
    body = client.post("/report", json={"scope": scope, "max_records": 10}).json()
    assert body["record_count"] > 0
    assert set(body["analysis"]) >= {"burnout_summary", "confidence_note"}


def test_report_employee_scope_requires_an_id(client):
    assert client.post("/report", json={"scope": "employee"}).status_code == 422


def test_report_includes_statistics_for_the_pdf(client):
    body = client.post("/report", json={"scope": "all"}).json()
    assert body["statistics"]["total_records"] == 10


# ------------------------------------------------------------------------------
# POST /refresh
# ------------------------------------------------------------------------------
def test_refresh_rebuilds_when_forced(client):
    body = client.post("/refresh", json={"force": True}).json()
    assert body["rebuilt"] is True
    assert body["records_indexed"] == 10


def test_refresh_skips_when_the_index_is_current(client):
    client.post("/refresh", json={"force": True})
    body = client.post("/refresh", json={"force": False}).json()
    assert body["rebuilt"] is False
    assert "already matches" in body["message"]


# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------
def test_unknown_path_returns_404(client):
    assert client.get("/nope").status_code == 404


def test_openapi_documents_all_six_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/", "/health", "/statistics", "/query", "/report",
                          "/refresh"}
