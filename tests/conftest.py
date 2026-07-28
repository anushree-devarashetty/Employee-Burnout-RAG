"""
tests/conftest.py — shared fixtures
==============================================================================

WHY THE EMBEDDING MODEL IS STUBBED

    The real `all-MiniLM-L6-v2` is an ~80 MB download from HuggingFace. Letting
    the suite fetch it would make a first CI run slow, make every run require
    network access, and make the tests fail for reasons unrelated to this code.

    `stub_embeddings` installs a deterministic bag-of-words encoder in its
    place. It produces genuine 384-dimension unit vectors with sensible
    similarity behaviour, so ChromaDB, the retriever and the pipeline are all
    exercised for real — only the transformer weights are substituted.

    Semantic *quality* is a property of the model, not of this code, so it is
    not what these tests are for. Everything around it is.

RUN
    pytest -q                      all tests
    pytest -q -m "not slow"        skip the ChromaDB-backed ones
    pytest -q tests/test_schemas.py
==============================================================================
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==============================================================================
# Embedding stub
# ==============================================================================
class _StubSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer.

    Hashes each word to one of 384 slots and counts occurrences, then
    L2-normalises. Shared vocabulary produces higher cosine similarity, which
    is enough for retrieval ordering to be meaningful without any model weights.
    """

    _vocab: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vector = np.zeros(384)
            for word in str(text).lower().replace(".", " ").replace(",", " ").split():
                index = self._vocab.setdefault(word, len(self._vocab) % 384)
                vector[index] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.array(vectors)


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch):
    """Replace sentence-transformers for every test. Applied automatically."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    # Drop cached singletons so each test builds against the stub.
    from backend.rag import embeddings, llm_client, rag_pipeline, retriever, vectorstore

    embeddings.reset_embedding_model()
    vectorstore.reset_vector_store()
    retriever.reset_retriever()
    llm_client.reset_llm_client()
    rag_pipeline.reset_pipeline()
    yield
    embeddings.reset_embedding_model()
    vectorstore.reset_vector_store()
    retriever.reset_retriever()
    llm_client.reset_llm_client()
    rag_pipeline.reset_pipeline()


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Point the vector store at a throwaway directory.

    Without this, tests would share (and corrupt) the developer's real
    `backend/chroma_db`.
    """
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.rag import retriever, vectorstore

    vectorstore.reset_vector_store()
    retriever.reset_retriever()
    yield tmp_path / "chroma"
    get_settings.cache_clear()


@pytest.fixture
def burnout_frame():
    """Person 2's real committed data, enriched by the loader."""
    from backend.data_loader import load_burnout_dataframe

    return load_burnout_dataframe(force_reload=True)


@pytest.fixture
def statistics(burnout_frame):
    from backend.data_loader import compute_statistics

    return compute_statistics(burnout_frame)


@pytest.fixture
def sample_analysis() -> dict:
    """A valid eight-key analysis payload."""
    return {
        "burnout_summary": "One employee shows moderate strain.",
        "emotional_analysis": "Sadness dominant; joy absent.",
        "possible_causes": ["Sustained overtime"],
        "risk_level": "Medium",
        "suggested_interventions": ["Workload review 1:1"],
        "hr_recommendations": ["Audit overtime patterns"],
        "mental_wellness_suggestions": ["Share EAP details"],
        "confidence_note": "Based on 10 records. Indicative only.",
    }
