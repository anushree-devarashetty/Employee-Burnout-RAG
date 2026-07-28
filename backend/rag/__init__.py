"""
backend.rag — Retrieval-Augmented Generation components (Person 3)
==============================================================================

PIPELINE (as specified)

    User Question
         |
         v
    embeddings.py    all-MiniLM-L6-v2 encodes the question -> 384-dim vector
         |
         v
    vectorstore.py   ChromaDB persistent collection of employee records
         |
         v
    retriever.py     Top-K nearest employee records + metadata filters
         |
         v
    ../prompts/      Structured prompt assembled from the retrieved context
         |
         v
    llm_client.py    Provider-agnostic call (OpenAI | Gemini | Ollama)
         |
         v
    rag_pipeline.py  Parses and validates the response
         |
         v
    Structured JSON

MODULE BOUNDARIES
    Each module owns exactly one stage and has no knowledge of the stage after
    it. `embeddings.py` does not know ChromaDB exists; `vectorstore.py` does not
    know which LLM is configured; `llm_client.py` does not know what burnout is.
    `rag_pipeline.py` is the only module that wires them together, which is what
    makes each stage independently testable and independently replaceable.

IMPORT ORDER
    Importing this package first triggers ``backend/__init__.py``, which places
    the repository root on sys.path. That is what allows `data_loader.py` to
    reuse Person 2's `models.utils.loader.load_data`.
==============================================================================
"""

from __future__ import annotations

# Importing the parent package guarantees the sys.path bootstrap has run before
# any submodule of backend.rag attempts `from models.utils.loader import ...`.
# This is an explicit dependency rather than an accident of import ordering.
from backend import REPO_ROOT  # noqa: F401  (re-exported for convenience)

__all__ = ["REPO_ROOT"]
