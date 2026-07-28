"""
tests/test_vectorstore.py — ChromaDB version compatibility and index lifecycle
==============================================================================

REGRESSION THIS FILE EXISTS FOR

    `chromadb.list_collections()` returns `Collection` objects on the 1.x line
    and plain strings on 0.6.x. `[c.name for c in ...]` therefore raises
    AttributeError on one of them.

    That alone would be a clear failure. What made it genuinely hard to
    diagnose was `count()` swallowing the error in a bare `except: return 0`.
    A successful build logged:

        Writing 10 records to ChromaDB...
        Vector store built: 0 records

    ...and then every downstream test failed on an index that was actually
    fine. Nine tests, one masked exception.

    These tests pin both halves: the shape handling, and the fact that a
    failure is now reported rather than hidden behind a phantom zero.
==============================================================================
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.rag.vectorstore import BurnoutVectorStore, compute_fingerprint


class _CollectionObject:
    """Stands in for a chromadb 1.x Collection."""

    def __init__(self, name: str) -> None:
        self.name = name


def _store(list_collections_result):
    store = BurnoutVectorStore(Path("/tmp/unused"), "burnout_records", MagicMock())
    store._client = MagicMock()
    store._client.list_collections.return_value = list_collections_result
    return store


# ------------------------------------------------------------------------------
# Version compatibility
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "returned,label",
    [
        ([_CollectionObject("burnout_records")], "chromadb 1.x: Collection objects"),
        (["burnout_records"], "chromadb 0.6.x: plain strings"),
        ([_CollectionObject("burnout_records"), "other"], "mixed"),
    ],
)
def test_collection_names_handles_every_chromadb_return_shape(returned, label):
    assert "burnout_records" in _store(returned)._collection_names(), label


def test_collection_names_on_an_empty_database():
    assert _store([])._collection_names() == []


def test_collection_names_drops_unnamed_entries():
    assert _store([_CollectionObject(""), "real"])._collection_names() == ["real"]


# ------------------------------------------------------------------------------
# The masked-error half of the bug
# ------------------------------------------------------------------------------
def test_count_failure_is_logged_not_silently_zero(caplog):
    """A broken count must explain itself.

    Returning 0 quietly turned a version mismatch into "the index is empty",
    which is a completely different and much harder problem to chase.
    """
    store = _store(["burnout_records"])
    store._get_collection = MagicMock(side_effect=RuntimeError("version mismatch"))

    with caplog.at_level("WARNING"):
        assert store.count() == 0

    assert "Could not read the record count" in caplog.text
    assert "chromadb" in caplog.text.lower()


def test_exists_is_false_when_the_collection_is_absent():
    assert _store(["something_else"]).exists() is False


# ------------------------------------------------------------------------------
# Fingerprint
# ------------------------------------------------------------------------------
def test_fingerprint_is_stable_for_identical_inputs(burnout_frame):
    from backend.data_loader import load_burnout_dataset

    dataset = load_burnout_dataset()
    first = compute_fingerprint(dataset, "all-MiniLM-L6-v2", 384)
    second = compute_fingerprint(dataset, "all-MiniLM-L6-v2", 384)
    assert first == second


def test_fingerprint_changes_when_the_embedding_model_changes():
    """A 768-dim index cannot answer a 384-dim query; catch it at startup."""
    from backend.data_loader import load_burnout_dataset

    dataset = load_burnout_dataset()
    assert compute_fingerprint(dataset, "all-MiniLM-L6-v2", 384) != compute_fingerprint(
        dataset, "all-mpnet-base-v2", 768
    )


def test_fingerprint_changes_when_person2_reruns_the_pipeline():
    import dataclasses

    from backend.data_loader import load_burnout_dataset

    dataset = load_burnout_dataset()
    newer = dataclasses.replace(dataset, source_mtime=dataset.source_mtime + 1)
    assert compute_fingerprint(dataset, "m", 384) != compute_fingerprint(newer, "m", 384)


# ------------------------------------------------------------------------------
# Real index lifecycle
# ------------------------------------------------------------------------------
@pytest.mark.slow
def test_build_indexes_every_record(isolated_chroma):
    from backend.rag.vectorstore import get_vector_store

    store = get_vector_store()
    result = store.build(force=True)

    assert result.rebuilt is True
    assert result.records_indexed == 10
    # The exact assertion that failed on chromadb 0.6.x before the fix.
    assert store.count() == 10


@pytest.mark.slow
def test_warm_start_skips_re_embedding(isolated_chroma):
    from backend.rag.vectorstore import get_vector_store

    store = get_vector_store()
    store.build(force=True)
    second = store.build(force=False)

    assert second.rebuilt is False
    assert "already matches" in second.reason
    assert store.count() == 10


@pytest.mark.slow
def test_fingerprint_survives_a_new_store_instance(isolated_chroma):
    """Simulates a backend restart: the persisted index must be reused."""
    from backend.rag import vectorstore

    first = vectorstore.get_vector_store()
    built = first.build(force=True)

    vectorstore.reset_vector_store()
    second = vectorstore.get_vector_store()

    assert second.stored_fingerprint() == built.fingerprint
    assert second.build(force=False).rebuilt is False


@pytest.mark.slow
def test_duplicate_ids_raise_instead_of_silently_overwriting(isolated_chroma):
    """Chroma upserts on duplicate IDs, which would drop records with no error."""
    import dataclasses

    import pandas as pd

    from backend.data_loader import load_burnout_dataset
    from backend.rag.vectorstore import VectorStoreError, get_vector_store

    dataset = load_burnout_dataset()
    duplicated = dataclasses.replace(
        dataset, frame=pd.concat([dataset.frame, dataset.frame.head(1)])
    )

    with pytest.raises(VectorStoreError, match="Duplicate employee_id"):
        get_vector_store().build(dataset=duplicated, force=True)


@pytest.mark.slow
def test_query_returns_bounded_similarity(isolated_chroma):
    """Cosine space must give a 0-1 score, not an unbounded L2 distance."""
    from backend.rag.embeddings import get_embedding_model
    from backend.rag.vectorstore import get_vector_store

    store = get_vector_store()
    store.build(force=True)

    vector = get_embedding_model().embed_query("exhausted and overworked")
    hits = store.query(vector, top_k=3)

    assert len(hits) == 3
    for hit in hits:
        assert 0.0 <= hit["similarity"] <= 1.0
