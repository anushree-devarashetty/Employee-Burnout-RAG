"""
backend/rag/vectorstore.py — Persistent ChromaDB index (Person 3, stage 2)
==============================================================================

ROLE
    Owns the vector database. Builds a Chroma collection from Person 2's CSV,
    persists it to disk, and answers nearest-neighbour queries. It knows
    nothing about prompts or LLMs.

WHY PERSISTENT, NOT IN-MEMORY
    An in-memory client re-embeds every record on each backend restart. That is
    invisible with 10 rows and intolerable once the Enron dataset is plugged in.
    `PersistentClient` writes to CHROMA_PERSIST_DIR (git-ignored), so a restart
    reuses the existing index.

WHY COSINE SPACE
    Chroma defaults to squared L2. `embeddings.py` emits unit-length vectors,
    so configuring `hnsw:space = "cosine"` makes the returned distance exactly
    `1 - cosine_similarity` — a bounded, explainable 0-1 relevance score for
    the dashboard rather than an unbounded distance.

THE FINGERPRINT
    Re-embedding on every startup is wasteful; never re-embedding means the
    index silently goes stale after Person 2 re-runs their pipeline. The
    collection therefore stores a fingerprint of the source data (row count,
    file mtime, embedding model, dimension). On startup the current fingerprint
    is compared with the stored one and the index is rebuilt only when they
    differ. `POST /refresh?force=true` overrides this.

    The embedding model is part of the fingerprint on purpose: switching
    EMBEDDING_MODEL changes vector dimensionality, and querying a 384-dim index
    with a 768-dim vector fails deep inside Chroma with an opaque error. This
    catches it at startup instead.
==============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from backend.config import Settings, get_settings
from backend.data_loader import (
    BurnoutDataset,
    load_burnout_dataset,
    row_to_metadata,
)
from backend.rag.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

#: Chroma rejects an entire add() if any batch exceeds its internal limit.
#: 512 keeps peak memory modest while staying far below that ceiling.
_ADD_BATCH_SIZE = 512

#: Key under which the fingerprint is stored in collection metadata.
_FINGERPRINT_KEY = "source_fingerprint"


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be created, populated or queried."""


@dataclass(frozen=True)
class IndexResult:
    """Outcome of a build/refresh operation."""

    rebuilt: bool
    records_indexed: int
    collection: str
    persist_dir: str
    fingerprint: str
    reason: str


# ==============================================================================
# Fingerprinting
# ==============================================================================
def compute_fingerprint(
    dataset: BurnoutDataset, model_name: str, dimension: Optional[int]
) -> str:
    """Stable hash of everything that would invalidate the index.

    Deliberately excludes row *content*: hashing 500k Enron rows on every
    startup would cost more than the rebuild it is meant to avoid. Row count
    plus file mtime is a sound proxy, because any edit to the CSV updates its
    mtime.
    """
    payload = json.dumps(
        {
            "path": str(dataset.source_path),
            "mtime": round(dataset.source_mtime, 3),
            "rows": dataset.record_count,
            "model": model_name,
            "dimension": dimension,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# ==============================================================================
# Vector store
# ==============================================================================
class BurnoutVectorStore:
    """ChromaDB-backed store of embedded employee burnout records."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedder: EmbeddingModel,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedder = embedder

        self._client = None
        self._collection = None
        self._lock = threading.RLock()

    # --------------------------------------------------------------------------
    # Client / collection
    # --------------------------------------------------------------------------
    def _get_client(self):
        """Create the persistent Chroma client on first use."""
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
            except ImportError as exc:
                raise VectorStoreError(
                    "chromadb is not installed.\n"
                    "  Fix: pip install -r backend/requirements.txt"
                ) from exc

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Opening ChromaDB at %s", self.persist_dir)

            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,  # no phoning home with HR data
                        allow_reset=True,
                    ),
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not open ChromaDB at {self.persist_dir}: {exc}\n"
                    f"  If the directory is corrupt, delete it and rebuild:\n"
                    f"    rm -rf {self.persist_dir}"
                ) from exc

            return self._client

    def _get_collection(self, create: bool = True):
        """Fetch (or create) the collection, configured for cosine distance."""
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection

            client = self._get_client()
            try:
                if not create:
                    self._collection = client.get_collection(self.collection_name)
                    return self._collection

                if self.collection_name in self._collection_names():
                    # Fetch WITHOUT metadata. Passing `hnsw:space` to an
                    # existing collection makes Chroma attempt a modify, which
                    # raises "Changing the distance function ... is not
                    # supported" — even though the value is identical, and even
                    # though we only wanted to read it.
                    self._collection = client.get_collection(self.collection_name)
                else:
                    self._collection = client.create_collection(
                        name=self.collection_name,
                        # Cosine, not the squared-L2 default. Only settable at
                        # creation; see the module docstring.
                        metadata={"hnsw:space": "cosine"},
                    )
            except Exception as exc:
                raise VectorStoreError(
                    f"Could not open collection '{self.collection_name}': {exc}"
                ) from exc

            return self._collection

    # --------------------------------------------------------------------------
    # State
    # --------------------------------------------------------------------------
    def _collection_names(self) -> List[str]:
        """Names of existing collections, across ChromaDB versions.

        `list_collections()` changed return type between releases: some
        versions yield `Collection` objects (use `.name`), others yield plain
        strings. Assuming either one breaks on the other — and because the
        resulting AttributeError used to be swallowed by a bare `except`, the
        symptom was a silent "0 records" after a successful write rather than
        an error naming the cause.
        """
        result: List[str] = []
        for item in self._get_client().list_collections():
            result.append(item if isinstance(item, str) else getattr(item, "name", ""))
        return [name for name in result if name]

    def exists(self) -> bool:
        """True if the collection exists and holds at least one record."""
        try:
            if self.collection_name not in self._collection_names():
                return False
            return self.count() > 0
        except Exception as exc:
            logger.debug("exists() check failed: %s", exc)
            return False

    def count(self) -> int:
        """Number of indexed records, or 0 if the collection is unavailable.

        Failures are logged rather than silently swallowed. A bare
        `except: return 0` here previously turned a version-compatibility error
        into a phantom empty index: the build reported "Writing 10 records"
        followed by "built: 0 records", with nothing explaining the gap.
        """
        try:
            return int(self._get_collection().count())
        except Exception as exc:
            logger.warning(
                "Could not read the record count for collection '%s': %s. "
                "Reporting 0. This usually indicates a ChromaDB version "
                "mismatch — check `pip show chromadb` against "
                "backend/requirements.txt.",
                self.collection_name,
                exc,
            )
            return 0

    def stored_fingerprint(self) -> Optional[str]:
        """Fingerprint recorded at the last build, if any.

        Re-fetches the collection from the client rather than reading the
        cached handle. A `Collection` object snapshots its metadata at fetch
        time, so a handle obtained before `_write_fingerprint` ran keeps
        reporting the old (or absent) value forever — which silently defeated
        the rebuild-skip and forced a full re-embed on every startup.
        """
        try:
            collection = self._get_client().get_collection(self.collection_name)
            value = (collection.metadata or {}).get(_FINGERPRINT_KEY)
            return str(value) if value else None
        except Exception:
            return None

    # --------------------------------------------------------------------------
    # Building
    # --------------------------------------------------------------------------
    def build(
        self,
        dataset: Optional[BurnoutDataset] = None,
        force: bool = False,
    ) -> IndexResult:
        """Build or refresh the index from Person 2's CSV.

        Args:
            dataset: Pre-loaded dataset. Loaded via `data_loader` if omitted.
            force: Rebuild even when the fingerprint matches.

        Returns:
            An :class:`IndexResult` describing what happened and why.
        """
        dataset = dataset or load_burnout_dataset()
        frame = dataset.frame

        fingerprint = compute_fingerprint(
            dataset, self.embedder.model_name, self.embedder.dimension
        )

        with self._lock:
            if not force and self.exists():
                stored = self.stored_fingerprint()
                if stored == fingerprint:
                    count = self.count()
                    logger.info(
                        "Vector store already current (%d records, fingerprint %s).",
                        count,
                        fingerprint[:8],
                    )
                    return IndexResult(
                        rebuilt=False,
                        records_indexed=count,
                        collection=self.collection_name,
                        persist_dir=str(self.persist_dir),
                        fingerprint=fingerprint,
                        reason="Index already matches the source data.",
                    )
                logger.info(
                    "Fingerprint changed (%s -> %s); rebuilding.",
                    (stored or "none")[:8],
                    fingerprint[:8],
                )
                reason = "Source data or embedding model changed."
            else:
                reason = "Forced rebuild." if force else "No existing index."

            self._drop_collection()
            collection = self._get_collection(create=True)

            if frame.empty:
                logger.warning("Source CSV has no rows; index left empty.")
                self._write_fingerprint(fingerprint)
                return IndexResult(
                    rebuilt=True,
                    records_indexed=0,
                    collection=self.collection_name,
                    persist_dir=str(self.persist_dir),
                    fingerprint=fingerprint,
                    reason="Source CSV contained no records.",
                )

            documents = frame["document_text"].astype(str).tolist()
            ids = frame["employee_id"].astype(str).tolist()
            metadatas = [row_to_metadata(row) for _, row in frame.iterrows()]

            if len(set(ids)) != len(ids):
                # Chroma silently upserts on duplicate IDs, which would drop
                # records without any error. Fail loudly instead.
                duplicates = [i for i in set(ids) if ids.count(i) > 1]
                raise VectorStoreError(
                    f"Duplicate employee_id values would silently overwrite "
                    f"records: {duplicates[:5]}. "
                    f"Check the employee_id column in {dataset.source_path}."
                )

            logger.info(
                "Embedding %d records with %s...",
                len(documents),
                self.embedder.model_name,
            )
            vectors = self.embedder.embed_documents(documents)

            logger.info("Writing %d records to ChromaDB...", len(documents))
            try:
                for start in range(0, len(documents), _ADD_BATCH_SIZE):
                    end = start + _ADD_BATCH_SIZE
                    collection.add(
                        ids=ids[start:end],
                        documents=documents[start:end],
                        embeddings=vectors[start:end],
                        metadatas=metadatas[start:end],
                    )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed writing records to ChromaDB: {exc}"
                ) from exc

            self._write_fingerprint(fingerprint)
            indexed = self.count()
            logger.info(
                "Vector store built: %d records, collection '%s'.",
                indexed,
                self.collection_name,
            )
            return IndexResult(
                rebuilt=True,
                records_indexed=indexed,
                collection=self.collection_name,
                persist_dir=str(self.persist_dir),
                fingerprint=fingerprint,
                reason=reason,
            )

    def _write_fingerprint(self, fingerprint: str) -> None:
        """Persist the fingerprint into collection metadata.

        Two Chroma behaviours are worked around here, both verified against
        chromadb 1.5:

        1. ``modify()`` raises "Changing the distance function of a collection
           once it is created is not supported" if ``hnsw:space`` is present in
           the payload — even when re-sending the identical value. So the key
           is deliberately omitted. The distance function is fixed at creation
           and is unaffected by its absence from metadata.

        2. ``modify()`` REPLACES metadata rather than merging it, so nothing
           else may be assumed to survive. Only the fingerprint is stored here,
           which is all this class reads back.

        The cached collection handle is then dropped, because a `Collection`
        object snapshots metadata at fetch time and would otherwise keep
        serving the pre-modify value.
        """
        try:
            self._get_collection().modify(metadata={_FINGERPRINT_KEY: fingerprint})
            self._collection = None      # force a re-fetch with fresh metadata
            logger.debug("Stored fingerprint %s", fingerprint[:8])
        except Exception as exc:  # non-fatal: costs one rebuild next start
            logger.warning("Could not store fingerprint: %s", exc)

    def _drop_collection(self) -> None:
        """Delete the collection so a rebuild starts from empty."""
        try:
            self._get_client().delete_collection(self.collection_name)
            logger.debug("Dropped existing collection '%s'.", self.collection_name)
        except Exception:
            pass  # absent is the desired end state
        finally:
            self._collection = None

    def reset(self) -> None:
        """Delete the entire on-disk store. Used by tests and hard resets."""
        with self._lock:
            self._collection = None
            self._client = None
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir, ignore_errors=True)
                logger.info("Removed vector store directory %s", self.persist_dir)

    # --------------------------------------------------------------------------
    # Querying
    # --------------------------------------------------------------------------
    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the ``top_k`` nearest records.

        Args:
            query_embedding: The encoded question.
            top_k: Number of neighbours.
            where: Chroma metadata filter, e.g. ``{"risk_level": "High"}``.

        Returns:
            Dicts with ``id``, ``document``, ``metadata``, ``distance`` and
            ``similarity`` (0-1), ordered most relevant first.
        """
        if not query_embedding:
            return []

        collection = self._get_collection(create=False)
        available = self.count()
        if available == 0:
            logger.warning("Query against an empty vector store.")
            return []

        # Chroma errors if n_results exceeds the collection size.
        n_results = max(1, min(int(top_k), available))

        try:
            raw = collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=n_results,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        return self._flatten(raw)

    @staticmethod
    def _flatten(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Chroma's parallel-array response into a list of dicts.

        Chroma returns ``{"ids": [[...]], "distances": [[...]], ...}`` — one
        outer list per query embedding. We always send one, so index 0 is taken
        throughout, with `or [[]]` guarding the empty-result case.
        """
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: List[Dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            results.append(
                {
                    "id": record_id,
                    "document": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distance,
                    "similarity": EmbeddingModel.distance_to_similarity(distance),
                }
            )
        return results

    def get_by_id(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one record by ID without a similarity search."""
        try:
            raw = self._get_collection(create=False).get(
                ids=[str(employee_id)], include=["documents", "metadatas"]
            )
        except Exception:
            return None

        ids = raw.get("ids") or []
        if not ids:
            return None
        return {
            "id": ids[0],
            "document": (raw.get("documents") or [""])[0],
            "metadata": (raw.get("metadatas") or [{}])[0],
            "distance": 0.0,
            "similarity": 1.0,
        }

    def info(self) -> Dict[str, Any]:
        """State summary for ``GET /health``."""
        return {
            "backend": "chromadb",
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "exists": self.persist_dir.is_dir(),
            "record_count": self.count(),
            "fingerprint": self.stored_fingerprint(),
            "distance_metric": "cosine",
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.info().get("dimension"),
        }


# ==============================================================================
# Singleton accessor
# ==============================================================================
_INSTANCE: Optional[BurnoutVectorStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_vector_store(settings: Optional[Settings] = None) -> BurnoutVectorStore:
    """Return the process-wide :class:`BurnoutVectorStore`.

    Manual singleton rather than ``@lru_cache``: Pydantic v2 ``Settings`` is
    unhashable, so lru_cache raises ``TypeError: unhashable type: 'Settings'``
    when settings are passed explicitly.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            resolved = settings or get_settings()
            _INSTANCE = BurnoutVectorStore(
                persist_dir=resolved.chroma_dir,
                collection_name=resolved.chroma_collection,
                embedder=get_embedding_model(resolved),
            )
    return _INSTANCE


def reset_vector_store() -> None:
    """Drop the cached instance. Used by tests."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


def build_index(force: bool = False) -> IndexResult:
    """Convenience wrapper used by startup and ``POST /refresh``."""
    return get_vector_store().build(force=force)


__all__ = [
    "BurnoutVectorStore",
    "VectorStoreError",
    "IndexResult",
    "get_vector_store",
    "build_index",
    "compute_fingerprint",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    result = build_index(force=True)
    print(json.dumps(result.__dict__, indent=2))
    print(json.dumps(get_vector_store().info(), indent=2))
