"""
backend/rag/embeddings.py — Sentence-Transformers encoder (Person 3, stage 1)
==============================================================================

ROLE IN THE PIPELINE

    User Question -> [ embeddings.py ] -> vectorstore.py -> retriever.py
                                       -> prompts -> llm_client -> JSON

    This module turns text into vectors. That is all it does. It has no
    knowledge of ChromaDB, of burnout, or of which LLM is configured — which
    is what makes it independently testable and swappable.

MODEL: all-MiniLM-L6-v2  (as specified)
    384 dimensions, ~80 MB, 6 transformer layers. Encodes a few hundred short
    messages per second on a laptop CPU. Downloaded once from HuggingFace on
    first use and cached in ~/.cache/huggingface, so only the first run needs
    a network connection.

THREE DECISIONS WORTH KNOWING ABOUT

1.  LAZY, THREAD-SAFE SINGLETON LOAD
        Constructing a SentenceTransformer takes seconds and allocates real
        memory. Loading it at import time would make `import backend.app` slow
        and would download 80 MB during test collection. It is therefore loaded
        on first actual use, behind a lock — FastAPI serves from a thread pool,
        so two simultaneous first requests would otherwise load two copies of
        the model into memory.

2.  EMBEDDINGS ARE L2-NORMALISED
        With unit-length vectors, cosine similarity reduces to a plain dot
        product, and Chroma's cosine distance becomes exactly `1 - similarity`.
        That gives `retriever.py` a clean, bounded 0-1 similarity score to show
        the user, instead of an unbounded L2 distance that means nothing on a
        dashboard.

3.  THE INTERFACE IS LANGCHAIN'S `Embeddings` INTERFACE
        `embed_documents(list[str]) -> list[list[float]]` and
        `embed_query(str) -> list[float]` are exactly the two methods LangChain
        expects. This class can therefore be handed straight to a LangChain
        vector store or retriever with no adapter, while remaining a plain
        object with no LangChain import of its own — so a LangChain version
        bump cannot break embedding.

WHY THE MODEL IS NOT REGISTERED AS A CHROMA `EmbeddingFunction`
    Chroma's EmbeddingFunction signature has changed across releases. Encoding
    here and passing explicit vectors to Chroma keeps this project pinned to
    one stable calling convention, and keeps the encoder reusable outside
    Chroma entirely.
==============================================================================
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable, List, Optional, Sequence

# ------------------------------------------------------------------------------
# Script-mode self-bootstrap (see backend/config.py for the full rationale)
# ------------------------------------------------------------------------------
if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from backend.config import EmbeddingDevice, Settings, get_settings

logger = logging.getLogger(__name__)

#: Known output dimensionality, used for fast sanity checks and error messages.
#: Verified at runtime against the loaded model rather than trusted blindly.
_KNOWN_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


# ==============================================================================
# Device resolution
# ==============================================================================
def _resolve_device(requested: EmbeddingDevice) -> str:
    """Return a torch device string that is actually available on this machine.

    Silently falling back to CPU is the right behaviour here rather than
    raising: four collaborators on three operating systems will have different
    hardware, and `EMBEDDING_DEVICE=cuda` committed by the teammate with an
    NVIDIA GPU must not crash the backend for everyone else. The fallback is
    logged at WARNING so it is never invisible.
    """
    device = requested.value

    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with sentence-transformers
        logger.warning("torch not importable; falling back to CPU.")
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "EMBEDDING_DEVICE=cuda but no CUDA device is available. Using CPU. "
            "all-MiniLM-L6-v2 is small enough that CPU is fine for this dataset."
        )
        return "cpu"

    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            logger.warning(
                "EMBEDDING_DEVICE=mps but Apple Metal is unavailable. Using CPU."
            )
            return "cpu"

    return device


# ==============================================================================
# Encoder
# ==============================================================================
class EmbeddingModel:
    """Thin, lazily-loaded wrapper around a SentenceTransformer.

    Implements LangChain's ``Embeddings`` interface (``embed_documents`` and
    ``embed_query``) without importing LangChain.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self._model = None                      # populated on first use
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

    # --------------------------------------------------------------------------
    # Lazy loading
    # --------------------------------------------------------------------------
    def _load(self):
        """Load the SentenceTransformer once, safely, on first use.

        Double-checked locking: the fast path is a lock-free attribute read,
        and the lock is taken only on the very first call. The second check
        inside the lock is what prevents a thread that queued behind the lock
        from loading a second copy.
        """
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:      # another thread won the race
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed.\n"
                    "  Fix: pip install -r backend/requirements.txt\n"
                    "  (or: pip install sentence-transformers)"
                ) from exc

            logger.info(
                "Loading embedding model '%s' on %s "
                "(first run downloads ~80MB from HuggingFace)...",
                self.model_name,
                self.device,
            )
            try:
                model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}' on "
                    f"device '{self.device}': {exc}\n"
                    f"  If this is the first run, check your internet connection "
                    f"— the model is downloaded from HuggingFace.\n"
                    f"  Or set a different model in .env: EMBEDDING_MODEL=..."
                ) from exc

            self._model = model

            # Trust the loaded model over the lookup table. A user who changes
            # EMBEDDING_MODEL must not silently get a wrong dimension, because
            # an existing Chroma collection built at another dimension would
            # then fail confusingly deep inside the vector store.
            self._dimension = int(model.get_sentence_embedding_dimension())
            expected = _KNOWN_DIMENSIONS.get(self.model_name)
            if expected and expected != self._dimension:
                logger.warning(
                    "Model '%s' reported %d dimensions; %d was expected.",
                    self.model_name,
                    self._dimension,
                    expected,
                )

            logger.info(
                "Embedding model ready: %s (%d dimensions, device=%s)",
                self.model_name,
                self._dimension,
                self.device,
            )
            return self._model

    @property
    def is_loaded(self) -> bool:
        """True if the model is in memory. Lets /health report without loading."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Output dimensionality. Triggers the lazy load if needed."""
        if self._dimension is None:
            self._load()
        return int(self._dimension or 0)

    # --------------------------------------------------------------------------
    # Encoding
    # --------------------------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Encode texts into L2-normalised vectors.

        Args:
            texts: Strings to encode. May be empty.
            batch_size: Override the configured batch size.
            show_progress: Show a progress bar (useful for a large rebuild).

        Returns:
            One vector per input, in the same order. ``[]`` for empty input.

        Raises:
            EmbeddingError: if the model is unavailable or encoding fails.
        """
        if texts is None:
            return []

        items = list(texts)
        if not items:
            # Short-circuit: SentenceTransformer.encode([]) returns an
            # inconsistently-shaped array across versions, and callers rebuilding
            # an empty collection hit this legitimately.
            return []

        # None and NaN reach here from CSV columns; the tokenizer raises on
        # non-strings, so coerce at the boundary instead of trusting callers.
        cleaned = [("" if text is None else str(text)) for text in items]

        model = self._load()
        try:
            vectors = model.encode(
                cleaned,
                batch_size=batch_size or self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                # The key setting: unit-length output makes cosine similarity a
                # dot product and turns Chroma's cosine distance into 1 - sim.
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to encode {len(cleaned)} text(s) with "
                f"'{self.model_name}': {exc}"
            ) from exc

        return [vector.tolist() for vector in vectors]

    # --------------------------------------------------------------------------
    # LangChain-compatible surface
    # --------------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of documents. LangChain ``Embeddings`` interface."""
        return self.encode(texts, show_progress=len(texts) > 500)

    def embed_query(self, text: str) -> List[float]:
        """Encode a single query. LangChain ``Embeddings`` interface.

        Returns a zero vector for empty input rather than raising: an empty
        query is a caller bug that `QueryRequest` already rejects at the API
        boundary, and a zero vector degrades to "no meaningful match" instead
        of a 500.
        """
        if not text or not str(text).strip():
            logger.warning("embed_query called with empty text; returning zeros.")
            return [0.0] * self.dimension
        return self.encode([text])[0]

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity between two vectors, clamped to [-1, 1].

        Implemented with the standard formula rather than assuming unit length,
        so it stays correct if handed vectors from another source. Clamping
        absorbs floating-point overshoot such as 1.0000000000000002, which
        would otherwise fail the ``le=1.0`` bound on `RetrievedRecord`.
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """Convert Chroma's cosine distance into a 0-1 similarity score.

        Chroma returns ``distance = 1 - cosine_similarity`` for a collection
        configured with ``hnsw:space = cosine``. Because embeddings here are
        normalised and the text is non-negative in practice, similarity lands
        in [0, 1]; the clamp guards the edge cases so the value always
        satisfies `RetrievedRecord.similarity`'s bounds.
        """
        return max(0.0, min(1.0, 1.0 - float(distance)))

    def info(self) -> dict:
        """Summary for ``GET /health``. Does NOT trigger a model load."""
        return {
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "dimension": self._dimension,      # None until first use
            "loaded": self.is_loaded,
            "normalised": True,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "loaded" if self.is_loaded else "not loaded"
        return (
            f"EmbeddingModel({self.model_name!r}, device={self.device!r}, "
            f"{state})"
        )


# ==============================================================================
# Singleton accessor
# ==============================================================================
_INSTANCE: Optional[EmbeddingModel] = None
_INSTANCE_LOCK = threading.Lock()


def get_embedding_model(settings: Optional[Settings] = None) -> EmbeddingModel:
    """Return the process-wide :class:`EmbeddingModel`.

    Cached so the ~80 MB model is loaded at most once per process. The instance
    is returned immediately; the underlying transformer still loads lazily on
    first encode, so importing this module stays cheap.

    NOT implemented with ``@lru_cache``. Pydantic v2 models are unhashable, so
    ``lru_cache`` raises ``TypeError: unhashable type: 'Settings'`` the moment
    a caller passes settings explicitly — which `vectorstore.get_vector_store`
    and `retriever.get_retriever` both do. A manual double-checked singleton
    caches on identity instead of on argument hashing, and is thread-safe.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE

    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            resolved = settings or get_settings()
            _INSTANCE = EmbeddingModel(
                model_name=resolved.embedding_model,
                device=_resolve_device(resolved.embedding_device),
                batch_size=resolved.embedding_batch_size,
            )
    return _INSTANCE


def reset_embedding_model() -> None:
    """Drop the cached instance. Used by tests and after a config change."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Module-level convenience wrapper around the singleton."""
    return get_embedding_model().embed_documents(list(texts))


def embed_question(question: str) -> List[float]:
    """Module-level convenience wrapper around the singleton."""
    return get_embedding_model().embed_query(question)


__all__ = [
    "EmbeddingModel",
    "EmbeddingError",
    "get_embedding_model",
    "reset_embedding_model",
    "embed_texts",
    "embed_question",
]


# ==============================================================================
# Manual inspection:  python backend/rag/embeddings.py
# ==============================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    model = get_embedding_model()
    print(f"Config: {model.info()}")

    print("\nLoading model (first run downloads ~80MB)...")
    print(f"Dimension: {model.dimension}")

    samples = [
        "I am completely exhausted after working late.",
        "There is too much pressure this week.",
        "I love working with my team.",
        "The meeting was productive.",
    ]
    vectors = model.embed_documents(samples)
    print(f"\nEncoded {len(vectors)} samples -> {len(vectors[0])} dimensions each")

    norm = sum(v * v for v in vectors[0]) ** 0.5
    print(f"L2 norm of first vector: {norm:.6f}  (must be ~1.0)")

    question = "Who is burned out and overworked?"
    q_vector = model.embed_query(question)

    print(f"\nQuery: {question!r}")
    print("Similarity to each sample (higher = more relevant):")
    ranked = sorted(
        ((EmbeddingModel.cosine_similarity(q_vector, v), s)
         for v, s in zip(vectors, samples)),
        reverse=True,
    )
    for score, sample in ranked:
        print(f"  {score:+.4f}  {sample}")
