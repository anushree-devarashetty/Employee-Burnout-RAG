"""
Person 3 — RAG Backend
==============================================================================
Explainable Employee Burnout Detection using Machine Learning,
Sentiment Analysis and Retrieval-Augmented Generation.

WHAT THIS PACKAGE DOES
    Consumes `outputs/burnout_scores.csv` — the final artefact produced by
    Person 2's ML pipeline — and serves Retrieval-Augmented explanations over
    it through a FastAPI application.

WHAT THIS PACKAGE DELIBERATELY DOES NOT DO
    It never re-runs preprocessing (Person 1) and never re-runs sentiment,
    emotion, behaviour or scoring (Person 2). The CSV is read-only input.
    If a burnout number looks wrong, the fix belongs in models/, not here.

WHY THIS FILE CONTAINS LOGIC INSTEAD OF BEING EMPTY
    `backend/data_loader.py` reuses Person 2's `models/utils/loader.load_data`
    instead of writing its own `pd.read_csv`. That import only resolves if the
    REPOSITORY ROOT is on `sys.path`. It is on sys.path when you launch from
    the repo root, but NOT when uvicorn or pytest is launched from inside
    `backend/`, and NOT when Streamlit runs `frontend/app.py`.

    Rather than force every teammate to remember a working directory, this
    module — the first thing Python executes for anything under `backend.` —
    resolves the repo root from its own file location and puts it on sys.path
    exactly once. After this runs, `from models.utils.loader import load_data`
    works from any launch directory on Windows, macOS and Linux.

    Verified on Python 3.10: `models/` and `models/utils/` need NO __init__.py
    files. PEP 420 implicit namespace packages resolve them as-is, so Person 2's
    directory tree is left completely untouched — zero files added, zero
    modified. This bootstrap is the whole reason that is possible.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Person 3 — RAG Backend"

# ------------------------------------------------------------------------------
# Repository root resolution
# ------------------------------------------------------------------------------
# This file lives at <repo_root>/backend/__init__.py, so the repo root is
# exactly one directory above this file's parent.
#
#   Path(__file__)          -> <repo_root>/backend/__init__.py
#   Path(__file__).resolve()-> absolute, with symlinks expanded
#   .parent                 -> <repo_root>/backend
#   .parent.parent          -> <repo_root>                      <-- what we want
#
# .resolve() matters: without it, launching via a symlinked venv or via
# `python -m` from an unusual directory can yield a relative path that breaks
# the sys.path insertion.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# The backend package directory itself, used by config.py to place chroma_db.
BACKEND_DIR: Path = REPO_ROOT / "backend"


def _bootstrap_repo_root_on_syspath() -> None:
    """Put the repository root on ``sys.path`` so ``models.*`` is importable.

    Idempotent and safe to call repeatedly — Python caches modules, so this
    body executes only once per interpreter regardless of how many backend
    modules are imported.

    Inserted at position 0 rather than appended, so that Person 2's ``models``
    package always wins against any unrelated third-party distribution that
    happens to publish a top-level module named ``models``. That collision is
    not hypothetical: several ML packages ship exactly that name.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# Executed at import time. Any module doing `from backend.config import ...`
# or `import backend.rag.retriever` triggers this first.
_bootstrap_repo_root_on_syspath()


__all__ = [
    "REPO_ROOT",
    "BACKEND_DIR",
    "__version__",
]
