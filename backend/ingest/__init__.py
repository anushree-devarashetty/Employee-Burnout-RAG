"""
backend.ingest — Document ingestion (Person 3 extension)
==============================================================================

Lets a user upload real workplace communication — PDF email exports, chat
transcripts, screenshots of conversations — and run it through the same
analysis pipeline that produced `outputs/burnout_scores.csv`.

    Upload ──> extract ──> split into messages ──> score ──> index ──> RAG
               (PDF/OCR)   (email/chat aware)      (P2 logic) (Chroma)

MODULES
    extractors.py       PDF text, OCR for scans and screenshots, plain text
    message_splitter.py Raw text -> individual messages
    pipeline_runner.py  Sentiment, emotion, behaviour, burnout score

DESIGN CONSTRAINT THAT SHAPED ALL OF THIS
    Uploaded messages must be scored by *exactly* the same rules as the
    committed dataset, or the dashboard would be comparing two different
    things and the comparison would be meaningless.

    Person 2's scoring lives in `models/burnout/burnout_score.py` as three
    module-level functions — but that file also executes top-level code that
    reads and OVERWRITES the CSVs. Importing it normally would destroy their
    output.

    `pipeline_runner.py` therefore parses their file with `ast`, keeps only the
    function definitions, and executes those in an isolated namespace. Their
    real scoring logic is reused, their file is untouched, and no side effects
    occur. See that module for the details.

SESSION SCOPE
    Uploaded documents are indexed into a SEPARATE ChromaDB collection and are
    never written back into `outputs/burnout_scores.csv`. Person 2's dataset
    remains the system of record; an upload is an overlay, not a mutation.
==============================================================================
"""

from __future__ import annotations

from backend import REPO_ROOT  # noqa: F401  (ensures the sys.path bootstrap ran)

__all__ = ["REPO_ROOT"]
