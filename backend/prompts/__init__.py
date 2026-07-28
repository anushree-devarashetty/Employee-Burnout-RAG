"""
backend.prompts — Prompt templates for the RAG pipeline (Person 3)
==============================================================================

WHY PROMPTS LIVE IN THEIR OWN PACKAGE
    The prompt is the single highest-leverage, most-frequently-edited artefact
    in a RAG system. Isolating it means a teammate can tune wording, tighten the
    JSON contract or adjust the safety framing without opening — or risking —
    any retrieval, vector-store or HTTP code.

    It also keeps prompts reviewable in isolation: a pull request that changes
    only `burnout_prompt.py` is legible to a non-engineer reviewing the
    psychological framing, which matters for a wellbeing-adjacent project.

CONTRACT PRODUCED BY THESE TEMPLATES
    Every prompt in this package instructs the LLM to return a single JSON
    object with exactly these eight keys, matching the project specification:

        burnout_summary          Plain-language overview of the situation
        emotional_analysis       What the NRC emotion signals actually indicate
        possible_causes          Workload, recognition, autonomy, clarity, etc.
        risk_level              Low | Medium | High  (echoed from the data)
        suggested_interventions  Concrete, actionable, manager-level steps
        hr_recommendations       Policy and process level actions for HR
        mental_wellness_suggestions  Supportive, non-clinical wellbeing steps
        confidence_note          Honest statement of the evidence's limits

    `confidence_note` is not decoration. This system infers a sensitive human
    state from a small sample of text, and the specification requires the
    limitation to be surfaced to the HR user rather than buried. The prompt
    forbids the model from omitting it.

GROUNDING RULE
    Templates instruct the model to answer ONLY from the retrieved records and
    to say so explicitly when the context does not support an answer, rather
    than inventing plausible-sounding detail. Retrieved employee text is
    delimited and labelled as data, so that message content can never be read
    as an instruction to the model.
==============================================================================
"""

from __future__ import annotations

from backend import REPO_ROOT  # noqa: F401  (ensures sys.path bootstrap ran)

__all__ = ["REPO_ROOT"]
