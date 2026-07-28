"""
tests/test_schemas.py — the eight-key contract and its defensive coercion
==============================================================================
LLMs return *nearly* the requested shape. These tests pin down which
near-misses are absorbed and which are rejected.
==============================================================================
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.schemas import BurnoutAnalysis, QueryRequest, RiskLevel

EIGHT_KEYS = {
    "burnout_summary",
    "emotional_analysis",
    "possible_causes",
    "risk_level",
    "suggested_interventions",
    "hr_recommendations",
    "mental_wellness_suggestions",
    "confidence_note",
}


# ------------------------------------------------------------------------------
# The contract
# ------------------------------------------------------------------------------
def test_serialises_to_exactly_the_eight_specified_keys(sample_analysis):
    payload = json.loads(BurnoutAnalysis(**sample_analysis).model_dump_json())
    assert set(payload) == EIGHT_KEYS


def test_json_round_trip_is_lossless(sample_analysis):
    original = BurnoutAnalysis(**sample_analysis)
    assert BurnoutAnalysis(**json.loads(original.model_dump_json())) == original


# ------------------------------------------------------------------------------
# List coercion — shapes real models actually emit
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (["a", "b"], ["a", "b"]),
        ("a\nb", ["a", "b"]),
        ("- a\n- b", ["a", "b"]),
        ("1. a\n2) b", ["a", "b"]),
        ("• a\n– b", ["a", "b"]),
        ("one sentence", ["one sentence"]),
        ("a\n\n\nb", ["a", "b"]),
        (None, []),
        ([{"cause": "overtime"}], ["overtime"]),
        ({"x": "overtime"}, ["overtime"]),
    ],
)
def test_list_fields_absorb_llm_formatting(sample_analysis, raw, expected):
    analysis = BurnoutAnalysis(**{**sample_analysis, "possible_causes": raw})
    assert analysis.possible_causes == expected


def test_prose_field_flattens_an_over_structured_list(sample_analysis):
    analysis = BurnoutAnalysis(
        **{**sample_analysis, "burnout_summary": ["Part one.", "Part two."]}
    )
    assert analysis.burnout_summary == "Part one. Part two."


# ------------------------------------------------------------------------------
# Risk coercion
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("high", RiskLevel.HIGH),
        ("HIGH RISK", RiskLevel.HIGH),
        ("Risk: Medium", RiskLevel.MEDIUM),
        ("moderate", RiskLevel.MEDIUM),
        ("critical", RiskLevel.HIGH),
        ("minimal", RiskLevel.LOW),
        ("banana", RiskLevel.UNKNOWN),
        (None, RiskLevel.UNKNOWN),
    ],
)
def test_risk_level_coercion(sample_analysis, raw, expected):
    assert BurnoutAnalysis(**{**sample_analysis, "risk_level": raw}).risk_level == expected


def test_high_wins_over_low_in_an_ambiguous_string(sample_analysis):
    """Under-reporting someone's risk is worse than over-reporting it."""
    analysis = BurnoutAnalysis(**{**sample_analysis, "risk_level": "not low, high"})
    assert analysis.risk_level == RiskLevel.HIGH


# ------------------------------------------------------------------------------
# confidence_note — the one field that may never go missing
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("missing", ["", "   ", None, []])
def test_absent_confidence_note_is_substituted_not_dropped(sample_analysis, missing):
    """This system infers a sensitive human state from a little text.

    Serving an analysis with no statement of its limits is precisely the
    black-box behaviour the project exists to avoid, so a caveat is supplied
    rather than the field being left empty.
    """
    analysis = BurnoutAnalysis(**{**sample_analysis, "confidence_note": missing})
    assert "not a clinical or diagnostic" in analysis.confidence_note


def test_a_real_confidence_note_is_left_untouched(sample_analysis):
    analysis = BurnoutAnalysis(**sample_analysis)
    assert analysis.confidence_note == sample_analysis["confidence_note"]


# ------------------------------------------------------------------------------
# Request validation
# ------------------------------------------------------------------------------
def test_whitespace_only_question_is_rejected():
    """`min_length=3` alone accepts three spaces, which would query the LLM
    with nothing and return a confident, meaningless answer."""
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_too_short_question_is_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="hi")


def test_question_is_trimmed():
    assert QueryRequest(question="  Who is at risk?  ").question == "Who is at risk?"


def test_risk_levels_accept_a_comma_separated_string():
    request = QueryRequest(question="who?", risk_levels="high, medium")
    assert request.risk_levels == [RiskLevel.HIGH, RiskLevel.MEDIUM]


def test_blank_employee_id_becomes_none():
    assert QueryRequest(question="who?", employee_id="   ").employee_id is None


def test_top_k_must_be_positive():
    with pytest.raises(ValidationError):
        QueryRequest(question="who?", top_k=0)
