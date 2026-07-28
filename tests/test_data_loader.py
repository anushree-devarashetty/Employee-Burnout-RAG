"""
tests/test_data_loader.py — the integration contract with Person 2
==============================================================================
These tests guard the boundary between Person 2's ML pipeline and Person 3/4.
If Person 2 changes the CSV schema, these fail first and say why.
==============================================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.data_loader import EMOTION_COLUMNS, compute_statistics


# ------------------------------------------------------------------------------
# Reuse of Person 2's code
# ------------------------------------------------------------------------------
def test_reuses_person2_loader_rather_than_reimplementing():
    """`backend` must call Person 2's `load_data`, not its own `read_csv`."""
    import inspect

    from backend import data_loader

    source = inspect.getsource(data_loader._read_with_person2_loader)
    assert "models.utils.loader" in source
    assert "load_data" in source


def test_person2_loader_still_satisfies_its_contract():
    """Person 2's loader must keep rejecting a frame without `clean_text`."""
    from models.utils.loader import load_data

    assert callable(load_data)


# ------------------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------------------
def test_required_columns_present(burnout_frame):
    for column in ("clean_text", "sentiment", "burnout_score", "risk_level"):
        assert column in burnout_frame.columns


def test_all_eight_emotions_available_for_charting(burnout_frame):
    """Absent emotion columns are materialised as 0 so charts never gap."""
    for emotion in EMOTION_COLUMNS:
        assert emotion in burnout_frame.columns


def test_behaviour_columns_are_never_fabricated(burnout_frame):
    """A missing behavioural measurement must NOT be invented as 0.

    `sentence_count` is absent because Person 2's `behavior_features.py`
    crashes on a non-existent `message` column — not because these messages
    contain zero sentences. Writing 0 would embed a false measurement into the
    vector store and present it to the LLM as evidence.

    Passes both before and after Person 2 applies docs/person2_fixes.patch:
    either the column is genuinely present, or it is absent everywhere.
    """
    from backend.data_loader import build_document_text, row_to_metadata

    row = burnout_frame.iloc[0]
    if "sentence_count" not in burnout_frame.columns:
        assert "sentence count" not in build_document_text(row)
        assert "sentence_count" not in row_to_metadata(row)
    else:
        assert float(row["sentence_count"]) >= 0


# ------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------
def test_employee_ids_are_stable_and_zero_padded(burnout_frame):
    ids = list(burnout_frame["employee_id"])
    assert ids == sorted(ids)
    assert all(i.startswith("EMP_") and len(i) == 8 for i in ids)


def test_employee_lookup_is_case_insensitive(burnout_frame):
    from backend.data_loader import get_employee_record

    assert get_employee_record("emp_0001", burnout_frame) is not None
    assert get_employee_record("EMP_0001", burnout_frame) is not None
    assert get_employee_record("EMP_9999", burnout_frame) is None


def test_real_employee_id_column_wins_if_person2_adds_one(monkeypatch, tmp_path):
    """A real `employee_id` must be used instead of synthetic IDs."""
    from backend.data_loader import _attach_employee_ids

    frame = pd.DataFrame({"clean_text": ["a", "b"], "employee_id": ["alice", "bob"]})
    result = _attach_employee_ids(frame, "EMP")
    assert list(result["employee_id"]) == ["alice", "bob"]


# ------------------------------------------------------------------------------
# Dominant emotion — the all-zero trap
# ------------------------------------------------------------------------------
def test_all_zero_emotion_rows_are_neutral_not_anger(burnout_frame):
    """Pandas `idxmax` breaks ties on the FIRST column, which is `anger`.

    A naive implementation therefore labels every emotionally-neutral record
    as angry. In this dataset that is 6 of 10 records, and in a wellbeing
    dashboard it would be actively harmful.
    """
    zero_rows = burnout_frame[
        burnout_frame[list(EMOTION_COLUMNS)].sum(axis=1) == 0
    ]
    assert len(zero_rows) > 0, "fixture no longer covers this case"
    assert set(zero_rows["dominant_emotion"]) == {"neutral"}

    naive = burnout_frame[list(EMOTION_COLUMNS)].idxmax(axis=1)
    assert (naive.loc[zero_rows.index] == "anger").all()


# ------------------------------------------------------------------------------
# Vector store metadata contract
# ------------------------------------------------------------------------------
def test_metadata_contains_only_chroma_legal_scalars(burnout_frame):
    """ChromaDB rejects an entire batch on one None, list or dict."""
    from backend.data_loader import row_to_metadata

    for _, row in burnout_frame.iterrows():
        for key, value in row_to_metadata(row).items():
            assert value is not None, f"{key} is None"
            assert isinstance(value, (str, int, float, bool)), f"{key}={type(value)}"


def test_long_text_is_truncated_for_metadata(burnout_frame):
    from backend.data_loader import row_to_metadata

    for value in row_to_metadata(burnout_frame.iloc[0]).values():
        assert len(str(value)) <= 1000


# ------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------
def test_overall_percent_is_risk_share_not_mean_score(burnout_frame, statistics):
    at_risk = len(burnout_frame[burnout_frame["risk_level"].isin(["Medium", "High"])])
    expected = round(at_risk / len(burnout_frame) * 100, 1)
    assert statistics["overall_burnout_percent"] == expected


def test_zero_count_risk_levels_are_explicit_not_missing(statistics):
    """`High: 0` is information. A missing key would leave a chart gap."""
    for level in ("Low", "Medium", "High"):
        assert level in statistics["risk_distribution"]


def test_empty_frame_does_not_divide_by_zero(burnout_frame):
    stats = compute_statistics(burnout_frame.head(0))
    assert stats["total_records"] == 0
    assert stats["overall_burnout_percent"] == 0.0


def test_burnout_percent_stays_within_bounds(burnout_frame):
    assert burnout_frame["burnout_percent"].between(0, 100).all()


# ------------------------------------------------------------------------------
# Search
# ------------------------------------------------------------------------------
def test_search_is_literal_not_regex(burnout_frame):
    """A user typing `C++` or `(a|b)` must not trigger a regex error."""
    from backend.data_loader import search_records

    assert search_records("C++ (a|b)*", burnout_frame) == []


def test_search_matches_exact_employee_id(burnout_frame):
    from backend.data_loader import search_records

    assert len(search_records("EMP_0001", burnout_frame)) == 1


# ------------------------------------------------------------------------------
# Caching
# ------------------------------------------------------------------------------
def test_cache_invalidates_when_person2_reruns_the_pipeline():
    """A new mtime must be picked up without a backend restart."""
    import os

    from backend.data_loader import clear_cache, load_burnout_dataset

    clear_cache()
    first = load_burnout_dataset()
    assert load_burnout_dataset() is first

    stat = first.source_path.stat()
    os.utime(first.source_path, (stat.st_atime, stat.st_mtime + 10))
    try:
        assert load_burnout_dataset() is not first
    finally:
        os.utime(first.source_path, (stat.st_atime, stat.st_mtime))


def test_concurrent_first_loads_share_one_object():
    import threading

    from backend.data_loader import clear_cache, load_burnout_dataset

    clear_cache()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(load_burnout_dataset()))
        for _ in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(r) for r in results}) == 1


# ------------------------------------------------------------------------------
# Missing data
# ------------------------------------------------------------------------------
def test_missing_csv_names_the_command_to_regenerate_it(monkeypatch):
    from backend.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        type(settings), "burnout_csv",
        property(lambda self: __import__("pathlib").Path("/nonexistent/x.csv")),
    )
    with pytest.raises(FileNotFoundError) as exc:
        settings.ensure_data_available()
    assert "burnout_score.py" in str(exc.value)
