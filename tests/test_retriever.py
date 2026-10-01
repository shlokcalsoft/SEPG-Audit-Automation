"""
tests/test_retriever.py
=======================
Unit tests for retriever.retrieve_evidence().
Uses a small synthetic corpus to test scoring, window expansion, and deduplication.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retriever import (
    _expand_window,
    _keyword_overlap,
    _tokenize,
    retrieve_evidence,
)

# ---------------------------------------------------------------------------
# Small synthetic data
# ---------------------------------------------------------------------------

SEGMENTS = [
    {"segment_id": 0, "timestamp": "00:01:00", "speaker": "PM", "text": "The SOW was reviewed and signed by all parties."},
    {"segment_id": 1, "timestamp": "00:02:00", "speaker": "PM", "text": "We confirmed the project kick-off meeting was conducted last Monday."},
    {"segment_id": 2, "timestamp": "00:03:00", "speaker": "Dev", "text": "Sprint planning was done and all stories are estimated."},
    {"segment_id": 3, "timestamp": "00:04:00", "speaker": "Dev", "text": "Unit testing and integration testing are complete."},
    {"segment_id": 4, "timestamp": "00:05:00", "speaker": "QA",  "text": "Test cases have been executed and defects logged."},
    {"segment_id": 5, "timestamp": "00:06:00", "speaker": "PM", "text": "The closure report was signed off by the client."},
    {"segment_id": 6, "timestamp": "00:07:00", "speaker": "PM", "text": "Risk register was updated and shared with stakeholders."},
    {"segment_id": 7, "timestamp": "00:08:00", "speaker": "Dev", "text": "Code review process is followed for all pull requests."},
    {"segment_id": 8, "timestamp": "00:09:00", "speaker": "QA",  "text": "UAT was completed and sign-off received from the client."},
    {"segment_id": 9, "timestamp": "00:10:00", "speaker": "PM", "text": "Invoice has been raised and payment tracking is up to date."},
]

CHECKPOINTS = [
    {"checkpoint_id": "Q2_2026_12",  "checkpoint_text": "Signed SOW is available",        "weight": 5, "activity": "SOW"},
    {"checkpoint_id": "Q2_2026_48",  "checkpoint_text": "Kick-off conducted for project", "weight": 5, "activity": "Kick-off"},
    {"checkpoint_id": "Q2_2026_146", "checkpoint_text": "Test cases executed",            "weight": 5, "activity": "Testing"},
]

BASE_CONFIG = {
    "retrieval": {
        "tfidf_weight": 0.70,
        "keyword_weight": 0.30,
        "top_k": 3,
        "window_radius": 1,
    }
}


@pytest.fixture(scope="module")
def results():
    out = tempfile.mktemp(suffix=".json")
    return retrieve_evidence(CHECKPOINTS, SEGMENTS, BASE_CONFIG, output_path=out)


# ---------------------------------------------------------------------------
# Tests — scoring primitives
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("The SOW was reviewed and signed.")
    assert "sow" in tokens
    assert "reviewed" in tokens
    assert "the" not in tokens  # stopword removed


def test_tokenize_punctuation_stripped():
    tokens = _tokenize("kick-off, signed; tested.")
    # Punctuation stripped, tokens lowercased
    assert all(t.isalpha() or "-" not in t for t in tokens)


def test_keyword_overlap_full():
    a = frozenset({"test", "cases", "executed"})
    b = frozenset({"test", "cases", "executed", "defects"})
    assert _keyword_overlap(a, b) == 1.0


def test_keyword_overlap_partial():
    a = frozenset({"test", "cases", "executed"})
    b = frozenset({"test", "defects"})
    assert abs(_keyword_overlap(a, b) - 1/3) < 1e-9


def test_keyword_overlap_empty_checkpoint():
    assert _keyword_overlap(frozenset(), frozenset({"anything"})) == 0.0


# ---------------------------------------------------------------------------
# Tests — window expansion
# ---------------------------------------------------------------------------

def test_expand_window_center():
    r = list(_expand_window(5, 10, 2))
    assert r == [3, 4, 5, 6, 7]


def test_expand_window_at_start():
    r = list(_expand_window(0, 10, 2))
    assert r == [0, 1, 2]


def test_expand_window_at_end():
    r = list(_expand_window(9, 10, 2))
    assert r == [7, 8, 9]


def test_expand_window_radius_zero():
    r = list(_expand_window(5, 10, 0))
    assert r == [5]


# ---------------------------------------------------------------------------
# Tests — retrieve_evidence output
# ---------------------------------------------------------------------------

def test_output_count_matches_checkpoints(results):
    assert len(results) == len(CHECKPOINTS)


def test_all_have_excerpts(results):
    for r in results:
        assert r["retrieved_excerpts"], f"{r['checkpoint_id']} has no excerpts"


def test_confidence_in_range(results):
    for r in results:
        assert 0.0 <= r["confidence"] <= 1.0, (
            f"{r['checkpoint_id']} confidence {r['confidence']} out of [0,1]"
        )


def test_excerpts_in_original_order(results):
    for r in results:
        indices = r["window_indices"]
        assert indices == sorted(indices), f"{r['checkpoint_id']} excerpts not in order"


def test_sow_checkpoint_finds_sow_segment(results):
    """The SOW checkpoint should retrieve the segment mentioning SOW."""
    sow_result = next(r for r in results if r["checkpoint_id"] == "Q2_2026_12")
    all_excerpts = " ".join(sow_result["retrieved_excerpts"]).lower()
    assert "sow" in all_excerpts or "statement" in all_excerpts


def test_testing_checkpoint_finds_test_segment(results):
    """The testing checkpoint should retrieve the segment about test execution."""
    test_result = next(r for r in results if r["checkpoint_id"] == "Q2_2026_146")
    all_excerpts = " ".join(test_result["retrieved_excerpts"]).lower()
    assert "test" in all_excerpts


def test_no_duplicate_excerpts(results):
    for r in results:
        assert len(r["window_indices"]) == len(set(r["window_indices"])), (
            f"{r['checkpoint_id']} has duplicate window indices"
        )


def test_output_written_to_file():
    out = tempfile.mktemp(suffix=".json")
    retrieve_evidence(CHECKPOINTS, SEGMENTS, BASE_CONFIG, output_path=out)
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert len(data) == 3
