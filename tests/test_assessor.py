"""
tests/test_assessor.py
======================
Unit tests for assessor.py — all tests mock the Groq client.
No real API calls are made here.
"""
import json
import sys
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.assessor import (
    CheckpointVerdict,
    assess_batch,
    build_system_prompt,
    build_user_prompt,
    choose_model,
)
from src.rate_limiter import GroqRateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


@pytest.fixture
def limiter(config):
    return GroqRateLimiter(config)


def _mock_groq_response(json_list: list[dict]) -> MagicMock:
    """Create a mock Groq API response returning the given JSON array."""
    response = MagicMock()
    response.choices[0].message.content = json.dumps(json_list)
    response.usage.total_tokens = 500
    return response


# ---------------------------------------------------------------------------
# Model routing tests
# ---------------------------------------------------------------------------

def test_choose_model_below_threshold(config):
    """Low avg confidence → escalate to 70B."""
    model = choose_model([0.1, 0.2, 0.3], threshold=0.35, config=config)
    assert model == config["groq"]["escalation_model"]


def test_choose_model_above_threshold(config):
    """High avg confidence → use default 8B model."""
    model = choose_model([0.5, 0.6, 0.7], threshold=0.35, config=config)
    assert model == config["groq"]["default_model"]


def test_choose_model_at_exact_threshold(config):
    """avg == threshold → use default model (not < threshold)."""
    model = choose_model([0.35], threshold=0.35, config=config)
    assert model == config["groq"]["default_model"]


def test_choose_model_empty_confidences(config):
    """Empty list → use default model."""
    model = choose_model([], threshold=0.35, config=config)
    assert model == config["groq"]["default_model"]


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

def test_system_prompt_has_all_six_verdicts():
    prompt = build_system_prompt()
    for verdict in ["Yes", "Partial", "In Progress", "Not clarified", "No", "NA"]:
        assert verdict in prompt, f"Verdict '{verdict}' missing from system prompt"


def test_system_prompt_has_observation_format():
    prompt = build_system_prompt()
    assert "[HH:MM:SS | Speaker]" in prompt


def test_system_prompt_lowercase_not_clarified():
    """Ensure 'Not clarified' has lowercase 'c' (matches template formula)."""
    prompt = build_system_prompt()
    assert '"Not clarified"' in prompt
    assert '"Not Clarified"' not in prompt


# ---------------------------------------------------------------------------
# User prompt tests
# ---------------------------------------------------------------------------

def test_user_prompt_contains_activity():
    batch = [{"checkpoint_id": "Q2_2026_12", "checkpoint_text": "SOW signed", "weight": 5, "confidence": 0.8, "retrieved_excerpts": ["SOW was signed."]}]
    prompt = build_user_prompt("SOW", batch)
    assert "SOW" in prompt
    assert "Q2_2026_12" in prompt


def test_user_prompt_contains_excerpts():
    batch = [{"checkpoint_id": "Q2_2026_12", "checkpoint_text": "SOW signed", "weight": 5, "confidence": 0.8, "retrieved_excerpts": ["The SOW was reviewed."]}]
    prompt = build_user_prompt("SOW", batch)
    assert "The SOW was reviewed." in prompt


# ---------------------------------------------------------------------------
# Pydantic validation tests
# ---------------------------------------------------------------------------

def test_valid_verdict_accepted():
    for v in ["Yes", "Partial", "In Progress", "Not clarified", "No", "NA"]:
        cv = CheckpointVerdict(checkpoint_id="X", verdict=v, confidence=0.5, observation="test")
        assert cv.verdict == v


def test_invalid_verdict_rejected():
    with pytest.raises(Exception):  # pydantic ValidationError
        CheckpointVerdict(checkpoint_id="X", verdict="Maybe", confidence=0.5, observation="test")


def test_not_clarified_lowercase_accepted():
    """'Not clarified' (lowercase c) must be valid."""
    cv = CheckpointVerdict(checkpoint_id="X", verdict="Not clarified", confidence=0.3, observation="")
    assert cv.verdict == "Not clarified"


def test_not_clarified_uppercase_rejected():
    """'Not Clarified' (capital C) must be REJECTED — it doesn't match template formula."""
    with pytest.raises(Exception):
        CheckpointVerdict(checkpoint_id="X", verdict="Not Clarified", confidence=0.3, observation="")


def test_in_progress_accepted():
    cv = CheckpointVerdict(checkpoint_id="X", verdict="In Progress", confidence=0.6, observation="obs")
    assert cv.verdict == "In Progress"


# ---------------------------------------------------------------------------
# assess_batch tests (mocked Groq client)
# ---------------------------------------------------------------------------

SAMPLE_BATCH = [
    {
        "checkpoint_id": "Q2_2026_12",
        "checkpoint_text": "Signed SOW is available",
        "weight": 5,
        "confidence": 0.80,
        "retrieved_excerpts": ["SOW was signed by the client."],
    },
    {
        "checkpoint_id": "Q2_2026_13",
        "checkpoint_text": "Approved SOW is available",
        "weight": 5,
        "confidence": 0.75,
        "retrieved_excerpts": ["The SOW was approved."],
    },
]

SAMPLE_RESPONSE = [
    {"checkpoint_id": "Q2_2026_12", "verdict": "Yes", "confidence": 0.9, "observation": "[00:05:23 | PM] SOW was signed."},
    {"checkpoint_id": "Q2_2026_13", "verdict": "Yes", "confidence": 0.85, "observation": "[00:05:45 | PM] SOW was approved."},
]


def test_assess_batch_success(config, limiter):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(SAMPLE_RESPONSE)

    verdicts, failed = assess_batch("SOW", SAMPLE_BATCH, config, limiter, groq_client=mock_client)

    assert not failed
    assert len(verdicts) == 2
    assert verdicts[0].verdict == "Yes"
    assert verdicts[0].checkpoint_id == "Q2_2026_12"
    assert "[00:05:23 | PM]" in verdicts[0].observation


def test_assess_batch_na_verdict_included(config, limiter):
    """NA verdicts come back from the assessor — filtering is excel_writer's job."""
    na_response = [
        {"checkpoint_id": "Q2_2026_12", "verdict": "NA", "confidence": 0.0, "observation": ""},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(na_response)

    batch = [SAMPLE_BATCH[0]]
    verdicts, failed = assess_batch("SOW", batch, config, limiter, groq_client=mock_client)

    assert not failed
    assert verdicts[0].verdict == "NA"


def test_assess_batch_malformed_json_triggers_retry(config, limiter):
    """Malformed JSON should trigger exactly one retry, then flag."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "NOT VALID JSON {{{"
    mock_client.chat.completions.create.return_value.usage.total_tokens = 100

    verdicts, failed = assess_batch("SOW", SAMPLE_BATCH, config, limiter, groq_client=mock_client)

    assert failed is True
    assert verdicts == []
    # Should have been called exactly 2 times (initial + 1 retry)
    assert mock_client.chat.completions.create.call_count == 2


def test_assess_batch_invalid_verdict_triggers_retry(config, limiter):
    """Invalid verdict string should cause Pydantic failure → retry → flag."""
    bad_response = [
        {"checkpoint_id": "Q2_2026_12", "verdict": "UNKNOWN", "confidence": 0.5, "observation": "x"},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(bad_response)

    verdicts, failed = assess_batch("SOW", [SAMPLE_BATCH[0]], config, limiter, groq_client=mock_client)

    assert failed is True
    assert mock_client.chat.completions.create.call_count == 2


def test_assess_batch_no_crash_on_double_failure(config, limiter):
    """Double failure must not raise — returns ([], True) cleanly."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API down")

    verdicts, failed = assess_batch("SOW", SAMPLE_BATCH, config, limiter, groq_client=mock_client)

    assert failed is True
    assert verdicts == []
