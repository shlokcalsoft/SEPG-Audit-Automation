"""
tests/test_template_reader.py
============================
Unit tests for template_reader.extract_checkpoints().
All tests run against the real template at data/template.xlsx.
"""
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.template_reader import extract_checkpoints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def checkpoints(config, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("out") / "registry.json"
    return extract_checkpoints(config, output_path=str(tmp))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_correct_count(checkpoints):
    assert len(checkpoints) == 248, f"Expected 248 checkpoints, got {len(checkpoints)}"


def test_unique_ids(checkpoints):
    ids = [c["checkpoint_id"] for c in checkpoints]
    assert len(ids) == len(set(ids)), "Duplicate checkpoint_ids detected"


def test_no_empty_checkpoint_text(checkpoints):
    bad = [c for c in checkpoints if not c.get("checkpoint_text", "").strip()]
    assert not bad, f"Entries with empty checkpoint_text: {[c['checkpoint_id'] for c in bad]}"


def test_all_weights_numeric_and_in_range(checkpoints):
    for c in checkpoints:
        w = c["weight"]
        assert isinstance(w, (int, float)), f"{c['checkpoint_id']} weight is not numeric: {w!r}"
        assert 1 <= w <= 5, f"{c['checkpoint_id']} weight {w} out of range 1-5"


def test_excel_cell_format_column_L(checkpoints):
    pattern = re.compile(r"^L\d+$")
    bad = [c for c in checkpoints if not pattern.match(c.get("excel_cell", ""))]
    assert not bad, f"Entries with wrong excel_cell format: {[c['checkpoint_id'] for c in bad[:5]]}"


def test_observation_cell_format_column_Q(checkpoints):
    pattern = re.compile(r"^Q\d+$")
    bad = [c for c in checkpoints if not pattern.match(c.get("observation_cell", ""))]
    assert not bad, f"Entries with wrong observation_cell format: {[c['checkpoint_id'] for c in bad[:5]]}"


def test_forward_fill_no_blanks(checkpoints):
    for c in checkpoints:
        assert c.get("phase_name"), f"{c['checkpoint_id']} has blank phase_name"
        assert c.get("phase_number"), f"{c['checkpoint_id']} has blank phase_number"
        assert c.get("activity"), f"{c['checkpoint_id']} has blank activity"


def test_all_have_correct_sheet(checkpoints):
    for c in checkpoints:
        assert c["excel_sheet"] == "Q2 2026", f"{c['checkpoint_id']} has wrong sheet: {c['excel_sheet']}"


# --- Spot-check entries (discovered from real template inspection) ---

def _find(checkpoints, checkpoint_id):
    matches = [c for c in checkpoints if c["checkpoint_id"] == checkpoint_id]
    assert matches, f"checkpoint_id '{checkpoint_id}' not found"
    return matches[0]


def test_spot_check_row_12(checkpoints):
    c = _find(checkpoints, "Q2_2026_12")
    assert "SOW" in c["checkpoint_text"] or "sow" in c["checkpoint_text"].lower(), \
        f"Row 12 text unexpected: {c['checkpoint_text']}"
    assert c["weight"] == 5
    assert c["excel_cell"] == "L12"
    assert c["observation_cell"] == "Q12"
    assert c["activity"] == "SOW"


def test_spot_check_row_25(checkpoints):
    c = _find(checkpoints, "Q2_2026_25")
    assert "methodology" in c["checkpoint_text"].lower() or "execution" in c["checkpoint_text"].lower()
    assert c["weight"] == 4
    assert c["excel_cell"] == "L25"
    assert c["observation_cell"] == "Q25"


def test_spot_check_row_48(checkpoints):
    c = _find(checkpoints, "Q2_2026_48")
    assert "kick" in c["checkpoint_text"].lower()
    assert c["weight"] == 5
    assert c["excel_cell"] == "L48"
    assert c["activity"] == "Kick-off"


def test_spot_check_row_146(checkpoints):
    c = _find(checkpoints, "Q2_2026_146")
    assert "test" in c["checkpoint_text"].lower()
    assert c["weight"] == 5
    assert c["excel_cell"] == "L146"


def test_spot_check_row_293(checkpoints):
    c = _find(checkpoints, "Q2_2026_293")
    assert "closure" in c["checkpoint_text"].lower() or "signed" in c["checkpoint_text"].lower()
    assert c["weight"] == 5
    assert c["excel_cell"] == "L293"


def test_no_boundary_rows_included(checkpoints):
    # Known boundary rows (have N formula but no D text)
    boundary_rows = {23, 28, 37, 47, 59, 70, 84, 94, 106, 112, 120, 124, 138,
                     154, 162, 169, 172, 178, 185, 192}
    ids_in_result = {c["checkpoint_id"] for c in checkpoints}
    for row in boundary_rows:
        cid = f"Q2_2026_{row}"
        assert cid not in ids_in_result, f"Boundary row {row} was incorrectly included"


def test_activity_count(checkpoints):
    activities = {c["activity"] for c in checkpoints}
    assert len(activities) == 43, f"Expected 43 distinct activities, got {len(activities)}"


def test_phase_count(checkpoints):
    phases = {c["phase_number"] for c in checkpoints}
    assert len(phases) == 23, f"Expected 23 distinct phase numbers, got {len(phases)}"


def test_last_checkpoint(checkpoints):
    last = checkpoints[-1]
    # Last checkpoint should be in row 299 (SST — Project Specific Checkpoints)
    assert last["checkpoint_id"] == "Q2_2026_299"
    assert last["excel_cell"] == "L299"
    assert last["observation_cell"] == "Q299"
