"""
tests/test_transcript_parser.py
================================
Unit tests for transcript_parser.parse_transcript().

Tests that can run WITHOUT a real transcript file use synthetic DOCX fixtures.
Tests requiring the real transcript are marked and skipped if the file is absent.
"""
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from docx import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.transcript_parser import parse_transcript

# ---------------------------------------------------------------------------
# Helpers to build a synthetic DOCX in memory
# ---------------------------------------------------------------------------

def _make_docx(paragraphs: list[str]) -> str:
    """Write paragraphs to a temp .docx and return the file path."""
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    doc.save(tmp.name)
    tmp.close()
    return tmp.name


def _base_config(docx_path: str) -> dict:
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["transcript"]["path"] = docx_path
    return cfg


# ---------------------------------------------------------------------------
# Synthetic tests (no real transcript needed)
# ---------------------------------------------------------------------------

def test_segment_ids_sequential():
    path = _make_docx([
        "Alice: Hello everyone.",
        "Bob: Good morning.",
        "Alice: Let us begin.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    ids = [s["segment_id"] for s in segs]
    assert ids == list(range(len(segs))), f"Non-sequential IDs: {ids}"


def test_no_empty_segments():
    path = _make_docx([
        "Alice: First speaker.",
        "",  # blank — should be skipped
        "Bob: Second speaker.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    bad = [s for s in segs if not s["text"].strip()]
    assert not bad, "Segments with empty text found"


def test_speaker_attribution():
    path = _make_docx([
        "Alice: First line.",
        "Bob: Second line.",
        "Alice: Third line.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    speakers = [s["speaker"] for s in segs]
    assert "Alice" in speakers
    assert "Bob" in speakers


def test_multi_paragraph_turn_merges():
    """A paragraph without a speaker label should merge into the previous turn."""
    path = _make_docx([
        "Alice: First line.",
        "This is a continuation of Alice's turn.",
        "Bob: New speaker.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    # Should produce 2 segments (Alice + Bob), not 3
    assert len(segs) == 2, f"Expected 2 segments (multi-para merge), got {len(segs)}"
    alice_seg = segs[0]
    assert "continuation" in alice_seg["text"]


def test_timestamp_extracted():
    """Timestamps in [HH:MM:SS] format should populate the timestamp field."""
    path = _make_docx([
        "[00:01:30] Alice: First line with timestamp.",
        "[00:02:45] Bob: Second line.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    assert segs[0]["timestamp"] == "00:01:30", f"Got {segs[0]['timestamp']}"
    assert segs[1]["timestamp"] == "00:02:45", f"Got {segs[1]['timestamp']}"


def test_no_timestamp_defaults_empty():
    path = _make_docx(["Alice: No timestamp here."])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    assert segs[0]["timestamp"] == "", f"Expected empty timestamp, got {segs[0]['timestamp']!r}"


def test_fallback_no_speaker_labels():
    """If no speaker labels exist, fall back to one-segment-per-paragraph."""
    path = _make_docx([
        "This is just a paragraph with no speaker label.",
        "This is another paragraph.",
        "And a third one.",
    ])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    assert len(segs) == 3
    for s in segs:
        assert s["speaker"] is None


def test_output_written_to_file():
    path = _make_docx(["Alice: Testing output file write."])
    cfg = _base_config(path)
    out = tempfile.mktemp(suffix=".json")
    parse_transcript(cfg, output_path=out)
    data = json.loads(Path(out).read_text(encoding="utf-8"))
    assert len(data) >= 1


def test_segment_has_required_keys():
    path = _make_docx(["Alice: Checking all keys are present."])
    cfg = _base_config(path)
    segs = parse_transcript(cfg, output_path=tempfile.mktemp(suffix=".json"))
    required = {"segment_id", "timestamp", "speaker", "text"}
    for seg in segs:
        assert required.issubset(seg.keys()), f"Missing keys in segment: {seg}"


# ---------------------------------------------------------------------------
# Real-transcript test (skipped if file not present)
# ---------------------------------------------------------------------------

REAL_TRANSCRIPT = Path("data/sample_transcript.docx")

@pytest.mark.skipif(not REAL_TRANSCRIPT.exists(), reason="Real transcript not yet provided")
def test_real_transcript_segment_count():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = tempfile.mktemp(suffix=".json")
    segs = parse_transcript(cfg, output_path=out)
    # Should have a reasonable number of segments (at least 10)
    assert len(segs) >= 10, f"Too few segments from real transcript: {len(segs)}"

@pytest.mark.skipif(not REAL_TRANSCRIPT.exists(), reason="Real transcript not yet provided")
def test_real_transcript_majority_speakers():
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = tempfile.mktemp(suffix=".json")
    segs = parse_transcript(cfg, output_path=out)
    speaker_count = sum(1 for s in segs if s["speaker"] is not None)
    ratio = speaker_count / len(segs) if segs else 0
    # If the transcript has speaker labels, at least 60% of segments should have one
    assert ratio >= 0.60 or speaker_count == 0, (
        f"Low speaker attribution: {speaker_count}/{len(segs)} = {ratio:.1%}"
    )
