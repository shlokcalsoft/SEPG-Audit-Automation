"""
tests/test_pipeline_integration.py
====================================
Integration-level tests for the pipeline.
These are skipped when the real transcript is not available.
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

REAL_TRANSCRIPT = Path("data/sample_transcript.docx")

@pytest.mark.skipif(not REAL_TRANSCRIPT.exists(), reason="Real transcript not yet provided")
def test_full_pipeline_smoke():
    """
    Smoke test: pipeline runs end-to-end without crashing.
    Does NOT check verdict quality — just structural validity.
    """
    from src.pipeline import main
    scorecard_path, log_path, flags = main()

    assert Path(scorecard_path).exists(), "Scorecard file not created"
    assert Path(log_path).exists(), "Review log file not created"

    data = json.loads(Path(log_path).read_text(encoding="utf-8"))
    valid_reasons = {"not_clarified", "low_confidence", "evidence_mismatch", "batch_failed_twice"}
    for entry in data:
        assert entry.get("flag_reason") in valid_reasons, (
            f"Invalid flag_reason: {entry.get('flag_reason')}"
        )
