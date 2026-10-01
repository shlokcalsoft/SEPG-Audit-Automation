"""
review_log.py
=============
Generates the human-review log flagging checkpoints that need attention.

Flag conditions:
  1. verdict == "Not clarified"         → flag_reason: "not_clarified"
  2. confidence < flag_confidence_threshold → flag_reason: "low_confidence"
  3. retriever confidence low BUT model returned decisive "Yes"/"No"
                                         → flag_reason: "evidence_mismatch"
  4. Whole batch failed twice           → flag_reason: "batch_failed_twice"

A single checkpoint can appear multiple times (once per flag reason).
Each run produces its own log file — logs are never merged automatically.
"""

import json
import logging
from pathlib import Path

from src.assessor import CheckpointVerdict

logger = logging.getLogger(__name__)

# Threshold below which retriever confidence is "low" for evidence_mismatch detection
_MISMATCH_RETRIEVER_THRESHOLD = 0.25
# Verdicts considered "decisive" for evidence_mismatch detection
_DECISIVE_VERDICTS = {"Yes", "No"}


def build_review_log(
    verdicts: list[CheckpointVerdict],
    retrieval_results: list[dict],
    failed_batches: list[dict],           # list of {activity, checkpoint_ids}
    registry: dict[str, dict],            # checkpoint_id -> checkpoint dict
    config: dict,
    output_path: str,
) -> list[dict]:
    """
    Build the review log for a completed pipeline run.

    Args:
        verdicts:         All CheckpointVerdict objects from the run.
        retrieval_results: List of retrieval dicts (checkpoint_id, confidence, …).
        failed_batches:   Batches that failed twice — each has 'activity' and 'checkpoint_ids'.
        registry:         Checkpoint registry dict, keyed by checkpoint_id.
        config:           Loaded config.yaml.
        output_path:      Where to write the review log JSON.

    Returns:
        List of flag dicts.
    """
    flag_threshold = float(config["review"]["flag_confidence_threshold"])

    # Build a lookup: checkpoint_id -> retriever confidence
    retriever_conf: dict[str, float] = {
        r["checkpoint_id"]: r["confidence"] for r in retrieval_results
    }

    flags: list[dict] = []

    # -----------------------------------------------------------------------
    # Flag verdicts
    # -----------------------------------------------------------------------
    for v in verdicts:
        entry = registry.get(v.checkpoint_id, {})
        activity = entry.get("activity", "Unknown")
        ret_conf = retriever_conf.get(v.checkpoint_id, 0.0)

        base = {
            "checkpoint_id":  v.checkpoint_id,
            "activity":       activity,
            "verdict":        v.verdict,
            "confidence":     v.confidence,
            "observation":    v.observation,
        }

        # Condition 1: Not clarified
        if v.verdict == "Not clarified":
            flags.append({**base, "flag_reason": "not_clarified"})

        # Condition 2: Low model confidence
        if v.confidence < flag_threshold:
            flags.append({**base, "flag_reason": "low_confidence"})

        # Condition 3: Evidence mismatch (decisive verdict, low retriever confidence)
        if (
            v.verdict in _DECISIVE_VERDICTS
            and ret_conf < _MISMATCH_RETRIEVER_THRESHOLD
        ):
            flags.append({**base, "flag_reason": "evidence_mismatch"})

    # -----------------------------------------------------------------------
    # Flag failed batches
    # -----------------------------------------------------------------------
    for batch in failed_batches:
        activity    = batch.get("activity", "Unknown")
        chk_ids     = batch.get("checkpoint_ids", [])
        for cid in chk_ids:
            entry = registry.get(cid, {})
            flags.append({
                "checkpoint_id": cid,
                "activity":      activity,
                "verdict":       None,
                "confidence":    None,
                "observation":   None,
                "flag_reason":   "batch_failed_twice",
            })

    logger.info(
        "Review log: %d flags (%d checkpoints reviewed, %d failed batches)",
        len(flags),
        len(verdicts),
        len(failed_batches),
    )

    # Summarise by reason
    from collections import Counter
    reason_counts = Counter(f["flag_reason"] for f in flags)
    for reason, count in sorted(reason_counts.items()):
        logger.info("  %s: %d", reason, count)

    # Write to disk
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Review log written to %s", output_path)

    return flags
