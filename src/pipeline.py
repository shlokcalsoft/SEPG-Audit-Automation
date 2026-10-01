"""
pipeline.py
===========
Orchestrator for the SEPG Audit Automation system.

Execution order:
  1. Load config.yaml + .env
  2. Extract checkpoints from template → checkpoint_registry.json
  3. Parse transcript → parsed_transcript.json
  4. Retrieve evidence for all 248 checkpoints → retrieval_results.json
  5. Group checkpoints by Activity (43 groups)
  6. For each Activity batch:
     a. Build prompt
     b. Choose model (routing)
     c. Rate-limit gate → Groq call
     d. Validate output; retry once on failure; flag if still failing
     e. Collect verdicts
  7. Write scorecard (only L and Q columns)
  8. Write review log
  9. Print run summary
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.assessor import CheckpointVerdict, assess_batch
from src.excel_writer import build_write_instructions, write_scorecard
from src.rate_limiter import GroqRateLimiter
from src.retriever import retrieve_evidence
from src.review_log import build_review_log
from src.template_reader import extract_checkpoints
from src.transcript_parser import parse_transcript

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    run_start = time.monotonic()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info("SEPG Audit Automation — Run ID: %s", run_id)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------------------------
    load_dotenv()
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2. Extract checkpoints
    # -----------------------------------------------------------------------
    logger.info("Step 2/9 — Extracting checkpoints from template …")
    checkpoints = extract_checkpoints(config, output_path="checkpoint_registry.json")
    registry = {c["checkpoint_id"]: c for c in checkpoints}
    logger.info("  %d checkpoints extracted.", len(checkpoints))

    # -----------------------------------------------------------------------
    # 3. Parse transcript
    # -----------------------------------------------------------------------
    logger.info("Step 3/9 — Parsing transcript …")
    segments = parse_transcript(config, output_path="parsed_transcript.json")
    logger.info("  %d segments parsed.", len(segments))

    # -----------------------------------------------------------------------
    # 4. Retrieve evidence
    # -----------------------------------------------------------------------
    logger.info("Step 4/9 — Retrieving evidence for %d checkpoints …", len(checkpoints))
    retrieval_results = retrieve_evidence(
        checkpoints, segments, config,
        output_path="retrieval_results.json"
    )
    retrieval_map = {r["checkpoint_id"]: r for r in retrieval_results}
    logger.info("  Retrieval complete.")

    # -----------------------------------------------------------------------
    # 5. Group by Activity
    # -----------------------------------------------------------------------
    logger.info("Step 5/9 — Grouping checkpoints by Activity …")
    # Preserve order as encountered
    activity_groups: dict[str, list[dict]] = defaultdict(list)
    for chk in checkpoints:
        activity_groups[chk["activity"]].append(chk)
    total_batches = len(activity_groups)
    logger.info("  %d activity groups.", total_batches)

    # -----------------------------------------------------------------------
    # 6. LLM assessment — one Groq call per Activity
    # -----------------------------------------------------------------------
    logger.info("Step 6/9 — Running LLM assessment (%d batches) …", total_batches)
    rate_limiter = GroqRateLimiter(config)

    all_verdicts: list[CheckpointVerdict] = []
    failed_batches: list[dict] = []
    escalation_count = 0
    total_groq_calls = 0

    for batch_idx, (activity_name, batch_checkpoints) in enumerate(activity_groups.items(), start=1):
        # Build enriched batch payload
        batch_payload = []
        batch_ret_confs = []
        for chk in batch_checkpoints:
            ret = retrieval_map.get(chk["checkpoint_id"], {})
            conf = ret.get("confidence", 0.0)
            batch_ret_confs.append(conf)
            batch_payload.append({
                "checkpoint_id":    chk["checkpoint_id"],
                "checkpoint_text":  chk["checkpoint_text"],
                "weight":           chk["weight"],
                "confidence":       conf,
                "retrieved_excerpts": ret.get("retrieved_excerpts", []),
            })

        # Track escalations
        from src.assessor import choose_model
        chosen_model = choose_model(
            batch_ret_confs,
            config["groq"]["escalation_confidence_threshold"],
            config
        )
        if chosen_model == config["groq"]["escalation_model"]:
            escalation_count += 1

        logger.info(
            "Batch %d/%d — '%s' (%d checkpoints, model=%s) …",
            batch_idx, total_batches, activity_name, len(batch_checkpoints), chosen_model
        )

        verdicts, batch_failed = assess_batch(
            activity_name, batch_payload, config, rate_limiter
        )
        total_groq_calls += 1

        if batch_failed:
            failed_batches.append({
                "activity":      activity_name,
                "checkpoint_ids": [c["checkpoint_id"] for c in batch_checkpoints],
            })
            logger.error("  FAILED — activity '%s'. Stopping early as requested.", activity_name)
            break
        else:
            all_verdicts.extend(verdicts)
            logger.info("  OK — %d verdicts received.", len(verdicts))
        
        if batch_idx == 2:
            logger.info("Stopping after 2 batches to provide partial Excel quickly.")
            break

    logger.info(
        "Assessment complete: %d verdicts from %d batches (%d failed, %d escalated to 70B).",
        len(all_verdicts), total_batches, len(failed_batches), escalation_count
    )

    # -----------------------------------------------------------------------
    # 7. Write scorecard
    # -----------------------------------------------------------------------
    logger.info("Step 7/9 — Writing scorecard …")
    scorecard_path = str(output_dir / f"{run_id}_scorecard.xlsx")
    instructions = build_write_instructions(all_verdicts, registry)
    write_scorecard(config["template"]["path"], instructions, scorecard_path)
    logger.info("  Scorecard: %s", scorecard_path)

    # -----------------------------------------------------------------------
    # 8. Write review log
    # -----------------------------------------------------------------------
    logger.info("Step 8/9 — Writing review log …")
    log_path = str(output_dir / f"{run_id}_review_log.json")
    flags = build_review_log(
        all_verdicts, retrieval_results, failed_batches, registry, config,
        output_path=log_path
    )
    logger.info("  Review log: %s (%d flags)", log_path, len(flags))

    # -----------------------------------------------------------------------
    # 9. Run summary
    # -----------------------------------------------------------------------
    from collections import Counter
    verdict_counts = Counter(v.verdict for v in all_verdicts)
    failed_chk_count = sum(len(b["checkpoint_ids"]) for b in failed_batches)
    escalation_pct = escalation_count / total_batches * 100 if total_batches else 0
    elapsed = time.monotonic() - run_start
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    flag_counts = Counter(f["flag_reason"] for f in flags)

    summary = f"""
============ Run Summary ============
Run ID:                 {run_id}
Total checkpoints:      {len(checkpoints)}

Verdicts:
  Yes:                  {verdict_counts.get("Yes", 0)}
  Partial:              {verdict_counts.get("Partial", 0)}
  In Progress:          {verdict_counts.get("In Progress", 0)}
  Not clarified:        {verdict_counts.get("Not clarified", 0)}
  No:                   {verdict_counts.get("No", 0)}
  NA:                   {verdict_counts.get("NA", 0)}
  Failed (no verdict):  {failed_chk_count}

Flagged for review:     {len(flags)}
  not_clarified:        {flag_counts.get("not_clarified", 0)}
  low_confidence:       {flag_counts.get("low_confidence", 0)}
  evidence_mismatch:    {flag_counts.get("evidence_mismatch", 0)}
  batch_failed_twice:   {flag_counts.get("batch_failed_twice", 0)}

70B escalations:        {escalation_count} / {total_batches} batches ({escalation_pct:.1f}%)
Total Groq calls:       {total_groq_calls}
Total run time:         {elapsed_str}

Output:
  Scorecard:            {scorecard_path}
  Review log:           {log_path}
=====================================
"""
    print(summary)
    logger.info("Step 9/9 — Run complete.")
    return scorecard_path, log_path, flags


if __name__ == "__main__":
    main()
