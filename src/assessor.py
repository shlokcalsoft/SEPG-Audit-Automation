"""
assessor.py
===========
Calls the Groq API to assess compliance verdicts for batches of checkpoints.

Design:
  - One Activity group = one Groq call (batch unit).
  - Model routing: default llama-3.1-8b-instant; escalate to llama-3.3-70b-versatile
    when batch average retrieval confidence < escalation_confidence_threshold.
  - Retry once on failure, then flag as batch_failed_twice.
  - Output validation via Pydantic (CheckpointVerdict).
  - Observation format: "[HH:MM:SS | Speaker] concise evidence sentence."

Valid verdict strings (case-sensitive, must match the N-column formula):
  Yes / Partial / In Progress / Not clarified / No / NA
"""

import json
import logging
import os
import time
from typing import Literal

import groq as groq_sdk
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from src.rate_limiter import GroqRateLimiter

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model for one checkpoint's verdict
# ---------------------------------------------------------------------------

class CheckpointVerdict(BaseModel):
    checkpoint_id: str
    verdict: Literal["Yes", "Partial", "In Progress", "Not clarified", "No", "NA"]
    confidence: float
    observation: str  # format: "[HH:MM:SS | Speaker] concise sentence."


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

def choose_model(batch_confidences: list[float], threshold: float, config: dict) -> str:
    """
    Choose Groq model based on average retrieval confidence.

    If avg_conf < threshold → escalate to the larger model.
    Otherwise → use the default (smaller, faster) model.

    Args:
        batch_confidences: List of per-checkpoint retrieval confidence scores.
        threshold:         Escalation threshold from config.
        config:            Loaded config.yaml.

    Returns:
        Model identifier string.
    """
    if not batch_confidences:
        return config["groq"]["default_model"]
    avg_conf = sum(batch_confidences) / len(batch_confidences)
    if avg_conf < threshold:
        return config["groq"]["escalation_model"]
    return config["groq"]["default_model"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return (
        "You are an SEPG compliance auditor. For each checkpoint below, assess "
        "compliance using ONLY the provided transcript excerpts. Do not use outside "
        "knowledge. Return one verdict per checkpoint.\n\n"
        "Valid verdicts (case-sensitive — use EXACTLY as shown):\n"
        '- "Yes": the excerpts clearly show the checkpoint requirement is met.\n'
        '- "Partial": the excerpts show some but not full compliance.\n'
        '- "In Progress": the excerpts show the requirement is being actively worked on but not yet complete.\n'
        '- "No": the excerpts clearly show the requirement is not met.\n'
        '- "NA": the checkpoint does not apply to this project/transcript context. '
        "This is a first-class answer, not a fallback for missing evidence.\n"
        '- "Not clarified": there is not enough evidence in the excerpts to judge '
        'either way. Use this — not "NA" — when evidence is simply absent.\n\n'
                "For every verdict, provide an \"observation\" that summarizes the evidence.\n"
        "If the verdict is \"Not clarified\", the observation MUST be exactly: \"No relevant discussion found.\"\n"
        "For all other verdicts, the observation MUST follow this exact two-line format:\n"
        "<Concise summary of the evidence>\n"
        "Evidence: [HH:MM:SS | Speaker]\n\n"
        "Example:\n"
        "The team clarifies they do not follow a specific agile methodology but use a Kanban board.\n"
        "Evidence: [00:11:09 | Rajshekar Chavakula]\n\n"
        "The timestamp and speaker MUST come from the actual excerpts provided.\n\n"
        "Respond with ONLY a valid JSON array, no prose, no markdown fences:\n"
        '[{"checkpoint_id": "...", "verdict": "...", "confidence": 0.0, "observation": "..."}]'
    )


def build_user_prompt(
    activity_name: str,
    checkpoints_with_excerpts: list[dict],
) -> str:
    """
    Build the per-batch user prompt.

    Args:
        activity_name:            Name of the Activity group being assessed.
        checkpoints_with_excerpts: List of dicts each containing:
            checkpoint_id, checkpoint_text, weight, retrieved_excerpts (list[str]).
    """
    lines = [f"Activity: {activity_name}\n", "Checkpoints:"]
    for item in checkpoints_with_excerpts:
        lines.append(
            f"- [{item['checkpoint_id']}] {item['checkpoint_text']} (weight: {item['weight']})"
        )

    lines.append("\nRetrieved transcript excerpts per checkpoint:")
    for item in checkpoints_with_excerpts:
        excerpts_joined = " | ".join(item["retrieved_excerpts"]) if item["retrieved_excerpts"] else "(no excerpts)"
        lines.append(f"[{item['checkpoint_id']}] excerpts: {excerpts_joined}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core batch assessor
# ---------------------------------------------------------------------------

def assess_batch(
    activity_name: str,
    checkpoints_with_excerpts: list[dict],
    config: dict,
    rate_limiter: GroqRateLimiter,
    groq_client=None,
) -> tuple[list[CheckpointVerdict], bool]:
    """
    Call Groq once for the given Activity batch.

    Returns:
        (verdicts, batch_failed): verdicts is a list of CheckpointVerdict.
        If batch_failed is True, verdicts is empty and the batch should be flagged.
    """
    groq_cfg = config["groq"]
    threshold = groq_cfg["escalation_confidence_threshold"]
    max_retries = int(groq_cfg.get("max_batch_retries", 1))  # = 1 (retry once)

    # Retrieve confidences for model routing
    batch_confidences = [item.get("confidence", 0.0) for item in checkpoints_with_excerpts]
    model = choose_model(batch_confidences, threshold, config)
    avg_conf = sum(batch_confidences) / len(batch_confidences) if batch_confidences else 0.0

    logger.info(
        "Activity '%s' (%d checkpoints) — avg_conf=%.3f → model=%s",
        activity_name, len(checkpoints_with_excerpts), avg_conf, model
    )

    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(activity_name, checkpoints_with_excerpts)

    # Initialise client if not injected (for testing)
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment.")
        groq_client = groq_sdk.Groq(api_key=api_key, max_retries=0)

    attempts = 0
    while attempts <= max_retries:
        attempts += 1
        try:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4 + 500
            rate_limiter.wait_if_needed(model, estimated_tokens)

            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            raw_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else estimated_tokens
            rate_limiter.record_call(model, tokens_used)
            rate_limiter.reset_backoff(model)

            # Parse JSON
            parsed = json.loads(raw_text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed)}")

            # Validate with Pydantic
            verdicts = [CheckpointVerdict(**item) for item in parsed]

            logger.info(
                "Activity '%s' — %d verdicts received (attempt %d)",
                activity_name, len(verdicts), attempts
            )
            return verdicts, False  # success

        except groq_sdk.RateLimitError:
            logger.warning("429 from Groq for activity '%s' (attempt %d)", activity_name, attempts)
            rate_limiter.handle_429(model)
            # Count the 429 as using the retry budget
            if attempts > max_retries:
                break
            continue

        except (json.JSONDecodeError, ValueError, ValidationError, KeyError) as e:
            logger.warning(
                "Activity '%s' attempt %d failed: %s", activity_name, attempts, e
            )
            if attempts > max_retries:
                break

        except Exception as e:
            logger.error(
                "Unexpected error for activity '%s' attempt %d: %s", activity_name, attempts, e
            )
            if attempts > max_retries:
                break

    # All attempts exhausted
    logger.error(
        "Activity '%s' failed after %d attempt(s) — flagging as batch_failed_twice.",
        activity_name, attempts
    )
    return [], True  # batch_failed


# ---------------------------------------------------------------------------
# CLI entry point (dry-run test with one synthetic batch)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    limiter = GroqRateLimiter(cfg)

    test_batch = [
        {
            "checkpoint_id": "Q2_2026_12",
            "checkpoint_text": "Signed SOW is available",
            "weight": 5,
            "confidence": 0.80,
            "retrieved_excerpts": [
                "[00:05:23 | Project Manager] The SOW was signed by the client and is available in the shared drive.",
                "[00:05:45 | PM] We have both the original and the approved version.",
            ],
        }
    ]

    verdicts, failed = assess_batch("SOW", test_batch, cfg, limiter)
    if failed:
        print("BATCH FAILED — would be flagged in review log.")
    else:
        for v in verdicts:
            print(f"  {v.checkpoint_id}: {v.verdict} (conf={v.confidence:.2f})")
            print(f"    obs: {v.observation}")
