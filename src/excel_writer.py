"""
excel_writer.py
===============
Writes verdict and observation values into a copy of the SEPG audit template.

The N/A Rule (two enforcement points):
  Point A: build_write_instructions() — NA verdicts produce NO instruction (skipped).
  Point B: write_scorecard() — assert verdict != "NA" before every write.

Only writes to:
  Column L (response_verdict cell)  — e.g. "L12"
  Column Q (observation cell)       — e.g. "Q12"

NEVER writes to:
  Columns N, O, P (formula cells) — score, weighted score, average score
  Column G9 (overall compliance % formula)
  Any other cell
"""

import logging
import shutil
from pathlib import Path

import openpyxl
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import TextBlock, CellRichText

from src.assessor import CheckpointVerdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction builder (Point A — NA filter)
# ---------------------------------------------------------------------------

def build_write_instructions(
    verdicts: list[CheckpointVerdict],
    registry: dict[str, dict],
) -> list[tuple[str, str, str, str, str]]:
    """
    Build a list of (sheet, verdict_cell, verdict, obs_cell, observation) tuples.
    NA verdicts are excluded — no instruction is generated for them.

    Args:
        verdicts:  List of CheckpointVerdict from assessor.
        registry:  Dict mapping checkpoint_id -> checkpoint dict (from template_reader).

    Returns:
        List of (sheet_name, excel_cell, verdict, observation_cell, observation) tuples.
    """
    instructions = []
    for v in verdicts:
        if v.verdict == "NA":
            logger.debug("Skipping NA verdict for %s (Point A)", v.checkpoint_id)
            continue  # Point A — no instruction generated at all

        entry = registry.get(v.checkpoint_id)
        if entry is None:
            logger.warning("checkpoint_id '%s' not found in registry — skipping.", v.checkpoint_id)
            continue

        instructions.append((
            entry["excel_sheet"],
            entry["excel_cell"],       # e.g. "L12"
            v.verdict,
            entry["observation_cell"],  # e.g. "Q12"
            v.observation,
        ))

    return instructions


# ---------------------------------------------------------------------------
# Scorecard writer
# ---------------------------------------------------------------------------

def write_scorecard(
    template_path: str,
    instructions: list[tuple[str, str, str, str, str]],
    output_path: str,
) -> None:
    """
    Write verdicts and observations into a copy of the template.

    Args:
        template_path: Path to the original template (never modified).
        instructions:  List of (sheet, verdict_cell, verdict, obs_cell, observation).
        output_path:   Where to save the filled-in scorecard.

    Raises:
        AssertionError: If any NA verdict somehow reaches this function (Point B).
        FileNotFoundError: If template_path does not exist.
    """
    src = Path(template_path)
    if not src.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Copy template to output path
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(out))
    logger.info("Copied template to %s", output_path)

    # Load the copy
    wb = openpyxl.load_workbook(str(out))

    written_count = 0
    for sheet_name, verdict_cell, verdict, obs_cell, observation in instructions:
        # Point B — defensive assertion
        assert verdict != "NA", (
            f"NA verdict for cell {verdict_cell} must never reach the Excel writer. "
            "Check build_write_instructions() — Point A should have excluded it."
        )

        if sheet_name not in wb.sheetnames:
            logger.error("Sheet '%s' not found in workbook — skipping cell %s", sheet_name, verdict_cell)
            continue

        ws = wb[sheet_name]

        # Write verdict to column L cell
        ws[verdict_cell] = verdict

        # Write observation to column Q cell
        if observation:
            if "Evidence:" in observation:
                parts = observation.split("Evidence:")
                summary = parts[0]
                evidence_text = "Evidence:" + parts[1]
                
                bold_font = InlineFont(b=True)
                # Create a rich text cell with the first part normal, and the Evidence part bold
                rich_text = CellRichText(
                    summary,
                    TextBlock(bold_font, evidence_text)
                )
                ws[obs_cell] = rich_text
            else:
                ws[obs_cell] = observation

        written_count += 1
        logger.debug("Wrote '%s' → %s | obs → %s", verdict, verdict_cell, obs_cell)

    wb.save(str(out))
    wb.close()

    logger.info(
        "Scorecard saved: %s (%d verdicts written, %d skipped due to NA)",
        output_path, written_count, 0  # NA were already excluded by build_write_instructions
    )
