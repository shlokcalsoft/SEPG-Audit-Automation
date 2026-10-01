"""
transcript_parser.py
====================
Parses a Microsoft Teams meeting transcript DOCX into a list of
speaker-tagged, timestamped segments.

Actual Teams DOCX format (observed from real file):
  Each paragraph contains a SINGLE string with embedded newlines:
    "\nSpeaker Name   M:SS\nText of what was said..."

  e.g. "\nAmit Gogate   0:16\nSandeep, what is Rakib's role in the team?"

  The first line (after stripping leading \n) is:  "Speaker Name   M:SS"
  The remaining lines are the speech text.

Also handles:
  - Generic inline "Speaker: text" (with or without timestamp prefix)
  - Plain paragraphs (fallback)

Output format per segment:
{
  "segment_id":  0,               # 0-based int
  "timestamp":   "00:00:16",      # zero-padded HH:MM:SS, or ""
  "speaker":     "Amit Gogate",
  "text":        "Sandeep, what is Rakib's role in the team?"
}
"""

import json
import logging
import re
from pathlib import Path

import docx
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_timestamp(ts: str) -> str:
    """
    Normalize Teams timestamps to HH:MM:SS.
    Teams uses M:SS (e.g. "0:16", "43:07") — NOT HH:MM:SS.
    """
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    elif len(parts) == 2:
        # Could be M:SS or H:MM — Teams uses M:SS exclusively
        m, s = parts
        return f"00:{int(m):02d}:{int(s):02d}"
    return ts


def _build_segment(idx: int, timestamp: str, speaker: str | None, text: str) -> dict:
    return {
        "segment_id": idx,
        "timestamp":  timestamp,
        "speaker":    speaker,
        "text":       text.strip(),
    }


# ---------------------------------------------------------------------------
# Strategy 1: Teams embedded-newline format
# e.g.  "\nAmit Gogate   0:16\nSandeep, what is Rakib's role?"
# ---------------------------------------------------------------------------

# Matches: "Speaker Name   M:SS"  (2+ spaces between name and timestamp)
_TEAMS_HEADER_RE = re.compile(
    r'^([A-Z][^\n]+?)\s{2,}(\d{1,2}:\d{2})$'
)


def _parse_teams_embedded(paragraphs: list[str]) -> list[dict]:
    """
    Parse Teams DOCX where each paragraph has embedded \\n:
      \\nSpeaker   M:SS\\nText...
    Returns segments, or [] if format not detected.
    """
    segments: list[dict] = []

    for raw_para in paragraphs:
        # Strip leading/trailing whitespace including \n
        para = raw_para.strip()
        if not para:
            continue

        # Split on embedded newlines
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue

        # Skip header/meta lines (no speaker format)
        if len(lines) == 1 and not _TEAMS_HEADER_RE.match(lines[0]):
            continue

        # Try to match first line as "Speaker   M:SS"
        m = _TEAMS_HEADER_RE.match(lines[0])
        if m and len(lines) >= 2:
            speaker   = m.group(1).strip()
            timestamp = _normalize_timestamp(m.group(2))
            text      = " ".join(lines[1:]).strip()
            if text:
                segments.append(_build_segment(len(segments), timestamp, speaker, text))

    return segments


# ---------------------------------------------------------------------------
# Strategy 2: Inline format  "[HH:MM:SS] Speaker: text"  or  "Speaker: text"
# ---------------------------------------------------------------------------

_SPEAKER_INLINE_RE   = re.compile(r'^([A-Z][a-zA-Z .]+):\s*(.+)$')
_TIMESTAMP_INLINE_RE = re.compile(r'^\[?(\d{1,2}:\d{2}:\d{2})\]?\s*')


def _parse_inline(paragraphs: list[str]) -> list[dict]:
    segments: list[dict] = []
    current_ts      = ""
    current_speaker = None
    current_lines: list[str] = []

    def _flush(idx: int) -> dict | None:
        joined = " ".join(current_lines).strip()
        return _build_segment(idx, current_ts, current_speaker, joined) if joined else None

    for raw in paragraphs:
        para = raw.strip()
        if not para:
            continue

        ts = ""
        ts_m = _TIMESTAMP_INLINE_RE.match(para)
        if ts_m:
            ts   = _normalize_timestamp(ts_m.group(1))
            para = para[ts_m.end():]

        m = _SPEAKER_INLINE_RE.match(para)
        if m:
            seg = _flush(len(segments))
            if seg:
                segments.append(seg)
            current_ts      = ts
            current_speaker = m.group(1).strip()
            current_lines   = [m.group(2).strip()]
        else:
            current_lines.append(para)

    seg = _flush(len(segments))
    if seg:
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def parse_transcript(config: dict, output_path: str = "parsed_transcript.json") -> list[dict]:
    """
    Load the DOCX transcript and return a list of segment dicts.

    Tries three strategies in order:
      1. Teams embedded-newline format  (real Teams DOCX)
      2. Inline "Speaker: text" format
      3. One-segment-per-paragraph fallback

    Args:
        config:      Loaded config.yaml as a dict.
        output_path: Where to write parsed_transcript.json.

    Returns:
        List of segment dicts.
    """
    tr_cfg = config["transcript"]
    path   = tr_cfg["path"]

    logger.info("Loading transcript: %s", path)
    document = docx.Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    logger.info("Total paragraphs in DOCX: %d", len(paragraphs))

    # --- Strategy 1: Teams embedded-newline ---
    segments = _parse_teams_embedded(paragraphs)
    speaker_count = sum(1 for s in segments if s["speaker"])

    if speaker_count > 0:
        logger.info("Teams embedded-newline format detected — %d segments.", len(segments))
    else:
        # --- Strategy 2: Inline ---
        logger.info("Teams format not detected — trying inline format …")
        segments = _parse_inline(paragraphs)
        speaker_count = sum(1 for s in segments if s["speaker"])

        if speaker_count > 0:
            logger.info("Inline format detected — %d segments.", len(segments))
        else:
            # --- Strategy 3: One-per-paragraph fallback ---
            logger.warning(
                "No speaker labels detected. Falling back to one-segment-per-paragraph. "
                "Retrieval will still work but observations won't have speaker/timestamp."
            )
            segments = []
            for para in paragraphs:
                p = para.strip()
                if p:
                    segments.append(_build_segment(len(segments), "", None, p))

    # Renumber sequentially
    for i, seg in enumerate(segments):
        seg["segment_id"] = i

    ts_count = sum(1 for s in segments if s["timestamp"])
    logger.info(
        "Final: %d segments | %d with speakers | %d with timestamps",
        len(segments), sum(1 for s in segments if s["speaker"]), ts_count
    )

    out = Path(output_path)
    out.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written %d segments to %s", len(segments), output_path)

    return segments


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = parse_transcript(cfg)

    print(f"\nSUCCESS — {len(result)} segments parsed.")
    print(f"  With speakers:   {sum(1 for s in result if s['speaker'])} / {len(result)}")
    print(f"  With timestamps: {sum(1 for s in result if s['timestamp'])} / {len(result)}")
    print(f"\nFirst 5 segments:")
    for seg in result[:5]:
        print(f"  [{seg['segment_id']:03d}] ts={seg['timestamp']!r}  speaker={seg['speaker']!r}")
        print(f"         text={seg['text'][:100]!r}")
    print(f"\nAll unique speakers ({len({s['speaker'] for s in result if s['speaker']})}):")
    for sp in sorted({s['speaker'] for s in result if s['speaker']}):
        print(f"  - {sp}")
