"""
retriever.py
============
For every checkpoint, finds the best-matching transcript segments using a
hybrid TF-IDF cosine similarity + keyword overlap score, then expands matches
to a ±2-segment conversational window.

Algorithm:
  combined_score = 0.70 * cosine_similarity + 0.30 * keyword_overlap
  keyword_overlap = |intersection(checkpoint_tokens, segment_tokens)| / |checkpoint_tokens|

Top-K = 5 segments per checkpoint.
Each top-5 match is expanded to indices [i-2, i+2], windows merged & deduped.
confidence = rank-1 (best single) combined score.
"""

import json
import logging
import re
import string
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (minimal built-in set — avoids NLTK dependency)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "for",
    "in", "of", "on", "at", "to", "by", "up", "as", "if", "it", "its",
    "this", "that", "these", "those", "with", "from", "into", "through",
    "during", "before", "after", "above", "below", "between", "each",
    "all", "any", "both", "few", "more", "most", "other", "some", "such",
    "only", "same", "than", "then", "too", "very", "just", "about",
    "per", "also", "been", "there", "their", "they", "them", "our",
    "we", "us", "you", "i", "he", "she", "his", "her", "who", "what",
    "which", "when", "where", "how", "why",
})


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase, strip punctuation, remove stopwords, return a frozenset."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return frozenset(t for t in tokens if t and t not in _STOPWORDS)


def _keyword_overlap(chk_tokens: frozenset, seg_tokens: frozenset) -> float:
    """
    Proportion of checkpoint tokens found in the segment.
    Returns 0.0 if checkpoint has no tokens.
    """
    if not chk_tokens:
        return 0.0
    return len(chk_tokens & seg_tokens) / len(chk_tokens)


def _expand_window(idx: int, total: int, radius: int) -> range:
    """Return the clamped index range [idx-radius, idx+radius]."""
    lo = max(0, idx - radius)
    hi = min(total - 1, idx + radius)
    return range(lo, hi + 1)


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_evidence(
    checkpoints: list[dict],
    segments: list[dict],
    config: dict,
    output_path: str = "retrieval_results.json",
) -> list[dict]:
    """
    For every checkpoint, retrieve the top-K transcript segments plus expanded
    conversational windows.

    Args:
        checkpoints:  List of checkpoint dicts from template_reader.
        segments:     List of segment dicts from transcript_parser.
        config:       Loaded config.yaml as a dict.
        output_path:  Where to write retrieval_results.json.

    Returns:
        List of retrieval result dicts, one per checkpoint.
    """
    ret_cfg = config["retrieval"]
    tfidf_w = float(ret_cfg["tfidf_weight"])   # 0.70
    kw_w    = float(ret_cfg["keyword_weight"])  # 0.30
    top_k   = int(ret_cfg["top_k"])             # 5
    radius  = int(ret_cfg["window_radius"])     # 2

    seg_texts   = [s["text"] for s in segments]
    seg_count   = len(seg_texts)

    logger.info(
        "Building TF-IDF vectorizer over %d segments (top_k=%d, window_radius=%d)",
        seg_count, top_k, radius
    )

    # Fit TF-IDF on all segment texts
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        stop_words="english",
        min_df=1,
    )
    seg_matrix = vectorizer.fit_transform(seg_texts)  # shape: (n_segs, vocab)

    # Pre-tokenize segments for keyword overlap
    seg_token_sets = [_tokenize(t) for t in seg_texts]

    results: list[dict] = []

    for chk in checkpoints:
        chk_id   = chk["checkpoint_id"]
        chk_text = chk["checkpoint_text"]

        # TF-IDF cosine similarity
        chk_vec     = vectorizer.transform([chk_text])          # (1, vocab)
        cos_scores  = cosine_similarity(chk_vec, seg_matrix)[0] # (n_segs,)

        # Keyword overlap
        chk_tokens  = _tokenize(chk_text)
        kw_scores   = np.array([
            _keyword_overlap(chk_tokens, st) for st in seg_token_sets
        ])

        # Combined score
        combined = tfidf_w * cos_scores + kw_w * kw_scores

        # Top-K indices by combined score (descending)
        top_indices = np.argsort(combined)[::-1][:top_k].tolist()

        # Confidence = rank-1 (best single) combined score
        confidence = float(combined[top_indices[0]]) if top_indices else 0.0

        # Expand each top-K to ±radius window, collect all segment indices
        window_indices: set[int] = set()
        for idx in top_indices:
            for wi in _expand_window(idx, seg_count, radius):
                window_indices.add(wi)

        # Sort and collect excerpt texts in original order with speaker/timestamp prefixes
        sorted_window = sorted(window_indices)
        excerpts = []
        for i in sorted_window:
            seg = segments[i]
            ts = seg.get("timestamp") or "00:00:00"
            spk = seg.get("speaker") or "Unknown"
            excerpts.append(f"[{ts} | {spk}] {seg['text']}")

        results.append({
            "checkpoint_id":    chk_id,
            "retrieved_excerpts": excerpts,
            "confidence":        confidence,
            "top_segment_indices": top_indices,  # kept for debugging
            "window_indices":    sorted_window,
        })

    logger.info("Retrieval complete for %d checkpoints.", len(results))

    # Write to disk
    out = Path(output_path)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Written retrieval results to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.template_reader import extract_checkpoints
    from src.transcript_parser import parse_transcript

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoints = extract_checkpoints(cfg)
    segments    = parse_transcript(cfg)
    results     = retrieve_evidence(checkpoints, segments, cfg)

    print(f"\nSUCCESS — {len(results)} retrieval results.")
    confs = [r["confidence"] for r in results]
    print(f"Confidence — min: {min(confs):.3f}, max: {max(confs):.3f}, mean: {sum(confs)/len(confs):.3f}")

    # Show top and bottom 3
    sorted_r = sorted(results, key=lambda x: x["confidence"], reverse=True)
    print("\nTop 3 most confident:")
    for r in sorted_r[:3]:
        chk = next(c for c in checkpoints if c["checkpoint_id"] == r["checkpoint_id"])
        print(f"  {r['checkpoint_id']}: conf={r['confidence']:.3f}  text={chk['checkpoint_text'][:60]}")
        print(f"    first excerpt: {r['retrieved_excerpts'][0][:80]!r}")

    print("\nBottom 3 (lowest evidence):")
    for r in sorted_r[-3:]:
        chk = next(c for c in checkpoints if c["checkpoint_id"] == r["checkpoint_id"])
        print(f"  {r['checkpoint_id']}: conf={r['confidence']:.3f}  text={chk['checkpoint_text'][:60]}")
