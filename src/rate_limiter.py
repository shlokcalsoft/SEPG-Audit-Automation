"""
rate_limiter.py
===============
Per-model sliding-window rate limiter for the Groq free tier.

Tracks requests and tokens used in the trailing 60 seconds per model.
Before each call, sleeps long enough to stay under both RPM and TPM caps.
Handles 429 responses with exponential backoff (starting at 2s).

All limits are read from config.yaml — never hardcoded here.
"""

import logging
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60  # Groq rate-limit window is per-minute


class GroqRateLimiter:
    """
    Sliding-window rate limiter for multiple Groq models.

    Usage:
        limiter = GroqRateLimiter(config)
        limiter.wait_if_needed("llama-3.1-8b-instant", estimated_tokens=2000)
        # ... make the API call ...
        limiter.record_call("llama-3.1-8b-instant", tokens_used=1850)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Loaded config.yaml as a dict.
                    Reads groq.rate_limits per model (rpm, tpm, rpd).
        """
        groq_cfg = config["groq"]
        raw_limits = groq_cfg.get("rate_limits", {})

        # Structure: model -> {"rpm": int, "tpm": int, "rpd": int}
        self._limits: dict[str, dict] = {}
        for model, lims in raw_limits.items():
            self._limits[model] = {
                "rpm": int(lims.get("rpm", 30)),
                "tpm": int(lims.get("tpm", 6000)),
                "rpd": int(lims.get("rpd", 14400)),
            }

        # Sliding window: model -> deque of (timestamp, tokens_used)
        self._window: dict[str, deque] = defaultdict(deque)

        # Daily request counter: model -> int
        self._daily_count: dict[str, int] = defaultdict(int)

        # Backoff state: model -> current_backoff_seconds
        self._backoff: dict[str, float] = defaultdict(lambda: 2.0)

        logger.info("RateLimiter initialised for models: %s", list(self._limits.keys()))

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _prune(self, model: str) -> None:
        """Remove entries older than the sliding window."""
        cutoff = time.monotonic() - _WINDOW_SECONDS
        q = self._window[model]
        while q and q[0][0] < cutoff:
            q.popleft()

    def _requests_in_window(self, model: str) -> int:
        self._prune(model)
        return len(self._window[model])

    def _tokens_in_window(self, model: str) -> int:
        self._prune(model)
        return sum(tokens for _, tokens in self._window[model])

    def _limits_for(self, model: str) -> dict:
        """Return limits for model, falling back to a safe default."""
        if model in self._limits:
            return self._limits[model]
        # Unknown model — use conservative defaults
        logger.warning("Unknown model '%s' — using conservative rate limits.", model)
        return {"rpm": 30, "tpm": 6000, "rpd": 1000}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def wait_if_needed(self, model: str, estimated_tokens: int = 0) -> None:
        """
        Sleep as long as needed to stay within RPM and TPM limits.

        Args:
            model:            Groq model identifier.
            estimated_tokens: Best-guess token count for the upcoming call.
        """
        lims = self._limits_for(model)
        rpm  = lims["rpm"]
        tpm  = lims["tpm"]

        while True:
            self._prune(model)
            req_count = len(self._window[model])
            tok_count = sum(t for _, t in self._window[model])

            # Check if we're under both limits
            req_ok = req_count < rpm
            tok_ok = (tok_count + estimated_tokens) <= tpm

            if req_ok and tok_ok:
                break

            # Calculate minimum sleep to free one slot (oldest entry leaves window)
            if self._window[model]:
                oldest_ts = self._window[model][0][0]
                sleep_s = max(0.1, (_WINDOW_SECONDS - (time.monotonic() - oldest_ts)) + 0.1)
            else:
                sleep_s = 1.0

            reason = "RPM" if not req_ok else "TPM"
            logger.info(
                "[RateLimiter] %s limit reached for '%s' — sleeping %.1fs",
                reason, model, sleep_s
            )
            time.sleep(sleep_s)

    def record_call(self, model: str, tokens_used: int) -> None:
        """
        Record a completed API call into the sliding window.

        Args:
            model:       Groq model identifier.
            tokens_used: Actual tokens consumed (prompt + completion).
        """
        now = time.monotonic()
        self._window[model].append((now, tokens_used))
        self._daily_count[model] += 1
        logger.debug(
            "[RateLimiter] Recorded call for '%s': tokens=%d, window_requests=%d",
            model, tokens_used, len(self._window[model])
        )

    def handle_429(self, model: str) -> None:
        """
        Handle a 429 Too Many Requests response with exponential backoff.
        Doubles the backoff time each call (starting at 2s, capped at 64s).
        """
        backoff = self._backoff[model]
        logger.warning(
            "[RateLimiter] 429 received for '%s' — backing off %.0fs", model, backoff
        )
        time.sleep(backoff)
        self._backoff[model] = min(backoff * 2, 64.0)

    def reset_backoff(self, model: str) -> None:
        """Reset exponential backoff counter after a successful call."""
        self._backoff[model] = 2.0

    def daily_count(self, model: str) -> int:
        """Return the number of calls made to this model today."""
        return self._daily_count[model]
