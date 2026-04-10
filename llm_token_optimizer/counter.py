"""Token counting (offline, no API calls required)."""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, model_id: Optional[str] = None) -> int:
    """
    Estimate token count using a fast heuristic (~4 chars/token for English).

    For production use, install tiktoken and use count_tokens_tiktoken().
    """
    if not text:
        return 0
    # Rough heuristic: ~4 chars per token on average for English LLM input
    words = len(re.findall(r'\S+', text))
    chars = len(text)
    # Blend word and char estimates
    return max(1, int(chars / 4 * 0.6 + words * 0.4 * 1.3))


def count_tokens_tiktoken(text: str, model_id: str = "gpt-4o") -> int:
    """
    Exact token count via tiktoken (if installed).
    Falls back to estimate_tokens if tiktoken is not available.
    """
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.encoding_for_model(model_id)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        logger.debug("tiktoken not installed — using estimate_tokens")
        return estimate_tokens(text, model_id)
