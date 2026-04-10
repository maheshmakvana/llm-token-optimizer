"""Prompt compression and token optimization strategies."""
from __future__ import annotations

import logging
import re
from typing import List

from llm_token_optimizer.counter import estimate_tokens
from llm_token_optimizer.exceptions import OptimizationError
from llm_token_optimizer.models import OptimizationResult

logger = logging.getLogger(__name__)


def _remove_redundant_whitespace(text: str) -> str:
    """Collapse multiple spaces/blank lines."""
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _remove_filler_phrases(text: str) -> str:
    """Strip common filler phrases that add tokens without value."""
    fillers = [
        r"Please note that\b",
        r"It is important to note that\b",
        r"As an AI language model,?\b",
        r"Certainly[!.]?\s*",
        r"Of course[!.]?\s*",
        r"Sure[!.]?\s*",
        r"I'd be happy to help[.!]?\s*",
        r"I hope this helps[.!]?\s*",
        r"Feel free to ask[^.]*\.",
        r"Let me know if you[^.]*\.",
    ]
    for f in fillers:
        text = re.sub(f, "", text, flags=re.IGNORECASE)
    return text.strip()


def _compress_repeated_content(text: str) -> str:
    """Remove repeated paragraphs (exact duplicates)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen: set = set()
    unique = []
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return "\n\n".join(unique)


def _abbreviate_examples(text: str, max_examples: int = 3) -> str:
    """Keep only first N few-shot examples (detect by numbered list patterns)."""
    lines = text.split("\n")
    example_count = 0
    result_lines = []
    in_example = False
    for line in lines:
        if re.match(r"^(Example|Input|Output|Q|A)[\s:]", line, re.IGNORECASE):
            if not in_example:
                in_example = True
                example_count += 1
            if example_count > max_examples:
                continue
        else:
            in_example = False
        result_lines.append(line)
    return "\n".join(result_lines)


_STRATEGIES = {
    "whitespace": _remove_redundant_whitespace,
    "fillers": _remove_filler_phrases,
    "dedup": _compress_repeated_content,
    "examples": _abbreviate_examples,
}


def optimize_prompt(
    text: str,
    strategies: List[str] = ("whitespace", "fillers", "dedup"),
    model_id: str = "gpt-4o",
) -> OptimizationResult:
    """
    Apply compression strategies to reduce token count.

    Args:
        text: Raw prompt text.
        strategies: List of strategy names to apply in order.
        model_id: Target model (affects token count estimate).

    Returns:
        OptimizationResult with before/after token counts and applied strategies.
    """
    original_tokens = estimate_tokens(text, model_id)
    optimized = text
    applied: List[str] = []

    for strategy in strategies:
        if strategy not in _STRATEGIES:
            raise OptimizationError(f"Unknown strategy: '{strategy}'. Valid: {list(_STRATEGIES.keys())}")
        before_len = len(optimized)
        optimized = _STRATEGIES[strategy](optimized)
        if len(optimized) != before_len:
            applied.append(strategy)

    optimized_tokens = estimate_tokens(optimized, model_id)
    tokens_saved = original_tokens - optimized_tokens

    logger.info(
        "optimize_prompt: %d→%d tokens (saved %d, strategies=%s)",
        original_tokens, optimized_tokens, tokens_saved, applied
    )

    return OptimizationResult(
        original_text=text,
        optimized_text=optimized,
        original_tokens=original_tokens,
        optimized_tokens=optimized_tokens,
        tokens_saved=tokens_saved,
        compression_ratio=optimized_tokens / max(original_tokens, 1),
        strategies_applied=applied,
    )
