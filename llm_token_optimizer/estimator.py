"""Cost estimation utilities."""
from __future__ import annotations

import logging
from typing import Optional

from llm_token_optimizer.counter import estimate_tokens
from llm_token_optimizer.models import CostEstimate, ModelTier
from llm_token_optimizer.pricing import PricingRegistry, default_registry

logger = logging.getLogger(__name__)


class CostEstimator:
    """Estimate token cost before making an LLM call."""

    def __init__(self, registry: Optional[PricingRegistry] = None) -> None:
        self._registry = registry or default_registry

    def estimate(
        self,
        model_id: str,
        prompt: str,
        estimated_output_tokens: int = 256,
    ) -> CostEstimate:
        """
        Estimate cost for a prompt + expected output length.

        Args:
            model_id: LLM model identifier.
            prompt: Input prompt text.
            estimated_output_tokens: Expected output token count.

        Returns:
            CostEstimate with per-component cost breakdown.
        """
        pricing = self._registry.get(model_id)
        input_tokens = estimate_tokens(prompt, model_id)
        input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (estimated_output_tokens / 1000) * pricing.output_cost_per_1k

        est = CostEstimate(
            model_id=model_id,
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(input_cost + output_cost, 8),
            tier=pricing.tier,
        )
        logger.debug("Cost estimate for '%s': $%.6f", model_id, est.total_cost_usd)
        return est

    def cheapest_model(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.CHEAP,
        estimated_output_tokens: int = 256,
    ) -> CostEstimate:
        """Return cost estimate for the cheapest model in a given tier."""
        pricing = self._registry.cheapest_for_tier(tier)
        return self.estimate(pricing.model_id, prompt, estimated_output_tokens)

    def compare_models(
        self,
        model_ids: list,
        prompt: str,
        estimated_output_tokens: int = 256,
    ) -> list:
        """Return sorted list of CostEstimates for a list of models (cheapest first)."""
        estimates = [self.estimate(m, prompt, estimated_output_tokens) for m in model_ids]
        return sorted(estimates, key=lambda e: e.total_cost_usd)
