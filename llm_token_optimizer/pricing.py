"""Token pricing registry for all major LLM providers (2026)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from llm_token_optimizer.exceptions import ModelNotSupportedError
from llm_token_optimizer.models import ModelPricing, ModelTier

logger = logging.getLogger(__name__)

# Pre-built pricing catalog (2026 reference rates)
_CATALOG: List[ModelPricing] = [
    # OpenAI
    ModelPricing(model_id="gpt-4o", tier=ModelTier.FRONTIER, input_cost_per_1k=0.005, output_cost_per_1k=0.015, context_window=128000, supports_batch=True, batch_discount=0.50),
    ModelPricing(model_id="gpt-4o-mini", tier=ModelTier.CHEAP, input_cost_per_1k=0.00015, output_cost_per_1k=0.0006, context_window=128000, supports_batch=True, batch_discount=0.50),
    ModelPricing(model_id="o1", tier=ModelTier.FRONTIER, input_cost_per_1k=0.015, output_cost_per_1k=0.060, context_window=200000, supports_batch=False, batch_discount=0.0),
    ModelPricing(model_id="o3-mini", tier=ModelTier.STANDARD, input_cost_per_1k=0.0011, output_cost_per_1k=0.0044, context_window=200000, supports_batch=True, batch_discount=0.50),
    # Anthropic
    ModelPricing(model_id="claude-opus-4-6", tier=ModelTier.FRONTIER, input_cost_per_1k=0.015, output_cost_per_1k=0.075, context_window=200000, supports_batch=True, batch_discount=0.50),
    ModelPricing(model_id="claude-sonnet-4-6", tier=ModelTier.STANDARD, input_cost_per_1k=0.003, output_cost_per_1k=0.015, context_window=200000, supports_batch=True, batch_discount=0.50),
    ModelPricing(model_id="claude-haiku-4-5-20251001", tier=ModelTier.CHEAP, input_cost_per_1k=0.00025, output_cost_per_1k=0.00125, context_window=200000, supports_batch=True, batch_discount=0.50),
    # Google
    ModelPricing(model_id="gemini-2.0-flash", tier=ModelTier.CHEAP, input_cost_per_1k=0.0001, output_cost_per_1k=0.0004, context_window=1048576, supports_batch=True, batch_discount=0.50),
    ModelPricing(model_id="gemini-2.0-pro", tier=ModelTier.FRONTIER, input_cost_per_1k=0.00125, output_cost_per_1k=0.005, context_window=1048576, supports_batch=True, batch_discount=0.50),
    # Mistral
    ModelPricing(model_id="mistral-large", tier=ModelTier.STANDARD, input_cost_per_1k=0.002, output_cost_per_1k=0.006, context_window=131072, supports_batch=False, batch_discount=0.0),
    ModelPricing(model_id="mistral-small", tier=ModelTier.CHEAP, input_cost_per_1k=0.0002, output_cost_per_1k=0.0006, context_window=131072, supports_batch=False, batch_discount=0.0),
]


class PricingRegistry:
    """Registry of LLM model pricing info."""

    def __init__(self) -> None:
        self._registry: Dict[str, ModelPricing] = {m.model_id: m for m in _CATALOG}

    def register(self, pricing: ModelPricing) -> None:
        """Add or update a model pricing entry."""
        self._registry[pricing.model_id] = pricing
        logger.info("Registered pricing for '%s'", pricing.model_id)

    def get(self, model_id: str) -> ModelPricing:
        """Get pricing for a model."""
        if model_id not in self._registry:
            raise ModelNotSupportedError(f"Model '{model_id}' not in pricing registry")
        return self._registry[model_id]

    def list_models(self) -> List[str]:
        """Return all registered model IDs."""
        return list(self._registry.keys())

    def cheapest_for_tier(self, tier: ModelTier) -> ModelPricing:
        """Return cheapest model in a given tier by input cost."""
        candidates = [m for m in self._registry.values() if m.tier == tier]
        if not candidates:
            raise ModelNotSupportedError(f"No models found for tier '{tier}'")
        return sorted(candidates, key=lambda m: m.input_cost_per_1k)[0]


# Module-level default registry
default_registry = PricingRegistry()
