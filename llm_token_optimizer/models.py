"""Pydantic models for llm-token-optimizer."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Model cost tier."""
    CHEAP = "cheap"
    STANDARD = "standard"
    FRONTIER = "frontier"


class ModelPricing(BaseModel):
    """Per-model token pricing."""
    model_id: str
    tier: ModelTier
    input_cost_per_1k: float   # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens
    context_window: int        # max tokens
    supports_batch: bool = False
    batch_discount: float = 0.0  # fraction, e.g. 0.5 for 50% off


class TokenUsage(BaseModel):
    """Token usage for a single LLM call."""
    model_id: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Result of prompt optimization."""
    original_text: str
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    compression_ratio: float  # optimized/original
    strategies_applied: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    """Cost estimate for a prompt + expected output."""
    model_id: str
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    tier: ModelTier


class BatchJob(BaseModel):
    """A deferred batch API job."""
    job_id: str
    model_id: str
    prompts: List[str]
    submitted_at: float = Field(default_factory=time.time)
    estimated_cost_usd: float = 0.0
    discount_applied: float = 0.0
    status: str = "pending"
