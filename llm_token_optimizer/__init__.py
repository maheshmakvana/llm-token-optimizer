"""
llm-token-optimizer — Token Cost Control & Auto-Optimization.

Compress prompts, estimate costs, enforce budgets, route to cheap models,
and cut LLM spend by up to 60% without sacrificing output quality.
"""
from llm_token_optimizer.models import (
    ModelTier,
    ModelPricing,
    TokenUsage,
    OptimizationResult,
    CostEstimate,
    BatchJob,
)
from llm_token_optimizer.pricing import PricingRegistry, default_registry
from llm_token_optimizer.counter import estimate_tokens, count_tokens_tiktoken
from llm_token_optimizer.optimizer import optimize_prompt
from llm_token_optimizer.estimator import CostEstimator
from llm_token_optimizer.exceptions import (
    TokenOptimizerError,
    BudgetExceededError,
    CompressionError,
    ModelNotSupportedError,
    OptimizationError,
)
from llm_token_optimizer.advanced import (
    OptimizationCache,
    SemanticCache,
    OptimizationPipeline,
    PromptConstraint,
    PromptConstraintValidator,
    ConfidenceScorer,
    RateLimiter,
    CancellationToken,
    batch_optimize,
    abatch_optimize,
    optimize_with_budget,
    OperationProfiler,
    CostTelemetry,
    DriftDetector,
    stream_optimize,
    results_to_ndjson,
    results_to_csv,
    OptimizationDiff,
    diff_optimizations,
    RegressionTracker,
    ScoreTrend,
    PIIScrubber,
    AuditLog,
    BatchAPIRouter,
    CostLedger,
    ModelRouter,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "ModelTier", "ModelPricing", "TokenUsage", "OptimizationResult",
    "CostEstimate", "BatchJob",
    # Core
    "PricingRegistry", "default_registry",
    "estimate_tokens", "count_tokens_tiktoken",
    "optimize_prompt", "CostEstimator",
    # Exceptions
    "TokenOptimizerError", "BudgetExceededError", "CompressionError",
    "ModelNotSupportedError", "OptimizationError",
    # Advanced
    "OptimizationCache", "SemanticCache",
    "OptimizationPipeline",
    "PromptConstraint", "PromptConstraintValidator", "ConfidenceScorer",
    "RateLimiter", "CancellationToken",
    "batch_optimize", "abatch_optimize", "optimize_with_budget",
    "OperationProfiler", "CostTelemetry", "DriftDetector",
    "stream_optimize", "results_to_ndjson", "results_to_csv",
    "OptimizationDiff", "diff_optimizations",
    "RegressionTracker", "ScoreTrend",
    "PIIScrubber", "AuditLog", "BatchAPIRouter", "CostLedger", "ModelRouter",
]
