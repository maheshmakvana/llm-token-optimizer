"""Exceptions for llm-token-optimizer."""


class TokenOptimizerError(Exception):
    """Base exception for llm-token-optimizer."""


class BudgetExceededError(TokenOptimizerError):
    """Raised when token or cost budget is exceeded."""


class CompressionError(TokenOptimizerError):
    """Raised when prompt compression fails."""


class ModelNotSupportedError(TokenOptimizerError):
    """Raised when a model is not in the pricing registry."""


class OptimizationError(TokenOptimizerError):
    """Raised when optimization fails to produce a valid result."""
