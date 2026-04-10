"""
Advanced features for llm-token-optimizer — 2026 Standard.

Covers: Caching, Pipeline, Validation & Schema, Async & Concurrency,
Observability, Streaming & Storage, Diff & Regression, Security & Cost.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from llm_token_optimizer.counter import estimate_tokens
from llm_token_optimizer.exceptions import BudgetExceededError, OptimizationError
from llm_token_optimizer.models import (
    BatchJob, CostEstimate, ModelTier, OptimizationResult, TokenUsage,
)
from llm_token_optimizer.pricing import PricingRegistry, default_registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. CACHING
# ─────────────────────────────────────────────

class OptimizationCache:
    """LRU + TTL cache for OptimizationResult, keyed by SHA-256 of input."""

    def __init__(self, max_size: int = 512, ttl: float = 600.0) -> None:
        self.max_size = max_size
        self.ttl = ttl
        self._store: Dict[str, Tuple[OptimizationResult, float]] = {}
        self._order: deque = deque()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(text: str, strategies: List[str]) -> str:
        raw = json.dumps({"text": text, "strategies": sorted(strategies)})
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, text: str, strategies: List[str]) -> Optional[OptimizationResult]:
        k = self._key(text, strategies)
        with self._lock:
            if k in self._store:
                result, ts = self._store[k]
                if time.time() - ts <= self.ttl:
                    self._hits += 1
                    return result
                del self._store[k]
            self._misses += 1
        return None

    def put(self, text: str, strategies: List[str], result: OptimizationResult) -> None:
        k = self._key(text, strategies)
        with self._lock:
            if k in self._store:
                self._order.remove(k)
            elif len(self._store) >= self.max_size:
                oldest = self._order.popleft()
                self._store.pop(oldest, None)
            self._store[k] = (result, time.time())
            self._order.append(k)

    def memoize(self, optimize_fn: Callable) -> Callable:
        """Decorator: cache optimization results."""
        def wrapper(text: str, strategies: List[str] = ("whitespace", "fillers"), **kw) -> OptimizationResult:
            cached = self.get(text, list(strategies))
            if cached is not None:
                return cached
            result = optimize_fn(text, strategies, **kw)
            self.put(text, list(strategies), result)
            return result
        return wrapper

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def save(self, path: str) -> None:
        with self._lock:
            data = {k: (v[0].model_dump(), v[1]) for k, v in self._store.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("OptimizationCache saved to %s", path)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            for k, (res_dict, ts) in data.items():
                self._store[k] = (OptimizationResult(**res_dict), ts)
                self._order.append(k)
        logger.info("OptimizationCache loaded from %s", path)


class SemanticCache:
    """Cosine-similarity based semantic cache for optimization results."""

    def __init__(self, threshold: float = 0.92) -> None:
        self.threshold = threshold
        self._store: List[Tuple[List[float], OptimizationResult]] = []

    @staticmethod
    def _bow_vector(text: str) -> List[float]:
        """Simple bag-of-words vector (normalized)."""
        words = re.findall(r'\w+', text.lower())
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        total = max(sum(freq.values()), 1)
        return [v / total for v in freq.values()]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0

    def get(self, text: str) -> Optional[OptimizationResult]:
        vec = self._bow_vector(text)
        for stored_vec, result in self._store:
            if self._cosine(vec, stored_vec) >= self.threshold:
                return result
        return None

    def put(self, text: str, result: OptimizationResult) -> None:
        self._store.append((self._bow_vector(text), result))


# ─────────────────────────────────────────────
# 2. PIPELINE
# ─────────────────────────────────────────────

@dataclass
class _OptPipelineStep:
    name: str
    fn: Callable[[str], str]
    retries: int = 0


class OptimizationPipeline:
    """Fluent, auditable token optimization pipeline."""

    def __init__(self) -> None:
        self._steps: List[_OptPipelineStep] = []
        self._audit_log: List[Dict[str, Any]] = []

    def map(self, name: str, fn: Callable[[str], str]) -> "OptimizationPipeline":
        """Apply a text transformation."""
        self._steps.append(_OptPipelineStep(name, fn))
        return self

    def filter(self, name: str, pred: Callable[[str], bool]) -> "OptimizationPipeline":
        """Keep text only if predicate returns True; else raises OptimizationError."""
        def _filter_fn(text: str) -> str:
            if not pred(text):
                raise OptimizationError(f"Filter '{name}' rejected input.")
            return text
        self._steps.append(_OptPipelineStep(name, _filter_fn))
        return self

    def branch(
        self,
        condition: Callable[[str], bool],
        true_fn: Callable[[str], str],
        false_fn: Callable[[str], str],
    ) -> "OptimizationPipeline":
        """Route to true_fn or false_fn based on condition."""
        def _branch_fn(text: str) -> str:
            return true_fn(text) if condition(text) else false_fn(text)
        self._steps.append(_OptPipelineStep("branch", _branch_fn))
        return self

    def with_retry(self, step_name: str, retries: int = 2) -> "OptimizationPipeline":
        for step in self._steps:
            if step.name == step_name:
                step.retries = retries
        return self

    def run(self, text: str) -> str:
        """Execute all pipeline steps."""
        result = text
        for step in self._steps:
            start = time.time()
            attempt = 0
            last_exc: Optional[Exception] = None
            while attempt <= step.retries:
                try:
                    result = step.fn(result)
                    break
                except Exception as exc:
                    last_exc = exc
                    attempt += 1
            else:
                raise OptimizationError(f"Step '{step.name}' failed after {step.retries + 1} attempts") from last_exc
            elapsed = time.time() - start
            self._audit_log.append({"step": step.name, "output_len": len(result), "elapsed_s": elapsed})
        return result

    async def arun(self, text: str) -> str:
        """Async pipeline execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, text)

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)


# ─────────────────────────────────────────────
# 3. VALIDATION & SCHEMA
# ─────────────────────────────────────────────

@dataclass
class PromptConstraint:
    """Declarative constraint on a prompt's token budget."""
    name: str
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    model_id: str = "gpt-4o"

    def validate(self, text: str) -> List[str]:
        violations = []
        n = estimate_tokens(text, self.model_id)
        if self.max_tokens and n > self.max_tokens:
            violations.append(f"[{self.name}] {n} tokens exceeds max {self.max_tokens}")
        if self.min_tokens and n < self.min_tokens:
            violations.append(f"[{self.name}] {n} tokens below min {self.min_tokens}")
        return violations


class PromptConstraintValidator:
    """Validate prompts against a set of declared token constraints."""

    def __init__(self) -> None:
        self._constraints: List[PromptConstraint] = []

    def add(self, constraint: PromptConstraint) -> "PromptConstraintValidator":
        self._constraints.append(constraint)
        return self

    def validate(self, text: str) -> List[str]:
        violations = []
        for c in self._constraints:
            violations.extend(c.validate(text))
        return violations

    def is_valid(self, text: str) -> bool:
        return len(self.validate(text)) == 0


class ConfidenceScorer:
    """0–1 confidence that a prompt optimization is beneficial."""

    def score(self, result: OptimizationResult) -> float:
        if result.original_tokens == 0:
            return 0.0
        savings_ratio = result.tokens_saved / result.original_tokens
        quality_heuristic = min(1.0, len(result.optimized_text) / max(len(result.original_text), 1) + 0.2)
        return round(min(1.0, savings_ratio * 0.6 + quality_heuristic * 0.4), 4)


# ─────────────────────────────────────────────
# 4. ASYNC & CONCURRENCY
# ─────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter for LLM API calls."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    def acquire(self, tokens: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
        return False

    async def aacquire(self, tokens: float = 1.0) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.acquire, tokens)


class CancellationToken:
    """Cooperative cancellation for async batch operations."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


def batch_optimize(
    prompts: List[str],
    optimize_fn: Callable[[str], OptimizationResult],
    max_workers: int = 4,
) -> List[OptimizationResult]:
    """Sync concurrent batch optimization."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(optimize_fn, prompts))


async def abatch_optimize(
    prompts: List[str],
    optimize_fn: Callable[[str], OptimizationResult],
    concurrency: int = 8,
    token: Optional[CancellationToken] = None,
) -> List[OptimizationResult]:
    """Async concurrent batch optimization."""
    sem = asyncio.Semaphore(concurrency)

    async def _opt(p: str) -> OptimizationResult:
        if token and token.is_cancelled:
            raise asyncio.CancelledError()
        async with sem:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, optimize_fn, p)

    return list(await asyncio.gather(*[_opt(p) for p in prompts]))


def optimize_with_budget(
    prompts: List[str],
    optimize_fn: Callable[[str], OptimizationResult],
    budget_seconds: float = 10.0,
) -> List[OptimizationResult]:
    """Optimize prompts within a wall-clock time budget."""
    results: List[OptimizationResult] = []
    deadline = time.monotonic() + budget_seconds
    for p in prompts:
        if time.monotonic() >= deadline:
            raise BudgetExceededError(f"Time budget of {budget_seconds}s exceeded after {len(results)} prompts.")
        results.append(optimize_fn(p))
    return results


# ─────────────────────────────────────────────
# 5. OBSERVABILITY
# ─────────────────────────────────────────────

class OperationProfiler:
    """Track timing and token savings per optimization call."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def profile(self, optimize_fn: Callable) -> Callable:
        """Wrap optimize_fn with timing and savings recording."""
        def wrapper(*args: Any, **kwargs: Any) -> OptimizationResult:
            start = time.perf_counter()
            result = optimize_fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self._records.append({
                "elapsed_s": round(elapsed, 6),
                "tokens_saved": result.tokens_saved,
                "compression_ratio": result.compression_ratio,
                "strategies": result.strategies_applied,
            })
            return result
        return wrapper

    def report(self) -> Dict[str, Any]:
        if not self._records:
            return {"calls": 0}
        elapsed_vals = [r["elapsed_s"] for r in self._records]
        savings = [r["tokens_saved"] for r in self._records]
        return {
            "calls": len(self._records),
            "mean_elapsed_s": sum(elapsed_vals) / len(elapsed_vals),
            "total_tokens_saved": sum(savings),
            "mean_tokens_saved": sum(savings) / len(savings),
            "records": self._records,
        }


class CostTelemetry:
    """Track cumulative token usage and cost across all LLM calls."""

    def __init__(self) -> None:
        self._usages: List[TokenUsage] = []
        self._lock = threading.Lock()

    def record(self, usage: TokenUsage) -> None:
        with self._lock:
            self._usages.append(usage)

    def total_cost(self) -> float:
        with self._lock:
            return sum(u.total_cost_usd for u in self._usages)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(u.input_tokens + u.output_tokens for u in self._usages)

    def by_model(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for u in self._usages:
                if u.model_id not in result:
                    result[u.model_id] = {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0}
                result[u.model_id]["calls"] += 1
                result[u.model_id]["total_tokens"] += u.input_tokens + u.output_tokens
                result[u.model_id]["total_cost_usd"] += u.total_cost_usd
            return result

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self._usages),
                "total_tokens": self.total_tokens(),
                "total_cost_usd": self.total_cost(),
            }


class DriftDetector:
    """Detect drift in optimization effectiveness across runs."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self._baseline_ratio: Optional[float] = None

    def set_baseline(self, result: OptimizationResult) -> None:
        self._baseline_ratio = result.compression_ratio

    def detect(self, result: OptimizationResult) -> float:
        if self._baseline_ratio is None:
            return 0.0
        drift = abs(result.compression_ratio - self._baseline_ratio)
        if drift >= self.threshold:
            logger.warning("Optimization drift detected: %.4f", drift)
        return round(drift, 4)


# ─────────────────────────────────────────────
# 6. STREAMING & STORAGE
# ─────────────────────────────────────────────

def stream_optimize(
    prompts: List[str],
    optimize_fn: Callable[[str], OptimizationResult],
) -> Generator[OptimizationResult, None, None]:
    """Generator that yields OptimizationResult one at a time."""
    for p in prompts:
        yield optimize_fn(p)


def results_to_ndjson(
    prompts: List[str],
    optimize_fn: Callable[[str], OptimizationResult],
) -> Generator[str, None, None]:
    """Stream NDJSON lines of OptimizationResult."""
    for result in stream_optimize(prompts, optimize_fn):
        yield json.dumps(result.model_dump(), default=str)


def results_to_csv(results: List[OptimizationResult]) -> str:
    """Convert list of OptimizationResult to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["original_tokens", "optimized_tokens", "tokens_saved", "compression_ratio", "strategies"])
    for r in results:
        writer.writerow([r.original_tokens, r.optimized_tokens, r.tokens_saved, r.compression_ratio, ",".join(r.strategies_applied)])
    return buf.getvalue()


# ─────────────────────────────────────────────
# 7. DIFF & REGRESSION
# ─────────────────────────────────────────────

@dataclass
class OptimizationDiff:
    """Diff between two OptimizationResults."""
    added_tokens: int
    removed_tokens: int
    compression_delta: float  # positive = more compressed in b
    strategy_added: List[str]
    strategy_removed: List[str]

    def summary(self) -> str:
        return (
            f"Token delta: {self.removed_tokens - self.added_tokens:+d}, "
            f"compression delta: {self.compression_delta:+.3f}, "
            f"strategies added: {self.strategy_added}, "
            f"strategies removed: {self.strategy_removed}"
        )

    def to_json(self) -> str:
        return json.dumps({
            "added_tokens": self.added_tokens,
            "removed_tokens": self.removed_tokens,
            "compression_delta": self.compression_delta,
            "strategy_added": self.strategy_added,
            "strategy_removed": self.strategy_removed,
        }, indent=2)


def diff_optimizations(a: OptimizationResult, b: OptimizationResult) -> OptimizationDiff:
    """Compare two OptimizationResults."""
    set_a = set(a.strategies_applied)
    set_b = set(b.strategies_applied)
    return OptimizationDiff(
        added_tokens=max(0, b.optimized_tokens - a.optimized_tokens),
        removed_tokens=max(0, a.optimized_tokens - b.optimized_tokens),
        compression_delta=round(a.compression_ratio - b.compression_ratio, 4),
        strategy_added=list(set_b - set_a),
        strategy_removed=list(set_a - set_b),
    )


class RegressionTracker:
    """Track optimization effectiveness across runs and detect regressions."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._history: deque = deque(maxlen=window)

    def record(self, result: OptimizationResult) -> None:
        self._history.append(result)

    def trend(self) -> str:
        if len(self._history) < 2:
            return "stable"
        ratios = [r.compression_ratio for r in self._history]
        delta = ratios[-1] - ratios[0]
        if delta < -0.05:
            return "improving"   # lower ratio = more compressed = better
        if delta > 0.05:
            return "declining"
        return "stable"

    def latest_regression(self) -> Optional[OptimizationDiff]:
        if len(self._history) < 2:
            return None
        a, b = list(self._history)[-2], list(self._history)[-1]
        return diff_optimizations(a, b)


class ScoreTrend:
    """Rolling token-savings trend with volatility."""

    def __init__(self, window: int = 10) -> None:
        self.window = window
        self._savings: deque = deque(maxlen=window)

    def record(self, tokens_saved: int) -> None:
        self._savings.append(tokens_saved)

    def trend(self) -> str:
        if len(self._savings) < 2:
            return "stable"
        savings = list(self._savings)
        delta = savings[-1] - savings[0]
        if delta > 5:
            return "improving"
        if delta < -5:
            return "declining"
        return "stable"

    def volatility(self) -> float:
        if len(self._savings) < 2:
            return 0.0
        vals = list(self._savings)
        mean = sum(vals) / len(vals)
        return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


# ─────────────────────────────────────────────
# 8. SECURITY & COST
# ─────────────────────────────────────────────

class PIIScrubber:
    """Detect and mask PII in prompts before LLM submission."""

    _PATTERNS = [
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
        (re.compile(r'\b\d{16}\b'), "[CARD]"),
        (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "[EMAIL]"),
        (re.compile(r'\b\d{3}[\-.\s]?\d{3}[\-.\s]?\d{4}\b'), "[PHONE]"),
    ]

    def scrub(self, text: str) -> str:
        for pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def has_pii(self, text: str) -> bool:
        return any(p.search(text) for p, _ in self._PATTERNS)


class AuditLog:
    """Append-only audit log for token optimization events."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log(self, event: str, data: Dict[str, Any]) -> None:
        entry = {"event": event, "timestamp": time.time(), **data}
        with self._lock:
            self._entries.append(entry)

    def to_json(self, indent: int = 2) -> str:
        with self._lock:
            return json.dumps(self._entries, indent=indent, default=str)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)


class BatchAPIRouter:
    """Route LLM calls to batch API for 50% cost savings when latency allows."""

    def __init__(
        self,
        registry: Optional[PricingRegistry] = None,
        latency_sensitive: bool = False,
    ) -> None:
        self._registry = registry or default_registry
        self.latency_sensitive = latency_sensitive

    def route(self, model_id: str, prompt: str) -> Tuple[str, bool]:
        """
        Return (effective_model_id, use_batch).
        use_batch=True when batch API is available and latency is acceptable.
        """
        pricing = self._registry.get(model_id)
        use_batch = pricing.supports_batch and not self.latency_sensitive
        return (model_id, use_batch)

    def effective_cost(self, model_id: str, tokens: int) -> float:
        """Calculate effective cost considering batch discount."""
        pricing = self._registry.get(model_id)
        cost_per_1k = pricing.input_cost_per_1k
        if self.route(model_id, "")[1]:  # use_batch
            cost_per_1k *= (1 - pricing.batch_discount)
        return (tokens / 1000) * cost_per_1k


class CostLedger:
    """Track all LLM costs for budget enforcement and reporting."""

    def __init__(self, budget_usd: Optional[float] = None) -> None:
        self.budget_usd = budget_usd
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, model_id: str, tokens: int, cost_usd: float) -> None:
        with self._lock:
            if self.budget_usd is not None:
                current = sum(e["cost_usd"] for e in self._entries)
                if current + cost_usd > self.budget_usd:
                    raise BudgetExceededError(
                        f"Cost budget of ${self.budget_usd:.4f} would be exceeded. "
                        f"Current: ${current:.4f}, New: ${cost_usd:.4f}"
                    )
            self._entries.append({
                "model_id": model_id,
                "tokens": tokens,
                "cost_usd": cost_usd,
                "timestamp": time.time(),
            })

    def total_cost(self) -> float:
        with self._lock:
            return sum(e["cost_usd"] for e in self._entries)

    def total_tokens(self) -> int:
        with self._lock:
            return sum(e["tokens"] for e in self._entries)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": len(self._entries),
                "total_tokens": self.total_tokens(),
                "total_cost_usd": self.total_cost(),
                "budget_usd": self.budget_usd,
                "budget_remaining_usd": (
                    round(self.budget_usd - self.total_cost(), 6)
                    if self.budget_usd is not None else None
                ),
            }


class ModelRouter:
    """Route LLM calls to cheap vs. frontier model based on prompt complexity."""

    def __init__(
        self,
        registry: Optional[PricingRegistry] = None,
        cheap_token_threshold: int = 500,
        cheap_tier: ModelTier = ModelTier.CHEAP,
        frontier_tier: ModelTier = ModelTier.FRONTIER,
    ) -> None:
        self._registry = registry or default_registry
        self.cheap_token_threshold = cheap_token_threshold
        self.cheap_tier = cheap_tier
        self.frontier_tier = frontier_tier

    def route(self, prompt: str) -> str:
        """Return recommended model_id based on prompt token length."""
        n = estimate_tokens(prompt)
        tier = self.cheap_tier if n <= self.cheap_token_threshold else self.frontier_tier
        pricing = self._registry.cheapest_for_tier(tier)
        logger.debug("ModelRouter: %d tokens → tier=%s → %s", n, tier, pricing.model_id)
        return pricing.model_id
