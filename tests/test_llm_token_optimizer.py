"""Tests for llm-token-optimizer core and advanced features."""
import asyncio
import pytest
from llm_token_optimizer.models import ModelTier, ModelPricing, OptimizationResult, TokenUsage
from llm_token_optimizer.pricing import PricingRegistry, default_registry
from llm_token_optimizer.counter import estimate_tokens
from llm_token_optimizer.optimizer import optimize_prompt
from llm_token_optimizer.estimator import CostEstimator
from llm_token_optimizer.exceptions import (
    BudgetExceededError, ModelNotSupportedError, OptimizationError,
)
from llm_token_optimizer.advanced import (
    OptimizationCache, SemanticCache, OptimizationPipeline,
    PromptConstraint, PromptConstraintValidator, ConfidenceScorer,
    RateLimiter, CancellationToken, batch_optimize, abatch_optimize,
    optimize_with_budget, OperationProfiler, CostTelemetry, DriftDetector,
    stream_optimize, results_to_ndjson, results_to_csv,
    diff_optimizations, RegressionTracker, ScoreTrend,
    PIIScrubber, AuditLog, BatchAPIRouter, CostLedger, ModelRouter,
)

LONG_PROMPT = (
    "Please note that you should summarize the following document. "
    "As an AI language model, I would be happy to help. "
    "The quick brown fox jumps over the lazy dog. " * 10
)

SHORT_PROMPT = "Summarize: The cat sat on the mat."


# ──────────────── Counter ────────────────

def test_estimate_tokens_basic():
    n = estimate_tokens("Hello world this is a test")
    assert n > 0

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


# ──────────────── Pricing ────────────────

def test_default_registry_has_models():
    models = default_registry.list_models()
    assert "gpt-4o" in models
    assert "claude-sonnet-4-6" in models

def test_registry_get():
    pricing = default_registry.get("gpt-4o-mini")
    assert pricing.tier == ModelTier.CHEAP
    assert pricing.input_cost_per_1k > 0

def test_registry_model_not_found():
    reg = PricingRegistry()
    with pytest.raises(ModelNotSupportedError):
        reg.get("nonexistent-model-xyz")

def test_registry_cheapest_for_tier():
    pricing = default_registry.cheapest_for_tier(ModelTier.CHEAP)
    assert pricing.tier == ModelTier.CHEAP

def test_registry_register_custom():
    reg = PricingRegistry()
    custom = ModelPricing(model_id="custom-model", tier=ModelTier.STANDARD,
                          input_cost_per_1k=0.001, output_cost_per_1k=0.002,
                          context_window=8192)
    reg.register(custom)
    assert reg.get("custom-model").model_id == "custom-model"


# ──────────────── Optimizer ────────────────

def test_optimize_prompt_whitespace():
    text = "Hello    world\n\n\n\nTest"
    result = optimize_prompt(text, strategies=["whitespace"])
    assert result.original_tokens > 0
    assert result.optimized_tokens > 0

def test_optimize_prompt_fillers():
    text = "Please note that the sky is blue. Certainly! I'd be happy to help."
    result = optimize_prompt(text, strategies=["fillers"])
    assert result.tokens_saved >= 0

def test_optimize_prompt_dedup():
    block = "The quick brown fox.\n\n"
    text = block * 3
    result = optimize_prompt(text, strategies=["dedup"])
    assert result.optimized_tokens <= result.original_tokens

def test_optimize_prompt_unknown_strategy():
    with pytest.raises(OptimizationError):
        optimize_prompt("text", strategies=["unknown_strategy"])

def test_optimize_prompt_all_strategies():
    result = optimize_prompt(LONG_PROMPT, strategies=["whitespace", "fillers", "dedup"])
    assert isinstance(result, OptimizationResult)
    assert 0.0 <= result.compression_ratio <= 1.5


# ──────────────── Estimator ────────────────

def test_cost_estimator():
    estimator = CostEstimator()
    estimate = estimator.estimate("gpt-4o", "Hello world", estimated_output_tokens=50)
    assert estimate.input_tokens > 0
    assert estimate.total_cost_usd > 0

def test_cost_estimator_cheapest_model():
    estimator = CostEstimator()
    estimate = estimator.cheapest_model("Hello world", tier=ModelTier.CHEAP)
    assert estimate.tier == ModelTier.CHEAP

def test_compare_models():
    estimator = CostEstimator()
    results = estimator.compare_models(["gpt-4o", "gpt-4o-mini"], "Hello world")
    assert len(results) == 2
    assert results[0].total_cost_usd <= results[1].total_cost_usd


# ──────────────── Cache ────────────────

def test_optimization_cache_hit_miss():
    cache = OptimizationCache(max_size=10, ttl=60)
    strategies = ["whitespace"]
    assert cache.get(LONG_PROMPT, strategies) is None
    result = optimize_prompt(LONG_PROMPT, strategies=strategies)
    cache.put(LONG_PROMPT, strategies, result)
    assert cache.get(LONG_PROMPT, strategies) is not None
    stats = cache.stats()
    assert stats["hits"] == 1

def test_cache_memoize():
    cache = OptimizationCache()
    calls = []
    def fake_optimize(text, strategies=("whitespace",), **kw):
        calls.append(1)
        return optimize_prompt(text, strategies=list(strategies))
    memoized = cache.memoize(fake_optimize)
    memoized(SHORT_PROMPT)
    memoized(SHORT_PROMPT)
    assert len(calls) == 1

def test_semantic_cache():
    sc = SemanticCache(threshold=0.5)
    result = optimize_prompt(SHORT_PROMPT)
    sc.put(SHORT_PROMPT, result)
    # Same text should hit
    hit = sc.get(SHORT_PROMPT)
    assert hit is not None


# ──────────────── Pipeline ────────────────

def test_optimization_pipeline():
    pipeline = (
        OptimizationPipeline()
        .map("strip", lambda t: t.strip())
        .map("lower_check", lambda t: t)
    )
    result = pipeline.run(LONG_PROMPT)
    assert isinstance(result, str)
    assert len(pipeline.audit_log) == 2

def test_pipeline_branch():
    pipeline = OptimizationPipeline()
    pipeline.branch(
        condition=lambda t: len(t) > 50,
        true_fn=lambda t: t[:50],
        false_fn=lambda t: t.upper(),
    )
    long_result = pipeline.run("x" * 100)
    assert len(long_result) == 50

def test_pipeline_arun():
    pipeline = OptimizationPipeline().map("noop", lambda t: t)
    result = asyncio.run(pipeline.arun(SHORT_PROMPT))
    assert isinstance(result, str)


# ──────────────── Validation ────────────────

def test_constraint_validator_pass():
    v = PromptConstraintValidator()
    v.add(PromptConstraint("limit", max_tokens=10000, model_id="gpt-4o"))
    assert v.is_valid(SHORT_PROMPT)

def test_constraint_validator_fail():
    v = PromptConstraintValidator()
    v.add(PromptConstraint("tiny", max_tokens=1, model_id="gpt-4o"))
    violations = v.validate(SHORT_PROMPT)
    assert len(violations) > 0

def test_confidence_scorer():
    result = optimize_prompt(LONG_PROMPT, strategies=["whitespace", "fillers"])
    cs = ConfidenceScorer()
    score = cs.score(result)
    assert 0.0 <= score <= 1.0


# ──────────────── Rate Limiter ────────────────

def test_rate_limiter_sync():
    rl = RateLimiter(rate=100, capacity=10)
    assert rl.acquire(5)

def test_rate_limiter_async():
    rl = RateLimiter(rate=100, capacity=10)
    assert asyncio.run(rl.aacquire(3))


# ──────────────── Batch ────────────────

def test_batch_optimize():
    prompts = [SHORT_PROMPT] * 4
    results = batch_optimize(prompts, lambda p: optimize_prompt(p))
    assert len(results) == 4

def test_abatch_optimize():
    prompts = [SHORT_PROMPT] * 3
    results = asyncio.run(abatch_optimize(prompts, lambda p: optimize_prompt(p)))
    assert len(results) == 3

def test_optimize_with_budget_exceeded():
    prompts = [SHORT_PROMPT] * 100
    with pytest.raises(BudgetExceededError):
        optimize_with_budget(prompts, lambda p: optimize_prompt(p), budget_seconds=0.000001)


# ──────────────── Observability ────────────────

def test_operation_profiler():
    profiler = OperationProfiler()
    profiled = profiler.profile(optimize_prompt)
    profiled(SHORT_PROMPT)
    report = profiler.report()
    assert report["calls"] == 1
    assert "total_tokens_saved" in report

def test_cost_telemetry():
    telemetry = CostTelemetry()
    usage = TokenUsage(model_id="gpt-4o", input_tokens=100, output_tokens=50,
                       input_cost_usd=0.0005, output_cost_usd=0.00075, total_cost_usd=0.00125)
    telemetry.record(usage)
    assert telemetry.total_cost() > 0
    assert telemetry.total_tokens() == 150

def test_drift_detector():
    result = optimize_prompt(SHORT_PROMPT)
    dd = DriftDetector(threshold=0.0)
    dd.set_baseline(result)
    drift = dd.detect(result)
    assert drift == 0.0


# ──────────────── Streaming ────────────────

def test_stream_optimize():
    prompts = [SHORT_PROMPT] * 3
    results = list(stream_optimize(prompts, lambda p: optimize_prompt(p)))
    assert len(results) == 3

def test_results_to_ndjson():
    prompts = [SHORT_PROMPT] * 2
    import json
    lines = list(results_to_ndjson(prompts, lambda p: optimize_prompt(p)))
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "original_tokens" in obj

def test_results_to_csv():
    results = [optimize_prompt(SHORT_PROMPT)]
    csv_str = results_to_csv(results)
    assert "original_tokens" in csv_str


# ──────────────── Diff & Regression ────────────────

def test_diff_optimizations():
    r1 = optimize_prompt(LONG_PROMPT, strategies=["whitespace"])
    r2 = optimize_prompt(LONG_PROMPT, strategies=["whitespace", "fillers"])
    diff = diff_optimizations(r1, r2)
    assert diff.summary() is not None
    assert isinstance(diff.to_json(), str)

def test_regression_tracker():
    tracker = RegressionTracker(window=5)
    for _ in range(3):
        tracker.record(optimize_prompt(SHORT_PROMPT))
    assert tracker.trend() in ("improving", "declining", "stable")
    assert tracker.latest_regression() is not None

def test_score_trend():
    trend = ScoreTrend(window=5)
    for s in [10, 15, 20, 25, 30]:
        trend.record(s)
    assert trend.trend() == "improving"
    assert trend.volatility() >= 0.0


# ──────────────── Security & Cost ────────────────

def test_pii_scrubber():
    scrubber = PIIScrubber()
    text = "Email: user@example.com, SSN: 123-45-6789"
    scrubbed = scrubber.scrub(text)
    assert "[EMAIL]" in scrubbed
    assert "[SSN]" in scrubbed
    assert scrubber.has_pii(text)

def test_audit_log():
    log = AuditLog()
    log.log("optimize", {"tokens_saved": 50})
    assert len(log.entries) == 1
    assert "optimize" in log.to_json()

def test_batch_api_router():
    router = BatchAPIRouter(latency_sensitive=False)
    model_id, use_batch = router.route("gpt-4o", SHORT_PROMPT)
    assert model_id == "gpt-4o"
    assert use_batch is True
    cost = router.effective_cost("gpt-4o", 1000)
    assert cost > 0

def test_cost_ledger():
    ledger = CostLedger(budget_usd=1.0)
    ledger.record("gpt-4o", tokens=500, cost_usd=0.01)
    s = ledger.summary()
    assert s["calls"] == 1
    assert s["budget_remaining_usd"] is not None

def test_cost_ledger_budget_exceeded():
    ledger = CostLedger(budget_usd=0.001)
    with pytest.raises(BudgetExceededError):
        ledger.record("gpt-4o", tokens=10000, cost_usd=0.002)

def test_model_router():
    router = ModelRouter(cheap_token_threshold=500)
    model = router.route(SHORT_PROMPT)
    assert isinstance(model, str)
    assert len(model) > 0
