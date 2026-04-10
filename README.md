# llm-token-optimizer

**Token cost control and auto-optimization for LLM applications.**

Compress prompts, estimate costs before calls, enforce budgets, route to cheap models, and cut LLM spend by up to 60% — with no vendor lock-in.

```bash
pip install llm-token-optimizer
```

---

## Why llm-token-optimizer?

In 2026, LLM API costs are the #1 operational expense for AI teams. Every wasted token costs money. Teams have no easy way to:
- Estimate cost **before** making an API call
- Compress prompts without breaking them  
- Route requests to cheaper models automatically
- Enforce per-day or per-job token budgets
- Detect cost drift across model upgrades

`llm-token-optimizer` fixes all of this — with a clean, provider-agnostic API.

---

## Quickstart

```python
from llm_token_optimizer import (
    optimize_prompt, CostEstimator, estimate_tokens,
)

prompt = """
Please note that you should summarize the following document.
As an AI language model, I'd be happy to help.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
"""

# Step 1: Estimate cost before calling the LLM
estimator = CostEstimator()
estimate = estimator.estimate("gpt-4o", prompt, estimated_output_tokens=200)
print(f"Estimated cost: ${estimate.total_cost_usd:.6f}")
print(f"Input tokens: {estimate.input_tokens}")

# Step 2: Optimize the prompt
result = optimize_prompt(prompt, strategies=["whitespace", "fillers", "dedup"])
print(f"Tokens saved: {result.tokens_saved}")
print(f"Compression ratio: {result.compression_ratio:.2f}")
print(result.optimized_text)
```

---

## Built-in Optimization Strategies

| Strategy | Description |
|----------|-------------|
| `whitespace` | Collapse redundant spaces and blank lines |
| `fillers` | Remove filler phrases ("Please note that", "As an AI...") |
| `dedup` | Remove repeated paragraphs |
| `examples` | Trim few-shot examples to first N (default 3) |

---

## Model Pricing (2026 catalog)

Pre-loaded pricing for OpenAI, Anthropic, Google, and Mistral:

```python
from llm_token_optimizer import CostEstimator, ModelTier

estimator = CostEstimator()

# Compare models before choosing
results = estimator.compare_models(
    ["gpt-4o", "gpt-4o-mini", "claude-haiku-4-5-20251001"],
    prompt="Your prompt here",
    estimated_output_tokens=300,
)
for r in results:
    print(f"{r.model_id}: ${r.total_cost_usd:.6f}")

# Find cheapest in a tier
cheapest = estimator.cheapest_model(prompt, tier=ModelTier.CHEAP)
```

---

## Advanced Features

### Caching (LRU + TTL + SHA-256)

```python
from llm_token_optimizer.advanced import OptimizationCache

cache = OptimizationCache(max_size=1000, ttl=600)
memoized = cache.memoize(optimize_prompt)
result = memoized(prompt, ["whitespace", "fillers"])  # cached on second call
print(cache.stats())
```

### Semantic Cache (cosine similarity)

```python
from llm_token_optimizer.advanced import SemanticCache

sc = SemanticCache(threshold=0.92)
sc.put(prompt, result)
cached = sc.get("similar prompt text...")  # returns if similarity >= 0.92
```

### Optimization Pipeline

```python
from llm_token_optimizer.advanced import OptimizationPipeline

pipeline = (
    OptimizationPipeline()
    .map("strip", lambda t: t.strip())
    .filter("non_empty", lambda t: len(t) > 0)
    .branch(
        condition=lambda t: len(t) > 2000,
        true_fn=lambda t: t[:2000],
        false_fn=lambda t: t,
    )
    .with_retry("strip", retries=2)
)
optimized = pipeline.run(prompt)
print(pipeline.audit_log)

import asyncio
optimized = asyncio.run(pipeline.arun(prompt))
```

### Declarative Token Constraints

```python
from llm_token_optimizer.advanced import PromptConstraintValidator, PromptConstraint

validator = (
    PromptConstraintValidator()
    .add(PromptConstraint("context_limit", max_tokens=4096, model_id="gpt-4o"))
    .add(PromptConstraint("min_content", min_tokens=10, model_id="gpt-4o"))
)
violations = validator.validate(prompt)
```

### PII Scrubbing

```python
from llm_token_optimizer.advanced import PIIScrubber

scrubber = PIIScrubber()
clean = scrubber.scrub("Contact: john@example.com, SSN: 123-45-6789")
# → "Contact: [EMAIL], SSN: [SSN]"
```

### Rate Limiter (sync + async)

```python
from llm_token_optimizer.advanced import RateLimiter
import asyncio

limiter = RateLimiter(rate=10, capacity=10)  # 10 calls/s
if limiter.acquire():
    result = optimize_prompt(prompt)
```

### Async Batch Optimization

```python
from llm_token_optimizer.advanced import abatch_optimize, batch_optimize
import asyncio

prompts = [prompt1, prompt2, prompt3]
results = asyncio.run(abatch_optimize(prompts, optimize_prompt, concurrency=8))
results = batch_optimize(prompts, optimize_prompt, max_workers=4)
```

### Budget-Controlled Optimization

```python
from llm_token_optimizer.advanced import optimize_with_budget

results = optimize_with_budget(prompts, optimize_prompt, budget_seconds=5.0)
```

### Observability

```python
from llm_token_optimizer.advanced import OperationProfiler, CostTelemetry, DriftDetector

# Timing profiler
profiler = OperationProfiler()
profiled = profiler.profile(optimize_prompt)
profiled(prompt)
print(profiler.report())

# Cost tracking
telemetry = CostTelemetry()
from llm_token_optimizer.models import TokenUsage
telemetry.record(TokenUsage(model_id="gpt-4o", input_tokens=500, output_tokens=100,
                            input_cost_usd=0.0025, output_cost_usd=0.0015, total_cost_usd=0.004))
print(telemetry.summary())
print(telemetry.by_model())

# Drift detection
drift_detector = DriftDetector(threshold=0.05)
drift_detector.set_baseline(result_v1)
drift = drift_detector.detect(result_v2)
```

### Streaming

```python
from llm_token_optimizer.advanced import stream_optimize, results_to_ndjson, results_to_csv

for result in stream_optimize(prompts, optimize_prompt):
    print(result.tokens_saved)

for line in results_to_ndjson(prompts, optimize_prompt):
    print(line)

csv_str = results_to_csv(results)
```

### Diff & Regression Tracking

```python
from llm_token_optimizer.advanced import diff_optimizations, RegressionTracker, ScoreTrend

diff = diff_optimizations(result_v1, result_v2)
print(diff.summary())
print(diff.to_json())

tracker = RegressionTracker(window=20)
tracker.record(result_v1)
tracker.record(result_v2)
print(tracker.trend())  # "improving" / "declining" / "stable"

trend = ScoreTrend(window=10)
trend.record(result.tokens_saved)
print(trend.trend(), trend.volatility())
```

### Cost Ledger, Batch API Router, Model Router

```python
from llm_token_optimizer.advanced import CostLedger, BatchAPIRouter, ModelRouter

# Hard budget enforcement
ledger = CostLedger(budget_usd=5.0)
ledger.record("gpt-4o", tokens=1000, cost_usd=0.005)
print(ledger.summary())  # raises BudgetExceededError if over budget

# 50% batch discount routing
router = BatchAPIRouter(latency_sensitive=False)
model_id, use_batch = router.route("gpt-4o", prompt)
effective_cost = router.effective_cost("gpt-4o", tokens=10000)

# Auto-route cheap vs. frontier
model_router = ModelRouter(cheap_token_threshold=500)
recommended_model = model_router.route(prompt)  # e.g. "gemini-2.0-flash" for short prompts
```

### Audit Log

```python
from llm_token_optimizer.advanced import AuditLog

log = AuditLog()
log.log("optimize", {"tokens_saved": 150, "model": "gpt-4o"})
print(log.to_json())
```

---

## Custom Model Pricing

```python
from llm_token_optimizer import PricingRegistry, ModelPricing, ModelTier

registry = PricingRegistry()
registry.register(ModelPricing(
    model_id="my-fine-tuned-model",
    tier=ModelTier.STANDARD,
    input_cost_per_1k=0.002,
    output_cost_per_1k=0.006,
    context_window=32768,
    supports_batch=True,
    batch_discount=0.50,
))
```

---

## Installation

```bash
pip install llm-token-optimizer

# With exact tiktoken counting (optional):
pip install "llm-token-optimizer[tiktoken]"
```

Python 3.8+ · No external dependencies (stdlib + pydantic; tiktoken optional)

---

## Changelog

### v1.0.0 (2026-04-10)
- Added Changelog section to README for release traceability
- Initial release: token cost estimation, prompt compression, budget enforcement, model routing

## License

MIT

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository on [GitHub](https://github.com/maheshmakvana/llm-token-optimizer)
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

Please open an issue first for major changes to discuss the approach.

## Author

**Mahesh Makvana** — [GitHub](https://github.com/maheshmakvana) · [PyPI](https://pypi.org/user/maheshmakvana/)

MIT License
