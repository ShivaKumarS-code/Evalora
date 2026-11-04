from __future__ import annotations

import pytest

from evalora.llm.costs import (
    TokenUsage,
    aggregate_usage,
    estimate_cost,
    make_usage,
)


def test_estimate_cost_for_gemini() -> None:
    gemini = estimate_cost(
        "gemini",
        "gemini-2.0-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    assert gemini == pytest.approx(0.50)


def test_unknown_model_costs_zero() -> None:
    assert estimate_cost("unknown", "mystery", prompt_tokens=100, completion_tokens=100) == 0.0


def test_make_usage_sets_totals_and_cost() -> None:
    usage = make_usage("gemini", "gemini-2.0-flash", prompt_tokens=100, completion_tokens=50)
    assert usage.total_tokens == 150
    assert usage.cost_usd == pytest.approx((100 * 0.10 + 50 * 0.40) / 1_000_000)


def test_aggregate_usage() -> None:
    totals = aggregate_usage(
        [
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.1),
            TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30, cost_usd=0.2),
        ]
    )
    assert totals.calls == 2
    assert totals.prompt_tokens == 30
    assert totals.completion_tokens == 15
    assert totals.total_tokens == 45
    assert totals.cost_usd == pytest.approx(0.3)
