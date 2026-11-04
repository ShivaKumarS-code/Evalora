"""Token usage tracking and approximate LLM pricing.

Prices are public list prices in USD per million tokens kept in one place so
cost numbers stay reproducible. They are approximate: providers change pricing,
and cached calls do not incur new spend.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[tuple[str, str], ModelPrice] = {
    ("gemini", "gemini-2.0-flash"): ModelPrice(input_per_mtok=0.10, output_per_mtok=0.40),
}


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class UsageTotals(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def estimate_cost(
    provider: str, model: str, *, prompt_tokens: int, completion_tokens: int
) -> float:
    price = PRICES.get((provider, model))
    if price is None:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * price.input_per_mtok
        + completion_tokens / 1_000_000 * price.output_per_mtok
    )


def make_usage(
    provider: str,
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
) -> TokenUsage:
    total = prompt_tokens + completion_tokens if total_tokens is None else total_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
        cost_usd=estimate_cost(
            provider, model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def aggregate_usage(usages: Iterable[TokenUsage]) -> UsageTotals:
    totals = UsageTotals()
    for usage in usages:
        totals.calls += 1
        totals.prompt_tokens += usage.prompt_tokens
        totals.completion_tokens += usage.completion_tokens
        totals.total_tokens += usage.total_tokens
        totals.cost_usd += usage.cost_usd
    return totals
