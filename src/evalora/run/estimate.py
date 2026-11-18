"""Call-count and cost estimates for planning and budgeting runs.

The estimates are upper bounds built from documented constants: they assume
every case is judged (pipeline errors are not judged, so real spend is lower)
and every ambiguous abstention falls back to the LLM. They exist so a run can
be priced before it starts, not to predict exact spend.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from evalora.llm.costs import PRICES, estimate_cost
from evalora.schemas import TestCase

ESTIMATED_CLAIMS_PER_ANSWER = 3.0
ESTIMATED_ABSTENTION_LLM_FRACTION = 0.25
ESTIMATED_PROMPT_TOKENS_PER_CALL = 700
ESTIMATED_COMPLETION_TOKENS_PER_CALL = 80


class RunEstimate(BaseModel):
    pending_cases: int
    answerable_cases: int
    pipeline_calls: int
    judge_calls: int
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    price_known: bool
    estimated_cost_usd: float


def _judge_calls_for_case(case: TestCase) -> float:
    calls = 1.0 + ESTIMATED_CLAIMS_PER_ANSWER
    calls += ESTIMATED_ABSTENTION_LLM_FRACTION
    if case.gold_answer is not None:
        calls += 1.0
    return calls


def estimate_run(
    cases: Sequence[TestCase],
    *,
    provider: str,
    model: str,
    judge_enabled: bool,
) -> RunEstimate:
    """Estimate pipeline and judge calls, tokens, and approximate cost."""
    pending = len(cases)
    answerable = sum(1 for case in cases if case.gold_answer is not None)
    judge_calls = 0.0
    if judge_enabled:
        judge_calls = sum(_judge_calls_for_case(case) for case in cases)
    judge_call_count = int(judge_calls) if judge_calls else 0
    estimated_prompt_tokens = judge_call_count * ESTIMATED_PROMPT_TOKENS_PER_CALL
    estimated_completion_tokens = judge_call_count * ESTIMATED_COMPLETION_TOKENS_PER_CALL
    price_known = (provider, model) in PRICES
    estimated_cost_usd = (
        estimate_cost(
            provider,
            model,
            prompt_tokens=estimated_prompt_tokens,
            completion_tokens=estimated_completion_tokens,
        )
        if judge_enabled
        else 0.0
    )
    return RunEstimate(
        pending_cases=pending,
        answerable_cases=answerable,
        pipeline_calls=pending,
        judge_calls=judge_call_count,
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimated_completion_tokens=estimated_completion_tokens,
        price_known=price_known,
        estimated_cost_usd=estimated_cost_usd,
    )
