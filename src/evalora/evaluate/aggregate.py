"""Run aggregation and seeded bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

from evalora.llm.costs import UsageTotals
from evalora.schemas import CaseResult, FailureCategory, QuestionType

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42


class MetricSummary(BaseModel):
    mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = 0


class TypeAggregate(BaseModel):
    cases: int
    faithfulness: MetricSummary
    correctness: MetricSummary
    hit_rate: MetricSummary
    mrr: MetricSummary
    context_recall: MetricSummary


class RunAggregate(BaseModel):
    cases: int
    faithfulness: MetricSummary
    correctness: MetricSummary
    hit_rate: MetricSummary
    mrr: MetricSummary
    context_recall: MetricSummary
    failure_counts: dict[FailureCategory, int]
    by_type: dict[QuestionType, TypeAggregate]
    latency_p50: float | None
    latency_p95: float | None
    token_usage: UsageTotals
    ci_confidence: float
    bootstrap_resamples: int
    seed: int


def _metric(
    values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> MetricSummary:
    if not values:
        return MetricSummary()
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(values) == 1:
        return MetricSummary(mean=mean, ci_low=mean, ci_high=mean, n=1)

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = array[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return MetricSummary(
        mean=mean,
        ci_low=float(np.quantile(bootstrap_means, alpha)),
        ci_high=float(np.quantile(bootstrap_means, 1.0 - alpha)),
        n=len(values),
    )


def _faithfulness_values(results: Sequence[CaseResult]) -> list[float]:
    return [result.faithfulness for result in results if result.faithfulness is not None]


def _correctness_values(results: Sequence[CaseResult]) -> list[float]:
    return [1.0 if result.correct else 0.0 for result in results if result.correct is not None]


def _retrieval_values(
    results: Sequence[CaseResult],
    extract: Callable[[CaseResult], float],
) -> list[float]:
    return [extract(result) for result in results if result.retrieval.applicable]


def _hit_value(result: CaseResult) -> float:
    return 1.0 if result.retrieval.hit else 0.0


def _mrr_value(result: CaseResult) -> float:
    return result.retrieval.reciprocal_rank


def _recall_value(result: CaseResult) -> float:
    return result.retrieval.context_recall


def _type_aggregate(
    results: Sequence[CaseResult],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> TypeAggregate:
    return TypeAggregate(
        cases=len(results),
        faithfulness=_metric(
            _faithfulness_values(results),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        correctness=_metric(
            _correctness_values(results),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        hit_rate=_metric(
            _retrieval_values(results, _hit_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        mrr=_metric(
            _retrieval_values(results, _mrr_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        context_recall=_metric(
            _retrieval_values(results, _recall_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
    )


def _sum_usage(results: Sequence[CaseResult]) -> UsageTotals:
    totals = UsageTotals()
    for result in results:
        totals.calls += result.usage.calls
        totals.prompt_tokens += result.usage.prompt_tokens
        totals.completion_tokens += result.usage.completion_tokens
        totals.total_tokens += result.usage.total_tokens
        totals.cost_usd += result.usage.cost_usd
    return totals


def aggregate_run(
    results: Sequence[CaseResult],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> RunAggregate:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    latencies = [result.response.latency_ms for result in results]
    failure_counts = {
        category: sum(1 for result in results if result.failure_category == category)
        for category in FailureCategory
    }
    return RunAggregate(
        cases=len(results),
        faithfulness=_metric(
            _faithfulness_values(results),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        correctness=_metric(
            _correctness_values(results),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        hit_rate=_metric(
            _retrieval_values(results, _hit_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        mrr=_metric(
            _retrieval_values(results, _mrr_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        context_recall=_metric(
            _retrieval_values(results, _recall_value),
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        ),
        failure_counts=failure_counts,
        by_type={
            question_type: _type_aggregate(
                [result for result in results if result.case.type == question_type],
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
            for question_type in QuestionType
        },
        latency_p50=None if not latencies else float(np.percentile(latencies, 50)),
        latency_p95=None if not latencies else float(np.percentile(latencies, 95)),
        token_usage=_sum_usage(results),
        ci_confidence=confidence,
        bootstrap_resamples=resamples,
        seed=seed,
    )
