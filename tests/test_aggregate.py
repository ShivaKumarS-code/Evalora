from __future__ import annotations

import pytest

from evalora.evaluate.aggregate import aggregate_run
from evalora.llm.costs import UsageTotals
from evalora.schemas import FailureCategory, QuestionType
from tests.factories import make_result


def _results() -> list:
    return [
        make_result(
            "case-1",
            QuestionType.SINGLE_HOP,
            faithfulness=1.0,
            correct=True,
            hit=True,
            reciprocal_rank=1.0,
            context_recall=1.0,
            latency_ms=10.0,
            usage=UsageTotals(
                calls=1,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                cost_usd=0.01,
            ),
        ),
        make_result(
            "case-2",
            QuestionType.SINGLE_HOP,
            faithfulness=0.5,
            correct=False,
            hit=False,
            reciprocal_rank=0.0,
            context_recall=0.5,
            failure_category=FailureCategory.RETRIEVAL_FAILURE,
            latency_ms=20.0,
            usage=UsageTotals(
                calls=2,
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
                cost_usd=0.02,
            ),
        ),
        make_result(
            "case-3",
            QuestionType.UNANSWERABLE,
            applicable=False,
            abstained=True,
            latency_ms=30.0,
        ),
    ]


def test_aggregate_computes_means_counts_and_percentiles() -> None:
    aggregate = aggregate_run(_results(), resamples=200, seed=7)

    assert aggregate.cases == 3
    assert aggregate.faithfulness.mean == pytest.approx(0.75)
    assert aggregate.faithfulness.n == 2
    assert aggregate.correctness.mean == pytest.approx(0.5)
    assert aggregate.correctness.n == 2
    assert aggregate.hit_rate.mean == pytest.approx(0.5)
    assert aggregate.mrr.mean == pytest.approx(0.5)
    assert aggregate.context_recall.mean == pytest.approx(0.75)
    assert aggregate.failure_counts[FailureCategory.RETRIEVAL_FAILURE] == 1
    assert aggregate.failure_counts[FailureCategory.HALLUCINATION] == 0
    assert aggregate.latency_p50 == pytest.approx(20.0)
    assert aggregate.latency_p95 == pytest.approx(29.0)
    assert aggregate.token_usage.calls == 3
    assert aggregate.token_usage.total_tokens == 45
    assert aggregate.token_usage.cost_usd == pytest.approx(0.03)
    assert aggregate.by_type[QuestionType.SINGLE_HOP].cases == 2
    assert aggregate.by_type[QuestionType.UNANSWERABLE].cases == 1
    assert aggregate.by_type[QuestionType.MULTI_HOP].cases == 0


def test_bootstrap_ci_is_seeded_and_contains_the_mean() -> None:
    first = aggregate_run(_results(), resamples=200, seed=7)
    second = aggregate_run(_results(), resamples=200, seed=7)

    assert first.model_dump() == second.model_dump()
    assert first.faithfulness.ci_low is not None
    assert first.faithfulness.ci_high is not None
    assert first.faithfulness.ci_low <= first.faithfulness.mean <= first.faithfulness.ci_high


def test_empty_results_return_empty_metrics() -> None:
    aggregate = aggregate_run([])

    assert aggregate.cases == 0
    assert aggregate.faithfulness.mean is None
    assert aggregate.faithfulness.n == 0
    assert aggregate.latency_p50 is None
    assert aggregate.latency_p95 is None
    assert aggregate.token_usage.calls == 0
    assert all(count == 0 for count in aggregate.failure_counts.values())


def test_invalid_bootstrap_settings_raise() -> None:
    with pytest.raises(ValueError):
        aggregate_run([], confidence=1.0)
    with pytest.raises(ValueError):
        aggregate_run([], resamples=0)
