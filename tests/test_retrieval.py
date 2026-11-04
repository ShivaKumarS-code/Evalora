from __future__ import annotations

import pytest

from evalora.evaluate.retrieval import (
    aggregate_retrieval,
    compute_retrieval,
    match_contexts,
    text_match_score,
)
from evalora.schemas import (
    CaseResult,
    FailureCategory,
    PipelineResponse,
    QuestionType,
    RetrievalScores,
    RetrievedContext,
)
from evalora.schemas import (
    TestCase as EvalTestCase,
)

GOLD_TEXT = "TechMart was founded in 1994 in Portland, Oregon."


def _case(
    *,
    question_type: QuestionType = QuestionType.SINGLE_HOP,
    gold_chunk_ids: list[str] | None = None,
) -> EvalTestCase:
    return EvalTestCase(
        case_id="case-1",
        type=question_type,
        question="When was TechMart founded?",
        gold_answer="1994",
        gold_chunk_ids=["c1"] if gold_chunk_ids is None else gold_chunk_ids,
        source_model="fake",
        answerability_verdict="answerable",
        metadata={},
    )


def _context(
    context_id: str | None, text: str = "TechMart was founded in 1994."
) -> RetrievedContext:
    return RetrievedContext(id=context_id, text=text)


def _result(
    case: EvalTestCase,
    *,
    applicable: bool = True,
    hit: bool = False,
    reciprocal_rank: float = 0.0,
    context_recall: float = 0.0,
    contexts: list[RetrievedContext] | None = None,
    error: str | None = None,
) -> CaseResult:
    response = PipelineResponse(
        answer="answer",
        contexts=contexts or [],
        latency_ms=1.0,
        error=error,
    )
    retrieval = RetrievalScores(
        applicable=applicable,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        context_recall=context_recall,
    )
    return CaseResult(
        case=case,
        response=response,
        retrieval=retrieval,
        faithfulness=None,
        claim_verdicts=[],
        correct=None,
        abstained=False,
        failure_category=FailureCategory.PIPELINE_ERROR if error else None,
    )


def test_hit_at_rank_one() -> None:
    scores = compute_retrieval(
        _case(),
        [_context("c1")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.applicable is True
    assert scores.hit is True
    assert scores.reciprocal_rank == 1.0
    assert scores.context_recall == 1.0


def test_hit_at_rank_two() -> None:
    scores = compute_retrieval(
        _case(),
        [_context("other", "unrelated"), _context("c1")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.reciprocal_rank == 0.5


def test_partial_context_recall() -> None:
    scores = compute_retrieval(
        _case(gold_chunk_ids=["c1", "c2"]),
        [_context("c1")],
        {"c1": GOLD_TEXT, "c2": "The first store opened in March 1994."},
        threshold=0.8,
    )

    assert scores.hit is True
    assert scores.context_recall == 0.5


def test_duplicate_retrievals_count_once_for_recall() -> None:
    scores = compute_retrieval(
        _case(gold_chunk_ids=["c1", "c2"]),
        [_context("c1"), _context("c1"), _context("c2")],
        {"c1": GOLD_TEXT, "c2": "The first store opened in March 1994."},
        threshold=0.8,
    )

    assert scores.context_recall == 1.0
    assert scores.reciprocal_rank == 1.0


def test_tied_matches_use_the_first_rank() -> None:
    scores = compute_retrieval(
        _case(),
        [_context("c1"), _context("c1")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.reciprocal_rank == 1.0
    assert scores.context_recall == 1.0


def test_unanswerable_cases_are_not_scored() -> None:
    scores = compute_retrieval(
        _case(
            question_type=QuestionType.UNANSWERABLE,
            gold_chunk_ids=[],
        ),
        [_context("c1")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.applicable is False
    assert scores.hit is False
    assert scores.reciprocal_rank == 0.0
    assert scores.context_recall == 0.0


def test_answerable_case_without_gold_chunks_is_not_scored() -> None:
    scores = compute_retrieval(
        _case(gold_chunk_ids=[]),
        [_context("c1")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.applicable is False


def test_text_fallback_matches_by_containment() -> None:
    scores = compute_retrieval(
        _case(),
        [_context(None, f"Intro sentence. {GOLD_TEXT} More text follows.")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.hit is True
    assert scores.context_recall == 1.0


def test_text_fallback_matches_by_token_overlap() -> None:
    context = _context(None, "TechMart was founded in Portland in 1994 by Elena Park.")
    assert text_match_score(GOLD_TEXT, context.text) >= 0.8

    scores = compute_retrieval(_case(), [context], {"c1": GOLD_TEXT}, threshold=0.8)
    assert scores.hit is True


def test_text_fallback_below_threshold_does_not_match() -> None:
    scores = compute_retrieval(
        _case(),
        [_context(None, "The weather in Portland is often rainy.")],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.hit is False
    assert scores.context_recall == 0.0


def test_returned_id_is_authoritative() -> None:
    scores = compute_retrieval(
        _case(),
        [_context("wrong-id", GOLD_TEXT)],
        {"c1": GOLD_TEXT},
        threshold=0.8,
    )

    assert scores.hit is False


def test_missing_gold_text_without_ids_does_not_match() -> None:
    scores = compute_retrieval(
        _case(),
        [_context(None, GOLD_TEXT)],
        {},
        threshold=0.8,
    )

    assert scores.hit is False


def test_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError):
        match_contexts([_context(None)], ["c1"], {"c1": GOLD_TEXT}, threshold=1.5)


def test_aggregate_retrieval_computes_means_and_counts() -> None:
    hit = _result(_case(), hit=True, reciprocal_rank=1.0, context_recall=1.0)
    miss = _result(
        _case(gold_chunk_ids=["c1", "c2"]),
        hit=False,
        reciprocal_rank=0.0,
        context_recall=0.5,
    )
    unanswerable = _result(
        _case(question_type=QuestionType.UNANSWERABLE, gold_chunk_ids=[]),
        applicable=False,
        contexts=[_context("c1")],
    )
    failed = _result(_case(), error="HTTP 500: boom", contexts=[])

    summary = aggregate_retrieval([hit, miss, unanswerable, failed])

    assert summary.cases == 4
    assert summary.answerable_cases == 3
    assert summary.unanswerable_cases == 1
    assert summary.hit_rate == pytest.approx(1.0 / 3.0)
    assert summary.mrr == pytest.approx(1.0 / 3.0)
    assert summary.context_recall == pytest.approx(0.5)
    assert summary.unanswerable_with_retrieval == 1
    assert summary.pipeline_errors == 1


def test_aggregate_retrieval_returns_none_for_empty_rates() -> None:
    summary = aggregate_retrieval([])

    assert summary.cases == 0
    assert summary.hit_rate is None
    assert summary.mrr is None
    assert summary.context_recall is None
