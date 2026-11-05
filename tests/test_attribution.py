from __future__ import annotations

from evalora.evaluate.attribution import attribute_failure
from evalora.schemas import ClaimVerdict, FailureCategory, QuestionType
from tests.factories import make_result


def _claim(supported: bool) -> ClaimVerdict:
    return ClaimVerdict(
        claim="a claim",
        supported=supported,
        evidence_context_index=0 if supported else None,
        rationale="because",
    )


def test_pipeline_error_has_highest_priority() -> None:
    result = make_result(
        error="HTTP 500",
        abstained=False,
        hit=True,
        correct=True,
        claim_verdicts=[_claim(True)],
    )

    assert attribute_failure(result) == FailureCategory.PIPELINE_ERROR


def test_unanswerable_with_substantive_claims_is_failed_abstention() -> None:
    result = make_result(
        question_type=QuestionType.UNANSWERABLE,
        abstained=False,
        claim_verdicts=[_claim(False)],
    )

    assert attribute_failure(result) == FailureCategory.FAILED_ABSTENTION


def test_unanswerable_abstention_passes() -> None:
    result = make_result(
        question_type=QuestionType.UNANSWERABLE,
        abstained=True,
        claim_verdicts=[],
    )

    assert attribute_failure(result) is None


def test_answerable_abstention_with_gold_chunk_is_wrong_abstention() -> None:
    result = make_result(abstained=True, hit=True)

    assert attribute_failure(result) == FailureCategory.WRONG_ABSTENTION


def test_answerable_abstention_without_gold_chunk_has_no_label() -> None:
    result = make_result(abstained=True, hit=False)

    assert attribute_failure(result) is None


def test_no_gold_chunk_and_wrong_answer_is_retrieval_failure() -> None:
    result = make_result(abstained=False, hit=False, correct=False)

    assert attribute_failure(result) == FailureCategory.RETRIEVAL_FAILURE


def test_no_gold_chunk_and_correct_answer_has_no_label() -> None:
    result = make_result(abstained=False, hit=False, correct=True)

    assert attribute_failure(result) is None


def test_hallucination_beats_generation_failure() -> None:
    result = make_result(
        abstained=False,
        hit=True,
        correct=False,
        claim_verdicts=[_claim(False)],
    )

    assert attribute_failure(result) == FailureCategory.HALLUCINATION


def test_hallucination_even_when_answer_is_marked_correct() -> None:
    result = make_result(
        abstained=False,
        hit=True,
        correct=True,
        claim_verdicts=[_claim(False)],
    )

    assert attribute_failure(result) == FailureCategory.HALLUCINATION


def test_supported_claims_and_wrong_answer_is_generation_failure() -> None:
    result = make_result(
        abstained=False,
        hit=True,
        correct=False,
        claim_verdicts=[_claim(True)],
    )

    assert attribute_failure(result) == FailureCategory.GENERATION_FAILURE


def test_supported_claims_and_correct_answer_passes() -> None:
    result = make_result(
        abstained=False,
        hit=True,
        correct=True,
        claim_verdicts=[_claim(True)],
    )

    assert attribute_failure(result) is None


def test_unjudged_correctness_has_no_label() -> None:
    result = make_result(abstained=False, hit=True, correct=None)

    assert attribute_failure(result) is None
