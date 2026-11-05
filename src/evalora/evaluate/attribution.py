"""Rule-based failure attribution in the spec's priority order."""

from __future__ import annotations

from evalora.schemas import CaseResult, FailureCategory, QuestionType


def attribute_failure(result: CaseResult) -> FailureCategory | None:
    if result.response.error is not None:
        return FailureCategory.PIPELINE_ERROR

    if result.case.type == QuestionType.UNANSWERABLE:
        if not result.abstained and result.claim_verdicts:
            return FailureCategory.FAILED_ABSTENTION
        return None

    if result.abstained:
        if result.retrieval.hit:
            return FailureCategory.WRONG_ABSTENTION
        return None

    if result.correct is None:
        return None

    if not result.retrieval.hit:
        if result.correct is False:
            return FailureCategory.RETRIEVAL_FAILURE
        return None

    if any(not verdict.supported for verdict in result.claim_verdicts):
        return FailureCategory.HALLUCINATION

    if result.correct is False:
        return FailureCategory.GENERATION_FAILURE

    return None
