from __future__ import annotations

from datetime import UTC, datetime

from evalora.llm.costs import UsageTotals
from evalora.schemas import (
    CaseResult,
    ClaimVerdict,
    CorrectnessLabel,
    FailureCategory,
    PipelineResponse,
    QuestionType,
    RetrievalScores,
    RetrievedContext,
    TestCase,
)
from evalora.store.db import RunRecord


def make_case(
    case_id: str = "case-1",
    question_type: QuestionType = QuestionType.SINGLE_HOP,
    *,
    question: str = "When was TechMart founded?",
    gold_answer: str | None = None,
    gold_chunk_ids: list[str] | None = None,
) -> TestCase:
    answer = None if question_type == QuestionType.UNANSWERABLE else (gold_answer or "1994")
    return TestCase(
        case_id=case_id,
        type=question_type,
        question=question,
        gold_answer=answer,
        gold_chunk_ids=(
            [] if question_type == QuestionType.UNANSWERABLE else (gold_chunk_ids or ["c1"])
        ),
        source_model="fake",
        answerability_verdict=(
            "unanswerable" if question_type == QuestionType.UNANSWERABLE else "answerable"
        ),
        metadata={},
    )


def make_result(
    case_id: str = "case-1",
    question_type: QuestionType = QuestionType.SINGLE_HOP,
    *,
    answer: str = "TechMart was founded in 1994.",
    contexts: list[RetrievedContext] | None = None,
    error: str | None = None,
    question: str = "When was TechMart founded?",
    applicable: bool = True,
    hit: bool = False,
    reciprocal_rank: float = 0.0,
    context_recall: float = 0.0,
    faithfulness: float | None = None,
    claim_verdicts: list[ClaimVerdict] | None = None,
    correct: bool | None = None,
    correctness_label: CorrectnessLabel | None = None,
    abstained: bool = False,
    failure_category: FailureCategory | None = None,
    latency_ms: float = 1.0,
    usage: UsageTotals | None = None,
    gold_answer: str | None = None,
) -> CaseResult:
    return CaseResult(
        case=make_case(
            case_id,
            question_type,
            question=question,
            gold_answer=gold_answer,
        ),
        response=PipelineResponse(
            answer=answer,
            contexts=contexts or [],
            latency_ms=latency_ms,
            error=error,
        ),
        retrieval=RetrievalScores(
            applicable=applicable,
            hit=hit,
            reciprocal_rank=reciprocal_rank,
            context_recall=context_recall,
        ),
        faithfulness=faithfulness,
        claim_verdicts=claim_verdicts or [],
        correct=correct,
        correctness_label=correctness_label,
        abstained=abstained,
        failure_category=failure_category,
        usage=usage or UsageTotals(),
    )


def make_run_record(
    run_name: str = "run-1",
    *,
    results: list[CaseResult] | None = None,
    test_set_hash: str = "a" * 64,
    config_snapshot: dict[str, object] | None = None,
) -> RunRecord:
    count = len(results or [])
    now = datetime.now(UTC)
    return RunRecord(
        run_name=run_name,
        created_at=now,
        updated_at=now,
        config_snapshot=config_snapshot or {"gate": {"ci_confidence": 0.95}},
        test_set_hash=test_set_hash,
        git_sha="b" * 40,
        case_count=count,
        completed_count=count,
    )
