"""Per-case judge orchestration."""

from __future__ import annotations

from evalora.evaluate.abstention import detect_abstention
from evalora.evaluate.attribution import attribute_failure
from evalora.evaluate.correctness import judge_correctness
from evalora.evaluate.faithfulness import judge_faithfulness
from evalora.llm.base import LLMClient
from evalora.llm.costs import TokenUsage, aggregate_usage
from evalora.schemas import CaseResult, FailureCategory, QuestionType


async def judge_case(result: CaseResult, client: LLMClient) -> CaseResult:
    if result.response.error is not None:
        return result.model_copy(update={"failure_category": FailureCategory.PIPELINE_ERROR})

    usages: list[TokenUsage] = []

    faithfulness = await judge_faithfulness(
        client,
        result.response.answer,
        result.response.contexts,
    )
    usages.extend(faithfulness.usages)

    abstention = await detect_abstention(
        client,
        question=result.case.question,
        answer=result.response.answer,
        use_llm_fallback=True,
    )
    usages.extend(abstention.usages)

    correctness = None
    if result.case.type != QuestionType.UNANSWERABLE and result.case.gold_answer is not None:
        correctness = await judge_correctness(
            client,
            question=result.case.question,
            gold_answer=result.case.gold_answer,
            answer=result.response.answer,
        )
        usages.extend(correctness.usages)

    updated = result.model_copy(
        update={
            "faithfulness": faithfulness.faithfulness,
            "claim_verdicts": faithfulness.claim_verdicts,
            "correct": None if correctness is None else correctness.correct,
            "correctness_label": None if correctness is None else correctness.label,
            "abstained": abstention.abstained,
            "usage": aggregate_usage(usages),
        }
    )
    return updated.model_copy(update={"failure_category": attribute_failure(updated)})
