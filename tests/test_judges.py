from __future__ import annotations

from pathlib import Path

from evalora.evaluate.judges import judge_case
from evalora.llm.base import LLMClient
from evalora.llm.cache import ResponseCache
from evalora.llm.fake import FakeProvider
from evalora.schemas import (
    CorrectnessLabel,
    FailureCategory,
    QuestionType,
    RetrievedContext,
)
from tests.factories import make_case, make_result


async def test_answerable_case_is_fully_judged() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
            '{"label": "correct", "reason": "matches the gold answer"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)
    result = make_result(
        "case-1",
        QuestionType.SINGLE_HOP,
        answer="TechMart was founded in 1994.",
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland.")],
        hit=True,
        reciprocal_rank=1.0,
        context_recall=1.0,
    )

    judged = await judge_case(result, client)

    assert judged.faithfulness == 1.0
    assert judged.claim_verdicts[0].supported is True
    assert judged.correct is True
    assert judged.correctness_label == CorrectnessLabel.CORRECT
    assert judged.abstained is False
    assert judged.failure_category is None
    assert judged.usage.calls == 3
    assert provider.call_count == 3


async def test_pipeline_error_skips_the_judges() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)
    result = make_result("case-1", answer="", error="HTTP 500")

    judged = await judge_case(result, client)

    assert judged.failure_category == FailureCategory.PIPELINE_ERROR
    assert provider.call_count == 0


async def test_unanswerable_answered_with_claims_is_failed_abstention() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["Mars has two moons."]}',
            '{"supported": false, "evidence_context_index": null, "rationale": "absent"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)
    result = make_result(
        "case-1",
        QuestionType.UNANSWERABLE,
        answer="Mars has two moons.",
        applicable=False,
        abstained=False,
    )

    judged = await judge_case(result, client)

    assert judged.failure_category == FailureCategory.FAILED_ABSTENTION
    assert judged.usage.calls == 2
    assert provider.call_count == 2


async def test_answerable_refusal_with_gold_chunk_is_wrong_abstention() -> None:
    provider = FakeProvider(
        [
            '{"claims": []}',
            '{"label": "incorrect", "reason": "does not answer"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)
    result = make_result(
        "case-1",
        QuestionType.SINGLE_HOP,
        answer="I don't know.",
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland.")],
        hit=True,
        reciprocal_rank=1.0,
        context_recall=1.0,
    )

    judged = await judge_case(result, client)

    assert judged.abstained is True
    assert judged.failure_category == FailureCategory.WRONG_ABSTENTION
    assert provider.call_count == 2


async def test_unanswerable_refusal_passes() -> None:
    provider = FakeProvider(['{"claims": []}'])
    client = LLMClient(provider, max_retries=0)
    result = make_result(
        "case-1",
        QuestionType.UNANSWERABLE,
        answer="I don't know.",
        applicable=False,
        abstained=False,
    )

    judged = await judge_case(result, client)

    assert judged.abstained is True
    assert judged.failure_category is None
    assert provider.call_count == 1


def test_make_case_keeps_unanswerable_gold_empty() -> None:
    case = make_case("case-1", QuestionType.UNANSWERABLE)

    assert case.gold_answer is None
    assert case.gold_chunk_ids == []


async def test_cached_judge_calls_do_not_add_usage(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
            '{"label": "correct", "reason": "matches"}',
        ]
    )
    client = LLMClient(
        provider,
        cache=ResponseCache(tmp_path / "cache.db"),
        max_retries=0,
    )
    result = make_result(
        "case-1",
        QuestionType.SINGLE_HOP,
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland.")],
        hit=True,
    )

    first = await judge_case(result, client)
    second = await judge_case(result, client)

    assert first.usage.calls == 3
    assert second.usage.calls == 0
    assert provider.call_count == 3
