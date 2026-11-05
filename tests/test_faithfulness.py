from __future__ import annotations

import pytest

from evalora.evaluate.faithfulness import judge_faithfulness
from evalora.llm.base import LLMClient, StructuredOutputError
from evalora.llm.fake import FakeProvider
from evalora.schemas import RetrievedContext


def _contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland."),
        RetrievedContext(id="c2", text="The first store opened in March 1994."),
    ]


async def test_all_claims_supported() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994.", "The first store opened in 1994."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
            '{"supported": true, "evidence_context_index": 1, "rationale": "stated"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(
        client,
        "TechMart was founded in 1994 and opened its first store that year.",
        _contexts(),
    )

    assert result.faithfulness == 1.0
    assert len(result.claim_verdicts) == 2
    assert all(verdict.supported for verdict in result.claim_verdicts)
    assert len(result.usages) == 3
    assert provider.call_count == 3


async def test_mixed_claims() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994.", "TechMart has 500 stores."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
            '{"supported": false, "evidence_context_index": null, "rationale": "not in context"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(client, "answer", _contexts())

    assert result.faithfulness == 0.5
    assert [verdict.supported for verdict in result.claim_verdicts] == [True, False]


async def test_zero_claims_gives_null_faithfulness() -> None:
    provider = FakeProvider(['{"claims": []}'])
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(client, "I don't know.", _contexts())

    assert result.faithfulness is None
    assert result.claim_verdicts == []
    assert len(result.usages) == 1
    assert provider.call_count == 1


async def test_empty_answer_makes_no_llm_calls() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(client, "   ", _contexts())

    assert result.faithfulness is None
    assert result.claim_verdicts == []
    assert provider.call_count == 0


async def test_blank_claims_are_filtered_out() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["  ", "TechMart was founded in 1994."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(client, "answer", _contexts())

    assert len(result.claim_verdicts) == 1
    assert result.claim_verdicts[0].claim == "TechMart was founded in 1994."


async def test_out_of_range_evidence_index_is_nulled() -> None:
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994."]}',
            '{"supported": true, "evidence_context_index": 99, "rationale": "stated"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    result = await judge_faithfulness(client, "answer", _contexts())

    assert result.claim_verdicts[0].evidence_context_index is None


async def test_persistent_json_failure_raises() -> None:
    provider = FakeProvider(["bad", "still bad", "nope"])
    client = LLMClient(provider, max_retries=0)

    with pytest.raises(StructuredOutputError) as info:
        await judge_faithfulness(client, "answer", _contexts())

    assert info.value.attempts == 3
    assert provider.call_count == 3
