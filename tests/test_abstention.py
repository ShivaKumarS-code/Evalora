from __future__ import annotations

import pytest

from evalora.evaluate.abstention import detect_abstention, detect_abstention_rule_based
from evalora.llm.base import LLMClient
from evalora.llm.fake import FakeProvider


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I do not know the answer.",
        "The provided context does not contain that information.",
        "There is not enough information to answer.",
        "I cannot determine that from the context.",
        "No information is available.",
        "That fact is not mentioned in the provided context.",
        "I'm not sure.",
        "I don't have enough information to answer.",
        "",
    ],
)
def test_rule_based_abstention_matches(answer: str) -> None:
    assert detect_abstention_rule_based(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "TechMart was founded in 1994.",
        "1994",
        "The context says the store opened in March 1994.",
    ],
)
def test_rule_based_abstention_rejects_substantive_answers(answer: str) -> None:
    assert detect_abstention_rule_based(answer) is False


async def test_substantive_answer_uses_no_llm() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)

    result = await detect_abstention(
        client,
        question="When was TechMart founded?",
        answer="TechMart was founded in 1994.",
    )

    assert result.abstained is False
    assert result.method == "default"
    assert provider.call_count == 0


async def test_rule_match_uses_no_llm() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)

    result = await detect_abstention(
        client,
        question="When was TechMart founded?",
        answer="I don't know.",
    )

    assert result.abstained is True
    assert result.method == "rule"
    assert provider.call_count == 0


async def test_ambiguous_answer_uses_llm_fallback() -> None:
    provider = FakeProvider(['{"abstained": true, "reason": "hedges"}'])
    client = LLMClient(provider, max_retries=0)

    result = await detect_abstention(
        client,
        question="When was TechMart founded?",
        answer="Maybe, it depends.",
        use_llm_fallback=True,
    )

    assert result.abstained is True
    assert result.method == "llm"
    assert result.reason == "hedges"
    assert len(result.usages) == 1
    assert provider.call_count == 1


async def test_ambiguous_answer_defaults_without_fallback() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)

    result = await detect_abstention(
        client,
        question="When was TechMart founded?",
        answer="Maybe, it depends.",
        use_llm_fallback=False,
    )

    assert result.abstained is False
    assert result.method == "default"
    assert provider.call_count == 0


async def test_empty_answer_is_rule_based_abstention() -> None:
    provider = FakeProvider([])
    client = LLMClient(provider, max_retries=0)

    result = await detect_abstention(
        client,
        question="When was TechMart founded?",
        answer="",
    )

    assert result.abstained is True
    assert result.method == "rule"
    assert result.reason == "empty answer"
    assert provider.call_count == 0
