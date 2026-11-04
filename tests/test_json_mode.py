from __future__ import annotations

import pytest
from pydantic import BaseModel

from evalora.llm.base import LLMClient, Message, Role, StructuredOutputError
from evalora.llm.fake import FakeProvider
from evalora.llm.json_mode import complete_json, extract_json


class StructuredAnswer(BaseModel):
    answer: str
    confidence: int


def test_extract_json_handles_plain_json() -> None:
    assert extract_json('{"answer": "yes"}') == '{"answer": "yes"}'


def test_extract_json_handles_code_fences() -> None:
    assert extract_json('```json\n{"answer": "yes"}\n```') == '{"answer": "yes"}'


def test_extract_json_handles_surrounding_prose() -> None:
    assert extract_json('Here you go: {"answer": "yes"} thanks') == '{"answer": "yes"}'


async def test_valid_json_returns_model() -> None:
    provider = FakeProvider(['{"answer": "yes", "confidence": 2}'])
    client = LLMClient(provider)

    result = await complete_json(
        client,
        [Message(role=Role.USER, content="prompt")],
        StructuredAnswer,
    )

    assert result == StructuredAnswer(answer="yes", confidence=2)
    assert provider.call_count == 1


async def test_fenced_json_is_accepted() -> None:
    provider = FakeProvider(['```json\n{"answer": "yes", "confidence": 2}\n```'])
    client = LLMClient(provider)

    result = await complete_json(
        client,
        [Message(role=Role.USER, content="prompt")],
        StructuredAnswer,
    )

    assert result.answer == "yes"


async def test_malformed_json_triggers_repair_retry() -> None:
    provider = FakeProvider(["not json", '{"answer": "yes", "confidence": 2}'])
    client = LLMClient(provider)

    result = await complete_json(
        client,
        [Message(role=Role.USER, content="prompt")],
        StructuredAnswer,
    )

    assert result.answer == "yes"
    assert provider.call_count == 2
    repair_turn = provider.calls[1]
    assert repair_turn[-2].role == Role.ASSISTANT
    assert repair_turn[-2].content == "not json"
    assert "Validation error" in repair_turn[-1].content


async def test_schema_validation_error_is_appended_to_prompt() -> None:
    provider = FakeProvider(
        ['{"answer": "yes", "confidence": "high"}', '{"answer": "yes", "confidence": 2}']
    )
    client = LLMClient(provider)

    result = await complete_json(
        client,
        [Message(role=Role.USER, content="prompt")],
        StructuredAnswer,
    )

    assert result.confidence == 2
    assert "confidence" in provider.calls[1][-1].content


async def test_persistent_failure_raises_typed_error() -> None:
    provider = FakeProvider(["bad", "still bad", "nope"])
    client = LLMClient(provider)

    with pytest.raises(StructuredOutputError) as info:
        await complete_json(
            client,
            [Message(role=Role.USER, content="prompt")],
            StructuredAnswer,
        )

    assert info.value.attempts == 3
    assert provider.call_count == 3
    assert info.value.last_error
