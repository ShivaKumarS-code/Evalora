from __future__ import annotations

import pytest

from evalora.llm.base import Message, Role
from evalora.llm.fake import FakeProvider


async def test_queue_is_returned_in_order() -> None:
    provider = FakeProvider(["one", "two"])
    messages = [Message(role=Role.USER, content="hi")]

    first = await provider.complete(messages)
    second = await provider.complete(messages)

    assert first.text == "one"
    assert second.text == "two"
    assert provider.call_count == 2
    assert provider.calls[0] == messages


async def test_prompt_keyed_response() -> None:
    messages = [Message(role=Role.USER, content="hi")]
    provider = FakeProvider(prompt_responses={FakeProvider.prompt_key(messages): "keyed"})

    response = await provider.complete(messages)

    assert response.text == "keyed"


async def test_exception_script_is_raised() -> None:
    error = RuntimeError("boom")
    provider = FakeProvider([error])

    with pytest.raises(RuntimeError, match="boom"):
        await provider.complete([Message(role=Role.USER, content="hi")])


async def test_default_response_after_queue_is_empty() -> None:
    provider = FakeProvider([], default="fallback")
    response = await provider.complete([Message(role=Role.USER, content="hi")])
    assert response.text == "fallback"
