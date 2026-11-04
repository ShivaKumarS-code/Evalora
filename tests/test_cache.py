from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from evalora.llm.base import LLMClient, Message, Role, make_cache_key
from evalora.llm.cache import ResponseCache, hash_payload
from evalora.llm.fake import FakeProvider


class Schema(BaseModel):
    answer: str


def test_hash_payload_is_key_order_insensitive() -> None:
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_cache_key_changes_with_inputs() -> None:
    messages = [Message(role=Role.USER, content="hi")]
    base = {
        "provider": "fake",
        "model": "m",
        "messages": messages,
        "temperature": 0.0,
        "response_schema": None,
    }
    key = make_cache_key(**base)
    assert key == make_cache_key(**base)
    assert key != make_cache_key(**(base | {"temperature": 1.0}))
    assert key != make_cache_key(**(base | {"response_schema": Schema}))
    assert key != make_cache_key(**(base | {"messages": [Message(role=Role.USER, content="yo")]}))


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    cache.put("key", provider="fake", model="m", payload='{"text": "cached"}')
    assert cache.get("key") == '{"text": "cached"}'
    assert cache.get("missing") is None


async def test_same_prompt_hits_cache(tmp_path: Path) -> None:
    provider = FakeProvider(default="hello")
    client = LLMClient(provider, cache=ResponseCache(tmp_path / "cache.db"))
    messages = [Message(role=Role.USER, content="what is rag?")]

    first = await client.complete(messages, temperature=0.0, max_tokens=16)
    second = await client.complete(messages, temperature=0.0, max_tokens=16)

    assert provider.call_count == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == "hello"


async def test_without_cache_calls_provider_every_time() -> None:
    provider = FakeProvider(default="hello")
    client = LLMClient(provider)
    messages = [Message(role=Role.USER, content="what is rag?")]

    await client.complete(messages)
    await client.complete(messages)

    assert provider.call_count == 2


async def test_different_temperature_misses_cache(tmp_path: Path) -> None:
    provider = FakeProvider(default="hello")
    client = LLMClient(provider, cache=ResponseCache(tmp_path / "cache.db"))
    messages = [Message(role=Role.USER, content="what is rag?")]

    await client.complete(messages, temperature=0.0)
    await client.complete(messages, temperature=0.7)

    assert provider.call_count == 2
