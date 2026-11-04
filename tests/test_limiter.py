from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from evalora.llm.base import (
    AuthError,
    LLMClient,
    LLMResponse,
    Message,
    RateLimitError,
    RetriesExhaustedError,
    Role,
)
from evalora.llm.fake import FakeProvider
from evalora.llm.limiter import AsyncRateLimiter


async def test_429_backs_off_and_then_succeeds() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    provider = FakeProvider([RateLimitError("429", status_code=429), "recovered"])
    client = LLMClient(
        provider,
        max_retries=2,
        base_delay=1.0,
        sleep=fake_sleep,
        rng=lambda: 0.5,
    )

    response = await client.complete([Message(role=Role.USER, content="hi")])

    assert response.text == "recovered"
    assert provider.call_count == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(1.0)


async def test_retries_exhausted_raises_typed_error() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    provider = FakeProvider([RateLimitError("429") for _ in range(3)])
    client = LLMClient(provider, max_retries=2, base_delay=1.0, sleep=fake_sleep, rng=lambda: 0.5)

    with pytest.raises(RetriesExhaustedError) as info:
        await client.complete([Message(role=Role.USER, content="hi")])

    assert info.value.attempts == 3
    assert provider.call_count == 3
    assert len(sleeps) == 2


async def test_non_retryable_error_is_not_retried() -> None:
    provider = FakeProvider([AuthError("bad key")])
    client = LLMClient(provider, max_retries=3)

    with pytest.raises(AuthError):
        await client.complete([Message(role=Role.USER, content="hi")])

    assert provider.call_count == 1


async def test_concurrency_is_bounded() -> None:
    provider = TrackingProvider()
    limiter = AsyncRateLimiter(max_concurrency=2, requests_per_minute=6000)
    client = LLMClient(provider, limiter=limiter)

    await asyncio.gather(
        *(client.complete([Message(role=Role.USER, content="x")]) for _ in range(6))
    )

    assert provider.max_active <= 2
    assert provider.calls == 6


async def test_token_bucket_paces_requests() -> None:
    clock = FakeClock()
    limiter = AsyncRateLimiter(
        max_concurrency=2,
        requests_per_minute=1,
        clock=clock,
        sleep=clock.sleep,
    )

    await limiter.acquire()
    limiter.release()
    await limiter.acquire()
    limiter.release()

    assert clock.now == pytest.approx(60.0)


def test_limiter_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        AsyncRateLimiter(max_concurrency=0)
    with pytest.raises(ValueError):
        AsyncRateLimiter(requests_per_minute=0)


class TrackingProvider:
    name = "tracking"
    model = "tracking-1"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.calls += 1
        self.active -= 1
        return LLMResponse(text="ok", provider=self.name, model=self.model)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
