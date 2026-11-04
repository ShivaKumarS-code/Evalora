"""LLM provider protocol, shared response types, and the single client wrapper."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field, ValidationError

from evalora.llm.cache import ResponseCache, hash_payload
from evalora.llm.costs import TokenUsage
from evalora.llm.limiter import AsyncRateLimiter, SleepFn


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = False


class LLMError(Exception):
    """Base class for LLM-layer failures."""


class ProviderError(LLMError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetryableProviderError(ProviderError):
    """A failure worth retrying: rate limits, server errors, and transport errors."""


class RateLimitError(RetryableProviderError):
    pass


class ServerError(RetryableProviderError):
    pass


class AuthError(ProviderError):
    pass


class RetriesExhaustedError(LLMError):
    def __init__(self, *, attempts: int, last_error: Exception) -> None:
        super().__init__(f"provider call failed after {attempts} attempt(s): {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class StructuredOutputError(LLMError):
    def __init__(self, *, attempts: int, last_error: str) -> None:
        super().__init__(f"structured output failed after {attempts} attempt(s): {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class LLMProvider(Protocol):
    name: str
    model: str

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse: ...


T = TypeVar("T")


def make_cache_key(
    *,
    provider: str,
    model: str,
    messages: Sequence[Message],
    temperature: float,
    response_schema: type[BaseModel] | None,
) -> str:
    return hash_payload(
        {
            "provider": provider,
            "model": model,
            "messages": [message.model_dump(mode="json") for message in messages],
            "temperature": temperature,
            "schema": response_schema.model_json_schema() if response_schema is not None else None,
        }
    )


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    sleep: SleepFn = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except RetryableProviderError as exc:
            if attempt == attempts:
                raise RetriesExhaustedError(attempts=attempt, last_error=exc) from exc
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = min(max_delay, delay * (0.5 + rng()))
            await sleep(delay)
    raise AssertionError("retry loop exited without returning or raising")


class LLMClient:
    """The only entry point for LLM calls: cache, rate limit, retry, then provider."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        cache: ResponseCache | None = None,
        limiter: AsyncRateLimiter | None = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        sleep: SleepFn = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.provider = provider
        self._cache = cache
        self._limiter = limiter
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._rng = rng

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        key = make_cache_key(
            provider=self.provider.name,
            model=self.provider.model,
            messages=messages,
            temperature=temperature,
            response_schema=response_schema,
        )
        if self._cache is not None:
            cached_payload = self._cache.get(key)
            if cached_payload is not None:
                try:
                    cached_response = LLMResponse.model_validate_json(cached_payload)
                except ValidationError:
                    cached_response = None
                if cached_response is not None:
                    return cached_response.model_copy(update={"cached": True})

        async def attempt() -> LLMResponse:
            if self._limiter is None:
                return await self.provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                )
            async with self._limiter:
                return await self.provider.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                )

        response = await retry_async(
            attempt,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            sleep=self._sleep,
            rng=self._rng,
        )
        if self._cache is not None:
            self._cache.put(
                key,
                provider=self.provider.name,
                model=self.provider.model,
                payload=response.model_dump_json(),
            )
        return response
