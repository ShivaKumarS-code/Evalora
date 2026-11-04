"""Construct the configured LLM client."""

from __future__ import annotations

from evalora.llm.base import LLMClient
from evalora.llm.cache import ResponseCache
from evalora.llm.gemini import GeminiProvider
from evalora.llm.limiter import AsyncRateLimiter


def build_llm_client(
    *,
    model: str,
    no_cache: bool = False,
    max_concurrency: int = 4,
    requests_per_minute: int = 60,
) -> LLMClient:
    provider = GeminiProvider(model=model)
    cache = None if no_cache else ResponseCache()
    limiter = AsyncRateLimiter(
        max_concurrency=max_concurrency,
        requests_per_minute=requests_per_minute,
    )
    return LLMClient(provider, cache=cache, limiter=limiter)
