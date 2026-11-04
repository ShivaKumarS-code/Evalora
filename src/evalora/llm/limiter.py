"""Async concurrency limiting and token-bucket rate limiting."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from types import TracebackType

SleepFn = Callable[[float], Awaitable[None]]


class AsyncRateLimiter:
    """Bounds concurrent in-flight calls and paces them to a requests-per-minute budget."""

    def __init__(
        self,
        *,
        max_concurrency: int = 4,
        requests_per_minute: int = 60,
        clock: Callable[[], float] = time.monotonic,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1")
        self.max_concurrency = max_concurrency
        self.requests_per_minute = requests_per_minute
        self._rate_per_second = requests_per_minute / 60.0
        self._capacity = float(requests_per_minute)
        self._tokens = self._capacity
        self._updated_at = clock()
        self._clock = clock
        self._sleep = sleep
        self._token_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        try:
            await self._acquire_token()
        except BaseException:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()

    async def __aenter__(self) -> AsyncRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    async def _acquire_token(self) -> None:
        while True:
            async with self._token_lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
                self._updated_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_s = (1.0 - self._tokens) / self._rate_per_second
            await self._sleep(wait_s)
