"""Deterministic in-memory provider for tests."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel

from evalora.llm.base import LLMResponse, Message
from evalora.llm.costs import make_usage

DEFAULT_TEXT = "fake response"

ResponseHandler = Callable[[Sequence[Message], type[BaseModel] | None], str | Exception]


class FakeProvider:
    name = "fake"

    def __init__(
        self,
        responses: Sequence[str | Exception] | None = None,
        *,
        prompt_responses: Mapping[str, str | Exception] | None = None,
        default: str = DEFAULT_TEXT,
        model: str = "fake-1",
        handler: ResponseHandler | None = None,
    ) -> None:
        self.model = model
        self._responses: deque[str | Exception] = deque(responses or ())
        self._prompt_responses = dict(prompt_responses or {})
        self._default = default
        self._handler = handler
        self.call_count = 0
        self.calls: list[list[Message]] = []

    @staticmethod
    def prompt_key(messages: Sequence[Message]) -> str:
        payload = [message.model_dump(mode="json") for message in messages]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.calls.append(list(messages))
        key = self.prompt_key(messages)
        if self._handler is not None:
            item = self._handler(messages, response_schema)
        elif key in self._prompt_responses:
            item = self._prompt_responses[key]
        elif self._responses:
            item = self._responses.popleft()
        else:
            item = self._default
        if isinstance(item, Exception):
            raise item
        prompt_tokens = sum(len(message.content.split()) for message in messages)
        completion_tokens = len(item.split())
        return LLMResponse(
            text=item,
            provider=self.name,
            model=self.model,
            usage=make_usage(
                self.name,
                self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )
