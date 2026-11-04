"""Google Gemini provider using the REST API through httpx."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, cast

import httpx
from pydantic import BaseModel

from evalora.llm.base import (
    AuthError,
    LLMResponse,
    Message,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
    Role,
    ServerError,
)
from evalora.llm.costs import make_usage

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        if not key:
            raise AuthError("GEMINI_API_KEY is not set")
        self.model = model
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, temperature, max_tokens, response_schema)
        url = f"{self._base_url}/v1beta/models/{self.model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                response = await client.post(url, params={"key": self._api_key}, json=payload)
        except httpx.HTTPError as exc:
            raise RetryableProviderError(f"Gemini request failed: {exc}") from exc
        self._raise_for_status(response)
        return self._parse_response(response)

    @staticmethod
    def _build_payload(
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        response_schema: type[BaseModel] | None,
    ) -> dict[str, object]:
        contents: list[dict[str, object]] = []
        system_parts: list[str] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                system_parts.append(message.content)
                continue
            role = "model" if message.role == Role.ASSISTANT else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        generation_config: dict[str, object] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
        payload: dict[str, object] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        status = response.status_code
        detail = response.text[:500]
        message = f"Gemini returned HTTP {status}: {detail}"
        if status == 429:
            raise RateLimitError(message, status_code=status)
        if 500 <= status < 600:
            raise ServerError(message, status_code=status)
        if status in (401, 403):
            raise AuthError(message, status_code=status)
        raise ProviderError(message, status_code=status)

    def _parse_response(self, response: httpx.Response) -> LLMResponse:
        try:
            data = cast(dict[str, Any], response.json())
        except ValueError as exc:
            raise ProviderError("Gemini returned a non-JSON response") from exc
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError("Gemini response contained no candidates")
        first = candidates[0]
        if not isinstance(first, dict):
            raise ProviderError("Gemini response contained a malformed candidate")
        content = first.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        text = "".join(str(part.get("text", "")) for part in parts or [] if isinstance(part, dict))
        metadata = data.get("usageMetadata")
        usage_meta = metadata if isinstance(metadata, dict) else {}
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0))
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0))
        total_tokens = int(usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens))
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            usage=make_usage(
                self.name,
                self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )
