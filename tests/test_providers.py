from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from evalora.llm.base import (
    AuthError,
    Message,
    ProviderError,
    RateLimitError,
    Role,
    ServerError,
)
from evalora.llm.gemini import GeminiProvider

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
)


class JsonSchema(BaseModel):
    answer: str


@respx.mock
async def test_gemini_success_parses_text_and_usage() -> None:
    route = respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "Gemini answer"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 19,
                },
            },
        )
    )
    provider = GeminiProvider(api_key="test-key")

    response = await provider.complete(
        [Message(role=Role.USER, content="hi")],
        temperature=0.0,
        max_tokens=64,
    )

    assert response.text == "Gemini answer"
    assert response.provider == "gemini"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 19
    request = route.calls[0].request
    assert request.url.params["key"] == "test-key"
    body = json.loads(request.content)
    assert body["contents"][0]["parts"][0]["text"] == "hi"


@respx.mock
async def test_gemini_moves_system_messages_and_requests_json() -> None:
    route = respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": []}}]})
    )
    provider = GeminiProvider(api_key="test-key")

    await provider.complete(
        [
            Message(role=Role.SYSTEM, content="be terse"),
            Message(role=Role.USER, content="hi"),
        ],
        temperature=0.0,
        max_tokens=64,
        response_schema=JsonSchema,
    )

    body = json.loads(route.calls[0].request.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, RateLimitError),
        (500, ServerError),
        (401, AuthError),
        (400, ProviderError),
    ],
)
@respx.mock
async def test_gemini_maps_http_errors(status: int, error: type[Exception]) -> None:
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(status, text="nope"))
    provider = GeminiProvider(api_key="test-key")

    with pytest.raises(error):
        await provider.complete([Message(role=Role.USER, content="hi")])


def test_missing_gemini_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(AuthError, match="GEMINI_API_KEY"):
        GeminiProvider()
