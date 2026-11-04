"""Structured LLM output: request JSON, validate, and repair on failure."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from evalora.llm.base import LLMClient, LLMResponse, Message, Role, StructuredOutputError

ModelT = TypeVar("ModelT", bound=BaseModel)

REPAIR_PROMPT = (
    "The previous response could not be parsed as valid JSON matching the required schema.\n"
    "Validation error:\n{error}\n\n"
    "Return only corrected JSON that matches the schema. Do not include prose or code fences."
)


def extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    stripped = stripped.strip()
    object_start = stripped.find("{")
    array_start = stripped.find("[")
    starts = [index for index in (object_start, array_start) if index != -1]
    if starts:
        start = min(starts)
        closing = "}" if stripped[start] == "{" else "]"
        end = stripped.rfind(closing)
        if end > start:
            stripped = stripped[start : end + 1]
    return stripped


async def complete_json_with_usage(
    client: LLMClient,
    messages: Sequence[Message],
    schema: type[ModelT],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_attempts: int = 3,
) -> tuple[ModelT, list[LLMResponse]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    conversation = list(messages)
    responses: list[LLMResponse] = []
    last_error = ""
    for _ in range(max_attempts):
        response = await client.complete(
            conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema,
        )
        responses.append(response)
        try:
            return schema.model_validate_json(extract_json(response.text)), responses
        except ValueError as exc:
            last_error = str(exc)
            conversation.append(Message(role=Role.ASSISTANT, content=response.text))
            conversation.append(
                Message(role=Role.USER, content=REPAIR_PROMPT.format(error=last_error))
            )
    raise StructuredOutputError(attempts=max_attempts, last_error=last_error)


async def complete_json(
    client: LLMClient,
    messages: Sequence[Message],
    schema: type[ModelT],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_attempts: int = 3,
) -> ModelT:
    result, _ = await complete_json_with_usage(
        client,
        messages,
        schema,
        temperature=temperature,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
    )
    return result
