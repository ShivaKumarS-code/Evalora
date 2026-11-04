"""Pipeline HTTP response mapping and contract validation."""

from __future__ import annotations

from typing import Any

from evalora.config import ResponseMapping
from evalora.schemas import PipelineResponse, RetrievedContext


class ContractError(Exception):
    """Raised when a pipeline response does not match the configured contract."""


def extract_path(payload: object, path: str) -> object:
    normalized = path.strip()
    if normalized.startswith("$"):
        normalized = normalized[1:]
    if normalized.startswith("."):
        normalized = normalized[1:]
    if not normalized:
        raise ContractError("response mapping path must not be empty")

    current: object = payload
    for part in normalized.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise ContractError(f"response mapping path '{path}' was not found")
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise ContractError(
                    f"response mapping path '{path}' expects a list index at '{part}'"
                ) from exc
            if index < 0 or index >= len(current):
                raise ContractError(f"response mapping path '{path}' index {index} is out of range")
            current = current[index]
        else:
            raise ContractError(f"response mapping path '{path}' cannot traverse '{part}'")
    return current


def parse_pipeline_response(
    payload: Any,
    mapping: ResponseMapping,
    latency_ms: float,
) -> PipelineResponse:
    answer = extract_path(payload, mapping.answer)
    if not isinstance(answer, str):
        raise ContractError(f"response mapping '{mapping.answer}' must resolve to a string answer")

    raw_contexts = extract_path(payload, mapping.contexts)
    if not isinstance(raw_contexts, list):
        raise ContractError(
            f"response mapping '{mapping.contexts}' must resolve to a list of contexts"
        )

    contexts: list[RetrievedContext] = []
    for index, raw_context in enumerate(raw_contexts):
        if not isinstance(raw_context, dict):
            raise ContractError(f"context at index {index} must be an object")
        text = raw_context.get("text")
        if not isinstance(text, str):
            raise ContractError(f"context at index {index} must contain a string 'text' field")
        context_id = raw_context.get("id")
        if context_id is not None and not isinstance(context_id, str):
            raise ContractError(f"context at index {index} has a non-string 'id' field")
        contexts.append(RetrievedContext(id=context_id, text=text))

    return PipelineResponse(answer=answer, contexts=contexts, latency_ms=latency_ms)
